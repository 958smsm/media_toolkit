import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import unittest

import numpy as np

import ffmpeg_manager


class BitrateTests(unittest.TestCase):
    def test_estimates_expected_h264_bitrate(self) -> None:
        bitrate = ffmpeg_manager.auto_video_kbps(
            1920,
            1080,
            30,
            codec_family="h264",
            quality="medium",
        )
        self.assertEqual(bitrate, 6221)

    def test_normalizes_codec_and_quality_aliases(self) -> None:
        self.assertEqual(ffmpeg_manager.normalize_codec("h265"), "hevc")
        self.assertEqual(
            ffmpeg_manager.normalize_quality("very high"),
            "very-high",
        )

    def test_auto_video_kbps_accepts_legacy_positional_arguments(self) -> None:
        expected = ffmpeg_manager.auto_video_kbps(
            1920, 1080, 30, codec_family="hevc", quality="medium"
        )
        legacy = ffmpeg_manager.auto_video_kbps(
            1920, 1080, 30, "hevc", "medium"
        )
        self.assertEqual(legacy, expected)


class PipeWriterCommandTests(unittest.TestCase):
    def test_builds_command_with_scale_filter_and_overwrite(self) -> None:
        writer = ffmpeg_manager.FFmpegPipeWriter(
            out_path=Path("output.mp4"),
            in_w=1920,
            in_h=1080,
            fps=30,
            v_kbps=4000,
            scale_filter="scale=1280:720",
            overwrite=True,
            ffmpeg_binary="my_ffmpeg",
        )
        cmd = writer._build_command()
        self.assertEqual(cmd[0], "my_ffmpeg")
        self.assertIn("-y", cmd)
        self.assertIn("-vf", cmd)
        vf_index = cmd.index("-vf")
        self.assertEqual(cmd[vf_index + 1], "scale=1280:720")


class FFprobeTests(unittest.TestCase):
    @patch.object(ffmpeg_manager.subprocess, "run")
    def test_returns_video_metadata(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 1920,
                            "height": 1080,
                            "avg_frame_rate": "30000/1001",
                            "r_frame_rate": "30/1",
                            "nb_frames": "375",
                            "duration": "12.5",
                        }
                    ],
                    "format": {"duration": "12.4", "size": "123456"},
                }
            ).encode(),
            stderr=b"",
        )

        metadata = ffmpeg_manager.ffprobe_video(Path("video.mp4"), timeout=12)

        self.assertEqual(metadata["codec"], "h264")
        self.assertEqual((metadata["width"], metadata["height"]), (1920, 1080))
        self.assertAlmostEqual(metadata["fps"], 30000 / 1001)
        self.assertEqual(metadata["frames"], 375)
        self.assertEqual(metadata["duration"], 12.5)
        self.assertEqual(metadata["size"], 123456)
        self.assertEqual(run.call_args.kwargs["timeout"], 12)

    @patch.object(ffmpeg_manager.subprocess, "run")
    def test_raises_ffprobe_error(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=b"",
            stderr=b"invalid media",
        )

        with self.assertRaisesRegex(RuntimeError, "invalid media"):
            ffmpeg_manager.ffprobe_video("broken.mp4")


class HardwareAccelerationTests(unittest.TestCase):
    @patch.object(ffmpeg_manager, "cuda_works_for_file", return_value=True)
    @patch.object(ffmpeg_manager, "nvidia_gpu_present", return_value=True)
    @patch.object(
        ffmpeg_manager,
        "ffmpeg_supports_hwaccel",
        return_value=True,
    )
    def test_auto_enables_working_cuda(
        self,
        _supports_cuda,
        _gpu_present,
        _cuda_works,
    ) -> None:
        arguments = ffmpeg_manager.hardware_acceleration_args(
            Path("video.mp4"),
            "auto",
        )
        self.assertEqual(arguments, ["-hwaccel", "cuda"])

    def test_cpu_mode_does_not_probe_hardware(self) -> None:
        with patch.object(
            ffmpeg_manager,
            "ffmpeg_supports_hwaccel",
        ) as probe:
            arguments = ffmpeg_manager.hardware_acceleration_args(
                Path("video.mp4"),
                "cpu",
            )
        self.assertEqual(arguments, [])
        probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
