from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json, unittest

import video_compressor


class VideoInfoTests(unittest.TestCase):
    @patch.object(video_compressor, "run_capture")
    def test_parses_ffprobe_json(self, run_capture) -> None:
        run_capture.return_value = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30000/1001",
                        "bit_rate": "5268145",
                    }
                ],
                "format": {"duration": "12.5"},
            }
        )
        info = video_compressor.ffprobe_video_info("video.mp4")
        self.assertEqual((info.width, info.height), (1920, 1080))
        self.assertAlmostEqual(info.fps, 29.970, places=3)
        self.assertEqual(info.duration, 12.5)
        self.assertEqual(info.bitrate_kbps, 5268)

    @patch.object(video_compressor, "run_capture")
    def test_unavailable_source_bitrate_does_not_break_quality_encoding(self, run_capture) -> None:
        for bitrate in (None, "N/A", "0", "-1"):
            with self.subTest(bitrate=bitrate):
                run_capture.return_value = json.dumps({
                    "streams": [{
                        "codec_type": "video", "width": 1920, "height": 1080,
                        "avg_frame_rate": "30/1", "bit_rate": bitrate,
                    }],
                    "format": {"duration": "5"},
                })
                self.assertIsNone(video_compressor.ffprobe_video_info("video.mp4").bitrate_kbps)


class ScalingTests(unittest.TestCase):
    def test_scales_to_even_dimensions(self) -> None:
        scale_filter, width, height = video_compressor.build_scale_filter(
            1920,
            1080,
            721,
        )
        self.assertEqual(height, 720)
        self.assertEqual(width % 2, 0)
        self.assertEqual(scale_filter, f"scale={width}:{height}")

    def test_does_not_upscale(self) -> None:
        result = video_compressor.build_scale_filter(640, 480, 1080)
        self.assertEqual(result, (None, 640, 480))


class InputResolutionTests(unittest.TestCase):
    def test_expands_directories_and_text_lists(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            first = nested / "first.mp4"
            second = root / "second.mkv"
            first.touch()
            second.touch()
            list_file = root / "inputs.txt"
            list_file.write_text("second.mkv\n", encoding="utf-8")

            directory_results = video_compressor.resolve_input_videos([root])
            list_results = video_compressor.resolve_input_videos([list_file])

            expected = sorted(
                [first.resolve(), second.resolve()],
                key=lambda path: str(path).casefold(),
            )
            self.assertEqual(directory_results, expected)
            self.assertEqual(list_results, [second.resolve()])


class CompressionTests(unittest.TestCase):
    def test_publishes_completed_output_atomically(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "input.mp4"
            output = root / "output.mp4"
            source.write_bytes(b"source")
            info = video_compressor.VideoInfo(
                width=1280,
                height=720,
                fps=30.0,
                duration=5.0,
                codec="h264",
            )

            def complete_command(command, *_args, **_kwargs) -> None:
                self.assertEqual(command[command.index("-crf") + 1], "26")
                self.assertNotIn("-b:v", command)
                self.assertNotIn("-maxrate", command)
                self.assertNotIn("-pass", command)
                Path(command[-1]).write_bytes(b"encoded")

            options = video_compressor.CompressionOptions(
                hardware="cpu",
                show_progress=False,
            )
            with (
                patch.object(
                    video_compressor,
                    "require_executable",
                    side_effect=lambda binary: binary,
                ),
                patch.object(
                    video_compressor,
                    "ffprobe_video_info",
                    return_value=info,
                ),
                patch.object(
                    video_compressor,
                    "hardware_acceleration_args",
                    return_value=[],
                ),
                patch.object(
                    video_compressor,
                    "run_ffmpeg_with_progress",
                    side_effect=complete_command,
                ),
            ):
                result = video_compressor.compress_video(
                    source,
                    output,
                    options,
                )

            self.assertEqual(output.read_bytes(), b"encoded")
            self.assertEqual(result.output_path, output.resolve())
            self.assertFalse((root / "output.partial.mp4").exists())
            self.assertEqual(result.crf, 26)
            self.assertIsNone(result.video_kbps)
            self.assertEqual(result.input_size_bytes, len(b"source"))
            self.assertEqual(result.output_size_bytes, len(b"encoded"))

    def test_two_pass_uses_source_capped_bitrate_and_no_crf(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, output = root / "input.mp4", root / "output.mp4"
            source.write_bytes(b"source")
            info = video_compressor.VideoInfo(1920, 1080, 30, 5, "h264", 2000)
            commands = []

            def complete_command(command, *_args, **_kwargs) -> None:
                commands.append(command)
                if command[command.index("-pass") + 1] == "2":
                    Path(command[-1]).write_bytes(b"encoded")

            with (
                patch.object(video_compressor, "require_executable", side_effect=lambda binary: binary),
                patch.object(video_compressor, "ffprobe_video_info", return_value=info),
                patch.object(video_compressor, "hardware_acceleration_args", return_value=[]),
                patch.object(video_compressor, "run_ffmpeg_with_progress", side_effect=complete_command),
            ):
                result = video_compressor.compress_video(
                    source, output,
                    video_compressor.CompressionOptions(two_pass=True, show_progress=False),
                )
            self.assertEqual(len(commands), 2)
            for index, command in enumerate(commands, start=1):
                self.assertEqual(command[command.index("-pass") + 1], str(index))
                self.assertEqual(command[command.index("-b:v") + 1], "2000k")
                self.assertNotIn("-crf", command)
            self.assertEqual(output.read_bytes(), b"encoded")
            self.assertEqual(result.video_kbps, 2000)
            self.assertIsNone(result.crf)

    def test_existing_output_is_protected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, output = root / "input.mp4", root / "output.mp4"
            source.write_bytes(b"source")
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                video_compressor.compress_video(source, output)
            self.assertEqual(output.read_bytes(), b"keep")


class EncodingQualityTests(unittest.TestCase):
    def test_all_codecs_use_quality_mode_without_positive_bitrate_target(self) -> None:
        for codec in ("h264", "hevc", "av1"):
            values = []
            for quality in ("low", "medium", "high", "very-high"):
                with self.subTest(codec=codec, quality=quality):
                    crf = video_compressor.QUALITY_CRF[codec][quality]
                    values.append(crf)
                    args = video_compressor._video_encoding_args(codec, "medium", None, crf=crf)
                    self.assertEqual(args[args.index("-crf") + 1], str(crf))
                    self.assertNotIn("-maxrate", args)
                    self.assertNotIn("-bufsize", args)
                    if codec == "av1":
                        self.assertEqual(args[args.index("-b:v") + 1], "0")
                        self.assertIn("-cpu-used", args)
                        self.assertNotIn("-preset", args)
                    else:
                        self.assertNotIn("-b:v", args)
            self.assertTrue(all(a > b for a, b in zip(values, values[1:])))


if __name__ == "__main__":
    unittest.main()
