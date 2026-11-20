"""Test suite for video_preview.py."""

from __future__ import annotations

import os, tempfile, unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2, numpy as np, yaml

import video_preview


class DisplayAndWindowTests(unittest.TestCase):
    def test_is_display_available_windows_and_linux(self) -> None:
        with patch("platform.system", return_value="Windows"):
            self.assertTrue(video_preview.is_display_available())

        with patch("platform.system", return_value="Linux"):
            with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True):
                self.assertTrue(video_preview.is_display_available())
            with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True):
                self.assertTrue(video_preview.is_display_available())
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(video_preview.is_display_available())

    def test_setup_opencv_window_calls_named_resize_move(self) -> None:
        with (
            patch("video_preview.is_display_available", return_value=True),
            patch("cv2.namedWindow") as named_mock,
            patch("cv2.resizeWindow") as resize_mock,
            patch("cv2.moveWindow") as move_mock,
        ):
            ok = video_preview.setup_opencv_window("PreviewWin", 1280, 720, 150, 200)
            self.assertTrue(ok)
            named_mock.assert_called_once_with("PreviewWin", cv2.WINDOW_NORMAL)
            resize_mock.assert_called_once_with("PreviewWin", 1280, 720)
            move_mock.assert_called_once_with("PreviewWin", 150, 200)


class GridAndSamplingTests(unittest.TestCase):
    def test_resolve_grid_shape_for_16_32_64(self) -> None:
        self.assertEqual(video_preview.resolve_grid_shape(16), (4, 4))
        self.assertEqual(video_preview.resolve_grid_shape(32), (4, 8))
        self.assertEqual(video_preview.resolve_grid_shape(48), (6, 8))
        self.assertEqual(video_preview.resolve_grid_shape(64), (8, 8))

    def test_source_stem_extracts_query_filename_from_url(self) -> None:
        url = (
            "https://fsc2.cdn.cr/v2/d6173ca4-6833-4db9-9985-fd40630ff605.m4v"
            "?token=abc&download=true&n=303303524_after-my-last-vacation_25.m4v"
        )
        self.assertEqual(
            video_preview.source_stem(url),
            "303303524_after-my-last-vacation_25",
        )

    def test_resolve_preview_sources_accepts_yaml_list_and_home_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vid = Path(tmp_dir) / "clip.m4v"
            vid.write_bytes(b"fake")
            resolved = video_preview.resolve_preview_sources([str(vid)])
            self.assertEqual(resolved, [str(vid.resolve())])

    def test_create_video_preview_16_32_64_on_synthetic_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (160, 90),
            )
            for idx in range(80):
                frame = np.full((90, 160, 3), idx * 3 % 255, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            for count, expected_grid in ((16, (4, 4)), (32, (4, 8)), (64, (8, 8))):
                out_img = tmp_path / f"grid_{count}.jpg"
                result = video_preview.create_video_preview(
                    video_path,
                    num_frames=count,
                    mode="grid",
                    output_path=out_img,
                    save_grid=True,
                    show_display=False,
                    show_progress=False,
                )
                self.assertEqual(result.sampled_count, count)
                self.assertEqual(result.grid_shape, expected_grid)
                self.assertTrue(out_img.is_file())


class CliConfigTests(unittest.TestCase):
    def test_three_execution_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            custom_yaml = tmp_path / "custom_args.yaml"
            custom_yaml.write_text(
                yaml.safe_dump(
                    {
                        "video_preview": {
                            "input": "test_video.mp4",
                            "num_frames": 32,
                            "mode": "grid",
                        }
                    }
                ),
                encoding="utf-8",
            )
            # Mode 2: explicit YAML path
            args_dict, cli_argv, yf = video_preview.load_yaml_args_dict(
                video_preview.HERE, [str(custom_yaml)]
            )
            self.assertEqual(yf, custom_yaml.resolve())
            self.assertEqual(cli_argv, [])
            self.assertEqual(args_dict["num_frames"], 32)

            # Mode 3: CLI flags override YAML defaults
            parser = video_preview.build_parser()
            parser.set_defaults(**args_dict)
            parsed, _ = parser.parse_known_args(["-n", "64", "-m", "playback"])
            self.assertEqual(parsed.num_frames, 64)
            self.assertEqual(parsed.mode, "playback")


if __name__ == "__main__":
    unittest.main()

