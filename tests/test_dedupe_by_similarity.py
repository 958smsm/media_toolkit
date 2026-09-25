from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from image_similarity import dedupe_by_similarity


class OptionValidationTests(unittest.TestCase):
    def test_accepts_default_options(self) -> None:
        dedupe_by_similarity.validate_options(
            dedupe_by_similarity.DedupeOptions()
        )

    def test_rejects_invalid_threshold(self) -> None:
        options = dedupe_by_similarity.DedupeOptions(threshold=1.1)
        with self.assertRaises(ValueError):
            dedupe_by_similarity.validate_options(options)


class OutputTests(unittest.TestCase):
    def test_builds_mp4_output_name(self) -> None:
        output = dedupe_by_similarity.output_path_for(
            Path("input.mkv"),
            Path("outputs"),
            "_deduped",
        )

        expected = (Path("outputs") / "input_deduped.mp4").resolve()
        self.assertEqual(output, expected)

    def test_moves_to_trash_without_overwriting(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "video.mp4"
            trash = root / "trash"
            trash.mkdir()
            source.write_bytes(b"new")
            (trash / "video.mp4").write_bytes(b"old")

            moved = dedupe_by_similarity.safe_move_to_trash(source, trash)

            self.assertEqual(moved.name, "video_1.mp4")
            self.assertEqual(moved.read_bytes(), b"new")
            self.assertFalse(source.exists())


class DeduplicationIntegrationTests(unittest.TestCase):
    def test_deduplicates_constant_frames(self) -> None:
        import cv2, numpy as np

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "sample.mp4"
            output_path = root / "sample_deduped.mp4"

            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (64, 64),
            )
            for _ in range(5):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()

            result = dedupe_by_similarity.deduplicate_video(
                input_path,
                output_path,
                dedupe_by_similarity.DedupeOptions(show_progress=False),
            )

            self.assertEqual(result.frames_read, 5)
            self.assertEqual(result.frames_kept, 1)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
