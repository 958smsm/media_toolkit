"""Shorten videos by dropping frames similar to the last retained frame."""

from __future__ import annotations

import argparse, logging, shutil, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffmpeg_manager import FFmpegPipeWriter, auto_video_kbps
from toolkit_runtime import (
    ProgressBar, configure_logging, parse_yaml_args, partial_output_path,
    progress_iter,
)
from video_compressor import (
    build_scale_filter, ffprobe_video_info, resolve_input_videos,
)

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem
LOGGER = logging.getLogger(FEATURE_NAME)


@dataclass(frozen=True)
class DedupeOptions:
    """Frame comparison and output encoding settings."""

    threshold: float = 0.95
    pixel_tolerance: int = 8
    resize_width: int = 320
    blur_kernel_size: int = 3
    minimum_keep_frames: int = 0
    quality: str = "medium"
    preset: str = "medium"
    max_height: int | None = None
    overwrite: bool = False
    move_to_trash: bool = False
    trash_dir: Path | None = None
    show_progress: bool = True
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"


@dataclass(frozen=True)
class DedupeResult:
    """Summary of a completed frame deduplication."""

    input_path: Path
    output_path: Path
    frames_read: int
    frames_kept: int
    fps: float
    elapsed_seconds: float
    moved_input_to: Path | None = None

    @property
    def removed_frames(self) -> int:
        return self.frames_read - self.frames_kept


def validate_options(options: DedupeOptions) -> None:
    """Validate frame comparison settings before opening a video."""

    if not 0.0 <= options.threshold <= 1.0:
        raise ValueError("Similarity threshold must be between 0 and 1.")
    if not 0 <= options.pixel_tolerance <= 255:
        raise ValueError("Pixel tolerance must be between 0 and 255.")
    if options.resize_width < 0:
        raise ValueError("Resize width cannot be negative.")
    if options.blur_kernel_size < 0:
        raise ValueError("Blur kernel size cannot be negative.")
    if options.minimum_keep_frames < 0:
        raise ValueError("Minimum keep frames cannot be negative.")


def preprocess_for_compare(
    frame_bgr: Any,
    resize_width: int,
    blur_kernel_size: int,
) -> Any:
    """Create a small grayscale frame for efficient comparison."""

    import cv2

    height, width = frame_bgr.shape[:2]
    if resize_width and width > resize_width:
        resized_height = max(1, round(height * resize_width / width))
        comparison_frame = cv2.resize(
            frame_bgr,
            (resize_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        comparison_frame = frame_bgr

    grayscale = cv2.cvtColor(comparison_frame, cv2.COLOR_BGR2GRAY)
    if blur_kernel_size:
        kernel_size = (
            blur_kernel_size
            if blur_kernel_size % 2 == 1
            else blur_kernel_size + 1
        )
        grayscale = cv2.GaussianBlur(
            grayscale,
            (kernel_size, kernel_size),
            0,
        )
    return grayscale


def similarity_percent(
    previous_gray: Any,
    current_gray: Any,
    pixel_tolerance: int,
) -> float:
    """Return the fraction of pixels within an absolute difference tolerance."""

    import cv2, numpy as np

    difference = cv2.absdiff(previous_gray, current_gray)
    return float(np.mean(difference <= pixel_tolerance))


def safe_move_to_trash(source: Path | str, trash_dir: Path | str) -> Path:
    """Move a source file to a trash directory without overwriting."""

    source_path = Path(source).expanduser().resolve()
    destination_dir = Path(trash_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Input to trash does not exist: {source_path}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name
    counter = 1
    while destination.exists():
        destination = destination_dir / (
            f"{source_path.stem}_{counter}{source_path.suffix}"
        )
        counter += 1
    return Path(shutil.move(str(source_path), str(destination)))


def deduplicate_video(
    input_path: Path | str,
    output_path: Path | str,
    options: DedupeOptions = DedupeOptions(),
) -> DedupeResult:
    """Drop similar frames, encode retained frames, and publish atomically."""

    import cv2

    validate_options(options)
    started_at = time.monotonic()
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if source == output:
        raise ValueError("Input and output paths must be different.")
    if output.exists() and not options.overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    video_info = ffprobe_video_info(
        source,
        ffprobe_binary=options.ffprobe_binary,
    )
    scale_filter, output_width, output_height = build_scale_filter(
        video_info.width,
        video_info.height,
        options.max_height,
    )
    video_kbps = auto_video_kbps(
        output_width,
        output_height,
        video_info.fps,
        codec_family="hevc",
        quality=options.quality,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = partial_output_path(output)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = ProgressBar(
        "Comparing frames",
        total=float(total_frames) if total_frames > 0 else None,
        unit="frame",
        enabled=options.show_progress,
    )
    writer: FFmpegPipeWriter | None = None
    frames_read = 0
    frames_kept = 0

    try:
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Could not read the first frame from {source}.")
        previous_comparison = preprocess_for_compare(
            frame,
            options.resize_width,
            options.blur_kernel_size,
        )
        writer = FFmpegPipeWriter(
            out_path=partial_output,
            in_w=video_info.width,
            in_h=video_info.height,
            fps=video_info.fps,
            v_kbps=video_kbps,
            preset=options.preset,
            codec_family="hevc",
            quality=options.quality,
            scale_filter=scale_filter,
            overwrite=True,
            ffmpeg_bin=options.ffmpeg_binary,
            logger=LOGGER,
        ).open()
        writer.write(frame)
        frames_read = 1
        frames_kept = 1
        last_kept_index = 0
        progress.update()

        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_index = frames_read
            frames_read += 1
            current_comparison = preprocess_for_compare(
                frame,
                options.resize_width,
                options.blur_kernel_size,
            )
            similarity = similarity_percent(
                previous_comparison,
                current_comparison,
                options.pixel_tolerance,
            )
            force_keep = (
                options.minimum_keep_frames > 0
                and frame_index - last_kept_index
                >= options.minimum_keep_frames
            )
            if force_keep or similarity < options.threshold:
                writer.write(frame)
                previous_comparison = current_comparison
                frames_kept += 1
                last_kept_index = frame_index
            progress.update()

        writer.close()
        writer = None
        partial_output.replace(output)
    except Exception:
        if writer is not None:
            writer.abort()
        partial_output.unlink(missing_ok=True)
        raise
    finally:
        progress.close()
        capture.release()

    moved_input_to = None
    if options.move_to_trash:
        trash_dir = (
            options.trash_dir
            if options.trash_dir is not None
            else HERE.parent / "trash"
        )
        moved_input_to = safe_move_to_trash(source, trash_dir)

    return DedupeResult(
        input_path=source,
        output_path=output,
        frames_read=frames_read,
        frames_kept=frames_kept,
        fps=video_info.fps,
        elapsed_seconds=time.monotonic() - started_at,
        moved_input_to=moved_input_to,
    )


def output_path_for(
    input_path: Path,
    output_dir: Path | None,
    suffix: str,
) -> Path:
    """Build an MP4 destination path for one source video."""

    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else input_path.parent
    )
    return destination / f"{input_path.stem}{suffix}.mp4"


def build_parser() -> argparse.ArgumentParser:
    """Build the frame-deduplication CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Shorten videos by dropping frames similar to the last kept frame."
        )
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="Video files, directories, globs, or TXT file lists.",
    )
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("-s", "--suffix")
    parser.add_argument("-t", "--threshold", type=float)
    parser.add_argument("-p", "--pixel-tolerance", type=int)
    parser.add_argument("-r", "--resize-width", type=int)
    parser.add_argument("-b", "--blur-kernel-size", type=int)
    parser.add_argument("-k", "--minimum-keep-frames", type=int)
    parser.add_argument(
        "-q",
        "--quality",
        choices=["low", "medium", "high", "very-high"],
    )
    parser.add_argument("-e", "--preset")
    parser.add_argument("-H", "--max-height", type=int)
    parser.add_argument(
        "-w",
        "--overwrite",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-T",
        "--trash",
        action=argparse.BooleanOptionalAction,
        dest="move_to_trash",
    )
    parser.add_argument("-d", "--trash-dir", type=Path)
    parser.add_argument(
        "-P",
        "--progress",
        action=argparse.BooleanOptionalAction,
        dest="show_progress",
    )
    parser.add_argument(
        "-f",
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("-F", "--ffmpeg-binary")
    parser.add_argument("-R", "--ffprobe-binary")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the frame-deduplication CLI."""

    try:
        args, unknown_args, yaml_path = parse_yaml_args(
            build_parser(),
            HERE,
            FEATURE_NAME,
            argv,
        )
        log = configure_logging(HERE, FEATURE_NAME, verbose=bool(args.verbose))
        log.debug("Loaded configuration from %s", yaml_path)
        if unknown_args:
            log.debug("Ignoring unknown arguments: %s", unknown_args)

        raw_inputs = args.inputs or []
        if isinstance(raw_inputs, str):
            raw_inputs = [raw_inputs]
        if not raw_inputs:
            raise ValueError(
                "No inputs configured; use -i/--inputs or edit args.yaml."
            )
        input_paths = resolve_input_videos(raw_inputs)
        if not input_paths:
            raise ValueError("No input videos found.")
        output_dir = Path(args.output_dir) if args.output_dir else None
        options = DedupeOptions(
            threshold=args.threshold,
            pixel_tolerance=args.pixel_tolerance,
            resize_width=args.resize_width,
            blur_kernel_size=args.blur_kernel_size,
            minimum_keep_frames=args.minimum_keep_frames,
            quality=args.quality,
            preset=args.preset,
            max_height=args.max_height,
            overwrite=bool(args.overwrite),
            move_to_trash=bool(args.move_to_trash),
            trash_dir=Path(args.trash_dir) if args.trash_dir else None,
            show_progress=bool(args.show_progress),
            ffmpeg_binary=args.ffmpeg_binary,
            ffprobe_binary=args.ffprobe_binary,
        )
        validate_options(options)
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    failures = 0
    for input_path in progress_iter(
        input_paths,
        "Deduplicating videos",
        enabled=options.show_progress,
    ):
        output_path = output_path_for(input_path, output_dir, args.suffix)
        log.info("Deduplicating %s -> %s", input_path, output_path)
        try:
            result = deduplicate_video(input_path, output_path, options)
        except Exception as error:
            failures += 1
            log.error("Failed %s: %s", input_path, error, exc_info=True)
            if args.fail_fast:
                break
            continue
        log.info(
            "Completed %s: kept %d/%d frames, removed %d in %.1f seconds",
            result.output_path,
            result.frames_kept,
            result.frames_read,
            result.removed_frames,
            result.elapsed_seconds,
        )
        if result.moved_input_to:
            log.info("Moved input to %s", result.moved_input_to)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
