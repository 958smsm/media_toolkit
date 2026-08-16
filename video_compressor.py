"""Compress videos with FFmpeg using resolution- and FPS-aware bitrates."""

from __future__ import annotations

import argparse, json, logging, math, os, re, subprocess, sys, tempfile
import threading, time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from ffmpeg_manager import (
    CODEC_ENCODERS, FFmpegError, auto_video_kbps, hardware_acceleration_args,
    normalize_codec, normalize_quality, require_executable, run_capture,
)
from toolkit_runtime import (
    ProgressBar, configure_logging, parse_yaml_args, partial_output_path,
    progress_iter, resolve_files,
)

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem
LOGGER = logging.getLogger(FEATURE_NAME)
VIDEO_EXTENSIONS = frozenset(
    {
        ".avi",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".webm",
        ".wmv",
    }
)


@dataclass(frozen=True)
class VideoInfo:
    """Metadata required for bitrate estimation and progress reporting."""

    width: int
    height: int
    fps: float
    duration: float
    codec: str


@dataclass(frozen=True)
class CompressionOptions:
    """Settings for one compression operation."""

    codec: str = "h264"
    quality: str = "medium"
    max_height: int | None = None
    audio_kbps: int = 128
    preset: str = "medium"
    two_pass: bool = False
    hardware: str = "auto"
    overwrite: bool = False
    show_progress: bool = True
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"


@dataclass(frozen=True)
class CompressionResult:
    """Summary of a completed compression."""

    input_path: Path
    output_path: Path
    source_info: VideoInfo
    output_width: int
    output_height: int
    video_kbps: int
    used_cuda: bool
    elapsed_seconds: float


def resolve_input_videos(inputs: Sequence[Path | str]) -> list[Path]:
    """Expand video files, directories, globs, and text lists."""

    return resolve_files(inputs, extensions=VIDEO_EXTENSIONS)


def ffprobe_video_info(
    input_path: Path | str,
    *,
    ffprobe_binary: str = "ffprobe",
) -> VideoInfo:
    """Read the first video stream with ffprobe."""

    path = Path(input_path).expanduser().resolve()
    output = run_capture(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
    )
    data = json.loads(output)
    video_stream = next(
        (
            stream
            for stream in data.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}.")

    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid video dimensions reported for {path}.") from error

    frame_rate = (
        video_stream.get("avg_frame_rate")
        or video_stream.get("r_frame_rate")
        or "0/0"
    )
    try:
        fraction = Fraction(frame_rate)
        fps = (
            float(fraction)
            if fraction.numerator and fraction.denominator
            else 0.0
        )
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    try:
        duration = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if fps <= 0 and duration > 0 and video_stream.get("nb_frames"):
        try:
            fps = float(video_stream["nb_frames"]) / duration
        except (TypeError, ValueError, ZeroDivisionError):
            fps = 0.0
    if fps <= 0:
        raise ValueError(f"Could not determine the frame rate for {path}.")

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        codec=str(video_stream.get("codec_name") or ""),
    )


def make_even(value: int) -> int:
    """Return the nearest lower positive even integer."""

    return max(2, int(value) - int(value) % 2)


def build_scale_filter(
    width: int,
    height: int,
    max_height: int | None,
) -> tuple[str | None, int, int]:
    """Build a scale filter that preserves aspect ratio and even dimensions."""

    if not max_height or height <= max_height:
        return None, width, height
    if max_height < 2:
        raise ValueError("Maximum height must be at least 2 pixels.")
    output_height = make_even(max_height)
    output_width = make_even(round(width * output_height / height))
    return (
        f"scale={output_width}:{output_height}",
        output_width,
        output_height,
    )


def _read_stderr(
    stream,
    tail: deque[str],
    *,
    echo: bool,
) -> None:
    for line in stream:
        cleaned = line.rstrip()
        if cleaned:
            tail.append(cleaned)
            if echo:
                print(cleaned, file=sys.stderr, flush=True)


def run_ffmpeg_with_progress(
    command: Sequence[str],
    duration_seconds: float,
    description: str,
    *,
    show_progress: bool = True,
    echo_stderr: bool = False,
) -> None:
    """Run FFmpeg while consuming its machine-readable progress stream."""

    if not command:
        raise ValueError("FFmpeg command cannot be empty.")
    full_command = [
        str(command[0]),
        "-progress",
        "pipe:1",
        "-nostats",
        *[str(part) for part in command[1:]],
    ]
    LOGGER.debug("Running FFmpeg: %s", subprocess.list2cmdline(full_command))
    process = subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stderr_tail: deque[str] = deque(maxlen=100)
    stderr_thread = threading.Thread(
        target=_read_stderr,
        args=(process.stderr, stderr_tail),
        kwargs={"echo": echo_stderr},
        daemon=True,
    )
    stderr_thread.start()

    with ProgressBar(
        description,
        total=max(duration_seconds, 1.0),
        unit="s",
        enabled=show_progress,
    ) as progress:
        try:
            for line in process.stdout:
                key, separator, value = line.strip().partition("=")
                if not separator:
                    continue
                if key in {"out_time_us", "out_time_ms"}:
                    try:
                        progress.update_to(int(value) / 1_000_000.0)
                    except ValueError:
                        pass
                elif key == "out_time":
                    match = re.fullmatch(
                        r"(\d+):(\d+):(\d+(?:\.\d+)?)",
                        value,
                    )
                    if match:
                        hours, minutes, seconds = match.groups()
                        progress.update_to(
                            int(hours) * 3600
                            + int(minutes) * 60
                            + float(seconds)
                        )

            return_code = process.wait()
            stderr_thread.join(timeout=2)
            if return_code != 0:
                diagnostics = "\n".join(stderr_tail)
                raise FFmpegError(
                    f"FFmpeg failed with exit code {return_code}: "
                    f"{subprocess.list2cmdline(full_command)}\n{diagnostics}"
                )
        finally:
            process.stdout.close()
            process.stderr.close()


def _video_encoding_args(
    codec: str,
    preset: str,
    video_kbps: int,
) -> list[str]:
    encoder, _container_args = CODEC_ENCODERS[codec]
    maximum_rate = math.ceil(video_kbps * 1.20)
    buffer_size = math.ceil(video_kbps * 2.00)
    return [
        "-c:v",
        encoder,
        "-preset",
        preset,
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{maximum_rate}k",
        "-bufsize",
        f"{buffer_size}k",
    ]


def compress_video(
    input_path: Path | str,
    output_path: Path | str,
    options: CompressionOptions = CompressionOptions(),
) -> CompressionResult:
    """Compress one video and atomically publish the completed output."""

    started_at = time.monotonic()
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if source == output:
        raise ValueError("Input and output paths must be different.")
    if output.exists() and not options.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    if options.audio_kbps <= 0:
        raise ValueError("Audio bitrate must be positive.")

    ffmpeg_binary = require_executable(options.ffmpeg_binary)
    ffprobe_binary = require_executable(options.ffprobe_binary)
    codec = normalize_codec(options.codec)
    quality = normalize_quality(options.quality)
    source_info = ffprobe_video_info(source, ffprobe_binary=ffprobe_binary)
    scale_filter, output_width, output_height = build_scale_filter(
        source_info.width,
        source_info.height,
        options.max_height,
    )
    video_kbps = auto_video_kbps(
        output_width,
        output_height,
        source_info.fps,
        codec_family=codec,
        quality=quality,
    )
    hardware_args = hardware_acceleration_args(
        source,
        options.hardware,
        ffmpeg_binary=ffmpeg_binary,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = partial_output_path(output)
    input_args = [
        ffmpeg_binary,
        "-y",
        *hardware_args,
        "-i",
        str(source),
    ]
    mapping_args = ["-map", "0:v:0", "-map", "0:a?"]
    scale_args = ["-vf", scale_filter] if scale_filter else []
    video_args = _video_encoding_args(codec, options.preset, video_kbps)
    _encoder, codec_container_args = CODEC_ENCODERS[codec]
    audio_args = ["-c:a", "aac", "-b:a", f"{options.audio_kbps}k"]
    container_args = [
        *codec_container_args,
        *(
            ["-movflags", "+faststart"]
            if output.suffix.lower() in {".m4v", ".mov", ".mp4"}
            else []
        ),
    ]

    try:
        if options.two_pass:
            with tempfile.TemporaryDirectory(
                prefix="video_compress_",
                dir=output.parent,
            ) as temporary_directory:
                pass_log = str(Path(temporary_directory) / "ffmpeg2pass")
                first_pass = [
                    *input_args,
                    "-map",
                    "0:v:0",
                    *scale_args,
                    *video_args,
                    "-pass",
                    "1",
                    "-passlogfile",
                    pass_log,
                    "-an",
                    "-f",
                    "null",
                    os.devnull,
                ]
                run_ffmpeg_with_progress(
                    first_pass,
                    source_info.duration,
                    "Encoding pass 1/2",
                    show_progress=options.show_progress,
                )
                second_pass = [
                    *input_args,
                    *mapping_args,
                    *scale_args,
                    *video_args,
                    "-pass",
                    "2",
                    "-passlogfile",
                    pass_log,
                    *audio_args,
                    *container_args,
                    str(partial_output),
                ]
                run_ffmpeg_with_progress(
                    second_pass,
                    source_info.duration,
                    "Encoding pass 2/2",
                    show_progress=options.show_progress,
                )
        else:
            command = [
                *input_args,
                *mapping_args,
                *scale_args,
                *video_args,
                *audio_args,
                *container_args,
                str(partial_output),
            ]
            run_ffmpeg_with_progress(
                command,
                source_info.duration,
                "Encoding",
                show_progress=options.show_progress,
            )
        partial_output.replace(output)
    except Exception:
        partial_output.unlink(missing_ok=True)
        raise

    return CompressionResult(
        input_path=source,
        output_path=output,
        source_info=source_info,
        output_width=output_width,
        output_height=output_height,
        video_kbps=video_kbps,
        used_cuda=bool(hardware_args),
        elapsed_seconds=time.monotonic() - started_at,
    )


def output_path_for(
    input_path: Path,
    *,
    output_dir: Path | None,
    suffix: str,
) -> Path:
    """Build the destination path for one input."""

    destination_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else input_path.parent
    )
    return destination_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def build_parser() -> argparse.ArgumentParser:
    """Build the video-compression CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compress videos with an automatically estimated target bitrate."
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
    parser.add_argument("-c", "--codec", choices=sorted(CODEC_ENCODERS))
    parser.add_argument(
        "-q",
        "--quality",
        choices=["low", "medium", "high", "very-high"],
    )
    parser.add_argument("-H", "--max-height", type=int)
    parser.add_argument("-a", "--audio-kbps", type=int)
    parser.add_argument("-p", "--preset")
    parser.add_argument(
        "-2",
        "--two-pass",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("-g", "--hw", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "-w",
        "--overwrite",
        action=argparse.BooleanOptionalAction,
    )
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
    """Run the video-compression CLI."""

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
        options = CompressionOptions(
            codec=args.codec,
            quality=args.quality,
            max_height=args.max_height,
            audio_kbps=args.audio_kbps,
            preset=args.preset,
            two_pass=bool(args.two_pass),
            hardware=args.hw,
            overwrite=bool(args.overwrite),
            show_progress=bool(args.show_progress),
            ffmpeg_binary=args.ffmpeg_binary,
            ffprobe_binary=args.ffprobe_binary,
        )
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
        "Processing videos",
        enabled=options.show_progress,
    ):
        output_path = output_path_for(
            input_path,
            output_dir=output_dir,
            suffix=args.suffix,
        )
        log.info("Compressing %s -> %s", input_path, output_path)
        try:
            result = compress_video(input_path, output_path, options)
        except Exception as error:
            failures += 1
            log.error("Failed %s: %s", input_path, error)
            if args.fail_fast:
                break
            continue
        log.info(
            "Completed %s at %dx%d, %d kbps in %.1f seconds%s",
            result.output_path,
            result.output_width,
            result.output_height,
            result.video_kbps,
            result.elapsed_seconds,
            " using CUDA decoding" if result.used_cuda else "",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
