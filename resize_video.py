"""Resize a video to an exact width and height using FFmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from toolkit_runtime import parse_yaml_args

HERE = Path(__file__).resolve()


def resize_video(
    source: Path,
    output: Path,
    width: int,
    height: int,
    *,
    overwrite: bool = False,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Write an H.264 MP4; preserve audio as AAC and publish after success."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video not found: {source}")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    if source == output:
        raise ValueError("Input and output must be different files.")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
    ffmpeg = shutil.which(ffmpeg_binary)
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg not found. Install it on PATH or use --ffmpeg-binary.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # H.264 4:2:0 requires even dimensions; 4:4:4 supports exact odd sizes.
    pixel_format = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
    with tempfile.TemporaryDirectory(prefix="resize_", dir=output.parent) as temp_dir:
        temporary = Path(temp_dir) / "resized.mp4"
        command = [
            ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", f"scale={width}:{height}:flags=lanczos,setsar=1",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", pixel_format, "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(temporary),
        ]
        subprocess.run(command, check=True)
        if overwrite:
            temporary.replace(output)
        else:
            # Windows rename refuses to overwrite a file created while encoding.
            temporary.rename(output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output MP4; default: INPUT_WIDTHxHEIGHT.mp4")
    parser.add_argument("-W", "--width", type=int, help="Exact output width in pixels")
    parser.add_argument("-H", "--height", type=int, help="Exact output height in pixels")
    parser.add_argument("-y", "--overwrite", action="store_true", help="Replace an existing output")
    parser.add_argument("-F", "--ffmpeg-binary", default="ffmpeg")
    try:
        args, _, _ = parse_yaml_args(parser, HERE, "resize_video", argv)
        if not args.input or args.width is None or args.height is None:
            parser.error("--input, --width and --height are required (or set them in args.yaml).")
        source = Path(args.input)
        output = Path(args.output) if args.output else source.with_name(
            f"{source.stem}_{args.width}x{args.height}.mp4"
        )
        result = resize_video(
            source, output, int(args.width), int(args.height),
            overwrite=args.overwrite, ffmpeg_binary=args.ffmpeg_binary,
        )
    except subprocess.CalledProcessError:
        print("error: FFmpeg could not resize the video; see its diagnostic output above.", file=sys.stderr)
        return 1
    except (OSError, ValueError, ModuleNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Saved {result} ({args.width}x{args.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
