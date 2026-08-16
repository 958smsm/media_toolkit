#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, hardened FFmpeg raw-frame pipe writer.

The module intentionally keeps the API used by ``rewrite_reduce.py`` while
adding deterministic process cleanup, background stderr draining, useful error
messages, codec selection, and low-memory encoder settings.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


_CODEC_ALIASES = {
    "h264": "h264",
    "avc": "h264",
    "x264": "h264",
    "hevc": "hevc",
    "h265": "hevc",
    "x265": "hevc",
    "av1": "av1",
    "svtav1": "av1",
}

_CODEC_ENCODERS = {
    "h264": "libx264",
    "hevc": "libx265",
    "av1": "libsvtav1",
}

_BITS_PER_PIXEL = {
    "h264": {"low": 0.07, "medium": 0.1, "high": 0.200},
    "hevc": {"low": 0.05, "medium": 0.07, "high": 0.14},
    "av1": {"low": 0.04, "medium": 0.06, "high": 0.12},
}

# _BITS_PER_PIXEL = {
#     "h264": {"low": 0.040, "medium": 0.065, "high": 0.100},
#     "hevc": {"low": 0.027, "medium": 0.045, "high": 0.072},
#     "av1": {"low": 0.022, "medium": 0.037, "high": 0.060},
# }

_AV1_PRESETS = {
    "ultrafast": "12",
    "superfast": "10",
    "veryfast": "8",
    "faster": "7",
    "fast": "6",
    "medium": "5",
    "slow": "4",
    "slower": "3",
    "veryslow": "2",
}


def _normalise_codec(codec_family: str) -> str:
    key = str(codec_family or "hevc").strip().lower()
    try:
        return _CODEC_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_CODEC_ALIASES)))
        raise ValueError(f"unsupported codec_family {codec_family!r}; use one of: {allowed}") from exc


def auto_video_kbps(
    width: int,
    height: int,
    fps: float,
    bitrate_kbps: Optional[int] = None,
    codec_family: str = "hevc",
    quality: str = "medium",
) -> int:
    """Estimate a practical target bitrate for constant-frame-rate video.

    The estimate is based on pixels per second and codec efficiency.  A known
    source bitrate acts as a ceiling rather than forcing an unnecessary upscale.
    """
    width = max(2, int(width))
    height = max(2, int(height))
    fps = max(0.001, float(fps))
    codec = _normalise_codec(codec_family)
    quality_key = str(quality or "medium").strip().lower()
    if quality_key not in {"low", "medium", "high"}:
        raise ValueError("quality must be low, medium, or high")

    estimated = int(round(width * height * fps * _BITS_PER_PIXEL[codec][quality_key] / 1000.0))
    # Avoid pathological tiny streams and accidental multi-hundred-Mbit targets.
    estimated = max(160, min(120_000, estimated))

    try:
        source = int(bitrate_kbps or 0)
    except (TypeError, ValueError):
        source = 0
    if source > 0:
        quality_ceiling = {"low": 0.75, "medium": 1.00, "high": 1.15}[quality_key]
        estimated = min(estimated, max(160, int(round(source * quality_ceiling))))
    return estimated


def _parse_ffprobe_rate(value: Any) -> float:
    if value in (None, "", "N/A", "0/0"):
        return 0.0
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator) if float(denominator) else 0.0
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def ffprobe_video(
    path: os.PathLike[str] | str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Return basic metadata for the first video stream in a media file."""
    video_path = Path(path)
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration,size",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or f"ffprobe exited with {result.returncode}")

    data = json.loads(result.stdout.decode("utf-8", errors="replace") or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("no video stream")

    stream = streams[0]
    media_format = data.get("format") or {}
    duration = 0.0
    for candidate in (stream.get("duration"), media_format.get("duration")):
        try:
            duration = max(duration, float(candidate))
        except (TypeError, ValueError):
            pass
    fps = _parse_ffprobe_rate(stream.get("avg_frame_rate")) or _parse_ffprobe_rate(
        stream.get("r_frame_rate")
    )
    try:
        frame_count = int(stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "frames": frame_count,
        "duration": duration,
        "size": int(media_format.get("size") or video_path.stat().st_size),
    }


class FFmpegPipeWriter:
    """Encode BGR uint8 NumPy frames through an FFmpeg stdin pipe."""

    def __init__(
        self,
        out_path: os.PathLike[str] | str,
        in_w: int,
        in_h: int,
        fps: float,
        v_kbps: int,
        preset: Any = "veryfast",
        threads: int = 0,
        low_memory: bool = False,
        codec_family: str = "hevc",
        quality: str = "medium",
        crf: Optional[float] = None,
        ffmpeg_bin: str = "ffmpeg",
        logger: Any = None,
        echo_stderr: bool = False,
        pix_fmt_in: str = "bgr24",
        pix_fmt_out: str = "yuv420p",
        close_timeout: float = 180.0,
        extra_output_args: Optional[Sequence[str]] = None,
    ):
        self.out_path = Path(out_path)
        self.in_w = int(in_w)
        self.in_h = int(in_h)
        self.fps = float(fps)
        self.v_kbps = max(1, int(v_kbps))
        self.preset = preset
        self.threads = max(0, int(threads or 0))
        self.low_memory = bool(low_memory)
        self.codec_family = _normalise_codec(codec_family)
        self.quality = str(quality or "medium").strip().lower()
        self.crf = None if crf in (None, "") else float(crf)
        self.ffmpeg_bin = str(ffmpeg_bin)
        self.logger = logger
        self.echo_stderr = bool(echo_stderr)
        self.pix_fmt_in = str(pix_fmt_in)
        self.pix_fmt_out = str(pix_fmt_out)
        self.close_timeout = max(1.0, float(close_timeout))
        self.extra_output_args = list(extra_output_args or ())

        self.proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_tail = deque(maxlen=120)
        self._opened = False
        self._closed = False
        self.frames_written = 0
        self.command: list[str] = []

        if self.in_w <= 0 or self.in_h <= 0:
            raise ValueError(f"invalid input size {self.in_w}x{self.in_h}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        if self.pix_fmt_out == "yuv420p" and ((self.in_w % 2) or (self.in_h % 2)):
            raise ValueError("yuv420p output requires even width and height")

    def _log(self, level: str, message: str) -> None:
        target = getattr(self.logger, level, None) if self.logger is not None else None
        if callable(target):
            try:
                target(message)
                return
            except Exception:
                pass

    def _codec_args(self) -> list[str]:
        codec = self.codec_family
        encoder = _CODEC_ENCODERS[codec]
        args = ["-c:v", encoder]

        preset = str(self.preset)
        if codec == "av1":
            preset = _AV1_PRESETS.get(preset.lower(), preset)
        args += ["-preset", preset]

        effective_threads = 1 if self.low_memory else self.threads
        if effective_threads > 0:
            args += ["-threads", str(effective_threads)]

        if self.crf is not None:
            args += ["-crf", f"{self.crf:g}"]
        else:
            maxrate = max(self.v_kbps, int(round(self.v_kbps * 1.20)))
            bufsize_factor = 1.25 if self.low_memory else 2.0
            bufsize = max(self.v_kbps, int(round(self.v_kbps * bufsize_factor)))
            args += [
                "-b:v", f"{self.v_kbps}k",
                "-maxrate", f"{maxrate}k",
                "-bufsize", f"{bufsize}k",
            ]

        if self.low_memory:
            if codec == "hevc":
                args += [
                    "-x265-params",
                    "pools=1:frame-threads=1:wpp=0:rc-lookahead=8:bframes=2:ref=2",
                ]
            elif codec == "h264":
                args += ["-x264-params", "rc-lookahead=5:bframes=1:ref=2"]
            else:
                args += ["-svtav1-params", "lp=1:lookahead=8"]

        if codec == "hevc" and self.out_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
            args += ["-tag:v", "hvc1"]
        return args

    def _build_command(self) -> list[str]:
        fps_text = f"{self.fps:.12g}"
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-nostdin",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", self.pix_fmt_in,
            "-video_size", f"{self.in_w}x{self.in_h}",
            "-framerate", fps_text,
            "-i", "pipe:0",
            "-map", "0:v:0",
            "-an", "-sn", "-dn",
            *self._codec_args(),
            "-pix_fmt", self.pix_fmt_out,
            "-fps_mode", "cfr",
            *self.extra_output_args,
        ]
        if self.out_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(self.out_path))
        return cmd

    def open(self) -> "FFmpegPipeWriter":
        if self._opened and not self._closed:
            return self
        if self._closed:
            raise RuntimeError("writer cannot be reopened after close/abort")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.command = self._build_command()
        creationflags = 0
        if platform.system() == "Windows":
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))

        try:
            self.proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=creationflags,
            )
        except Exception as exc:
            raise RuntimeError(f"could not start FFmpeg encoder: {exc}") from exc
        if self.proc.stdin is None or self.proc.stderr is None:
            self.abort()
            raise RuntimeError("FFmpeg encoder did not create stdin/stderr pipes")

        def drain_stderr() -> None:
            assert self.proc is not None and self.proc.stderr is not None
            for raw in iter(self.proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                self._stderr_tail.append(line)
                self._log("warning", line)
                if self.echo_stderr:
                    print(f"[FFMPEG-ENC] {line}")

        self._stderr_thread = threading.Thread(
            target=drain_stderr,
            name=f"ffmpeg-encoder-stderr-{self.out_path.name}",
            daemon=True,
        )
        self._stderr_thread.start()
        self._opened = True
        return self

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def _raise_process_error(self, prefix: str) -> None:
        rc = self.proc.poll() if self.proc is not None else None
        tail = self.stderr_tail().strip()
        suffix = f" (exit code {rc})" if rc is not None else ""
        if tail:
            raise RuntimeError(f"{prefix}{suffix}:\n{tail}")
        raise RuntimeError(f"{prefix}{suffix}")

    def write(self, frame: np.ndarray) -> None:
        if not self._opened:
            self.open()
        if self._closed or self.proc is None or self.proc.stdin is None:
            raise RuntimeError("cannot write to a closed FFmpeg writer")
        if self.proc.poll() is not None:
            self._raise_process_error("FFmpeg encoder exited before frame write")

        array = np.asarray(frame)
        if array.dtype != np.uint8:
            raise TypeError(f"frame dtype must be uint8, got {array.dtype}")
        if array.ndim != 3 or array.shape != (self.in_h, self.in_w, 3):
            raise ValueError(
                f"frame shape must be {(self.in_h, self.in_w, 3)}, got {array.shape}"
            )
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)

        view = memoryview(array).cast("B")
        try:
            while view:
                written = self.proc.stdin.write(view)
                if written is None:
                    written = 0
                if written <= 0:
                    raise BrokenPipeError("FFmpeg stdin accepted no data")
                view = view[written:]
        except (BrokenPipeError, OSError, ValueError) as exc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
            tail = self.stderr_tail().strip()
            message = f"FFmpeg encoder pipe failed after {self.frames_written} frame(s): {exc}"
            if tail:
                message += f"\n{tail}"
            raise RuntimeError(message) from exc
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.proc is None:
            return

        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()
            try:
                returncode = self.proc.wait(timeout=self.close_timeout)
            except subprocess.TimeoutExpired as exc:
                self.proc.kill()
                self.proc.wait(timeout=10)
                raise RuntimeError(
                    f"FFmpeg encoder did not finish within {self.close_timeout:g}s"
                ) from exc
        finally:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=3)
            try:
                if self.proc.stderr is not None:
                    self.proc.stderr.close()
            except Exception:
                pass

        if returncode != 0:
            self._raise_process_error("FFmpeg encoder failed while closing")
        if self.frames_written > 0:
            try:
                size = self.out_path.stat().st_size
            except OSError:
                size = 0
            if size <= 0:
                raise RuntimeError(f"FFmpeg reported success but produced no output: {self.out_path}")

    def abort(self) -> None:
        if self._closed and self.proc is None:
            return
        self._closed = True
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:
            pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass

    def __enter__(self) -> "FFmpegPipeWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False

# --- Legacy Utilities (Restored for video_compressor.py) ---

CODEC_ENCODERS = {
    "h264": ("libx264", ()),
    "hevc": ("libx265", ("-tag:v", "hvc1")),
    "av1": ("libaom-av1", ("-tag:v", "av01")),
}

class FFmpegError(RuntimeError):
    pass

def normalize_codec(codec_family: str | None) -> str:
    codec = (codec_family or "h264").strip().lower()
    codec = {"h265": "hevc", "x264": "h264", "x265": "hevc"}.get(codec, codec)
    if codec not in CODEC_ENCODERS:
        supported = ", ".join(CODEC_ENCODERS)
        raise ValueError(f"Unsupported codec {codec_family!r}; choose {supported}.")
    return codec

def normalize_quality(quality: str | None) -> str:
    normalized = (quality or "medium").strip().lower().replace(" ", "-")
    if normalized not in {"low", "medium", "high", "very-high"}:
        raise ValueError(f"Unsupported quality {quality!r}; choose low, medium, high, or very-high.")
    return normalized

def require_executable(binary: str) -> str:
    import shutil
    candidate = Path(binary).expanduser()
    if candidate.parent != Path(".") and candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(binary)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Required executable {binary!r} was not found on PATH.")

def run_capture(command: Sequence[str]) -> str:
    rendered_command = [str(part) for part in command]
    result = subprocess.run(
        rendered_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise FFmpegError(
            f"Command failed with exit code {result.returncode}: "
            f"{subprocess.list2cmdline(rendered_command)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout

def try_run_capture(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "") + (result.stderr or "")

def ffmpeg_supports_hwaccel(name: str, *, ffmpeg_binary: str = "ffmpeg") -> bool:
    output = try_run_capture([ffmpeg_binary, "-hide_banner", "-hwaccels"])
    requested = name.strip().lower()
    return bool(output and any(line.strip().lower() == requested for line in output.splitlines()))

def nvidia_gpu_present(*, nvidia_smi_binary: str = "nvidia-smi") -> bool:
    output = try_run_capture([nvidia_smi_binary, "-L"])
    if output and "gpu" in output.lower():
        return True
    if os.name != "nt":
        return False
    candidates = (
        Path(r"C:\Windows\System32\nvidia-smi.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
    )
    for executable in candidates:
        if executable.is_file():
            output = try_run_capture([str(executable), "-L"])
            if output and "gpu" in output.lower():
                return True
    adapters = try_run_capture([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
    ])
    return bool(adapters and "nvidia" in adapters.lower())

_CUDA_PROBE_CACHE: dict[tuple[str, str], bool] = {}
def cuda_works_for_file(input_path: Path | str, *, ffmpeg_binary: str = "ffmpeg") -> bool:
    path = str(Path(input_path).expanduser().resolve())
    cache_key = (ffmpeg_binary, path)
    if cache_key in _CUDA_PROBE_CACHE:
        return _CUDA_PROBE_CACHE[cache_key]
    result = subprocess.run(
        [ffmpeg_binary, "-hide_banner", "-v", "error", "-hwaccel", "cuda",
         "-i", path, "-map", "0:v:0", "-frames:v", "1", "-an", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    stderr = result.stderr or ""
    failure_markers = ("Failed setup for format cuda", "hwaccel initialisation returned error",
                       "Hardware is lacking required capabilities", "not supported with this chroma format")
    supported = result.returncode == 0 and not any(m in stderr for m in failure_markers)
    _CUDA_PROBE_CACHE[cache_key] = supported
    return supported

def hardware_acceleration_args(input_path: Path | str, mode: str = "auto", *, ffmpeg_binary: str = "ffmpeg") -> list[str]:
    selected = (mode or "auto").strip().lower()
    if selected in {"cpu", "off"}: return []
    if selected not in {"auto", "cuda"}: raise ValueError("Hardware mode must be auto, cpu, off, or cuda.")
    has_cuda = ffmpeg_supports_hwaccel("cuda", ffmpeg_binary=ffmpeg_binary)
    has_gpu = nvidia_gpu_present()
    if selected == "cuda":
        if not has_cuda or not has_gpu:
            raise FFmpegError("CUDA requested but not supported/detected.")
        return ["-hwaccel", "cuda"]
    if not has_cuda or not has_gpu: return []
    if not cuda_works_for_file(input_path, ffmpeg_binary=ffmpeg_binary): return []
    return ["-hwaccel", "cuda"]
