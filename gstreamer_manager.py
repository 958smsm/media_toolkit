#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GStreamer raw-frame writer for NumPy/OpenCV frames.

This module is a GStreamer equivalent of an FFmpeg stdin raw-frame writer.
It accepts BGR uint8 NumPy frames and encodes them into MP4, MKV, WebM, or
MPEG-TS using GStreamer appsrc.

Required packages on Ubuntu/Debian:

    sudo apt install \
        python3-gi python3-gst-1.0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav

Example:

    import cv2
    from gstreamer_pipe_writer import GStreamerPipeWriter

    cap = cv2.VideoCapture("input.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    with GStreamerPipeWriter(
        "output.mp4",
        width,
        height,
        fps,
        v_kbps=4000,
        codec_family="h264",
    ) as writer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)

    cap.release()
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except (ImportError, ValueError):
    Gst = None  # type: ignore[assignment]


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

_DEFAULT_ENCODERS = {
    "h264": "x264enc",
    "hevc": "x265enc",
    "av1": "svtav1enc",
}

_DEFAULT_PARSERS = {
    "h264": "h264parse",
    "hevc": "h265parse",
    "av1": "av1parse",
}

_BITS_PER_PIXEL = {
    "h264": {"low": 0.040, "medium": 0.065, "high": 0.100},
    "hevc": {"low": 0.027, "medium": 0.045, "high": 0.072},
    "av1": {"low": 0.022, "medium": 0.037, "high": 0.060},
}

_AV1_PRESETS = {
    "ultrafast": 12,
    "superfast": 10,
    "veryfast": 8,
    "faster": 7,
    "fast": 6,
    "medium": 5,
    "slow": 4,
    "slower": 3,
    "veryslow": 2,
}

_INPUT_FORMAT_ALIASES = {
    "bgr24": "BGR",
    "bgr": "BGR",
    "rgb24": "RGB",
    "rgb": "RGB",
    "gray": "GRAY8",
    "gray8": "GRAY8",
}

_OUTPUT_FORMAT_ALIASES = {
    "yuv420p": "I420",
    "i420": "I420",
    "nv12": "NV12",
    "y444": "Y444",
    "bgra": "BGRA",
    "rgba": "RGBA",
}


def _require_gstreamer() -> None:
    if Gst is None:
        raise RuntimeError(
            "PyGObject/GStreamer is not installed. Install python3-gi, "
            "python3-gst-1.0, and the required GStreamer plugins."
        )
    Gst.init(None)


def _normalise_codec(codec_family: str) -> str:
    key = str(codec_family or "h264").strip().lower()
    try:
        return _CODEC_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_CODEC_ALIASES)))
        raise ValueError(
            f"unsupported codec_family {codec_family!r}; use one of: {allowed}"
        ) from exc


def auto_video_kbps(
    width: int,
    height: int,
    fps: float,
    bitrate_kbps: Optional[int] = None,
    codec_family: str = "h264",
    quality: str = "medium",
) -> int:
    """Estimate a practical target bitrate from pixels per second."""
    if isinstance(bitrate_kbps, str):
        candidate = bitrate_kbps.strip().lower()
        if candidate in _CODEC_ALIASES:
            if str(codec_family or "").strip().lower() in {"low", "medium", "high"}:
                quality = codec_family
            codec_family = candidate
            bitrate_kbps = None

    width = max(2, int(width))
    height = max(2, int(height))
    fps = max(0.001, float(fps))
    codec = _normalise_codec(codec_family)
    quality_key = str(quality or "medium").strip().lower()
    if quality_key not in {"low", "medium", "high"}:
        raise ValueError("quality must be low, medium, or high")

    estimated = int(
        round(width * height * fps * _BITS_PER_PIXEL[codec][quality_key] / 1000.0)
    )
    estimated = max(160, min(120_000, estimated))

    try:
        source = int(bitrate_kbps or 0)
    except (TypeError, ValueError):
        source = 0

    if source > 0:
        quality_ceiling = {"low": 0.75, "medium": 1.00, "high": 1.15}[quality_key]
        estimated = min(
            estimated,
            max(160, int(round(source * quality_ceiling))),
        )
    return estimated


def _make_element(factory_name: str, instance_name: str) -> Any:
    element = Gst.ElementFactory.make(factory_name, instance_name)
    if element is None:
        raise RuntimeError(
            f"GStreamer element {factory_name!r} is unavailable. "
            f"Check installed plugins with: gst-inspect-1.0 {factory_name}"
        )
    return element


def _has_property(element: Any, name: str) -> bool:
    return element.find_property(name) is not None


def _set_property(element: Any, name: str, value: Any, *, required: bool = False) -> bool:
    if not _has_property(element, name):
        if required:
            raise RuntimeError(
                f"{element.get_name()} does not support property {name!r}"
            )
        return False

    try:
        element.set_property(name, value)
    except (TypeError, ValueError):
        try:
            Gst.util_set_object_arg(element, name, str(value))
        except Exception as exc:
            if required:
                raise RuntimeError(
                    f"could not set {element.get_name()}.{name}={value!r}: {exc}"
                ) from exc
            return False
    return True


def _link_many(elements: list[Any]) -> None:
    for left, right in zip(elements, elements[1:]):
        if not left.link(right):
            raise RuntimeError(
                f"could not link GStreamer elements "
                f"{left.get_name()} -> {right.get_name()}"
            )


class GStreamerPipeWriter:
    """Encode NumPy frames through a GStreamer appsrc pipeline.

    Frames must be uint8 arrays matching the configured input format and size.
    The default input is OpenCV-compatible BGR.
    """

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
        codec_family: str = "h264",
        quality: str = "medium",
        crf: Optional[float] = None,
        logger: Any = None,
        echo_messages: bool = False,
        pix_fmt_in: str = "bgr24",
        pix_fmt_out: str = "yuv420p",
        close_timeout: float = 180.0,
        encoder_name: Optional[str] = None,
        encoder_properties: Optional[Mapping[str, Any]] = None,
        muxer_name: Optional[str] = None,
        muxer_properties: Optional[Mapping[str, Any]] = None,
        is_live: bool = False,
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
        self.logger = logger
        self.echo_messages = bool(echo_messages)
        self.close_timeout = max(1.0, float(close_timeout))
        self.encoder_name = encoder_name or _DEFAULT_ENCODERS[self.codec_family]
        self.encoder_properties = dict(encoder_properties or {})
        self.muxer_name = muxer_name
        self.muxer_properties = dict(muxer_properties or {})
        self.is_live = bool(is_live)

        input_key = str(pix_fmt_in).strip().lower()
        output_key = str(pix_fmt_out).strip().lower()
        self.gst_input_format = _INPUT_FORMAT_ALIASES.get(
            input_key, str(pix_fmt_in).strip().upper()
        )
        self.gst_output_format = _OUTPUT_FORMAT_ALIASES.get(
            output_key, str(pix_fmt_out).strip().upper()
        )

        self.pipeline: Any = None
        self.appsrc: Any = None
        self.bus: Any = None
        self._bus_thread: Optional[threading.Thread] = None
        self._stop_bus = threading.Event()
        self._eos_seen = threading.Event()
        self._fatal_error: Optional[str] = None
        self._messages = deque(maxlen=120)
        self._opened = False
        self._closed = False
        self._lock = threading.RLock()
        self.frames_written = 0
        self.pipeline_description = ""

        if self.in_w <= 0 or self.in_h <= 0:
            raise ValueError(f"invalid input size {self.in_w}x{self.in_h}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        if self.gst_output_format in {"I420", "NV12"}:
            if (self.in_w % 2) or (self.in_h % 2):
                raise ValueError(
                    f"{self.gst_output_format} output requires even width and height"
                )
        if self.quality not in {"low", "medium", "high"}:
            raise ValueError("quality must be low, medium, or high")

        channels = {
            "BGR": 3,
            "RGB": 3,
            "BGRA": 4,
            "RGBA": 4,
            "GRAY8": 1,
        }.get(self.gst_input_format)
        if channels is None:
            raise ValueError(
                f"unsupported input format {self.gst_input_format!r}; "
                "use BGR, RGB, BGRA, RGBA, or GRAY8"
            )
        self.channels = channels

        fraction = Fraction(self.fps).limit_denominator(1001)
        self.fps_num = fraction.numerator
        self.fps_den = fraction.denominator

    def _log(self, level: str, message: str) -> None:
        target = getattr(self.logger, level, None) if self.logger is not None else None
        if callable(target):
            try:
                target(message)
                return
            except Exception:
                pass

    def _append_message(self, level: str, message: str) -> None:
        line = f"[{level.upper()}] {message}"
        self._messages.append(line)
        self._log("warning" if level != "info" else "info", line)
        if self.echo_messages:
            print(f"[GSTREAMER] {line}")

    def message_tail(self) -> str:
        return "\n".join(self._messages)

    def stderr_tail(self) -> str:
        """Compatibility alias with subprocess-based writers."""
        return self.message_tail()

    def _select_muxer(self) -> str:
        if self.muxer_name:
            return self.muxer_name

        suffix = self.out_path.suffix.lower()
        if suffix in {".mp4", ".m4v", ".mov"}:
            return "mp4mux"
        if suffix in {".mkv", ".mka"}:
            return "matroskamux"
        if suffix == ".webm":
            if self.codec_family != "av1":
                raise ValueError(
                    "this writer supports WebM only with AV1; "
                    "use .mkv/.mp4 for H.264 or HEVC"
                )
            return "webmmux"
        if suffix in {".ts", ".mts", ".m2ts"}:
            return "mpegtsmux"
        return "matroskamux"

    def _configure_encoder(self, encoder: Any) -> None:
        name = self.encoder_name.lower()

        # Most software and hardware encoders expose one of these.
        if not _set_property(encoder, "bitrate", self.v_kbps):
            _set_property(encoder, "target-bitrate", self.v_kbps * 1000)

        if "svtav1" in name:
            av1_preset = _AV1_PRESETS.get(
                str(self.preset).lower(),
                int(self.preset) if str(self.preset).isdigit() else 8,
            )
            _set_property(encoder, "preset", av1_preset)
            if self.crf is not None:
                _set_property(encoder, "crf", int(round(self.crf)))
        else:
            _set_property(encoder, "speed-preset", str(self.preset))
            if self.crf is not None:
                # x264enc/x265enc accept libx264/libx265 options here.
                _set_property(encoder, "option-string", f"crf={self.crf:g}")

        if self.threads > 0:
            _set_property(encoder, "threads", self.threads)

        key_interval = max(1, int(round(self.fps * 2)))
        _set_property(encoder, "key-int-max", key_interval)
        _set_property(encoder, "gop-size", key_interval)

        if self.low_memory:
            _set_property(encoder, "tune", "zerolatency")
            _set_property(encoder, "bframes", 0)
            _set_property(encoder, "lookahead", 0)

        for key, value in self.encoder_properties.items():
            _set_property(encoder, key, value, required=True)

    def _build_pipeline(self) -> None:
        _require_gstreamer()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline = Gst.Pipeline.new("numpy-video-writer")
        if pipeline is None:
            raise RuntimeError("could not create GStreamer pipeline")

        appsrc = _make_element("appsrc", "source")
        queue = _make_element("queue", "input_queue")
        convert = _make_element("videoconvert", "convert")
        capsfilter = _make_element("capsfilter", "encoder_caps")
        encoder = _make_element(self.encoder_name, "encoder")
        parser = _make_element(_DEFAULT_PARSERS[self.codec_family], "parser")
        muxer_name = self._select_muxer()
        muxer = _make_element(muxer_name, "muxer")
        sink = _make_element("filesink", "sink")

        input_caps = Gst.Caps.from_string(
            "video/x-raw,"
            f"format={self.gst_input_format},"
            f"width={self.in_w},"
            f"height={self.in_h},"
            f"framerate={self.fps_num}/{self.fps_den}"
        )
        output_caps = Gst.Caps.from_string(
            f"video/x-raw,format={self.gst_output_format}"
        )

        appsrc.set_property("caps", input_caps)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("is-live", self.is_live)
        appsrc.set_property("block", True)
        appsrc.set_property("do-timestamp", False)

        queue.set_property("max-size-time", 0)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-buffers", 2 if self.low_memory else 8)

        capsfilter.set_property("caps", output_caps)
        self._configure_encoder(encoder)

        _set_property(parser, "config-interval", -1)

        if muxer_name in {"mp4mux", "qtmux"}:
            _set_property(muxer, "faststart", True)
        for key, value in self.muxer_properties.items():
            _set_property(muxer, key, value, required=True)

        sink.set_property("location", str(self.out_path))
        sink.set_property("sync", False)
        sink.set_property("async", False)

        elements = [appsrc, queue, convert, capsfilter, encoder, parser, muxer, sink]
        for element in elements:
            pipeline.add(element)
        _link_many(elements)

        self.pipeline = pipeline
        self.appsrc = appsrc
        self.bus = pipeline.get_bus()
        self.pipeline_description = (
            f"appsrc(BGR/NumPy) ! queue ! videoconvert ! "
            f"{self.gst_output_format} ! {self.encoder_name} ! "
            f"{_DEFAULT_PARSERS[self.codec_family]} ! {muxer_name} ! "
            f"filesink({self.out_path})"
        )

    def _bus_worker(self) -> None:
        message_types = (
            Gst.MessageType.ERROR
            | Gst.MessageType.WARNING
            | Gst.MessageType.EOS
        )
        while not self._stop_bus.is_set():
            message = self.bus.timed_pop_filtered(
                100 * Gst.MSECOND,
                message_types,
            )
            if message is None:
                continue

            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                text = str(error)
                if debug:
                    text += f"\n{debug}"
                self._fatal_error = text
                self._append_message("error", text)
                self._eos_seen.set()
                return

            if message.type == Gst.MessageType.WARNING:
                warning, debug = message.parse_warning()
                text = str(warning)
                if debug:
                    text += f"\n{debug}"
                self._append_message("warning", text)
                continue

            if message.type == Gst.MessageType.EOS:
                self._append_message("info", "end of stream")
                self._eos_seen.set()
                return

    def open(self) -> "GStreamerPipeWriter":
        with self._lock:
            if self._opened and not self._closed:
                return self
            if self._closed:
                raise RuntimeError("writer cannot be reopened after close/abort")

            self._build_pipeline()
            self._stop_bus.clear()
            self._eos_seen.clear()
            self._fatal_error = None

            self._bus_thread = threading.Thread(
                target=self._bus_worker,
                name=f"gstreamer-bus-{self.out_path.name}",
                daemon=True,
            )
            self._bus_thread.start()

            result = self.pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                self.abort()
                raise RuntimeError(
                    "GStreamer pipeline failed to enter PLAYING state:\n"
                    + self.message_tail()
                )

            state_result, current, pending = self.pipeline.get_state(10 * Gst.SECOND)
            if state_result == Gst.StateChangeReturn.FAILURE:
                self.abort()
                raise RuntimeError(
                    "GStreamer pipeline could not start:\n" + self.message_tail()
                )

            self._opened = True
            return self

    def _raise_if_failed(self) -> None:
        if self._fatal_error:
            raise RuntimeError(f"GStreamer pipeline failed:\n{self._fatal_error}")

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if not self._opened:
                self.open()
            if self._closed or self.pipeline is None or self.appsrc is None:
                raise RuntimeError("cannot write to a closed GStreamer writer")
            self._raise_if_failed()

            array = np.asarray(frame)
            if array.dtype != np.uint8:
                raise TypeError(f"frame dtype must be uint8, got {array.dtype}")

            expected_shape = (
                (self.in_h, self.in_w)
                if self.channels == 1
                else (self.in_h, self.in_w, self.channels)
            )
            if array.shape != expected_shape:
                raise ValueError(
                    f"frame shape must be {expected_shape}, got {array.shape}"
                )
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)

            payload = array.tobytes(order="C")
            buffer = Gst.Buffer.new_allocate(None, len(payload), None)
            if buffer is None:
                raise RuntimeError("could not allocate GStreamer buffer")
            buffer.fill(0, payload)

            pts = Gst.util_uint64_scale(
                self.frames_written * self.fps_den,
                Gst.SECOND,
                self.fps_num,
            )
            next_pts = Gst.util_uint64_scale(
                (self.frames_written + 1) * self.fps_den,
                Gst.SECOND,
                self.fps_num,
            )
            buffer.pts = pts
            buffer.dts = pts
            buffer.duration = max(1, next_pts - pts)
            buffer.offset = self.frames_written

            result = self.appsrc.emit("push-buffer", buffer)
            if result != Gst.FlowReturn.OK:
                self._raise_if_failed()
                raise RuntimeError(
                    f"GStreamer appsrc rejected frame {self.frames_written}: {result}"
                )
            self.frames_written += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self.pipeline is None:
                return

            try:
                self._raise_if_failed()
                flow = self.appsrc.emit("end-of-stream")
                if flow != Gst.FlowReturn.OK:
                    raise RuntimeError(
                        f"could not signal end-of-stream to GStreamer: {flow}"
                    )

                if not self._eos_seen.wait(timeout=self.close_timeout):
                    raise RuntimeError(
                        f"GStreamer did not finish within {self.close_timeout:g}s"
                    )
                self._raise_if_failed()
            finally:
                self._stop_bus.set()
                self.pipeline.set_state(Gst.State.NULL)
                if self._bus_thread is not None:
                    self._bus_thread.join(timeout=3)

            if self.frames_written > 0:
                try:
                    size = self.out_path.stat().st_size
                except OSError:
                    size = 0
                if size <= 0:
                    raise RuntimeError(
                        f"GStreamer reported success but produced no output: "
                        f"{self.out_path}"
                    )

    def abort(self) -> None:
        with self._lock:
            if self._closed and self.pipeline is None:
                return
            self._closed = True
            self._stop_bus.set()

            if self.pipeline is not None:
                try:
                    self.pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass

            if self._bus_thread is not None:
                self._bus_thread.join(timeout=2)

    def __enter__(self) -> "GStreamerPipeWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False
