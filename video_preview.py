"""Generate and display 16, 32, 48, or 64-frame video previews with fast MP4 keyframe range extraction."""

from __future__ import annotations

import argparse, glob, logging, math, os, platform, re, struct, sys, tempfile
import time, cv2, numpy as np, requests, yaml
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from requests.adapters import HTTPAdapter
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from toolkit_runtime import ProgressBar, configure_logging, natural_sort_key

HERE = Path(__file__).resolve()
FEATURE_NAME = HERE.stem
LOGGER = logging.getLogger(FEATURE_NAME)

GRID_LAYOUTS: dict[int, tuple[int, int]] = {
    4: (2, 2),
    6: (2, 3),
    8: (2, 4),
    9: (3, 3),
    12: (3, 4),
    16: (4, 4),
    20: (4, 5),
    24: (4, 6),
    30: (5, 6),
    32: (4, 8),
    36: (6, 6),
    40: (5, 8),
    48: (6, 8),
    64: (8, 8),
}
VIDEO_EXTENSIONS = frozenset(
    {".avi", ".flv", ".lrf", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".webm", ".wmv"}
)


@dataclass(frozen=True)
class SampledFrame:
    """One sampled video frame and its metadata."""

    slot_index: int
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray


@dataclass(frozen=True)
class PreviewResult:
    """Summary of a completed video preview operation."""

    source: str
    num_frames: int
    sampled_count: int
    source_fps: float
    total_frames: int
    grid_shape: tuple[int, int]
    output_path: Path | None
    displayed: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class _Mp4VideoIndex:
    """Parsed MP4/M4V video sample table for direct keyframe byte-range reads."""

    codec_ext: str
    codec_headers: bytes
    nal_length_size: int
    keyframes: list[int]
    offsets: list[int]
    sizes: list[int]
    total_frames: int
    fps: float


def is_display_available() -> bool:
    """Detect whether a graphical display is available on Windows or Linux."""
    system = platform.system()
    if system == "Windows":
        return True
    if system == "Linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def setup_opencv_window(
    window_name: str,
    width: int,
    height: int,
    x: int = 100,
    y: int = 100,
) -> bool:
    """Create, resize, and position an OpenCV window when a display is available."""
    if not is_display_available():
        return False
    if width <= 0 or height <= 0:
        raise ValueError("Window width and height must be positive integers.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(width), int(height))
    cv2.moveWindow(window_name, int(x), int(y))
    return True


def is_url(value: str) -> bool:
    """Return whether ``value`` is an HTTP or HTTPS URL."""
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_download_url(url: str, session: requests.Session | None = None) -> str:
    """Resolve a landing-page URL (such as filester.gg/d/<slug> download button) to a direct media URL."""
    raw = str(url).strip()
    if not is_url(raw):
        return raw

    parsed = urlparse(raw)
    path_parts = [p for p in parsed.path.split("/") if p]
    origin = f"{parsed.scheme}://{parsed.netloc}"
    client = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1. Direct support for filester (/d/<slug>) download button API (/v2/api/public/download)
    if len(path_parts) >= 2 and path_parts[0] == "d":
        file_slug = path_parts[-1]
        for endpoint in ("/v2/api/public/download", "/v2/api/public/view"):
            try:
                resp = client.post(
                    f"{origin}{endpoint}",
                    json={"file_slug": file_slug},
                    headers=headers,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("token") and data.get("file"):
                        base = data.get("server") or "https://cn1.filester.me"
                        direct = f"{base}/v2/{data['file']}?token={quote(str(data['token']))}&download=true"
                        if data.get("name"):
                            direct += f"&n={quote(str(data['name']))}"
                        LOGGER.info("Resolved download button URL for %s -> %s", raw, direct)
                        return direct
            except Exception as exc:
                LOGGER.debug("Download token API %s failed for %s: %s", endpoint, raw, exc)

    # 2. If already a direct media URL (has video extension or token= query), return as-is
    ext = Path(parsed.path).suffix.lower()
    if ext in VIDEO_EXTENSIONS or "token=" in parsed.query.lower():
        return raw

    # 3. Generic HTML page inspection for #downloadButton or <video>/<source>
    try:
        page = client.get(raw, headers=headers, timeout=15)
        ctype = page.headers.get("Content-Type", "").lower()
        if "text/html" in ctype and page.text:
            html = page.text
            for pattern in (
                r'id=["\']downloadButton["\'][^>]*href=["\']([^"\'#]+)["\']',
                r'href=["\']([^"\'#]+)["\'][^>]*id=["\']downloadButton["\']',
                r'window\.fileStreamURL\s*=\s*["\']([^"\']+)["\']',
                r'<source[^>]+src=["\']([^"\']+)["\']',
                r'<video[^>]+src=["\']([^"\']+)["\']',
            ):
                match = re.search(pattern, html, flags=re.IGNORECASE)
                if match and match.group(1).strip():
                    resolved = urljoin(raw, match.group(1).strip())
                    LOGGER.info("Extracted media URL from page %s -> %s", raw, resolved)
                    return resolved
    except Exception as exc:
        LOGGER.debug("HTML inspection failed for %s: %s", raw, exc)

    return raw


def _locate_local_path(raw: str) -> Path:
    """Resolve a local path against CWD, the user's home directory, or the script folder."""
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if not candidate.is_absolute():
        for base in (Path.home(), HERE.parent):
            alt = (base / candidate).expanduser()
            if alt.exists():
                return alt.resolve()
    raise FileNotFoundError(f"Input video not found: {candidate.resolve()}")


def resolve_preview_sources(
    raw_inputs: str | Path | Iterable[str | Path],
    *,
    recursive: bool = True,
) -> list[str]:
    """Expand a string, Path, or list of files/directories/globs/URLs into preview sources."""
    if isinstance(raw_inputs, (str, Path)):
        items: list[str | Path] = [raw_inputs]
    else:
        items = list(raw_inputs)

    resolved: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text.startswith("#"):
            continue
        if is_url(text):
            resolved.append(text)
            continue

        if glob.has_magic(text):
            matches = glob.glob(text, recursive=recursive)
            if not matches and not Path(text).is_absolute():
                for base in (Path.home(), HERE.parent):
                    matches = glob.glob(str(base / text), recursive=recursive)
                    if matches:
                        break
            if not matches:
                raise FileNotFoundError(f"Input pattern matched no files: {text}")
            for match in sorted(matches, key=natural_sort_key):
                match_path = Path(match).resolve()
                if match_path.is_file() and match_path.suffix.lower() in VIDEO_EXTENSIONS:
                    resolved.append(str(match_path))
            continue

        path = _locate_local_path(text)
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            videos = sorted(
                (p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
                key=natural_sort_key,
            )
            resolved.extend(str(v) for v in videos)
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    resolved.extend(resolve_preview_sources(stripped, recursive=recursive))
        else:
            resolved.append(str(path))

    if not resolved:
        raise ValueError("No valid input videos or stream URLs found.")
    return list(dict.fromkeys(resolved))


def source_stem(source: str | Path) -> str:
    """Derive a clean filename stem from a local path or URL."""
    raw = str(source).strip()
    if is_url(raw):
        parsed = urlparse(raw)
        query_params = parse_qs(parsed.query)
        named = query_params.get("n") or query_params.get("filename")
        candidate = unquote(named[0]) if named else unquote(Path(parsed.path).name)
        stem = Path(candidate).stem or "video_preview"
    else:
        stem = Path(raw).stem or "video_preview"
    cleaned = re.sub(r'[\\/*?:"<>|]+', "_", stem).strip(" ._")
    return cleaned[:100] if cleaned else "video_preview"


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1_000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def resolve_grid_shape(
    num_frames: int,
    rows: int | None = None,
    cols: int | None = None,
) -> tuple[int, int]:
    """Determine ``(rows, cols)`` for the frame mosaic."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if rows is not None and cols is not None:
        if rows <= 0 or cols <= 0 or rows * cols < num_frames:
            raise ValueError("rows * cols must be at least num_frames.")
        return int(rows), int(cols)
    if rows is not None:
        if rows <= 0:
            raise ValueError("rows must be positive.")
        return int(rows), math.ceil(num_frames / rows)
    if cols is not None:
        if cols <= 0:
            raise ValueError("cols must be positive.")
        return math.ceil(num_frames / cols), int(cols)
    if num_frames in GRID_LAYOUTS:
        return GRID_LAYOUTS[num_frames]
    for r in range(math.isqrt(num_frames), 0, -1):
        if num_frames % r == 0:
            c = num_frames // r
            if c / r <= 2.5:
                return r, c
    computed_cols = math.ceil(math.sqrt(num_frames))
    computed_rows = math.ceil(num_frames / computed_cols)
    return computed_rows, computed_cols


def resize_with_letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    pad_color: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    """Resize an image into ``(target_width, target_height)`` preserving aspect ratio."""
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("Invalid frame dimensions.")
    scale = min(target_width / w, target_height / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
    x_off = (target_width - new_w) // 2
    y_off = (target_height - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def annotate_tile(
    tile: np.ndarray,
    label: str,
) -> np.ndarray:
    """Draw a readable outlined label in the top-left corner of a frame tile."""
    output = tile.copy()
    h, w = output.shape[:2]
    font_scale = max(0.38, min(w, h) / 360.0)
    thickness = max(1, round(font_scale * 1.8))
    outline = max(2, thickness + 2)
    (_, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    origin = (8, max(text_h + 6, 18))
    cv2.putText(
        output, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (0, 0, 0), outline, lineType=cv2.LINE_AA,
    )
    cv2.putText(
        output, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA,
    )
    return output


def _iter_mp4_boxes(buf: bytes, start: int = 0, end: int | None = None):
    """Yield ``(box_type, payload_start, box_end)`` from an MP4 buffer."""
    limit = len(buf) if end is None else end
    pos = start
    while pos + 8 <= limit:
        size, btype = struct.unpack(">I4s", buf[pos : pos + 8])
        hdr = 8
        if size == 1:
            if pos + 16 > limit:
                break
            size = struct.unpack(">Q", buf[pos + 8 : pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = limit - pos
        if size < hdr or pos + size > limit:
            break
        yield btype, pos + hdr, pos + size
        pos += size


def _find_mp4_child(buf: bytes, target: bytes, start: int, end: int) -> tuple[int, int] | None:
    for btype, s, e in _iter_mp4_boxes(buf, start, end):
        if btype == target:
            return s, e
    return None


def _parse_codec_config(stsd_buf: bytes) -> tuple[str, bytes, int] | None:
    """Extract Annex-B SPS/PPS/VPS headers and NAL length size from ``stsd``."""
    avcc_pos = stsd_buf.find(b"avcC")
    if avcc_pos >= 4:
        avcc = stsd_buf[avcc_pos + 4 :]
        if len(avcc) >= 7:
            nal_len_size = (avcc[4] & 0x03) + 1
            num_sps = avcc[5] & 0x1F
            p = 6
            headers = bytearray()
            for _ in range(num_sps):
                if p + 2 > len(avcc):
                    return None
                length = struct.unpack(">H", avcc[p : p + 2])[0]
                p += 2
                headers += b"\x00\x00\x00\x01" + avcc[p : p + length]
                p += length
            if p < len(avcc):
                num_pps = avcc[p]
                p += 1
                for _ in range(num_pps):
                    if p + 2 > len(avcc):
                        return None
                    length = struct.unpack(">H", avcc[p : p + 2])[0]
                    p += 2
                    headers += b"\x00\x00\x00\x01" + avcc[p : p + length]
                    p += length
            return ".h264", bytes(headers), nal_len_size

    hvcc_pos = stsd_buf.find(b"hvcC")
    if hvcc_pos >= 4:
        hvcc = stsd_buf[hvcc_pos + 4 :]
        if len(hvcc) >= 23:
            nal_len_size = (hvcc[21] & 0x03) + 1
            num_arrays = hvcc[22]
            p = 23
            headers = bytearray()
            for _ in range(num_arrays):
                if p + 3 > len(hvcc):
                    return None
                num_nalus = struct.unpack(">H", hvcc[p + 1 : p + 3])[0]
                p += 3
                for _ in range(num_nalus):
                    if p + 2 > len(hvcc):
                        return None
                    length = struct.unpack(">H", hvcc[p : p + 2])[0]
                    p += 2
                    headers += b"\x00\x00\x00\x01" + hvcc[p : p + length]
                    p += length
            return ".hevc", bytes(headers), nal_len_size

    return None


def _parse_moov_video_index(moov: bytes) -> _Mp4VideoIndex | None:
    """Parse keyframe indices, byte offsets, and sizes from an MP4 ``moov`` box."""
    for btype, ts, te in _iter_mp4_boxes(moov, 0, len(moov)):
        if btype != b"trak":
            continue
        mdia = _find_mp4_child(moov, b"mdia", ts, te)
        if not mdia:
            continue
        hdlr = _find_mp4_child(moov, b"hdlr", *mdia)
        if not hdlr or moov[hdlr[0] + 8 : hdlr[0] + 12] != b"vide":
            continue

        mdhd = _find_mp4_child(moov, b"mdhd", *mdia)
        fps = 25.0
        duration_sec = 0.0
        if mdhd:
            ver = moov[mdhd[0]]
            if ver == 1 and mdhd[0] + 32 <= mdhd[1]:
                timescale, duration = struct.unpack(">IQ", moov[mdhd[0] + 20 : mdhd[0] + 32])
            elif mdhd[0] + 20 <= mdhd[1]:
                timescale, duration = struct.unpack(">II", moov[mdhd[0] + 12 : mdhd[0] + 20])
            else:
                timescale, duration = 0, 0
            if timescale > 0 and duration > 0:
                duration_sec = duration / timescale

        minf = _find_mp4_child(moov, b"minf", *mdia)
        if not minf:
            continue
        stbl = _find_mp4_child(moov, b"stbl", *minf)
        if not stbl:
            continue
        stsd = _find_mp4_child(moov, b"stsd", *stbl)
        stss = _find_mp4_child(moov, b"stss", *stbl)
        stsz = _find_mp4_child(moov, b"stsz", *stbl)
        stsc = _find_mp4_child(moov, b"stsc", *stbl)
        if not (stsd and stss and stsz and stsc):
            continue

        codec_info = _parse_codec_config(moov[stsd[0] : stsd[1]])
        if not codec_info:
            continue
        codec_ext, codec_headers, nal_len_size = codec_info

        kf_count = struct.unpack(">I", moov[stss[0] + 4 : stss[0] + 8])[0]
        keyframes = [
            x - 1
            for x in struct.unpack(f">{kf_count}I", moov[stss[0] + 8 : stss[0] + 8 + 4 * kf_count])
        ]

        fixed_sz, sample_cnt = struct.unpack(">II", moov[stsz[0] + 4 : stsz[0] + 12])
        if sample_cnt <= 0:
            continue
        if fixed_sz:
            sizes = [fixed_sz] * sample_cnt
        else:
            sizes = list(
                struct.unpack(f">{sample_cnt}I", moov[stsz[0] + 12 : stsz[0] + 12 + 4 * sample_cnt])
            )

        stsc_cnt = struct.unpack(">I", moov[stsc[0] + 4 : stsc[0] + 8])[0]
        stsc_entries = [
            struct.unpack(">III", moov[stsc[0] + 8 + i * 12 : stsc[0] + 20 + i * 12])
            for i in range(stsc_cnt)
        ]

        stco = _find_mp4_child(moov, b"stco", *stbl)
        co64 = _find_mp4_child(moov, b"co64", *stbl)
        if stco:
            co_cnt = struct.unpack(">I", moov[stco[0] + 4 : stco[0] + 8])[0]
            chunk_offsets = list(
                struct.unpack(f">{co_cnt}I", moov[stco[0] + 8 : stco[0] + 8 + 4 * co_cnt])
            )
        elif co64:
            co_cnt = struct.unpack(">I", moov[co64[0] + 4 : co64[0] + 8])[0]
            chunk_offsets = list(
                struct.unpack(f">{co_cnt}Q", moov[co64[0] + 8 : co64[0] + 8 + 8 * co_cnt])
            )
        else:
            continue

        offsets = [0] * sample_cnt
        s_idx = 0
        for i, (first_ch, spc, _) in enumerate(stsc_entries):
            next_ch = stsc_entries[i + 1][0] if i + 1 < len(stsc_entries) else len(chunk_offsets) + 1
            for ch in range(first_ch - 1, next_ch - 1):
                if ch >= len(chunk_offsets):
                    break
                off = chunk_offsets[ch]
                for _ in range(spc):
                    if s_idx >= sample_cnt:
                        break
                    offsets[s_idx] = off
                    off += sizes[s_idx]
                    s_idx += 1

        if duration_sec > 0:
            fps = sample_cnt / duration_sec
        return _Mp4VideoIndex(
            codec_ext=codec_ext,
            codec_headers=codec_headers,
            nal_length_size=nal_len_size,
            keyframes=[k for k in keyframes if 0 <= k < sample_cnt],
            offsets=offsets,
            sizes=sizes,
            total_frames=sample_cnt,
            fps=fps,
        )
    return None


def _packet_to_annexb(packet: bytes, nal_len_size: int) -> bytes:
    """Convert length-prefixed MP4 NAL units to Annex-B start-code format."""
    out = bytearray()
    pos = 0
    limit = len(packet)
    while pos + nal_len_size <= limit:
        if nal_len_size == 4:
            length = struct.unpack(">I", packet[pos : pos + 4])[0]
        elif nal_len_size == 2:
            length = struct.unpack(">H", packet[pos : pos + 2])[0]
        else:
            length = packet[pos]
        pos += nal_len_size
        if length <= 0 or pos + length > limit:
            break
        out += b"\x00\x00\x00\x01"
        out += packet[pos : pos + length]
        pos += length
    return bytes(out)


def _decode_annexb_keyframes(
    index: _Mp4VideoIndex,
    chosen_keyframes: Sequence[int],
    packets: Sequence[bytes],
) -> list[SampledFrame]:
    """Decode extracted keyframe packets in a single sequential pass."""
    stream = bytearray()
    valid_keyframes: list[int] = []
    for kf, pkt in zip(chosen_keyframes, packets):
        annexb = _packet_to_annexb(pkt, index.nal_length_size)
        if annexb:
            stream += index.codec_headers
            stream += annexb
            valid_keyframes.append(kf)

    if not stream:
        return []
    # Append a trailing copy of the last frame so raw H.264/HEVC parser flushes all N frames immediately.
    stream += index.codec_headers
    stream += _packet_to_annexb(packets[-1], index.nal_length_size)

    temp_file = tempfile.NamedTemporaryFile(suffix=index.codec_ext, delete=False)
    try:
        temp_file.write(stream)
        temp_file.close()
        cap = cv2.VideoCapture(temp_file.name)
        try:
            decoded: list[SampledFrame] = []
            for slot_idx, kf in enumerate(valid_keyframes):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                decoded.append(
                    SampledFrame(
                        slot_index=slot_idx,
                        frame_index=int(kf),
                        timestamp_seconds=kf / index.fps,
                        image=frame,
                    )
                )
            return decoded
        finally:
            cap.release()
    finally:
        Path(temp_file.name).unlink(missing_ok=True)


def _parse_multipart_byteranges(body: bytes, content_type: str) -> dict[tuple[int, int], bytes]:
    """Parse an HTTP 206 ``multipart/byteranges`` response into ``{(start, end): payload}``."""
    match = re.search(r"boundary=([^\s;]+)", content_type, flags=re.IGNORECASE)
    if not match:
        return {}
    boundary = b"--" + match.group(1).strip('"\'').encode("latin1")
    results: dict[tuple[int, int], bytes] = {}
    search_pos = 0
    while True:
        b_pos = body.find(boundary, search_pos)
        if b_pos == -1:
            break
        header_start = b_pos + len(boundary)
        if body[header_start : header_start + 2] == b"--":
            break
        header_end = body.find(b"\r\n\r\n", header_start)
        if header_end == -1:
            break
        header_text = body[header_start:header_end].decode("latin1", errors="replace")
        cr_match = re.search(r"Content-Range:\s*bytes\s+(\d+)-(\d+)/", header_text, flags=re.IGNORECASE)
        data_start = header_end + 4
        if cr_match:
            start_b = int(cr_match.group(1))
            end_b = int(cr_match.group(2))
            expected_len = end_b - start_b + 1
            results[(start_b, end_b)] = body[data_start : data_start + expected_len]
            search_pos = data_start + expected_len
        else:
            search_pos = data_start
    return results


def _try_fast_mp4_sample(
    source_str: str,
    num_frames: int,
    *,
    page_url: str | None = None,
    show_progress: bool = True,
) -> tuple[list[SampledFrame], float, int] | None:
    """Fast-path MP4/M4V keyframe byte-range extractor for local files and HTTP URLs."""
    try:
        if not is_url(source_str):
            path = Path(source_str)
            if path.suffix.lower() not in {".mp4", ".m4v", ".mov"}:
                return None
            file_size = path.stat().st_size
            with path.open("rb") as f:
                head = f.read(min(file_size, 2 * 1024 * 1024))
                moov: bytes | None = None
                pos = 0
                while pos + 8 <= len(head):
                    sz, bt = struct.unpack(">I4s", head[pos : pos + 8])
                    hdr = 8
                    if sz == 1 and pos + 16 <= len(head):
                        sz = struct.unpack(">Q", head[pos + 8 : head + 16])[0]
                        hdr = 16
                    if sz < hdr:
                        break
                    if bt == b"moov":
                        if pos + sz <= len(head):
                            moov = head[pos + hdr : pos + sz]
                        else:
                            f.seek(pos + hdr)
                            moov = f.read(sz - hdr)
                        break
                    pos += sz
                if moov is None and file_size > len(head):
                    tail_size = min(file_size, 4 * 1024 * 1024)
                    f.seek(file_size - tail_size)
                    tail = f.read(tail_size)
                    m_pos = tail.find(b"moov")
                    if m_pos >= 4:
                        sz = struct.unpack(">I", tail[m_pos - 4 : m_pos])[0]
                        moov = tail[m_pos + 4 : m_pos - 4 + sz]
                if not moov:
                    return None
                index = _parse_moov_video_index(moov)
                if not index or len(index.keyframes) < num_frames:
                    return None
                picks = np.linspace(0, len(index.keyframes) - 1, num=num_frames, dtype=int).tolist()
                chosen_kfs = [index.keyframes[p] for p in picks]
                packets: list[bytes] = []
                with ProgressBar(
                    f"Sampling {num_frames} keyframes",
                    total=float(num_frames),
                    unit="frame",
                    enabled=show_progress,
                ) as progress:
                    for kf in chosen_kfs:
                        f.seek(index.offsets[kf])
                        packets.append(f.read(index.sizes[kf]))
                        progress.update(1.0)
            decoded = _decode_annexb_keyframes(index, chosen_kfs, packets)
            if len(decoded) == num_frames:
                return decoded, index.fps, index.total_frames
            return None

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        r_head = session.get(source_str, headers={"Range": "bytes=0-1048575"}, timeout=15)
        if r_head.status_code not in {200, 206}:
            return None
        raw_head = r_head.content
        moov = None
        pos = 0
        while pos + 8 <= len(raw_head):
            sz, bt = struct.unpack(">I4s", raw_head[pos : pos + 8])
            hdr = 8
            if sz == 1 and pos + 16 <= len(raw_head):
                sz = struct.unpack(">Q", raw_head[pos + 8 : pos + 16])[0]
                hdr = 16
            if sz < hdr:
                break
            if bt == b"moov":
                if pos + sz <= len(raw_head):
                    moov = raw_head[pos + hdr : pos + sz]
                else:
                    r_moov = session.get(
                        source_str,
                        headers={"Range": f"bytes={pos + hdr}-{pos + sz - 1}"},
                        timeout=15,
                    )
                    if r_moov.status_code in {200, 206}:
                        moov = r_moov.content
                break
            pos += sz

        if not moov:
            return None
        index = _parse_moov_video_index(moov)
        if not index or len(index.keyframes) < max(4, num_frames // 2):
            return None

        picks = np.linspace(0, len(index.keyframes) - 1, num=num_frames, dtype=int).tolist()
        chosen_kfs = [index.keyframes[p] for p in picks]
        unique_kfs = sorted(set(chosen_kfs))
        kf_ranges = {kf: (index.offsets[kf], index.offsets[kf] + index.sizes[kf] - 1) for kf in unique_kfs}
        fetched_by_kf: dict[int, bytes] = {}

        with ProgressBar(
            f"Fetching {num_frames} keyframes",
            total=float(len(unique_kfs)),
            unit="frame",
            enabled=show_progress,
        ) as progress:
            active_url = [source_str]
            url_lock = __import__("threading").Lock()

            def _refresh_url(failed_url: str) -> str:
                with url_lock:
                    if active_url[0] != failed_url:
                        return active_url[0]
                    if page_url:
                        refreshed = resolve_download_url(page_url, session=session)
                        if refreshed and refreshed != failed_url:
                            active_url[0] = refreshed
                    return active_url[0]

            def _fetch_one(kf: int) -> tuple[int, bytes]:
                start_b, end_b = kf_ranges[kf]
                expected_len = end_b - start_b + 1
                for attempt in range(5):
                    cur_url = active_url[0]
                    with session.get(
                        cur_url,
                        headers={"Range": f"bytes={start_b}-{end_b}"},
                        stream=True,
                        timeout=15,
                    ) as r:
                        if r.status_code == 206:
                            return kf, r.raw.read(expected_len, decode_content=True)
                        if r.status_code == 200 and start_b == 0:
                            return kf, r.raw.read(expected_len, decode_content=True)
                        status = r.status_code
                    if status in {403, 429} and page_url:
                        _refresh_url(cur_url)
                    time.sleep(0.25 * (attempt + 1))
                raise RuntimeError(f"HTTP {status} fetching keyframe {kf}")

            with ThreadPoolExecutor(max_workers=4) as pool:
                for kf, data in pool.map(_fetch_one, unique_kfs):
                    fetched_by_kf[kf] = data
                    progress.update(1.0)

        if len(fetched_by_kf) < len(unique_kfs):
            return None
        packets = [fetched_by_kf[k] for k in chosen_kfs]
        decoded = _decode_annexb_keyframes(index, chosen_kfs, packets)
        if len(decoded) == num_frames:
            return decoded, index.fps, index.total_frames
    except Exception as exc:
        LOGGER.debug("Fast MP4 keyframe sampler fell back to OpenCV: %s", exc)
    return None


def sample_video_frames(
    source: str | Path,
    num_frames: int = 16,
    *,
    page_url: str | None = None,
    show_progress: bool = True,
) -> tuple[list[SampledFrame], float, int]:
    """Extract ``num_frames`` evenly spaced frames from a video file or stream URL."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be a positive integer, got {num_frames}.")

    source_str = str(source).strip()
    if is_url(source_str):
        if page_url is None:
            page_url = source_str
        source_str = resolve_download_url(source_str)
    else:
        source_str = str(_locate_local_path(source_str))

    fast_result = _try_fast_mp4_sample(
        source_str,
        num_frames,
        page_url=page_url,
        show_progress=show_progress,
    )
    if fast_result is not None:
        return fast_result

    capture = cv2.VideoCapture(source_str)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        sampled: list[SampledFrame] = []
        if total_frames > 0:
            target_indices = np.linspace(
                0, max(total_frames - 1, 0), num=num_frames, dtype=int
            ).tolist()
            last_frame: np.ndarray | None = None
            current_pos = 0

            with ProgressBar(
                f"Sampling {num_frames} frames",
                total=float(num_frames),
                unit="frame",
                enabled=show_progress,
            ) as progress:
                for slot_idx, target_idx in enumerate(target_indices):
                    gap = target_idx - current_pos
                    if gap < 0 or gap > 120:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
                        current_pos = target_idx
                    elif gap > 0:
                        for _ in range(gap):
                            capture.grab()
                            current_pos += 1
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        last_frame = frame
                        current_pos += 1
                    elif last_frame is not None:
                        frame = last_frame.copy()
                    else:
                        continue

                    pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                    timestamp = (
                        pos_msec / 1000.0
                        if pos_msec > 0
                        else target_idx / fps
                    )
                    sampled.append(
                        SampledFrame(
                            slot_index=slot_idx,
                            frame_index=int(target_idx),
                            timestamp_seconds=timestamp,
                            image=frame,
                        )
                    )
                    progress.update(1.0)
        else:
            raw_frames: list[tuple[int, float, np.ndarray]] = []
            idx = 0
            with ProgressBar(
                "Reading stream frames",
                total=None,
                unit="frame",
                enabled=show_progress,
            ) as progress:
                while True:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    raw_frames.append((idx, idx / fps, frame))
                    idx += 1
                    progress.update(1.0)
            total_frames = len(raw_frames)
            if total_frames > 0:
                chosen = np.linspace(
                    0, total_frames - 1, num=min(num_frames, total_frames), dtype=int
                ).tolist()
                for slot_idx, pick in enumerate(chosen):
                    f_idx, ts, img = raw_frames[pick]
                    sampled.append(
                        SampledFrame(
                            slot_index=slot_idx,
                            frame_index=f_idx,
                            timestamp_seconds=ts,
                            image=img,
                        )
                    )
    finally:
        capture.release()

    if not sampled:
        raise RuntimeError(f"No frames could be decoded from {source}.")
    return sampled, fps, total_frames


def build_contact_sheet(
    frames: Sequence[SampledFrame],
    num_frames: int,
    *,
    tile_width: int = 320,
    tile_height: int = 180,
    rows: int | None = None,
    cols: int | None = None,
    annotate: bool = True,
    border_px: int = 2,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Assemble sampled frames into a grid mosaic image."""
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile_width and tile_height must be positive.")
    grid_rows, grid_cols = resolve_grid_shape(num_frames, rows=rows, cols=cols)

    cell_w = tile_width + border_px * 2
    cell_h = tile_height + border_px * 2
    sheet = np.full(
        (grid_rows * cell_h, grid_cols * cell_w, 3),
        (12, 12, 12),
        dtype=np.uint8,
    )

    for idx in range(min(len(frames), grid_rows * grid_cols)):
        sample = frames[idx]
        r, c = divmod(idx, grid_cols)
        tile = resize_with_letterbox(sample.image, tile_width, tile_height)
        if annotate:
            label = f"#{sample.slot_index + 1}/{num_frames} | {format_timestamp(sample.timestamp_seconds)}"
            tile = annotate_tile(tile, label)
        y0 = r * cell_h + border_px
        x0 = c * cell_w + border_px
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile

    return sheet, (grid_rows, grid_cols)


def display_preview(
    frames: Sequence[SampledFrame],
    grid_image: np.ndarray,
    *,
    mode: str = "both",
    window_name: str = "Video Preview",
    window_width: int = 1280,
    window_height: int = 720,
    window_x: int = 100,
    window_y: int = 100,
    playback_fps: float = 4.0,
    wait_ms: int = 0,
    annotate: bool = True,
) -> bool:
    """Display sequential playback and/or contact sheet grid when a display is available."""
    if not setup_opencv_window(
        window_name, window_width, window_height, window_x, window_y
    ):
        LOGGER.info("No graphical display detected; skipping OpenCV window.")
        return False

    delay_ms = max(1, round(1000.0 / max(playback_fps, 0.1)))
    total = len(frames)

    try:
        if mode in {"playback", "both"}:
            for sample in frames:
                canvas = sample.image.copy()
                if annotate:
                    label = (
                        f"Frame {sample.slot_index + 1}/{total} "
                        f"(#{sample.frame_index}) | {format_timestamp(sample.timestamp_seconds)}"
                    )
                    canvas = annotate_tile(canvas, label)
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in {27, ord("q")}:
                    if mode == "playback":
                        return True
                    break

        if mode in {"grid", "both"}:
            cv2.imshow(window_name, grid_image)
            cv2.waitKey(max(0, int(wait_ms)))
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            cv2.destroyAllWindows()

    return True


def create_video_preview(
    source: str | Path,
    *,
    num_frames: int = 16,
    mode: str = "both",
    output_path: str | Path | None = None,
    save_grid: bool = True,
    tile_width: int = 320,
    tile_height: int = 180,
    rows: int | None = None,
    cols: int | None = None,
    annotate: bool = True,
    show_display: bool = True,
    window_name: str = "Video Preview",
    window_width: int = 1280,
    window_height: int = 720,
    window_x: int = 100,
    window_y: int = 100,
    playback_fps: float = 4.0,
    wait_ms: int = 0,
    overwrite: bool = True,
    show_progress: bool = True,
) -> PreviewResult:
    """Sample evenly spaced frames, build a contact sheet, optionally save and display it."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be a positive integer, got {num_frames}.")
    normalized_mode = str(mode or "both").strip().lower()
    if normalized_mode not in {"grid", "playback", "both"}:
        raise ValueError("mode must be 'grid', 'playback', or 'both'.")

    if isinstance(source, (list, tuple, set)):
        sources = resolve_preview_sources(source)
        resolved_source = sources[0]
    else:
        resolved_source = resolve_preview_sources([source])[0]

    page_url: str | None = None
    if is_url(resolved_source):
        page_url = resolved_source
        resolved_source = resolve_download_url(resolved_source)

    started_at = time.monotonic()
    frames, source_fps, total_frames = sample_video_frames(
        resolved_source,
        num_frames=num_frames,
        page_url=page_url,
        show_progress=show_progress,
    )
    grid_image, grid_shape = build_contact_sheet(
        frames,
        num_frames,
        tile_width=tile_width,
        tile_height=tile_height,
        rows=rows,
        cols=cols,
        annotate=annotate,
    )

    saved_path: Path | None = None
    if output_path or save_grid:
        if output_path:
            candidate = Path(output_path).expanduser().resolve()
            if candidate.is_dir() or not candidate.suffix:
                candidate.mkdir(parents=True, exist_ok=True)
                saved_path = candidate / f"{source_stem(resolved_source)}_preview_{num_frames}f.jpg"
            else:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                saved_path = candidate
        else:
            default_dir = HERE.parent / "previews"
            default_dir.mkdir(parents=True, exist_ok=True)
            saved_path = default_dir / f"{source_stem(resolved_source)}_preview_{num_frames}f.jpg"

        if saved_path.exists() and not overwrite:
            raise FileExistsError(f"Output preview already exists: {saved_path}")
        if not cv2.imwrite(str(saved_path), grid_image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write preview image to {saved_path}")
        LOGGER.info("Saved %d-frame preview grid to %s", num_frames, saved_path)

    displayed = False
    if show_display:
        displayed = display_preview(
            frames,
            grid_image,
            mode=normalized_mode,
            window_name=window_name,
            window_width=window_width,
            window_height=window_height,
            window_x=window_x,
            window_y=window_y,
            playback_fps=playback_fps,
            wait_ms=wait_ms,
            annotate=annotate,
        )

    return PreviewResult(
        source=resolved_source,
        num_frames=num_frames,
        sampled_count=len(frames),
        source_fps=source_fps,
        total_frames=total_frames,
        grid_shape=grid_shape,
        output_path=saved_path,
        displayed=displayed,
        elapsed_seconds=time.monotonic() - started_at,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with short and long option names."""
    parser = argparse.ArgumentParser(
        description="Preview 16, 32, 48, or 64 frames from a video file or stream URL."
    )
    parser.add_argument(
        "-i", "--input", "--inputs", dest="input", nargs="+",
        help="Input video file path(s), directory, glob, or stream URL(s).",
    )
    parser.add_argument("-o", "--output", help="Output image path or directory for grid preview.")
    parser.add_argument(
        "-n", "--num-frames", type=int,
        help="Number of frames to preview (e.g., 16, 32, 48, 64, or any positive integer).",
    )
    parser.add_argument(
        "-m", "--mode", choices=["grid", "playback", "both"],
        help="Preview mode: grid mosaic, sequential playback, or both.",
    )
    parser.add_argument("-t", "--tile-width", type=int, help="Tile width in pixels.")
    parser.add_argument("-g", "--tile-height", type=int, help="Tile height in pixels.")
    parser.add_argument("-r", "--rows", type=int, help="Optional custom grid rows.")
    parser.add_argument("-c", "--cols", type=int, help="Optional custom grid columns.")
    parser.add_argument(
        "-a", "--annotate", action=argparse.BooleanOptionalAction,
        help="Overlay frame index and timestamp on each frame.",
    )
    parser.add_argument(
        "-s", "--save-grid", action=argparse.BooleanOptionalAction,
        help="Save the generated contact-sheet grid image to disk.",
    )
    parser.add_argument(
        "-d", "--show-display", action=argparse.BooleanOptionalAction,
        help="Show preview in an OpenCV window when a display is available.",
    )
    parser.add_argument("-N", "--window-name", help="OpenCV window title.")
    parser.add_argument("-W", "--window-width", type=int, help="OpenCV window width.")
    parser.add_argument("-H", "--window-height", type=int, help="OpenCV window height.")
    parser.add_argument("-X", "--window-x", type=int, help="OpenCV window X coordinate.")
    parser.add_argument("-Y", "--window-y", type=int, help="OpenCV window Y coordinate.")
    parser.add_argument("-f", "--playback-fps", type=float, help="Playback speed in frames per second.")
    parser.add_argument("-k", "--wait-ms", type=int, help="Milliseconds to wait on grid window (0 = until key press).")
    parser.add_argument(
        "-w", "--overwrite", action=argparse.BooleanOptionalAction,
        help="Overwrite existing output preview image.",
    )
    parser.add_argument(
        "-P", "--progress", action=argparse.BooleanOptionalAction, dest="show_progress",
        help="Display a progress bar while sampling frames.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def load_yaml_args_dict(
    script_path: Path = HERE,
    argv: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[str], Path]:
    """Load YAML defaults from an explicit YAML argument, same directory, or parent directory."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if (
        len(raw_argv) == 1
        and not raw_argv[0].startswith("-")
        and Path(raw_argv[0]).suffix.lower() in {".yaml", ".yml"}
    ):
        yf = Path(raw_argv[0]).expanduser().resolve()
        cli_argv: list[str] = []
    else:
        yf = script_path.parents[0] / "args.yaml"
        if not yf.is_file() and (script_path.parents[1] / "args.yaml").is_file():
            yf = script_path.parents[1] / "args.yaml"
        cli_argv = raw_argv

    args_dict: dict[str, Any] = {
        "input": None,
        "output": None,
        "num_frames": 16,
        "mode": "both",
        "tile_width": 320,
        "tile_height": 180,
        "rows": None,
        "cols": None,
        "annotate": True,
        "save_grid": True,
        "show_display": True,
        "window_name": "Video Preview",
        "window_width": 1280,
        "window_height": 720,
        "window_x": 100,
        "window_y": 100,
        "playback_fps": 4.0,
        "wait_ms": 0,
        "overwrite": True,
        "show_progress": True,
        "verbose": False,
    }
    if yf.is_file():
        with open(yf, "r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        section = loaded.get(script_path.stem, {}) or {}
        if isinstance(section, dict):
            normalized = {str(k).replace("-", "_"): v for k, v in section.items()}
            if "inputs" in normalized and not normalized.get("input"):
                normalized["input"] = normalized.pop("inputs")
            args_dict.update(normalized)
    elif not cli_argv:
        raise FileNotFoundError(f"YAML configuration not found: {yf}")

    return args_dict, cli_argv, yf


def main(argv: Sequence[str] | None = None) -> int:
    """Run the video preview CLI."""
    parser = build_parser()
    try:
        args_dict, cli_argv, yf = load_yaml_args_dict(HERE, argv)
        parser.set_defaults(**args_dict)
        args, unknown_args = parser.parse_known_args(cli_argv)

        log = configure_logging(HERE, FEATURE_NAME, verbose=bool(args.verbose))
        log.debug("Loaded configuration from %s", yf)
        if unknown_args:
            log.debug("Ignoring unknown arguments: %s", unknown_args)

        if not args.input:
            raise ValueError("No input configured; pass -i/--input or set 'input' in args.yaml.")

        sources = resolve_preview_sources(args.input)
        for source in sources:
            result = create_video_preview(
                source,
                num_frames=int(args.num_frames),
                mode=str(args.mode),
                output_path=args.output,
                save_grid=bool(args.save_grid),
                tile_width=int(args.tile_width),
                tile_height=int(args.tile_height),
                rows=args.rows,
                cols=args.cols,
                annotate=bool(args.annotate),
                show_display=bool(args.show_display),
                window_name=str(args.window_name),
                window_width=int(args.window_width),
                window_height=int(args.window_height),
                window_x=int(args.window_x),
                window_y=int(args.window_y),
                playback_fps=float(args.playback_fps),
                wait_ms=int(args.wait_ms),
                overwrite=bool(args.overwrite),
                show_progress=bool(args.show_progress),
            )
            log.info(
                "Previewed %d/%d frames (%dx%d grid) from %s in %.2f seconds.%s",
                result.sampled_count,
                result.num_frames,
                result.grid_shape[0],
                result.grid_shape[1],
                result.source,
                result.elapsed_seconds,
                f" Saved to {result.output_path}" if result.output_path else "",
            )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
