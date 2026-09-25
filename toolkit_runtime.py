"""Shared configuration, logging, progress, and filesystem helpers."""

from __future__ import annotations

import argparse, glob, logging, re, sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

_BUILTIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "ffmpeg_manager": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "codec": "h264",
        "quality": "medium",
        "bitrate_kbps": None,
        "verbose": False,
    },
    "video_compressor": {
        "inputs": [],
        "output_dir": None,
        "suffix": "_compressed",
        "codec": "h264",
        "quality": "medium",
        "max_height": 1080,
        "audio_kbps": 128,
        "preset": "medium",
        "two_pass": False,
        "hw": "auto",
        "overwrite": False,
        "show_progress": True,
        "fail_fast": False,
        "ffmpeg_binary": "ffmpeg",
        "ffprobe_binary": "ffprobe",
        "verbose": False,
    },
    "youtube_download": {
        "inputs": [],
        "output": None,
        "format_selector": "bestvideo+bestaudio/best",
        "merge_format": "mp4",
        "archive": None,
        "no_playlist": False,
        "fail_fast": False,
        "quiet": False,
        "show_progress": True,
        "verbose": False,
    },
    "image_video_converter": {
        "mode": "video-to-images",
        "inputs": [],
        "output": None,
        "fps": 1,
        "interval_seconds": None,
        "image_extension": "jpg",
        "image_quality": 95,
        "codec": None,
        "resize_frames": False,
        "overwrite": False,
        "show_progress": True,
        "fail_fast": False,
        "verbose": False,
    },
    "image_similarity": {
        "source": None,
        "method": "average-hash",
        "threshold": None,
        "output": None,
        "move": False,
        "recursive": True,
        "include_singletons": True,
        "overwrite": False,
        "show_progress": True,
        "verbose": False,
    },
    "dedupe_by_similarity": {
        "inputs": [],
        "output_dir": None,
        "suffix": "_deduped",
        "threshold": 0.95,
        "pixel_tolerance": 8,
        "resize_width": 320,
        "blur_kernel_size": 3,
        "keep_dup_type": "consecutive",
        "minimum_keep_frames": 15,
        "quality": "medium",
        "preset": "medium",
        "max_height": 1080,
        "overwrite": False,
        "move_to_trash": False,
        "trash_dir": None,
        "show_progress": True,
        "fail_fast": False,
        "ffmpeg_binary": "ffmpeg",
        "ffprobe_binary": "ffprobe",
        "verbose": False,
    },
}

T = TypeVar("T")


class LineRotatingFileHandler(logging.Handler):
    """Rotate ``log_1.txt`` through ``log_5.txt`` by line count."""

    def __init__(
        self,
        log_directory: Path,
        *,
        max_lines: int = 3_000,
        backup_count: int = 5,
    ) -> None:
        super().__init__()
        if max_lines <= 0 or backup_count <= 0:
            raise ValueError("max_lines and backup_count must be positive.")

        self.log_directory = Path(log_directory)
        self.max_lines = int(max_lines)
        self.backup_count = int(backup_count)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.current_path = self.log_directory / "log_1.txt"
        self._line_count = self._count_lines(self.current_path)

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return sum(1 for _line in stream)

    def _rotate(self) -> None:
        oldest = self.log_directory / f"log_{self.backup_count}.txt"
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self.log_directory / f"log_{index}.txt"
            if source.exists():
                source.replace(
                    self.log_directory / f"log_{index + 1}.txt"
                )
        self._line_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._line_count >= self.max_lines:
                self._rotate()
            message = self.format(record)
            with self.current_path.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
            self._line_count += message.count("\n") + 1
        except Exception:
            self.handleError(record)


def configure_logging(
    script_path: Path,
    feature_name: str,
    *,
    verbose: bool = False,
) -> logging.Logger:
    """Configure an isolated console and rotating-file logger."""

    logger = logging.getLogger(feature_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    marker = f"media-processing-toolkit:{feature_name}"

    existing_handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_toolkit_marker", None) == marker
    ]
    if not existing_handlers:
        config_path = default_yaml_path(script_path)
        log_root = (
            config_path.parent if config_path.is_file() else Path.cwd()
        )
        file_handler = LineRotatingFileHandler(
            log_root / "logs" / feature_name
        )
        file_handler._toolkit_marker = marker
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler._toolkit_marker = marker
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s: %(message)s")
        )
        logger.addHandler(console_handler)

    console_level = logging.DEBUG if verbose else logging.INFO
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler,
            LineRotatingFileHandler,
        ):
            handler.setLevel(console_level)

    logger.debug("Logging initialized for %s", feature_name)
    return logger


def default_yaml_path(script_path: Path) -> Path:
    """Find ``args.yaml`` beside a script or one directory above it."""

    candidates = (
        script_path.parent / "args.yaml",
        script_path.parent.parent / "args.yaml",
    )
    return next(
        (candidate for candidate in candidates if candidate.is_file()),
        candidates[0],
    )


def load_yaml_defaults(
    script_path: Path,
    feature_name: str,
    argv: Sequence[str] | None,
) -> tuple[dict[str, Any], list[str], Path]:
    """Load defaults and remove a sole explicit YAML path from CLI arguments."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    fallback_path = default_yaml_path(script_path)
    if any(argument in {"-h", "--help"} for argument in arguments):
        return {}, arguments, fallback_path

    if (
        len(arguments) == 1
        and not arguments[0].startswith("-")
        and Path(arguments[0]).suffix.lower() in {".yaml", ".yml"}
    ):
        yaml_path = Path(arguments[0]).expanduser().resolve()
        parse_arguments: list[str] = []
    else:
        yaml_path = fallback_path
        parse_arguments = arguments

    if not yaml_path.is_file():
        if parse_arguments:
            return (
                dict(_BUILTIN_DEFAULTS.get(feature_name, {})),
                parse_arguments,
                yaml_path,
            )
        raise FileNotFoundError(f"YAML configuration not found: {yaml_path}")

    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyYAML is required; install packages from requirements.txt."
        ) from error

    with yaml_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")

    defaults = dict(_BUILTIN_DEFAULTS.get(feature_name, {}))
    configured_defaults = loaded.get(feature_name, {})
    if configured_defaults is None:
        configured_defaults = {}
    if not isinstance(configured_defaults, Mapping):
        raise ValueError(
            f"YAML section {feature_name!r} must be a mapping: {yaml_path}"
        )
    defaults.update(configured_defaults)
    return defaults, parse_arguments, yaml_path


def parse_yaml_args(
    parser: argparse.ArgumentParser,
    script_path: Path,
    feature_name: str,
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str], Path]:
    """Apply YAML defaults and parse known CLI arguments."""

    defaults, parse_arguments, yaml_path = load_yaml_defaults(
        script_path,
        feature_name,
        argv,
    )
    parser.set_defaults(**defaults)
    arguments, unknown_arguments = parser.parse_known_args(parse_arguments)
    return arguments, unknown_arguments, yaml_path


class ProgressBar:
    """Progress adapter with a lightweight fallback when tqdm is unavailable."""

    def __init__(
        self,
        description: str,
        *,
        total: float | None = None,
        unit: str = "item",
        enabled: bool = True,
    ) -> None:
        self.description = description
        self.total = total
        self.unit = unit
        self.enabled = enabled
        self.current = 0.0
        self._bar: Any = None
        if enabled:
            try:
                from tqdm import tqdm

                self._bar = tqdm(
                    total=total,
                    desc=description,
                    unit=unit,
                    dynamic_ncols=True,
                )
            except ModuleNotFoundError:
                self._bar = None

    def update(self, amount: float = 1.0) -> None:
        if amount <= 0:
            return
        self.current += amount
        if self._bar:
            self._bar.update(amount)
        elif self.enabled and int(self.current) % 100 == 0:
            total_text = f"/{self.total:g}" if self.total is not None else ""
            print(
                f"\r{self.description}: {self.current:g}{total_text} {self.unit}",
                end="",
                flush=True,
            )

    def update_to(self, value: float) -> None:
        self.update(max(value - self.current, 0.0))

    def close(self) -> None:
        if self._bar:
            self._bar.close()
        elif self.enabled and self.current >= 100:
            print()

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def progress_iter(
    values: Iterable[T],
    description: str,
    *,
    enabled: bool = True,
) -> Iterator[T]:
    """Iterate through materialized values with a progress bar."""

    items = list(values)
    with ProgressBar(
        description,
        total=float(len(items)),
        enabled=enabled,
    ) as progress:
        for item in items:
            yield item
            progress.update()


def natural_sort_key(path: Path | str) -> tuple[Any, ...]:
    """Sort paths naturally so item 2 appears before item 10."""

    parts = re.split(r"(\d+)", str(path).casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def resolve_files(
    inputs: Iterable[Path | str],
    *,
    extensions: Iterable[str],
    recursive: bool = True,
    allow_text_lists: bool = True,
) -> list[Path]:
    """Expand files, directories, globs, and optional text-file lists."""

    normalized_extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }
    discovered: list[Path] = []

    def add_input(value: Path | str, base_dir: Path | None = None) -> None:
        raw = str(value).strip()
        if not raw or raw.startswith("#"):
            return
        candidate_text = str(base_dir / raw) if base_dir else raw

        if glob.has_magic(candidate_text):
            matches = glob.glob(candidate_text, recursive=recursive)
            if not matches:
                raise FileNotFoundError(
                    f"Input pattern matched no files: {candidate_text}"
                )
            for match in matches:
                add_input(match)
            return

        path = Path(candidate_text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            discovered.extend(
                child.resolve()
                for child in iterator
                if child.is_file()
                and child.suffix.lower() in normalized_extensions
            )
            return
        if allow_text_lists and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                add_input(line, path.parent)
            return
        if path.suffix.lower() not in normalized_extensions:
            supported = ", ".join(sorted(normalized_extensions))
            raise ValueError(
                f"Unsupported input extension for {path}; choose {supported}."
            )
        discovered.append(path.resolve())

    for input_value in inputs:
        add_input(input_value)
    return sorted(set(discovered), key=natural_sort_key)


def increment_path(path: Path) -> Path:
    """Return a non-existing path by appending an incrementing suffix."""

    if not path.exists():
        return path
    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not allocate a unique path near {path}.")


def partial_output_path(output_path: Path) -> Path:
    """Build a sibling temporary path while preserving the final extension."""

    return output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )
