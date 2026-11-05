# Media Processing Toolkit

A focused Python toolkit for FFmpeg-based video processing, image/video
conversion, downloads, and visual-similarity workflows.

## Install

Python 3.10 or newer is required. FFmpeg commands also require `ffmpeg` and
`ffprobe` on `PATH`.

```powershell
pip install -r requirements.txt
```

For editable installation and command aliases:

```powershell
pip install -e .
```

## Run a tool

Every executable supports the same three configuration modes:

```powershell
# Load the script's section from the nearest args.yaml
python video_compressor.py

# Load the script's section from an explicit YAML file
python video_compressor.py D:\configs\video.yaml

# Load args.yaml, then override selected values on the command line
python video_compressor.py -i video.mp4 -c hevc -q high
```

The available YAML sections are `ffmpeg_manager`, `video_compressor`,
`youtube_download`, `image_video_converter`, `image_similarity`, and
`dedupe_by_similarity`. Parsers tolerate unknown arguments so these tools can
also be embedded in larger launchers.

All processing commands expose short and long option names; use `--help` for
the complete interface.

## Commands

| Task | Script | Installed command |
| --- | --- | --- |
| Estimate video bitrate | `ffmpeg_manager.py` | `media-bitrate` |
| Compress videos | `video_compressor.py` | `media-compress` |
| Convert videos/images | `image_video_converter.py` | `media-convert` |
| Download media | `youtube_download.py` | `media-download` |
| Cluster similar images | `python -m image_similarity` | `image-cluster` |
| Remove similar frames | `image_similarity/dedupe_by_similarity.py` | `video-dedupe` |

Examples:

```powershell
python video_compressor.py -i D:\videos -c hevc -H 1080
python image_video_converter.py -m images-to-video -i D:\frames -o movie.mp4 -r 25
python image_video_converter.py -m video-to-images -i movie.mp4 -o D:\frames -s 1
python youtube_download.py -i urls.txt -o D:\downloads
python -m image_similarity -i D:\images -m average-hash
python image_similarity\dedupe_by_similarity.py -i movie.mp4 -t 0.95
```

Input arguments accept individual files, directories, globs, and—where
applicable—TXT lists. Relative entries in a TXT list are resolved from that
list's directory. Completed video outputs are published atomically so failed
jobs do not replace a destination.

## Logs and progress

Each executable records debug, info, and error events under
`logs/<feature>/log_1.txt`. Logs rotate through `log_5.txt`, with at most 3,000
lines per file. Long-running and batch operations display progress bars; both
progress and verbose console output are configurable.

## Public modules

- `toolkit_runtime.py` owns YAML parsing, rotating logs, progress, input
  resolution, natural sorting, and safe output-path helpers.
- `ffmpeg_manager.py` owns bitrate calculation, hardware probes, FFmpeg command
  execution, and raw-frame writers.
- `video_compressor.py`, `image_video_converter.py`, and
  `youtube_download.py` expose reusable functions as well as CLIs.
- `image_similarity/` contains image clustering and video-frame deduplication.

See [image_similarity/README.md](image_similarity/README.md) for similarity
methods and safety behavior.

## Test

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Resize a video

```powershell
python resize_video.py -i "D:\videos\input.mp4" -o "D:\videos\resized.mp4" --width 1280 --height 720
```

The output uses the exact requested dimensions (stretching if the aspect ratio
changes). Audio is retained and encoded as AAC. Without `--output`, the result
is saved beside the input as `INPUT_WIDTHxHEIGHT.mp4`. Existing files are
protected unless `--overwrite` is supplied. Odd dimensions use H.264 4:4:4;
use even dimensions for wider player compatibility. FFmpeg must be on PATH.
After editable installation, `media-resize` provides the same command.
You can also set `input`, `output`, `width`, and `height` in a `resize_video`
section of `args.yaml`.
