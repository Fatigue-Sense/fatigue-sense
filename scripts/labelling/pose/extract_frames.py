"""Stage 0: sample frames from raw videos for pose retraining.

Walks the input videos directory, samples one frame every
``--sample-stride-s`` seconds, and drops near-duplicate frames using a
downscaled mean-squared-error check. Writes JPEGs named
``<video_stem>_<frame_idx:06d>.jpg`` to the output directory.

The flat staging directory is consumed by ``pseudo_label.py``.

Usage::

    python -m scripts.labelling.pose.extract_frames

    python -m scripts.labelling.pose.extract_frames \\
        --videos-dir videos \\
        --output-dir data/pose/frames
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from scripts.labelling.pose._defaults import DEFAULT_FRAMES_DIR, DEFAULT_VIDEOS_DIR

DEFAULT_VIDEO_EXTS = (".mp4", ".mov", ".MOV", ".avi", ".mkv")
DEFAULT_SAMPLE_STRIDE_S = 1.0
DEFAULT_DEDUPE_THUMB_SIZE = (64, 64)
DEFAULT_DEDUPE_MSE_THRESHOLD = 25.0
DEFAULT_JPEG_QUALITY = 92
DEFAULT_TRIM_LEAD_S = 0.5
DEFAULT_TRIM_TAIL_S = 0.5


@dataclass
class SampleConfig:
    videos_dir: Path
    output_dir: Path
    video_exts: tuple[str, ...]
    sample_stride_s: float
    dedupe_thumb_size: tuple[int, int]
    dedupe_mse_threshold: float
    jpeg_quality: int
    trim_lead_s: float
    trim_tail_s: float


@dataclass
class VideoStats:
    name: str
    candidates: int
    kept: int
    dropped_duplicate: int


def _list_videos(root: Path, exts: tuple[str, ...]) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.suffix in exts)


def _thumb_mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def _process_video(video_path: Path, cfg: SampleConfig) -> VideoStats:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(cfg.sample_stride_s * fps)))
    lead = int(round(cfg.trim_lead_s * fps))
    tail_cutoff = max(0, total_frames - int(round(cfg.trim_tail_s * fps)))

    stem = video_path.stem
    last_thumb: np.ndarray | None = None
    candidates = 0
    kept = 0
    dropped = 0

    pbar = tqdm(total=total_frames, desc=stem, leave=False)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        in_range = lead <= frame_idx < tail_cutoff
        on_stride = (frame_idx % stride) == 0
        if in_range and on_stride:
            candidates += 1
            thumb = cv2.resize(
                frame, cfg.dedupe_thumb_size, interpolation=cv2.INTER_AREA
            )
            thumb_gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)

            is_duplicate = (
                last_thumb is not None
                and _thumb_mse(thumb_gray, last_thumb) < cfg.dedupe_mse_threshold
            )

            if is_duplicate:
                dropped += 1
            else:
                out_path = cfg.output_dir / f"{stem}_{frame_idx:06d}.jpg"
                cv2.imwrite(
                    str(out_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
                )
                kept += 1
                last_thumb = thumb_gray

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return VideoStats(stem, candidates, kept, dropped)


def _parse_thumb_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"thumb-size must be WxH (e.g. 64x64), got {value!r}"
        )
    return int(parts[0]), int(parts[1])


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    p.add_argument(
        "--video-exts",
        nargs="+",
        default=list(DEFAULT_VIDEO_EXTS),
        help="Allowed video file extensions (with leading dot).",
    )
    p.add_argument("--sample-stride-s", type=float, default=DEFAULT_SAMPLE_STRIDE_S)
    p.add_argument(
        "--dedupe-thumb-size",
        type=_parse_thumb_size,
        default=DEFAULT_DEDUPE_THUMB_SIZE,
        help="Downscale size for similarity check, format WxH.",
    )
    p.add_argument(
        "--dedupe-mse-threshold",
        type=float,
        default=DEFAULT_DEDUPE_MSE_THRESHOLD,
        help="Below this MSE the new frame is treated as a duplicate.",
    )
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument("--trim-lead-s", type=float, default=DEFAULT_TRIM_LEAD_S)
    p.add_argument("--trim-tail-s", type=float, default=DEFAULT_TRIM_TAIL_S)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    cfg = SampleConfig(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        video_exts=tuple(args.video_exts),
        sample_stride_s=args.sample_stride_s,
        dedupe_thumb_size=tuple(args.dedupe_thumb_size),
        dedupe_mse_threshold=args.dedupe_mse_threshold,
        jpeg_quality=args.jpeg_quality,
        trim_lead_s=args.trim_lead_s,
        trim_tail_s=args.trim_tail_s,
    )

    if not cfg.videos_dir.exists():
        raise FileNotFoundError(f"{cfg.videos_dir} not found")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    videos = _list_videos(cfg.videos_dir, cfg.video_exts)
    if not videos:
        raise RuntimeError(
            f"No videos in {cfg.videos_dir} matching extensions {cfg.video_exts}"
        )

    print(f"Sampling {len(videos)} videos -> {cfg.output_dir}")
    all_stats: list[VideoStats] = []
    for v in videos:
        stats = _process_video(v, cfg)
        all_stats.append(stats)
        print(
            f"  {stats.name}: kept={stats.kept} "
            f"dropped_dup={stats.dropped_duplicate} "
            f"candidates={stats.candidates}"
        )

    total_kept = sum(s.kept for s in all_stats)
    total_dropped = sum(s.dropped_duplicate for s in all_stats)
    print(f"Done. kept={total_kept} dropped={total_dropped}")


if __name__ == "__main__":
    main()
