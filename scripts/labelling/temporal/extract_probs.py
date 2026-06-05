"""Stage 1: extract per-frame eye/mouth probabilities + upper-body keypoints.

``--video`` accepts either a single video file or a directory of videos;
``--output`` is always a directory. One Parquet per input video is written
as ``<output>/<video_stem>.parquet``, with one row per processed frame.

Model paths default to ``scripts/weights_path.py`` (repo-root landmarker,
HF eye/mouth/pose weights). Run from the repository root.

Usage::

    python -m scripts.labelling.temporal.extract_probs \\
        --video videos/vid_001.mp4 \\
        --output data/temporal/raw_probs

    python -m scripts.labelling.temporal.extract_probs \\
        --video videos \\
        --output data/temporal/raw_probs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from fatigue_pipeline import FatiguePipeline
from fatigue_pipeline.constants import DEFAULT_FPS, UPPER_BODY_KPT_NAMES
from fatigue_pipeline.pose_estimator import PoseEstimator

from scripts.labelling.temporal._defaults import (
    DEFAULT_RAW_PROBS_DIR,
    DEFAULT_VIDEOS_DIR,
    resolve_weights,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

KPT_COLUMNS: list[str] = []
for _name in UPPER_BODY_KPT_NAMES:
    KPT_COLUMNS.extend([f"kp_{_name}_x", f"kp_{_name}_y", f"kp_{_name}_conf"])


def _empty_kpt_row() -> dict[str, float]:
    return {col: np.nan for col in KPT_COLUMNS}


def _kpt_row(kpts: np.ndarray) -> dict[str, float]:
    row: dict[str, float] = {}
    for i, name in enumerate(UPPER_BODY_KPT_NAMES):
        x, y, c = kpts[i]
        row[f"kp_{name}_x"] = float(x) if np.isfinite(x) else np.nan
        row[f"kp_{name}_y"] = float(y) if np.isfinite(y) else np.nan
        row[f"kp_{name}_conf"] = float(c) if np.isfinite(c) else np.nan
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Single video file or directory of videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RAW_PROBS_DIR,
        help="Output directory; one Parquet (+ CSV) per input video.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Process every Nth video frame (default 5 ~ 6 fps from 30 fps).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process videos whose Parquet already exists.",
    )
    return parser.parse_args()


def collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {path.suffix}")
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file())
        videos = [p for p in files if p.suffix.lower() in VIDEO_EXTENSIONS]
        skipped = [p for p in files if p.suffix.lower() not in VIDEO_EXTENSIONS]
        if skipped:
            print(
                f"Skipped {len(skipped)} unsupported file(s) in {path}: "
                + ", ".join(p.name for p in skipped)
            )
        if not videos:
            raise FileNotFoundError(
                f"No supported videos found in {path}. "
                f"Supported extensions: {sorted(VIDEO_EXTENSIONS)}"
            )
        return videos
    raise FileNotFoundError(f"Video path missing: {path}")


def extract_one(
    pipeline: FatiguePipeline,
    pose_estimator: PoseEstimator,
    video_path: Path,
    output_path: Path,
    csv_path: Path,
    frame_step: int,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    effective_fps = fps / frame_step
    pipeline.reset(fps=effective_fps)

    rows: list[dict] = []
    video_frame_idx = 0

    try:
        with tqdm(total=total, desc=video_path.stem, unit="frame") as pbar:
            while cap.isOpened():
                if video_frame_idx % frame_step != 0:
                    if not cap.grab():
                        break
                    video_frame_idx += 1
                    pbar.update(1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                probs = pipeline.process_frame(frame)
                kpts = pose_estimator.predict(frame)
                row = {
                    "frame_idx": video_frame_idx,
                    "timestamp_s": video_frame_idx / fps,
                    "face_detected": probs.face_detected,
                    "p_eye_left_closed": (
                        probs.p_eye_left_closed
                        if probs.p_eye_left_closed is not None
                        else np.nan
                    ),
                    "p_eye_right_closed": (
                        probs.p_eye_right_closed
                        if probs.p_eye_right_closed is not None
                        else np.nan
                    ),
                    "p_mouth_open": (
                        probs.p_mouth_open if probs.p_mouth_open is not None else np.nan
                    ),
                    "person_detected": kpts is not None,
                }
                row.update(_empty_kpt_row() if kpts is None else _kpt_row(kpts))
                rows.append(row)

                video_frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()

    dtypes: dict[str, str] = {
        "frame_idx": "int32",
        "timestamp_s": "float32",
        "face_detected": "bool",
        "p_eye_left_closed": "float32",
        "p_eye_right_closed": "float32",
        "p_mouth_open": "float32",
        "person_detected": "bool",
    }
    for col in KPT_COLUMNS:
        dtypes[col] = "float32"
    df = pd.DataFrame(rows).astype(dtypes)
    df.to_parquet(output_path, index=False)
    df.to_csv(csv_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    if args.frame_step <= 0:
        raise ValueError(f"--frame-step must be > 0, got {args.frame_step}")
    videos = collect_videos(args.video)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    face_path, eye_path, mouth_path, pose_path = resolve_weights()
    pipeline = FatiguePipeline(
        face_landmarker_path=face_path,
        eye_classifier_path=eye_path,
        mouth_classifier_path=mouth_path,
    )
    pose_estimator = PoseEstimator(weights_path=pose_path)

    try:
        for video_path in videos:
            output_path = args.output / f"{video_path.stem}.parquet"
            csv_path = csv_dir / f"{video_path.stem}.csv"
            if output_path.exists() and not args.overwrite:
                print(f"Skip (exists): {output_path}")
                continue

            df = extract_one(
                pipeline,
                pose_estimator,
                video_path,
                output_path,
                csv_path,
                frame_step=args.frame_step,
            )
            face_pct = df["face_detected"].mean() * 100.0
            pose_pct = df["person_detected"].mean() * 100.0
            print(
                f"{video_path.stem}: {len(df):,} rows -> {output_path} "
                f"(face {face_pct:.1f}% / pose {pose_pct:.1f}%)"
            )
    finally:
        pipeline.close()
        pose_estimator.close()


if __name__ == "__main__":
    main()
