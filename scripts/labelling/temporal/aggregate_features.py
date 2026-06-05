"""Stage 2: aggregate per-frame probabilities into per-second feature steps.

Thin offline driver around ``fatigue_pipeline.feature_aggregator`` so
offline dataset building matches live inference.

``--probs`` accepts either a single Stage 1 Parquet file or a directory;
``--output`` is always a directory. One feature Parquet per input stem.

Usage::

    python -m scripts.labelling.temporal.aggregate_features \\
        --probs data/temporal/raw_probs/vid_001.parquet \\
        --output data/temporal/features

    python -m scripts.labelling.temporal.aggregate_features \\
        --probs data/temporal/raw_probs \\
        --output data/temporal/features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fatigue_pipeline.constants import (
    DEFAULT_FPS,
    DEFAULT_STEP_STRIDE_S,
    DEFAULT_SUB_WINDOW_S,
    POSE_KPT_DIM,
    POSE_NUM_KPTS,
    UPPER_BODY_KPT_NAMES,
)
from fatigue_pipeline.feature_aggregator import FeatureAggregator
from fatigue_pipeline.inference_pipeline import FrameProbs

from scripts.labelling.temporal._defaults import (
    DEFAULT_FEATURES_DIR,
    DEFAULT_RAW_PROBS_DIR,
)

POSE_FEATURE_COLUMNS = [
    "head_pitch",
    "head_roll",
    "shoulder_tilt",
    "head_size_ratio",
    "head_motion_energy",
    "head_drift_y",
    "posture_drift",
    "kpt_visibility",
]

OUTPUT_COLUMNS = [
    "step_idx",
    "timestamp_s",
    "perclos",
    "blink_rate_bpm",
    "mean_blink_duration",
    "eye_closure_variance",
    "yawn_rate_per_min",
    "mean_yawn_duration",
    "mouth_open_ratio",
    "mean_p_eye",
    "mean_p_mouth",
    *POSE_FEATURE_COLUMNS,
    "valid",
    "blink_count_total",
    "yawn_count_total",
]

OUTPUT_DTYPES = {
    "step_idx": "int32",
    "timestamp_s": "float32",
    "perclos": "float32",
    "blink_rate_bpm": "float32",
    "mean_blink_duration": "float32",
    "eye_closure_variance": "float32",
    "yawn_rate_per_min": "float32",
    "mean_yawn_duration": "float32",
    "mouth_open_ratio": "float32",
    "mean_p_eye": "float32",
    "mean_p_mouth": "float32",
    **{col: "float32" for col in POSE_FEATURE_COLUMNS},
    "valid": "bool",
    "blink_count_total": "int32",
    "yawn_count_total": "int32",
}


def _kpt_arrays_from_df(df: pd.DataFrame) -> np.ndarray | None:
    required = [
        f"kp_{name}_{axis}"
        for name in UPPER_BODY_KPT_NAMES
        for axis in ("x", "y", "conf")
    ]
    if not all(col in df.columns for col in required):
        return None
    arr = np.full((len(df), POSE_NUM_KPTS, POSE_KPT_DIM), np.nan, dtype=np.float32)
    for i, name in enumerate(UPPER_BODY_KPT_NAMES):
        arr[:, i, 0] = df[f"kp_{name}_x"].to_numpy(dtype=np.float32)
        arr[:, i, 1] = df[f"kp_{name}_y"].to_numpy(dtype=np.float32)
        arr[:, i, 2] = df[f"kp_{name}_conf"].to_numpy(dtype=np.float32)
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probs",
        type=Path,
        default=DEFAULT_RAW_PROBS_DIR,
        help="Stage 1 Parquet file or directory of them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Output directory; one feature Parquet per input.",
    )
    parser.add_argument("--sub-window-s", type=float, default=DEFAULT_SUB_WINDOW_S)
    parser.add_argument("--step-stride-s", type=float, default=DEFAULT_STEP_STRIDE_S)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process inputs whose output Parquet already exists.",
    )
    return parser.parse_args()


def collect_probs_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected .parquet input, got {path.suffix}")
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files found in {path}")
        return files
    raise FileNotFoundError(f"Probs path missing: {path}")


def aggregate(
    probs_df: pd.DataFrame,
    sub_window_s: float = DEFAULT_SUB_WINDOW_S,
    step_stride_s: float = DEFAULT_STEP_STRIDE_S,
) -> pd.DataFrame:
    if probs_df.empty:
        raise ValueError("Empty probs dataframe.")

    timestamps = probs_df["timestamp_s"].values
    duration_s = float(timestamps[-1] - timestamps[0])
    if duration_s <= 0:
        raise ValueError("Non-monotonic or single-frame timestamps.")

    fps = (len(probs_df) - 1) / duration_s if duration_s > 0 else DEFAULT_FPS

    eye_l_all = probs_df["p_eye_left_closed"].to_numpy(dtype=np.float32)
    eye_r_all = probs_df["p_eye_right_closed"].to_numpy(dtype=np.float32)
    mouth_all = probs_df["p_mouth_open"].to_numpy(dtype=np.float32)
    kpts_all = _kpt_arrays_from_df(probs_df)

    aggregator = FeatureAggregator(
        fps=fps,
        sub_window_s=sub_window_s,
        step_stride_s=step_stride_s,
    )

    def _to_optional(value: float) -> float | None:
        return None if np.isnan(value) else float(value)

    rows: list[dict] = []
    for i in range(len(probs_df)):
        kpts: np.ndarray | None = None
        if kpts_all is not None:
            row_kpts = kpts_all[i]
            if np.isfinite(row_kpts).any():
                kpts = row_kpts
        probs = FrameProbs(
            frame_idx=int(i),
            timestamp_s=float(timestamps[i]),
            face_detected=not (
                np.isnan(eye_l_all[i])
                and np.isnan(eye_r_all[i])
                and np.isnan(mouth_all[i])
            ),
            p_eye_left_closed=_to_optional(eye_l_all[i]),
            p_eye_right_closed=_to_optional(eye_r_all[i]),
            p_mouth_open=_to_optional(mouth_all[i]),
            keypoints=kpts,
        )
        step = aggregator.add(probs)
        if step is None:
            continue
        row = step.features.to_dict()
        row["step_idx"] = step.step_idx
        row["timestamp_s"] = step.timestamp_s
        row["blink_count_total"] = step.blink_count_total
        row["yawn_count_total"] = step.yawn_count_total
        rows.append(row)

    out = pd.DataFrame(rows)
    return out[OUTPUT_COLUMNS].astype(OUTPUT_DTYPES)


def main() -> None:
    args = parse_args()
    inputs = collect_probs_files(args.probs)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    for probs_path in inputs:
        output_path = args.output / f"{probs_path.stem}.parquet"
        csv_path = csv_dir / f"{probs_path.stem}.csv"
        if output_path.exists() and not args.overwrite:
            print(f"Skip (exists): {output_path}")
            continue

        probs_df = pd.read_parquet(probs_path)
        feats = aggregate(
            probs_df,
            sub_window_s=args.sub_window_s,
            step_stride_s=args.step_stride_s,
        )
        feats.to_parquet(output_path, index=False)
        feats.to_csv(csv_path, index=False)

        valid_pct = feats["valid"].mean() * 100.0
        print(
            f"{probs_path.stem}: {len(feats):,} steps -> {output_path} "
            f"(valid {valid_pct:.1f}%)"
        )


if __name__ == "__main__":
    main()
