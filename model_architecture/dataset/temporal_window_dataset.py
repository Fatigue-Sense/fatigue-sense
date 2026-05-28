"""
Dataset for training the temporal fatigue model.

Loads feature Parquet files, cuts them into fixed-length windows, and skips
windows that contain invalid feature rows
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fatigue_pipeline.constants import FEATURE_NAMES


# ---- Constants ----
DEFAULT_WINDOW_STEPS = 30
DEFAULT_WINDOW_STRIDE = 1
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 42

PERCLOS_WEIGHT = 0.55
YAWN_WEIGHT = 0.15
BLINK_DUR_WEIGHT = 0.10
SLUMP_WEIGHT = 0.10
LOW_VIS_WEIGHT = 0.10

PERCLOS_NORM = 0.30 # severe drowsiness threshold
YAWN_RATE_NORM = 1.0 # 1 yawn/min
BLINK_DUR_NORM_S = 0.5 # mean blink > 0.5s = drowsy (normal ~0.1-0.3s)

# Larger head_pitch means the nose is closer to, or below, the shoulder line
HEAD_PITCH_ALERT = -0.3
HEAD_PITCH_RANGE = 0.4

# Shoulder movement over the window, normalized by shoulder width
POSTURE_DRIFT_NORM = 0.30

LabelFn = Callable[[np.ndarray], float]

@dataclass
class _WindowRef:
    """Pointer into the per-file feature array for one accepted window."""

    file_idx: int
    start: int
    end: int


# Dataset
class TemporalWindowDataset(Dataset):
    """
    Sliding fixed-length windows over Stage 2 feature Parquets
    """
    def __init__(
        self,
        paths: list[Path],
        window_steps: int = DEFAULT_WINDOW_STEPS,
        stride: int = DEFAULT_WINDOW_STRIDE,
        label_fn: LabelFn | None = None,
        normalization: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.window_steps = window_steps
        self.stride = stride
        self.label_fn = label_fn or default_label_from_window
        self.normalization = normalization

        self._files: list[np.ndarray] = []
        self._refs: list[_WindowRef] = []
        self._labels: list[float] = []

        for path in paths:
            df = pd.read_parquet(path)
            features = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
            valid = df["valid"].to_numpy()
            self._files.append(features)
            file_idx = len(self._files) - 1

            # Skip windows with missing/invalid feature rows
            for start in range(0, len(df) - window_steps + 1, stride):
                end = start + window_steps
                if not bool(valid[start:end].all()):
                    continue
                window = features[start:end]
                self._refs.append(_WindowRef(file_idx, start, end))
                self._labels.append(self.label_fn(window))

    # ---- pytorch protocol ----
    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self._refs[idx]
        x = self._files[ref.file_idx][ref.start : ref.end].copy()
        if self.normalization is not None:
            x = (x - self.normalization["mean"]) / self.normalization["std"]
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.tensor(self._labels[idx], dtype=torch.float32),
        )

    # ---- helpers ----
    def stacked_features(self) -> np.ndarray:
        """
        Return all raw feature rows, mainly for fitting normalization
        """
        return np.concatenate(self._files, axis=0)


# Temporary label heuristic
def default_label_from_window(window: np.ndarray) -> float:
    """
    Estimate a focus score for a window until real labels are available.

    The score starts from 1.0 and subtracts fatigue evidence from eye closure,
    yawning, long blinks, slumped posture, and low pose visibility
    """
    perclos = float(window[:, FEATURE_NAMES.index("perclos")].mean())
    yawn_rate = float(window[:, FEATURE_NAMES.index("yawn_rate_per_min")].mean())
    blink_dur = float(window[:, FEATURE_NAMES.index("mean_blink_duration")].mean())
    head_pitch = float(window[:, FEATURE_NAMES.index("head_pitch")].mean())
    posture_drift = float(window[:, FEATURE_NAMES.index("posture_drift")].mean())
    kpt_visibility = float(window[:, FEATURE_NAMES.index("kpt_visibility")].mean())

    pitch_term = np.clip(
        (head_pitch - HEAD_PITCH_ALERT) / HEAD_PITCH_RANGE, 0.0, 1.0
    )
    drift_term = np.clip(posture_drift / POSTURE_DRIFT_NORM, 0.0, 1.0)
    slump_score = 0.5 * pitch_term + 0.5 * drift_term
    low_vis = np.clip(1.0 - kpt_visibility, 0.0, 1.0)

    fatigue = (
        PERCLOS_WEIGHT * np.clip(perclos / PERCLOS_NORM, 0.0, 1.0)
        + YAWN_WEIGHT * np.clip(yawn_rate / YAWN_RATE_NORM, 0.0, 1.0)
        + BLINK_DUR_WEIGHT * np.clip(blink_dur / BLINK_DUR_NORM_S, 0.0, 1.0)
        + SLUMP_WEIGHT * slump_score
        + LOW_VIS_WEIGHT * low_vis
    )
    return float(np.clip(1.0 - fatigue, 0.0, 1.0))


