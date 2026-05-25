"""Sliding-window dataset for the temporal model.

Walks one or more Stage 2 feature Parquets (per-second feature steps) and
emits fixed-length windows ready for the BiGRU. Windows that contain any
step with `valid=False` are dropped so the model never sees half-blank
rows from no-detection sub-windows.

Labels are bootstrapped from the features themselves via a simple heuristic
(replace `default_label_from_window` once human-annotated focus scores
exist).

Train/val splitting and normalization helpers live in
`model_architecture.utils` since they apply beyond this dataset.
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


# =============================================================================
# Constants
# =============================================================================

DEFAULT_WINDOW_STEPS = 30
DEFAULT_WINDOW_STRIDE = 1
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 42

# Bootstrap label heuristic - tune once real annotations exist.
# Weights sum to 1.0. PERCLOS is dominant (0.55) so eye closure can move
# the score on its own; pose is a 20% modifier (slump + low-visibility)
# rather than a baseline that would anchor "drowsy + upright" toward
# alert. Yawn / blink-duration sit in between as secondary eye/mouth
# signals.
PERCLOS_WEIGHT = 0.55
YAWN_WEIGHT = 0.15
BLINK_DUR_WEIGHT = 0.10
SLUMP_WEIGHT = 0.10
LOW_VIS_WEIGHT = 0.10

# Saturation norms - chosen to match clinical drowsiness thresholds, so the
# weighted contribution actually moves the score before the metric is
# physiologically extreme.
PERCLOS_NORM = 0.30           # PERCLOS that saturates the term (clinical
                              # drowsy threshold: PERCLOS > 0.3 = severe)
YAWN_RATE_NORM = 1.0          # 1 yawn/min already abnormal
BLINK_DUR_NORM_S = 0.5        # mean blink > 0.5s = drowsy (normal ~0.1-0.3s)

# Slump terms:
#   head_pitch = (nose_y - mid_shoulder_y) / shoulder_width.
#   - very alert / upright: ~-0.5 to -0.7 (nose well above shoulders)
#   - slumped: > -0.2 (nose drifting toward shoulder line)
#   - severe (head down): >= 0 (nose at/below shoulder line)
# Map (-0.3, +0.1) -> (0, 1) so anything more upright than -0.3 is "no
# penalty" and at/below shoulders is "fully slumped".
HEAD_PITCH_ALERT = -0.3
HEAD_PITCH_RANGE = 0.4

# posture_drift = std(mid_shoulder_y / shoulder_width) across 60s.
# Saturate around 0.30 of shoulder-width drift over the window.
POSTURE_DRIFT_NORM = 0.30


LabelFn = Callable[[np.ndarray], float]


# =============================================================================
# Window references
# =============================================================================


@dataclass
class _WindowRef:
    """Pointer into the per-file feature array for one accepted window."""

    file_idx: int
    start: int
    end: int


# =============================================================================
# Dataset
# =============================================================================


class TemporalWindowDataset(Dataset):
    """Sliding fixed-length windows over Stage 2 feature Parquets."""

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

        self._files: list[np.ndarray] = []   # per-file (T, F) raw feature arrays
        self._refs: list[_WindowRef] = []
        self._labels: list[float] = []

        for path in paths:
            df = pd.read_parquet(path)
            features = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
            valid = df["valid"].to_numpy()
            self._files.append(features)
            file_idx = len(self._files) - 1

            # Slide a fixed window. Drop windows that touch any invalid step
            # so the temporal model only ever sees fully-populated rows.
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
        """All raw feature rows across the dataset (used to fit normalization)."""
        return np.concatenate(self._files, axis=0)


# =============================================================================
# Label heuristic (replace once real annotations exist)
# =============================================================================


def default_label_from_window(window: np.ndarray) -> float:
    """Bootstrap focus score from feature aggregates inside the window.

    fatigue = 0.55 * clip(mean(PERCLOS)          / 0.30, 0, 1)
            + 0.15 * clip(mean(yawn_rate)        / 1.0,  0, 1)
            + 0.10 * clip(mean(mean_blink_dur_s) / 0.5,  0, 1)
            + 0.10 * slump_score
            + 0.10 * (1 - mean(kpt_visibility))
    focus_score = clip(1.0 - fatigue, 0, 1)

    Saturation norms target clinical drowsiness thresholds: PERCLOS >
    0.30 = severe, yawn_rate > 1/min = abnormal, blink > 0.5s = drowsy.

    Where:
        slump_score = 0.5 * pitch_term + 0.5 * drift_term
        pitch_term  = clip((head_pitch - HEAD_PITCH_ALERT) / HEAD_PITCH_RANGE, 0, 1)
        drift_term  = clip(posture_drift / POSTURE_DRIFT_NORM, 0, 1)

    The heuristic is intentionally rough - the BiGRU's job is to smooth
    and contextualize it across the window, not parrot it back. Pose
    contributes 20% total (10% slump + 10% low-visibility) so it acts as
    a modifier rather than a baseline that would anchor "drowsy +
    upright" toward alert. PERCLOS dominates at 55% so eye closure alone
    can drive the score into the drowsy band.
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


