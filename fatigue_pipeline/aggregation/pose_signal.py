"""
Stores recent upper-body keypoints and turns them into window-level pose features.

For each window, it calculates eight pose-related features that are added to the
existing eye and mouth features:

    head_pitch: average nose position relative to the shoulders
    head_roll: average left/right head tilt
    shoulder_tilt: average shoulder tilt
    head_size_ratio: average ear distance relative to shoulder width
    head_motion_energy: average frame-to-frame nose movement
    head_drift_y: vertical head movement over the window
    posture_drift: vertical shoulder movement over the window
    kpt_visibility: average keypoint confidence

When window visibility is too low (mean conf < POSE_MIN_VISIBILITY) the
pose features collapse to zero so the BiGRU sees a "pose unobserved"
signature instead of noisy garbage
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from fatigue_pipeline.constants import (
    POSE_KPT_DIM,
    POSE_MIN_VISIBILITY,
    POSE_NUM_KPTS,
)

# Index aliases into UPPER_BODY_KPT_INDICES order 
NOSE = 0
EAR_L = 1
EAR_R = 2
SHOULDER_L = 3
SHOULDER_R = 4


@dataclass
class PoseFeatures:
    head_pitch: float
    head_roll: float
    shoulder_tilt: float
    head_size_ratio: float
    head_motion_energy: float
    head_drift_y: float
    posture_drift: float
    kpt_visibility: float


_ZERO_FEATURES = PoseFeatures(
    head_pitch=0.0,
    head_roll=0.0,
    shoulder_tilt=0.0,
    head_size_ratio=0.0,
    head_motion_energy=0.0,
    head_drift_y=0.0,
    posture_drift=0.0,
    kpt_visibility=0.0,
)

"""Keeps a rolling window of upper-body keypoints."""
class PoseSignalBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)
        self._capacity = capacity

    def reconfigure(self, capacity: int) -> None:
        self._buf = deque(maxlen=capacity)
        self._capacity = capacity

    def clear(self) -> None:
        self._buf.clear()

    def push(self, kpts: np.ndarray | None) -> None:
        if kpts is None:
            kpts = np.full((POSE_NUM_KPTS, POSE_KPT_DIM), np.nan, dtype=np.float32)
        else:
            kpts = np.asarray(kpts, dtype=np.float32)
            if kpts.shape != (POSE_NUM_KPTS, POSE_KPT_DIM):
                raise ValueError(
                    f"Expected ({POSE_NUM_KPTS}, {POSE_KPT_DIM}) kpts, "
                    f"got {kpts.shape}"
                )
        self._buf.append(kpts)

    # Function to return the current keypoint window as a stacked array
    def snapshot(self) -> np.ndarray:
        if not self._buf:
            return np.empty((0, POSE_NUM_KPTS, POSE_KPT_DIM), dtype=np.float32)
        return np.stack(self._buf, axis=0)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buf)


def _safe_atan2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """atan2 that returns NaN where either input is NaN"""
    out = np.full_like(y, np.nan, dtype=np.float32)
    valid = ~(np.isnan(y) | np.isnan(x))
    out[valid] = np.arctan2(y[valid], x[valid]).astype(np.float32)
    return out


def _level_angle(dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    return _safe_atan2(dy, np.abs(dx) + 1e-6)


def compute_pose_features(window: np.ndarray) -> PoseFeatures:
    """
    Convert a window of keypoints into the eight pose features
    """
    if window.size == 0 or window.shape[0] == 0:
        return _ZERO_FEATURES

    xy = window[..., :2]
    conf = window[..., 2]

    # nan-aware reductions warn on all-NaN slices
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings(
            "ignore", r"All-NaN (slice|axis) encountered", RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore", r"Degrees of freedom <= 0 for slice", RuntimeWarning
        )
        return _compute_pose_features_inner(xy, conf)


def _compute_pose_features_inner(xy: np.ndarray, conf: np.ndarray) -> PoseFeatures:
    visibility_raw = np.nanmean(conf) if conf.size else 0.0
    visibility = _nan_to_zero(float(visibility_raw))
    if visibility < POSE_MIN_VISIBILITY:
        return PoseFeatures(
            **{**_ZERO_FEATURES.__dict__, "kpt_visibility": visibility}
        )

    # Per-frame derived signals
    nose_x = xy[:, NOSE, 0]
    nose_y = xy[:, NOSE, 1]
    sh_l = xy[:, SHOULDER_L, :]
    sh_r = xy[:, SHOULDER_R, :]
    ear_l = xy[:, EAR_L, :]
    ear_r = xy[:, EAR_R, :]

    mid_sh_y = np.nanmean(np.stack([sh_l[:, 1], sh_r[:, 1]], axis=0), axis=0)
    sh_dx = sh_r[:, 0] - sh_l[:, 0]
    sh_dy = sh_r[:, 1] - sh_l[:, 1]
    sh_width = np.sqrt(sh_dx**2 + sh_dy**2).astype(np.float32)
    # Guard divide-by-zero / tiny widths
    sh_width[sh_width < 1.0] = np.nan

    ear_dx = ear_r[:, 0] - ear_l[:, 0]
    ear_dy = ear_r[:, 1] - ear_l[:, 1]
    ear_dist = np.sqrt(ear_dx**2 + ear_dy**2).astype(np.float32)

    head_pitch_per_frame = (nose_y - mid_sh_y) / sh_width
    head_roll_per_frame = _level_angle(ear_dy, ear_dx)
    shoulder_tilt_per_frame = _level_angle(sh_dy, sh_dx)
    head_size_per_frame = ear_dist / sh_width

    head_pitch = float(np.nanmean(head_pitch_per_frame))
    head_roll = float(np.nanmean(head_roll_per_frame))
    shoulder_tilt = float(np.nanmean(shoulder_tilt_per_frame))
    head_size_ratio = float(np.nanmean(head_size_per_frame))

    # Window-level motion and drift
    nose_x_norm = nose_x / sh_width
    nose_y_norm = nose_y / sh_width
    mid_sh_y_norm = mid_sh_y / sh_width

    dx = np.diff(nose_x_norm)
    dy = np.diff(nose_y_norm)
    step_magnitude = np.sqrt(dx**2 + dy**2)
    head_motion_energy = (
        float(np.nanmean(step_magnitude)) if step_magnitude.size else 0.0
    )

    head_drift_y = float(np.nanstd(nose_y_norm)) if nose_y_norm.size else 0.0
    posture_drift = float(np.nanstd(mid_sh_y_norm)) if mid_sh_y_norm.size else 0.0

    return PoseFeatures(
        head_pitch=_nan_to_zero(head_pitch),
        head_roll=_nan_to_zero(head_roll),
        shoulder_tilt=_nan_to_zero(shoulder_tilt),
        head_size_ratio=_nan_to_zero(head_size_ratio),
        head_motion_energy=_nan_to_zero(head_motion_energy),
        head_drift_y=_nan_to_zero(head_drift_y),
        posture_drift=_nan_to_zero(posture_drift),
        kpt_visibility=visibility,
    )


def _nan_to_zero(value: float) -> float:
    return 0.0 if not np.isfinite(value) else float(value)
