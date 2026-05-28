from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from fatigue_pipeline.constants import FEATURE_NAMES

@dataclass
class StepFeatures:
    """One per-second aggregated feature vector for the temporal model"""

    perclos: float
    blink_rate_bpm: float
    mean_blink_duration: float
    eye_closure_variance: float
    yawn_rate_per_min: float
    mean_yawn_duration: float
    mouth_open_ratio: float
    mean_p_eye: float
    mean_p_mouth: float
    # Pose-derived (yolo11n-pose upper body subset)
    head_pitch: float = 0.0
    head_roll: float = 0.0
    shoulder_tilt: float = 0.0
    head_size_ratio: float = 0.0
    head_motion_energy: float = 0.0
    head_drift_y: float = 0.0
    posture_drift: float = 0.0
    kpt_visibility: float = 0.0
    valid: bool = False

    def to_array(self) -> np.ndarray:
        """Return the 9-dim feature vector (excludes the ``valid`` flag)."""
        return np.asarray(
            [getattr(self, name) for name in FEATURE_NAMES],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {name: getattr(self, name) for name in (*FEATURE_NAMES, "valid")}

@dataclass
class FeatureStep:
    """Live feature row with timestamp metadata."""

    step_idx: int
    timestamp_s: float
    features: StepFeatures
    blink_count_total: int = 0
    yawn_count_total: int = 0
