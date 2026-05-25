"""End-to-end Phase A + Phase B inference pipeline.

Connects `RegionCropper` (MediaPipe-based ROI extraction) with the trained
binary ROI classifiers and produces a single per-frame probability record:

    FrameProbs(frame_idx, timestamp_s, face_detected,
               p_eye_left_closed, p_eye_right_closed, p_mouth_open)

The mouth predictor is optional - the pipeline still runs (returning None
for `p_mouth_open`) when no mouth checkpoint is provided.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from fatigue_pipeline.cnn_predictors import (
    EyeStateClassifier,
    MouthStateClassifier,
)
from fatigue_pipeline.constants import DEFAULT_FPS
from fatigue_pipeline.region_cropper import RegionCropper


@dataclass
class FrameProbs:
    frame_idx: int
    timestamp_s: float
    face_detected: bool
    p_eye_left_closed: float | None
    p_eye_right_closed: float | None
    p_mouth_open: float | None
    # Upper-body pose keypoints (shape: (POSE_NUM_KPTS, 3), columns: x, y, conf).
    # None when YOLO pose estimator wasn't run, or no person detected.
    # NaN per-row when a single kpt fell below POSE_MIN_KPT_CONF.
    keypoints: np.ndarray | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class FatiguePipeline:
    """Phase A + Phase B fused inference."""

    def __init__(
        self,
        face_landmarker_path: str | Path,
        eye_classifier_path: str | Path,
        mouth_classifier_path: str | Path | None = None,
        device: str | torch.device | None = None,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.cropper = RegionCropper(face_landmarker_path, fps=fps)
        self.eye_classifier = EyeStateClassifier(eye_classifier_path, device=device)
        self.mouth_classifier = MouthStateClassifier(
            mouth_classifier_path, device=device
        )

    def reset(self, fps: float | None = None) -> None:
        self.cropper.reset(fps=fps)

    def close(self) -> None:
        self.cropper.close()

    @property
    def frame_idx(self) -> int:
        return self.cropper.frame_idx

    @property
    def fps(self) -> float:
        return self.cropper.fps

    def process_frame(self, frame: np.ndarray) -> FrameProbs:
        """Run Phase A + Phase B on a single BGR frame."""
        idx = self.cropper.frame_idx
        timestamp_s = idx / self.cropper.fps

        left_eye, right_eye, mouth = self.cropper.get_crops(frame)
        face_detected = any(c is not None for c in (left_eye, right_eye, mouth))

        # Get probs for both mouth and eyes
        p_left, p_right = self.eye_classifier.predict_probs_batch([left_eye, right_eye])
        p_mouth = self.mouth_classifier.predict_prob(mouth)

        return FrameProbs(
            frame_idx=idx,
            timestamp_s=timestamp_s,
            face_detected=face_detected,
            p_eye_left_closed=p_left,
            p_eye_right_closed=p_right,
            p_mouth_open=p_mouth,
        )
