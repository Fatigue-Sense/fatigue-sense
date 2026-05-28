"""
Upper-body pose estimator used by the fatigue pipeline.

Returns nose, ear, and shoulder keypoints as [x, y, confidence] rows.
Low-confidence keypoints are set to NaN so the aggregator can treat them as
missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

from fatigue_pipeline.constants import (
    POSE_MIN_KPT_CONF,
    POSE_NUM_KPTS,
    POSE_PERSON_CONF,
    UPPER_BODY_KPT_INDICES,
)

logger = logging.getLogger(__name__)

ULTRALYTICS_FALLBACK = "yolo11n-pose.pt"

class PoseEstimator:
    """
    Runs YOLO pose inference and keeps the upper-body keypoints
    """
    def __init__(
        self,
        weights_path: str | Path,
        device: str | None = None,
        person_conf: float = POSE_PERSON_CONF,
        kpt_min_conf: float = POSE_MIN_KPT_CONF,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        if self.weights_path.exists():
            resolved = str(self.weights_path)
        else:
            logger.warning(
                "Pose weights missing at %s - falling back to Ultralytics base "
                "model '%s' (auto-downloads on first use).",
                self.weights_path,
                ULTRALYTICS_FALLBACK,
            )
            resolved = ULTRALYTICS_FALLBACK

        self.model = YOLO(resolved)

        # Move the model once instead of doing it on every frame
        if device:
            self.model.to(device)
        self.device = device
        self.person_conf = float(person_conf)
        self.kpt_min_conf = float(kpt_min_conf)
        self._kpt_indices = list(UPPER_BODY_KPT_INDICES)
        self._closed = False

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Return upper-body keypoints for the best detected person, if any
        """
        if self._closed:
            raise RuntimeError("PoseEstimator is closed")
        results = self.model(frame_bgr, verbose=False, conf=self.person_conf)
        if not results:
            return None

        kpts = results[0].keypoints
        if kpts is None or kpts.data is None or kpts.data.shape[0] == 0:
            return None

        # Pick the person with the highest mean kpt confidence.
        kpts_np = kpts.data.detach().cpu().numpy()  # (N, K, 3)
        if kpts_np.shape[0] > 1:
            mean_conf = kpts_np[:, :, 2].mean(axis=1)
            best = int(np.argmax(mean_conf))
        else:
            best = 0

        n_model_kpts = kpts_np.shape[1]
        if n_model_kpts == POSE_NUM_KPTS:
            upper = kpts_np[best, :, :].astype(np.float32).copy()
        elif n_model_kpts > max(self._kpt_indices):
            upper = kpts_np[best, self._kpt_indices, :].astype(np.float32).copy()
        else:
            raise ValueError(
                f"Pose model emits {n_model_kpts} keypoints; expected "
                f"{POSE_NUM_KPTS} (retrained head) or "
                f">={max(self._kpt_indices) + 1} (stock COCO-17)."
            )

        low_conf = upper[:, 2] < self.kpt_min_conf
        upper[low_conf, :2] = np.nan
        return upper

    def close(self) -> None:
        """
        Mark the estimator as closed
        """
        self._closed = True
