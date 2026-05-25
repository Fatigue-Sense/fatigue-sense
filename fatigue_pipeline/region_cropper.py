"""Production face-region cropper.

Self-contained copy used by the inference pipeline. Wraps MediaPipe
FaceLandmarker and returns left-eye, right-eye, and mouth crops sized
geometrically from inter-eye distance.

This module duplicates the labelling-side cropper in `vision_pipeline/`
intentionally - production code must not depend on dev-only tooling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from fatigue_pipeline.constants import (
    DEFAULT_FPS,
    EYE_BOX_H_RATIO,
    EYE_BOX_W_RATIO,
    EYE_CROP_SIZE,
    LEFT_EYE_INDICES,
    LEFT_EYE_OUTER_CORNER,
    MOUTH_BOX_H_RATIO_HYBRID,
    MOUTH_CROP_SIZE,
    MOUTH_INDICES,
    MOUTH_LANDMARK_H_SCALE,
    MOUTH_LANDMARK_W_SCALE,
    MOUTH_VERTICAL_OFFSET_RATIO_HYBRID,
    MS_PER_SECOND,
    RIGHT_EYE_INDICES,
    RIGHT_EYE_OUTER_CORNER,
)


CropResult = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]


class RegionCropper:
    """Detect MediaPipe face landmarks and crop eyes/mouth regions."""

    def __init__(
        self,
        model_path: str | Path,
        num_faces: int = 1,
        eye_crop_size: tuple[int, int] = EYE_CROP_SIZE,
        mouth_crop_size: tuple[int, int] = MOUTH_CROP_SIZE,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.eye_crop_size = eye_crop_size
        self.mouth_crop_size = mouth_crop_size
        self.frame_idx = 0
        self.fps = fps if fps and fps > 0 else DEFAULT_FPS

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker_options = options
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def reset(self, fps: float | None = None) -> None:
        self.frame_idx = 0
        if fps is not None and fps > 0:
            self.fps = fps
        # Recreate the landmarker - VIDEO mode requires monotonic timestamps,
        # so a fresh instance is the only way to safely restart at t=0.
        if self.landmarker is not None:
            self.landmarker.close()
        self.landmarker = vision.FaceLandmarker.create_from_options(
            self._landmarker_options
        )

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()

    @staticmethod
    def landmark_to_pixel(lm: Any, w: int, h: int) -> tuple[int, int]:
        return int(lm.x * w), int(lm.y * h)

    @classmethod
    def get_region_center(
        cls, landmarks: list[Any], indices: list[int], w: int, h: int
    ) -> tuple[int, int]:
        pts = [cls.landmark_to_pixel(landmarks[i], w, h) for i in indices]
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        return cx, cy

    @classmethod
    def get_inter_eye_distance(
        cls, landmarks: list[Any], w: int, h: int
    ) -> int:
        left = cls.landmark_to_pixel(landmarks[LEFT_EYE_OUTER_CORNER], w, h)
        right = cls.landmark_to_pixel(landmarks[RIGHT_EYE_OUTER_CORNER], w, h)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        return max(1, int((dx**2 + dy**2) ** 0.5))

    @classmethod
    def get_face_roll_deg(
        cls, landmarks: list[Any], w: int, h: int
    ) -> float:
        """Roll angle (deg) of the eye line, positive = head tilted right.

        Computed from the two outer eye corners. Subtract this from the
        frame so the eye axis is horizontal before cropping.
        """
        left = cls.landmark_to_pixel(landmarks[LEFT_EYE_OUTER_CORNER], w, h)
        right = cls.landmark_to_pixel(landmarks[RIGHT_EYE_OUTER_CORNER], w, h)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    @staticmethod
    def crop_aligned_box(
        frame: np.ndarray,
        center: tuple[int, int],
        angle_deg: float,
        box_w: int,
        box_h: int,
        out_size: tuple[int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        """Rotate the frame so the eye axis is horizontal, then crop.

        The classifier was trained on (mostly) upright eyes; this brings
        tilted heads back onto the training manifold without retraining.

        Returned ``box`` is the axis-aligned slice in the *rotated*
        frame's coordinates. For display overlays on the original frame
        the caller should compute its own axis-aligned bbox.
        """
        h, w = frame.shape[:2]
        cx, cy = center
        # Negative angle = rotate eye axis back to horizontal.
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), -angle_deg, 1.0)
        rotated = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return RegionCropper.crop_fixed_box(
            rotated, center, box_w, box_h, out_size
        )

    @staticmethod
    def crop_fixed_box(
        frame: np.ndarray,
        center: tuple[int, int],
        box_w: int,
        box_h: int,
        out_size: tuple[int, int],
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        h, w = frame.shape[:2]
        cx, cy = center

        x1 = max(0, int(cx - box_w / 2))
        y1 = max(0, int(cy - box_h / 2))
        x2 = min(w, int(cx + box_w / 2))
        y2 = min(h, int(cy + box_h / 2))

        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_CUBIC)
        return crop, (x1, y1, x2, y2)

    def _detect_landmarks(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(self.frame_idx * (MS_PER_SECOND / self.fps))
        self.frame_idx += 1
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def get_crops(self, frame: np.ndarray) -> CropResult:
        face = self._detect_landmarks(frame)
        if face is None:
            return None, None, None

        h, w = frame.shape[:2]
        eye_dist = self.get_inter_eye_distance(face, w, h)
        roll_deg = self.get_face_roll_deg(face, w, h)

        eye_box_w = int(EYE_BOX_W_RATIO * eye_dist)
        eye_box_h = int(EYE_BOX_H_RATIO * eye_dist)

        left_center = self.get_region_center(face, LEFT_EYE_INDICES, w, h)
        right_center = self.get_region_center(face, RIGHT_EYE_INDICES, w, h)

        mouth_pts = [self.landmark_to_pixel(face[i], w, h) for i in MOUTH_INDICES]
        xs = [p[0] for p in mouth_pts]
        ys = [p[1] for p in mouth_pts]
        mouth_w_landmark = max(xs) - min(xs)
        mouth_h_landmark = max(ys) - min(ys)

        mouth_box_w = max(int(eye_dist), int(MOUTH_LANDMARK_W_SCALE * mouth_w_landmark))
        mouth_box_h = max(
            int(MOUTH_BOX_H_RATIO_HYBRID * eye_dist),
            int(MOUTH_LANDMARK_H_SCALE * mouth_h_landmark),
        )

        mcx, mcy = self.get_region_center(face, MOUTH_INDICES, w, h)
        mcy = int(mcy + MOUTH_VERTICAL_OFFSET_RATIO_HYBRID * mouth_box_h)

        left_eye, _ = self.crop_aligned_box(
            frame, left_center, roll_deg, eye_box_w, eye_box_h, self.eye_crop_size
        )
        right_eye, _ = self.crop_aligned_box(
            frame, right_center, roll_deg, eye_box_w, eye_box_h, self.eye_crop_size
        )
        mouth, _ = self.crop_aligned_box(
            frame, (mcx, mcy), roll_deg, mouth_box_w, mouth_box_h, self.mouth_crop_size
        )

        return left_eye, right_eye, mouth
