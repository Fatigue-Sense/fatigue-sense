from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


LEFT_EYE = [33,  133, 160, 144, 158, 153, 159, 145]
RIGHT_EYE = [362, 263, 387, 373, 385, 380, 386, 374]
MOUTH = [61, 291, 81, 178, 13, 14, 402, 311]

EAR_OPEN_THRESH = 0.13
EAR_CLOSE_THRESH = 0.10

MAR_OPEN_THRESH = 0.43
MAR_CLOSE_THRESH = 0.40

CROP_HEIGHT = 64
CROP_WIDTH = 64


@dataclass
class RegionOutputs:
    clean_frame: np.ndarray
    left_eye_crop: np.ndarray | None
    right_eye_crop: np.ndarray | None
    mouth_crop: np.ndarray | None
    ear: float | None
    ear_left: float | None
    ear_right: float | None
    mar: float | None
    left_eye_state: str | None
    right_eye_state: str | None
    mouth_state: str | None
    detected: bool


class MediaPipeRegionExtractor:
    def __init__(
        self,
        model_path: str | Path,
        num_faces: int = 1,
        eye_crop_size: tuple[int, int] = (CROP_WIDTH, CROP_HEIGHT),
        mouth_crop_size: tuple[int, int] = (CROP_WIDTH, CROP_HEIGHT),
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.eye_crop_size = eye_crop_size
        self.mouth_crop_size = mouth_crop_size
        self.mouth_state = "CLOSED"
        self.frame_idx = 0

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

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
    def get_inter_eye_distance(cls, landmarks: list[Any], w: int, h: int) -> int:
        left_outer = cls.landmark_to_pixel(landmarks[33], w, h)
        right_outer = cls.landmark_to_pixel(landmarks[263], w, h)
        dx = right_outer[0] - left_outer[0]
        dy = right_outer[1] - left_outer[1]
        return max(1, int((dx**2 + dy**2) ** 0.5))

    @staticmethod
    def crop_fixed_box(
        frame: np.ndarray,
        center: tuple[int, int],
        box_w: int,
        box_h: int,
        out_size: tuple[int, int],
    ) -> np.ndarray | None:
        h, w = frame.shape[:2]
        cx, cy = center

        x1 = max(0, int(cx - box_w / 2))
        y1 = max(0, int(cy - box_h / 2))
        x2 = min(w, int(cx + box_w / 2))
        y2 = min(h, int(cy + box_h / 2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_CUBIC)
        return crop

    @staticmethod
    def euclidean(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    @classmethod
    def compute_ear(
        cls, landmarks: list[Any], indices: list[int], w: int, h: int
    ) -> float:
        """
        indices order: [outer, inner, top1, bot1, top2, bot2, top3, bot3]
        Matches LEFT_EYE  = [33, 133, 160, 144, 158, 153, 159, 145]
                RIGHT_EYE = [362, 263, 387, 373, 385, 380, 386, 374]
        """
        pts = [cls.landmark_to_pixel(landmarks[i], w, h) for i in indices]

        # Horizontal distance
        horizontal = cls.euclidean(pts[0], pts[1])
        if horizontal == 0:
            return 0.0

        # Three vertical distances for better coverage across the lid
        v1 = cls.euclidean(pts[2], pts[3])
        v2 = cls.euclidean(pts[4], pts[5])
        v3 = cls.euclidean(pts[6], pts[7])

        return (v1 + v2 + v3) / (3.0 * horizontal)
    
    @classmethod
    def compute_mar(
        cls, landmarks: list[Any], indices: list[int], w: int, h: int
    ) -> float:
        """
        indices order: [left_corner, right_corner, top1, bot1, top2, bot2, top3, bot3]
        Matches MOUTH = [61, 291, 81, 178, 13, 14, 402, 311]
        """
        pts = [cls.landmark_to_pixel(landmarks[i], w, h) for i in indices]

        horizontal = cls.euclidean(pts[0], pts[1])
        if horizontal == 0:
            return 0.0

        v1 = cls.euclidean(pts[2], pts[3])
        v2 = cls.euclidean(pts[4], pts[5])
        v3 = cls.euclidean(pts[6], pts[7])

        return (v1 + v2 + v3) / (3.0 * horizontal)

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()

    def process_frame(self, frame: np.ndarray) -> RegionOutputs:
        clean_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = self.frame_idx * 33
        self.frame_idx += 1

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return RegionOutputs(
                clean_frame=clean_frame,
                left_eye_crop=None,
                right_eye_crop=None,
                mouth_crop=None,
                ear=None,
                ear_left=None,
                ear_right=None,
                mar=None,
                left_eye_state=None,
                right_eye_state=None,
                mouth_state=None,
                detected=False,
            )

        face = result.face_landmarks[0]
        h, w = frame.shape[:2]

        ear_left = self.compute_ear(face, LEFT_EYE, w, h)
        ear_right = self.compute_ear(face, RIGHT_EYE, w, h)
        ear = (ear_left + ear_right) / 2.0
        mar = self.compute_mar(face, MOUTH, w, h)

        left_eye_state = None
        right_eye_state = None

        if ear_left <= EAR_CLOSE_THRESH:
            left_eye_state = "CLOSED"
        elif ear_left >= EAR_OPEN_THRESH:
            left_eye_state = "OPEN"

        if ear_right <= EAR_CLOSE_THRESH:
            right_eye_state = "CLOSED"
        elif ear_right >= EAR_OPEN_THRESH:
            right_eye_state = "OPEN"

        if mar > MAR_OPEN_THRESH:
            self.mouth_state = "OPEN"
        elif mar < MAR_CLOSE_THRESH:
            self.mouth_state = "CLOSED"

        left_eye_center = self.get_region_center(face, LEFT_EYE, w, h)
        right_eye_center = self.get_region_center(face, RIGHT_EYE, w, h)
        mouth_center = self.get_region_center(face, MOUTH, w, h)

        eye_dist = self.get_inter_eye_distance(face, w, h)

        eye_box_w = int(0.50 * eye_dist)
        eye_box_h = int(0.35 * eye_dist)

        mar_clamped = max(0.3, min(mar, 1.5))
        mouth_scale = 1.0 + 4.0 * (mar_clamped - 0.3)

        mouth_box_w = int(1.0 * eye_dist)
        mouth_box_h = int(0.4 * eye_dist * mouth_scale)
        mouth_box_h = max(mouth_box_h, int(0.35 * eye_dist))

        cx, cy = mouth_center
        cy = int(cy + 0.05 * mouth_box_h)
        mouth_center = (cx, cy)

        left_eye_crop = self.crop_fixed_box(
            clean_frame, left_eye_center, eye_box_w, eye_box_h, self.eye_crop_size
        )
        right_eye_crop = self.crop_fixed_box(
            clean_frame, right_eye_center, eye_box_w, eye_box_h, self.eye_crop_size
        )
        mouth_crop = self.crop_fixed_box(
            clean_frame, mouth_center, mouth_box_w, mouth_box_h, self.mouth_crop_size
        )

        return RegionOutputs(
            clean_frame=clean_frame,
            left_eye_crop=left_eye_crop,
            right_eye_crop=right_eye_crop,
            mouth_crop=mouth_crop,
            ear=ear,
            ear_left=ear_left,
            ear_right=ear_right,
            mar=mar,
            left_eye_state=left_eye_state,
            right_eye_state=right_eye_state,
            mouth_state=self.mouth_state,
            detected=True,
        )


if __name__ == "__main__":
    MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
    if not MODEL_PATH.exists():
        print(f"Update MODEL_PATH in __main__ test block. Missing: {MODEL_PATH}")
        raise SystemExit(1)

    extractor = MediaPipeRegionExtractor(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            outputs = extractor.process_frame(frame)

            if outputs.left_eye_crop is not None:
                cv2.imshow("Left Eye", outputs.left_eye_crop)
            if outputs.right_eye_crop is not None:
                cv2.imshow("Right Eye", outputs.right_eye_crop)
            if outputs.mouth_crop is not None:
                cv2.imshow("Mouth", outputs.mouth_crop)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()