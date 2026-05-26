"""Webcam demo: MediaPipe face landmarks through eye/mouth CNN classifiers.

Visualizes ROI bounding boxes and per-frame probabilities (no pose, aggregator,
or temporal model). Run from the repository root:

    python scripts/vision_pipeline.py

Checkpoint paths come from ``scripts/weights_path.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from fatigue_pipeline.cnn_predictors import EyeStateClassifier, MouthStateClassifier
from fatigue_pipeline.constants import (
    EYE_BOX_H_RATIO,
    EYE_BOX_W_RATIO,
    EYE_CROP_SIZE,
    EYE_CLOSED_THRESH,
    LEFT_EYE_INDICES,
    MOUTH_BOX_H_RATIO_HYBRID,
    MOUTH_CROP_SIZE,
    MOUTH_INDICES,
    MOUTH_LANDMARK_H_SCALE,
    MOUTH_LANDMARK_W_SCALE,
    MOUTH_OPEN_THRESH,
    MOUTH_RISING_THRESH,
    MOUTH_VERTICAL_OFFSET_RATIO_HYBRID,
    RIGHT_EYE_INDICES,
)
from fatigue_pipeline.region_cropper import RegionCropper
from scripts.weights_path import EYE_CKPT, LANDMARKER_PATH, MOUTH_CKPT, resolve_asset

PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
WEBCAM_INDEX = 0
TARGET_FPS = 30.0
TARGET_DT = 1.0 / TARGET_FPS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_TITLE = "FatigueSense vision_pipeline (q to quit)"


def _resolve_device(d: str) -> str:
    if d != "cuda":
        return d
    if not torch.cuda.is_available():
        print("[vision_pipeline] CUDA not available - using CPU")
        return "cpu"
    try:
        probe = torch.zeros(1, device="cuda")
        (probe + 1).cpu()
    except RuntimeError as exc:
        print(f"[vision_pipeline] CUDA unusable ({exc}) - using CPU")
        return "cpu"
    return "cuda"


def _face_bbox(landmarks, w: int, h: int) -> tuple[int, int, int, int]:
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    pad = 8
    return (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(w, max(xs) + pad),
        min(h, max(ys) + pad),
    )


def _roi_boxes(
    frame: np.ndarray, landmarks
) -> tuple[
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int] | None,
]:
    """Axis-aligned boxes on the display frame (aligned crops feed the CNN)."""
    h, w = frame.shape[:2]
    eye_dist = RegionCropper.get_inter_eye_distance(landmarks, w, h)
    eye_box_w = int(EYE_BOX_W_RATIO * eye_dist)
    eye_box_h = int(EYE_BOX_H_RATIO * eye_dist)
    left_center = RegionCropper.get_region_center(landmarks, LEFT_EYE_INDICES, w, h)
    right_center = RegionCropper.get_region_center(landmarks, RIGHT_EYE_INDICES, w, h)

    mouth_pts = [RegionCropper.landmark_to_pixel(landmarks[i], w, h) for i in MOUTH_INDICES]
    xs = [p[0] for p in mouth_pts]
    ys = [p[1] for p in mouth_pts]
    mouth_w_landmark = max(xs) - min(xs)
    mouth_h_landmark = max(ys) - min(ys)
    mouth_box_w = max(int(eye_dist), int(MOUTH_LANDMARK_W_SCALE * mouth_w_landmark))
    mouth_box_h = max(
        int(MOUTH_BOX_H_RATIO_HYBRID * eye_dist),
        int(MOUTH_LANDMARK_H_SCALE * mouth_h_landmark),
    )
    mcx, mcy = RegionCropper.get_region_center(landmarks, MOUTH_INDICES, w, h)
    mcy = int(mcy + MOUTH_VERTICAL_OFFSET_RATIO_HYBRID * mouth_box_h)

    _, le_box = RegionCropper.crop_fixed_box(
        frame, left_center, eye_box_w, eye_box_h, EYE_CROP_SIZE
    )
    _, re_box = RegionCropper.crop_fixed_box(
        frame, right_center, eye_box_w, eye_box_h, EYE_CROP_SIZE
    )
    _, mo_box = RegionCropper.crop_fixed_box(
        frame, (mcx, mcy), mouth_box_w, mouth_box_h, MOUTH_CROP_SIZE
    )
    return _face_bbox(landmarks, w, h), le_box, re_box, mo_box


def _aligned_crops(
    frame: np.ndarray,
    landmarks,
    cropper: RegionCropper,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Same aligned crops as ``RegionCropper.get_crops`` (CNN input)."""
    h, w = frame.shape[:2]
    eye_dist = RegionCropper.get_inter_eye_distance(landmarks, w, h)
    roll_deg = RegionCropper.get_face_roll_deg(landmarks, w, h)
    eye_box_w = int(EYE_BOX_W_RATIO * eye_dist)
    eye_box_h = int(EYE_BOX_H_RATIO * eye_dist)
    left_center = RegionCropper.get_region_center(landmarks, LEFT_EYE_INDICES, w, h)
    right_center = RegionCropper.get_region_center(landmarks, RIGHT_EYE_INDICES, w, h)

    mouth_pts = [RegionCropper.landmark_to_pixel(landmarks[i], w, h) for i in MOUTH_INDICES]
    xs = [p[0] for p in mouth_pts]
    ys = [p[1] for p in mouth_pts]
    mouth_w_landmark = max(xs) - min(xs)
    mouth_h_landmark = max(ys) - min(ys)
    mouth_box_w = max(int(eye_dist), int(MOUTH_LANDMARK_W_SCALE * mouth_w_landmark))
    mouth_box_h = max(
        int(MOUTH_BOX_H_RATIO_HYBRID * eye_dist),
        int(MOUTH_LANDMARK_H_SCALE * mouth_h_landmark),
    )
    mcx, mcy = RegionCropper.get_region_center(landmarks, MOUTH_INDICES, w, h)
    mcy = int(mcy + MOUTH_VERTICAL_OFFSET_RATIO_HYBRID * mouth_box_h)

    left_eye, _ = RegionCropper.crop_aligned_box(
        frame, left_center, roll_deg, eye_box_w, eye_box_h, cropper.eye_crop_size
    )
    right_eye, _ = RegionCropper.crop_aligned_box(
        frame, right_center, roll_deg, eye_box_w, eye_box_h, cropper.eye_crop_size
    )
    mouth, _ = RegionCropper.crop_aligned_box(
        frame, (mcx, mcy), roll_deg, mouth_box_w, mouth_box_h, cropper.mouth_crop_size
    )
    return left_eye, right_eye, mouth


def _mirror_box(
    box: tuple[int, int, int, int] | None, width: int
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (width - 1 - x2, y1, width - 1 - x1, y2)


def _put_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x, y = xy
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_overlay(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    le_box: tuple[int, int, int, int] | None,
    re_box: tuple[int, int, int, int] | None,
    mo_box: tuple[int, int, int, int] | None,
    p_left_closed: float | None,
    p_right_closed: float | None,
    p_mouth_open: float | None,
    inference_fps: float,
) -> None:
    left_closed = p_left_closed is not None and p_left_closed >= EYE_CLOSED_THRESH
    right_closed = p_right_closed is not None and p_right_closed >= EYE_CLOSED_THRESH
    mouth_open = p_mouth_open is not None and p_mouth_open >= MOUTH_OPEN_THRESH

    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _put_text(frame, "face", (x1, max(0, y1 - 6)), 0.5, (0, 255, 0), 1)

    eye_color = (0, 0, 255) if (left_closed or right_closed) else (255, 200, 0)
    mouth_color = (0, 165, 255) if mouth_open else (255, 200, 0)
    eye_thick = 2 if (left_closed or right_closed) else 1
    mouth_thick = 2 if mouth_open else 1

    for box, label, p_closed, p_open, color, thick in (
        (le_box, "L eye", p_left_closed, None, eye_color, eye_thick),
        (re_box, "R eye", p_right_closed, None, eye_color, eye_thick),
        (mo_box, "mouth", None, p_mouth_open, mouth_color, mouth_thick),
    ):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        if p_open is not None:
            p_open_f = float(p_open)
            p_closed_f = 1.0 - p_open_f
            tag = "OPEN" if p_open_f >= MOUTH_RISING_THRESH else "closed"
            caption = f"{label} {tag} o={p_open_f:.2f} c={p_closed_f:.2f}"
        elif p_closed is not None:
            p_closed_f = float(p_closed)
            p_open_f = 1.0 - p_closed_f
            tag = "CLOSED" if p_closed_f >= EYE_CLOSED_THRESH else "open"
            caption = f"{label} {tag} c={p_closed_f:.2f} o={p_open_f:.2f}"
        else:
            caption = f"{label} --"
        _put_text(frame, caption, (x1, max(0, y1 - 4)), 0.42, color, 1)

    _put_text(
        frame, f"inference fps {inference_fps:5.1f}", (12, 24), 0.55, (200, 220, 0), 1
    )
    lines = [
        (
            f"P(L closed) {p_left_closed:.3f}  open {1 - p_left_closed:.3f}"
            if p_left_closed is not None
            else "P(L closed) --"
        ),
        (
            f"P(R closed) {p_right_closed:.3f}  open {1 - p_right_closed:.3f}"
            if p_right_closed is not None
            else "P(R closed) --"
        ),
        (
            f"P(mouth open) {p_mouth_open:.3f}  closed {1 - p_mouth_open:.3f}"
            if p_mouth_open is not None
            else "P(mouth open) --"
        ),
    ]
    y = 48
    for line in lines:
        _put_text(frame, line, (12, y), 0.48, (255, 255, 255), 1)
        y += 20

    _put_text(frame, "q quit", (12, frame.shape[0] - 12), 0.5, (180, 180, 180), 1)


def _preview_frame(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Scale to fit max size without stretching (webcam aspect varies)."""
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if new_w == max_w and new_h == max_h:
        return resized
    canvas = np.zeros((max_h, max_w, 3), dtype=frame.dtype)
    y0 = (max_h - new_h) // 2
    x0 = (max_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def main() -> None:
    device = _resolve_device(DEVICE)
    landmarker_path = resolve_asset(LANDMARKER_PATH)
    print(f"[vision_pipeline] device={device}")
    print(f"[vision_pipeline] landmarker={landmarker_path}")
    print(f"[vision_pipeline] eye={EYE_CKPT}")
    print(f"[vision_pipeline] mouth={MOUTH_CKPT}")

    cropper = RegionCropper(landmarker_path, fps=TARGET_FPS)
    eye_clf = EyeStateClassifier(EYE_CKPT, device=device)
    mouth_clf = MouthStateClassifier(MOUTH_CKPT, device=device)

    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {WEBCAM_INDEX}")

    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    inference_dt_avg = 0.0
    inference_fps = 0.0

    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            p_left = p_right = p_mouth = None
            face_box = le_box = re_box = mo_box = None
            t_inf_start = time.perf_counter()
            landmarks = cropper._detect_landmarks(frame)
            if landmarks is not None:
                face_box, le_box, re_box, mo_box = _roi_boxes(frame, landmarks)
                le, re, mo = _aligned_crops(frame, landmarks, cropper)
                p_left, p_right = eye_clf.predict_probs_batch([le, re])
                p_mouth = mouth_clf.predict_prob(mo)
            inference_dt = time.perf_counter() - t_inf_start
            inference_dt_avg = (
                inference_dt
                if inference_dt_avg == 0.0
                else 0.9 * inference_dt_avg + 0.1 * inference_dt
            )
            inference_fps = 1.0 / inference_dt_avg if inference_dt_avg > 0 else 0.0

            display = cv2.flip(frame, 1)
            disp_w = display.shape[1]
            _draw_overlay(
                display,
                _mirror_box(face_box, disp_w),
                _mirror_box(le_box, disp_w),
                _mirror_box(re_box, disp_w),
                _mirror_box(mo_box, disp_w),
                p_left,
                p_right,
                p_mouth,
                inference_fps,
            )
            preview = _preview_frame(display, PREVIEW_WIDTH, PREVIEW_HEIGHT)
            cv2.imshow(WINDOW_TITLE, preview)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            loop_dt = time.perf_counter() - t_loop
            sleep_for = TARGET_DT - loop_dt
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cropper.close()


if __name__ == "__main__":
    main()
