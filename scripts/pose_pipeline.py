"""
Webcam demo for the upper-body pose model.

Shows the five pose keypoints and their confidence scores.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from fatigue_pipeline.constants import POSE_NUM_KPTS, UPPER_BODY_KPT_NAMES
from fatigue_pipeline.pose_estimator import PoseEstimator
from scripts.weights_path import POSE_MODEL_PATH, resolve_asset

PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
WEBCAM_INDEX = 0
TARGET_FPS = 30.0
TARGET_DT = 1.0 / TARGET_FPS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_TITLE = "FatigueSense pose_pipeline (q to quit)"

POSE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (3, 4),
)


def _resolve_device(d: str) -> str:
    if d != "cuda":
        return d
    if not torch.cuda.is_available():
        print("[pose_pipeline] CUDA not available - using CPU")
        return "cpu"
    try:
        probe = torch.zeros(1, device="cuda")
        (probe + 1).cpu()
    except RuntimeError as exc:
        print(f"[pose_pipeline] CUDA unusable ({exc}) - using CPU")
        return "cpu"
    return "cuda"


def _mirror_kpts(kpts: np.ndarray | None, width: int) -> np.ndarray | None:
    if kpts is None:
        return None
    out = kpts.copy()
    valid = np.isfinite(out[:, 0])
    out[valid, 0] = (width - 1) - out[valid, 0]
    return out


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


def _draw_pose(frame: np.ndarray, kpts: np.ndarray | None) -> None:
    if kpts is None:
        return
    h, w = frame.shape[:2]
    pts: list[tuple[int, int] | None] = []
    for i in range(POSE_NUM_KPTS):
        x, y, c = kpts[i]
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(c):
            pts.append(None)
            continue
        px = int(np.clip(x, 0, w - 1))
        py = int(np.clip(y, 0, h - 1))
        pts.append((px, py))

    for a, b in POSE_EDGES:
        if pts[a] is None or pts[b] is None:
            continue
        cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)

    for i, pt in enumerate(pts):
        if pt is None:
            continue
        cv2.circle(frame, pt, 4, (0, 200, 255), -1, cv2.LINE_AA)
        _put_text(
            frame,
            UPPER_BODY_KPT_NAMES[i],
            (pt[0] + 6, pt[1] - 6),
            0.4,
            (0, 200, 255),
            1,
        )


def _mean_kpt_conf(kpts: np.ndarray | None) -> float | None:
    if kpts is None:
        return None
    confs = [float(kpts[i, 2]) for i in range(POSE_NUM_KPTS) if np.isfinite(kpts[i, 2])]
    if not confs:
        return None
    return sum(confs) / len(confs)


def _draw_overlay(
    frame: np.ndarray,
    kpts: np.ndarray | None,
    inference_fps: float,
) -> None:
    _draw_pose(frame, kpts)
    _put_text(
        frame, f"inference fps {inference_fps:5.1f}", (12, 24), 0.55, (200, 220, 0), 1
    )

    mean_conf = _mean_kpt_conf(kpts)
    if kpts is None:
        _put_text(frame, "pose: no person", (12, 48), 0.48, (180, 180, 180), 1)
    else:
        _put_text(
            frame,
            f"pose conf mean {mean_conf:.3f}" if mean_conf is not None else "pose conf mean --",
            (12, 48),
            0.48,
            (255, 255, 255),
            1,
        )
        y = 72
        for i, name in enumerate(UPPER_BODY_KPT_NAMES):
            x, y_k, c = kpts[i]
            if np.isfinite(x) and np.isfinite(y_k) and np.isfinite(c):
                line = f"{name}: conf={c:.3f}"
            else:
                line = f"{name}: --"
            _put_text(frame, line, (12, y), 0.42, (255, 255, 255), 1)
            y += 18

    _put_text(frame, "q quit", (12, frame.shape[0] - 12), 0.5, (180, 180, 180), 1)


def _preview_frame(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Scale to fit max size without stretching"""
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
    pose_weights = resolve_asset(POSE_MODEL_PATH)
    print(f"[pose_pipeline] device={device}")
    print(f"[pose_pipeline] pose model={pose_weights}")

    pose_estimator = PoseEstimator(
        weights_path=str(pose_weights),
        device=device if device != "mps" else None,
    )

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

            t_inf_start = time.perf_counter()
            kpts = pose_estimator.predict(frame)
            inference_dt = time.perf_counter() - t_inf_start
            inference_dt_avg = (
                inference_dt
                if inference_dt_avg == 0.0
                else 0.9 * inference_dt_avg + 0.1 * inference_dt
            )
            inference_fps = 1.0 / inference_dt_avg if inference_dt_avg > 0 else 0.0

            display = cv2.flip(frame, 1)
            _draw_overlay(display, _mirror_kpts(kpts, display.shape[1]), inference_fps)
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
        pose_estimator.close()


if __name__ == "__main__":
    main()
