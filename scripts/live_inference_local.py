"""Local-only webcam demo with the full FatigueSense pipeline.

Runs MediaPipe FaceLandmarker + the eye/mouth CNN classifiers +
YOLO11n-pose upper-body kpts in-process, feeds them through the same
``FeatureAggregator`` the offline pipeline uses, and then through the
BiGRU temporal model to display a continuous focus score.

The preview shows:
    - left HUD: per-frame raw probs + all 17 aggregated feature values
    - right HUD: big focus score + cumulative blink / yawn counters
    - three FPS values:
        inference_fps  - frame capture -> per-frame probs + aggregator update
        temporal_fps   - feature step -> BiGRU forward (only when buffer full)
        overall_fps    - whole loop iteration (capture -> display)

Aggregator defaults mirror the production pipeline: ``sub_window_s=60``
and ``step_stride_s=1``. With the BiGRU's ``window_steps=30`` that maps
to a ~90 s cold-start budget (60 s to fill the first sub-window + 30 s
to fill the temporal buffer).

Set ``DEMO_SUB_WINDOW_S=5`` for faster iteration; metric values will be
noisier than at the production 60 s window.

Model checkpoint paths live in ``scripts/weights_path.py``.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import vision

from fatigue_pipeline.aggregation.step_features import FeatureStep
from fatigue_pipeline.cnn_predictors import EyeStateClassifier, MouthStateClassifier
from fatigue_pipeline.constants import (
    DEFAULT_BIGRU_INFERENCE_STRIDE,
    DEFAULT_STEP_STRIDE_S,
    DEFAULT_SUB_WINDOW_S,
    EYE_BOX_H_RATIO,
    FEATURE_NAMES,
    EYE_BOX_W_RATIO,
    EYE_CROP_SIZE,
    LEFT_EYE_INDICES,
    MOUTH_BOX_H_RATIO_HYBRID,
    MOUTH_CROP_SIZE,
    MOUTH_INDICES,
    MOUTH_LANDMARK_H_SCALE,
    MOUTH_LANDMARK_W_SCALE,
    MOUTH_VERTICAL_OFFSET_RATIO_HYBRID,
    MS_PER_SECOND,
    POSE_NUM_KPTS,
    RIGHT_EYE_INDICES,
    UPPER_BODY_KPT_NAMES,
)
from fatigue_pipeline.feature_aggregator import FeatureAggregator
from fatigue_pipeline.inference_pipeline import FrameProbs
from fatigue_pipeline.pose_estimator import PoseEstimator
from fatigue_pipeline.region_cropper import RegionCropper
from model_architecture.dataset.temporal_window_dataset import DEFAULT_WINDOW_STEPS
from model_architecture.models.bigru_temporal import build_temporal_model
from model_architecture.utils.normalization import load_normalization
from scripts.weights_path import (
    EYE_CKPT,
    LANDMARKER_PATH,
    MOUTH_CKPT,
    NORMALIZATION_PATH,
    POSE_MODEL_PATH,
    TEMPORAL_MODEL_PATH,
    resolve_asset,
)

# ---- display / capture ----
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 960
WEBCAM_INDEX = 0
TARGET_FPS = 30.0
TARGET_DT = 1.0 / TARGET_FPS

# ---- device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- temporal window / smoothing ----
WINDOW_STEPS = DEFAULT_WINDOW_STEPS
BIGRU_INFERENCE_STRIDE = DEFAULT_BIGRU_INFERENCE_STRIDE
EMA_ALPHA = 0.4
LATENCY_WINDOW = 120
DEMO_SUB_WINDOW_S = DEFAULT_SUB_WINDOW_S
DEMO_STEP_STRIDE_S = DEFAULT_STEP_STRIDE_S

# ---- logging / debug ----
STEP_LOG_PATH = Path("runs/live_inference_steps.csv")
EYE_LOG_EVERY_N = 0
POSE_LOG_EVERY_N = 0
MOUTH_LOG_EVERY_N = 0
EYE_DUMP_DIR = None
EYE_DUMP_EVERY_N = 30

# ---- pose skeleton edges (over the 5-kpt upper-body subset) ----
POSE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),  # nose <-> ear_left
    (0, 2),  # nose <-> ear_right
    (0, 3),  # nose <-> shoulder_left
    (0, 4),  # nose <-> shoulder_right
    (3, 4),  # shoulder_left <-> shoulder_right
)


def _resolve_device(d: str) -> str:
    if d != "cuda":
        return d
    if not torch.cuda.is_available():
        print("[live_local] CUDA not available - falling back to CPU")
        return "cpu"
    try:
        probe = torch.zeros(1, device="cuda")
        (probe + 1).cpu()
    except RuntimeError as e:
        print(f"[live_local] CUDA present but unusable ({e.__class__.__name__}: {e}) - falling back to CPU")
        return "cpu"
    return "cuda"


def _build_landmarker(path: str) -> vision.FaceLandmarker:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Landmarker model missing: {p}")
    base = mp.tasks.BaseOptions(model_asset_path=str(p))
    opts = vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _face_bbox(landmarks, w: int, h: int) -> tuple[int, int, int, int]:
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    pad = 8
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    return x1, y1, x2, y2


def _crops_from_landmarks(frame: np.ndarray, landmarks) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int] | None,
]:
    h, w = frame.shape[:2]
    eye_dist = RegionCropper.get_inter_eye_distance(landmarks, w, h)
    roll_deg = RegionCropper.get_face_roll_deg(landmarks, w, h)

    eye_box_w = int(EYE_BOX_W_RATIO * eye_dist)
    eye_box_h = int(EYE_BOX_H_RATIO * eye_dist)

    left_center = RegionCropper.get_region_center(landmarks, LEFT_EYE_INDICES, w, h)
    right_center = RegionCropper.get_region_center(landmarks, RIGHT_EYE_INDICES, w, h)

    mouth_pts = [
        RegionCropper.landmark_to_pixel(landmarks[i], w, h) for i in MOUTH_INDICES
    ]
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

    # Aligned crops -> classifier (eye axis canonicalized to horizontal).
    le, _ = RegionCropper.crop_aligned_box(
        frame, left_center, roll_deg, eye_box_w, eye_box_h, EYE_CROP_SIZE
    )
    re, _ = RegionCropper.crop_aligned_box(
        frame, right_center, roll_deg, eye_box_w, eye_box_h, EYE_CROP_SIZE
    )
    mo, _ = RegionCropper.crop_aligned_box(
        frame, (mcx, mcy), roll_deg, mouth_box_w, mouth_box_h, MOUTH_CROP_SIZE
    )
    # Axis-aligned bboxes -> HUD overlay on the un-rotated display frame.
    _, le_box = RegionCropper.crop_fixed_box(
        frame, left_center, eye_box_w, eye_box_h, EYE_CROP_SIZE
    )
    _, re_box = RegionCropper.crop_fixed_box(
        frame, right_center, eye_box_w, eye_box_h, EYE_CROP_SIZE
    )
    _, mo_box = RegionCropper.crop_fixed_box(
        frame, (mcx, mcy), mouth_box_w, mouth_box_h, MOUTH_CROP_SIZE
    )
    return le, re, mo, le_box, re_box, mo_box


def _mirror_box(
    box: tuple[int, int, int, int] | None, width: int
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (width - 1 - x2, y1, width - 1 - x1, y2)


def _mirror_kpts(kpts: np.ndarray | None, width: int) -> np.ndarray | None:
    if kpts is None:
        return None
    out = kpts.copy()
    valid = np.isfinite(out[:, 0])
    out[valid, 0] = (width - 1) - out[valid, 0]
    return out


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
        cv2.putText(
            frame,
            UPPER_BODY_KPT_NAMES[i],
            (pt[0] + 6, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )


def _status_label(score: float | None) -> tuple[str, tuple[int, int, int]]:
    """Map focus score -> short label + BGR color.

    Thresholds calibrated to the bootstrap label heuristic in
    ``temporal_window_dataset.default_label_from_window`` (PERCLOS-
    dominant, pose as 20% modifier). DROWSY band starts at 0.55 so
    PERCLOS ~= 0.6 alone is enough to leave ALERT.
    """
    if score is None:
        return ("WARM-UP", (180, 180, 180))
    if score < 0.30:
        return ("FATIGUED", (0, 0, 255))
    if score < 0.55:
        return ("DROWSY", (0, 165, 255))
    return ("ALERT", (0, 200, 0))


def _put_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    *,
    align: str = "left",
) -> None:
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = xy
    if align == "right":
        x = x - tw
    elif align == "center":
        x = x - tw // 2
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


def _draw_left_hud(
    frame: np.ndarray,
    p_left: float | None,
    p_right: float | None,
    p_mouth: float | None,
    kpts: np.ndarray | None,
    inference_fps: float,
    temporal_fps: float,
    overall_fps: float,
    latency_ms: float,
    last_step: FeatureStep | None,
    warmup_progress: int,
    warmup_target: int,
) -> None:
    if kpts is not None and kpts.size:
        kpt_mean_conf = float(np.nanmean(kpts[:, 2]))
    else:
        kpt_mean_conf = 0.0

    lines: list[tuple[str, tuple[int, int, int]]] = []
    white = (255, 255, 255)
    gray = (200, 200, 200)
    cyan = (200, 220, 0)

    lines.append((f"inference fps : {inference_fps:5.1f}", cyan))
    lines.append((f"temporal  fps : {temporal_fps:5.1f}", cyan))
    lines.append(
        (f"overall   fps : {overall_fps:5.1f}  (target {TARGET_FPS:.0f})", cyan)
    )
    lines.append((f"latency mean  : {latency_ms:6.2f} ms", gray))
    lines.append(("", white))
    lines.append(("--- raw per-frame ---", white))
    lines.append(
        (
            (
                f"P(left closed) : {p_left:.3f}"
                if p_left is not None
                else "P(left closed) : --"
            ),
            white,
        )
    )
    lines.append(
        (
            (
                f"P(right closed): {p_right:.3f}"
                if p_right is not None
                else "P(right closed): --"
            ),
            white,
        )
    )
    lines.append(
        (
            (
                f"P(mouth open)  : {p_mouth:.3f}"
                if p_mouth is not None
                else "P(mouth open)  : --"
            ),
            white,
        )
    )
    lines.append(
        (
            (
                f"pose conf      : {kpt_mean_conf:.3f}"
                if kpts is not None
                else "pose           : (no person)"
            ),
            white,
        )
    )
    lines.append(("", white))
    lines.append(("--- aggregated step ---", white))

    if last_step is None:
        pct = (warmup_progress / warmup_target * 100.0) if warmup_target else 0.0
        lines.append((f"WARMUP: {warmup_progress}/{warmup_target} ({pct:5.1f}%)", gray))
        lines.append(("(no step emitted yet)", gray))
    else:
        f = last_step.features
        valid_tag = "valid" if f.valid else "INVALID"
        lines.append(
            (
                f"step #{last_step.step_idx}  t={last_step.timestamp_s:6.2f}s [{valid_tag}]",
                gray,
            )
        )
        lines.append((f"PERCLOS        : {f.perclos:.3f}", white))
        lines.append((f"blink rate     : {f.blink_rate_bpm:5.2f} bpm", white))
        lines.append((f"mean blink dur : {f.mean_blink_duration:.3f} s", white))
        lines.append((f"eye var        : {f.eye_closure_variance:.4f}", white))
        lines.append((f"yawn rate      : {f.yawn_rate_per_min:5.2f} /min", white))
        lines.append((f"mean yawn dur  : {f.mean_yawn_duration:.3f} s", white))
        lines.append((f"mouth open     : {f.mouth_open_ratio:.3f}", white))
        lines.append((f"mean P(eye)    : {f.mean_p_eye:.3f}", white))
        lines.append((f"mean P(mouth)  : {f.mean_p_mouth:.3f}", white))
        lines.append(("--- pose (60s) ---", white))
        lines.append((f"head pitch     : {f.head_pitch:+.3f}", white))
        lines.append((f"head roll      : {f.head_roll:+.3f}", white))
        lines.append((f"shoulder tilt  : {f.shoulder_tilt:+.3f}", white))
        lines.append((f"head size      : {f.head_size_ratio:.3f}", white))
        lines.append((f"head motion E  : {f.head_motion_energy:.4f}", white))
        lines.append((f"head drift_y   : {f.head_drift_y:.4f}", white))
        lines.append((f"posture drift  : {f.posture_drift:.4f}", white))
        lines.append((f"kpt visibility : {f.kpt_visibility:.3f}", white))

    x = 12
    y = 22
    line_h = 16
    scale = 0.42
    for text, color in lines:
        if text:
            _put_text(frame, text, (x, y), scale, color, thickness=1)
        y += line_h


def _draw_right_hud(
    frame: np.ndarray,
    score_raw: float | None,
    score_ema: float | None,
    blink_total: int,
    yawn_total: int,
    buffer_fill: int,
    buffer_target: int,
) -> None:
    h, w = frame.shape[:2]
    right_x = w - 30  # right margin anchor

    label, color = _status_label(score_ema)

    # Big focus score
    if score_ema is None:
        pct = (buffer_fill / buffer_target * 100.0) if buffer_target else 0.0
        _put_text(frame, "FOCUS", (right_x, 60), 0.7, color, 2, align="right")
        _put_text(frame, "--", (right_x, 130), 2.4, color, 4, align="right")
        _put_text(
            frame,
            f"warmup {buffer_fill}/{buffer_target} ({pct:.0f}%)",
            (right_x, 165),
            0.6,
            (180, 180, 180),
            1,
            align="right",
        )
    else:
        _put_text(frame, "FOCUS", (right_x, 60), 0.7, color, 2, align="right")
        _put_text(
            frame,
            f"{score_ema:.2f}",
            (right_x, 140),
            2.6,
            color,
            5,
            align="right",
        )
        _put_text(frame, label, (right_x, 175), 0.9, color, 2, align="right")
        if score_raw is not None:
            _put_text(
                frame,
                f"raw {score_raw:.2f}",
                (right_x, 200),
                0.55,
                (180, 180, 180),
                1,
                align="right",
            )

        # Score bar (right-justified)
        bar_w = 260
        bar_h = 16
        bar_x2 = right_x
        bar_x1 = bar_x2 - bar_w
        bar_y = 215
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * float(np.clip(score_ema, 0.0, 1.0)))
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + fill, bar_y + bar_h), color, -1)

    # Counters
    counter_y = 290
    _put_text(
        frame, "BLINKS", (right_x, counter_y), 0.6, (200, 200, 200), 1, align="right"
    )
    _put_text(
        frame,
        f"{blink_total}",
        (right_x, counter_y + 50),
        1.5,
        (255, 255, 255),
        3,
        align="right",
    )

    _put_text(
        frame,
        "YAWNS",
        (right_x, counter_y + 90),
        0.6,
        (200, 200, 200),
        1,
        align="right",
    )
    _put_text(
        frame,
        f"{yawn_total}",
        (right_x, counter_y + 140),
        1.5,
        (255, 255, 255),
        3,
        align="right",
    )

    _put_text(
        frame,
        "q to quit",
        (right_x, h - 18),
        0.5,
        (180, 180, 180),
        1,
        align="right",
    )


def _draw_geometry(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    le_box: tuple[int, int, int, int] | None,
    re_box: tuple[int, int, int, int] | None,
    mo_box: tuple[int, int, int, int] | None,
    p_left: float | None,
    p_right: float | None,
    p_mouth: float | None,
    kpts: np.ndarray | None,
    is_yawning: bool = False,
    is_blinking: bool = False,
) -> None:
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _put_text(frame, "face", (x1, max(0, y1 - 6)), 0.5, (0, 255, 0), 1)

    eye_color = (0, 0, 255) if is_blinking else (255, 200, 0)
    eye_label_prefix = "BLINK " if is_blinking else ""
    mouth_color = (0, 0, 255) if is_yawning else (255, 200, 0)
    mouth_label_prefix = "YAWN " if is_yawning else ""

    eye_thickness = 2 if is_blinking else 1
    mouth_thickness = 2 if is_yawning else 1

    for box, label, prob, color, thickness, prefix in (
        (le_box, "L eye", p_left, eye_color, eye_thickness, eye_label_prefix),
        (re_box, "R eye", p_right, eye_color, eye_thickness, eye_label_prefix),
        (mo_box, "mouth", p_mouth, mouth_color, mouth_thickness, mouth_label_prefix),
    ):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        body = f"{label} {prob:.2f}" if prob is not None else f"{label} --"
        text = f"{prefix}{body}"
        _put_text(frame, text, (x1, max(0, y1 - 4)), 0.42, color, 1)

    _draw_pose(frame, kpts)


def main() -> None:
    device = _resolve_device(DEVICE)
    print(f"[live_local] device={device}")
    print(f"[live_local] eye ckpt = {EYE_CKPT}")
    print(f"[live_local] mouth ckpt = {MOUTH_CKPT}")
    print(f"[live_local] landmarker = {LANDMARKER_PATH}")
    print(f"[live_local] pose model = {POSE_MODEL_PATH}")
    print(f"[live_local] temporal model = {TEMPORAL_MODEL_PATH}")
    print(f"[live_local] normalization = {NORMALIZATION_PATH}")
    print(f"[live_local] step log = {STEP_LOG_PATH}")

    STEP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    step_log_file = open(STEP_LOG_PATH, "w", newline="", encoding="utf-8")
    step_log_writer = csv.writer(step_log_file)
    step_log_writer.writerow(
        [
            "timestamp_s",
            "step_idx",
            "valid",
            *FEATURE_NAMES,
            "blink_total",
            "yawn_total",
            "score_raw",
            "score_ema",
        ]
    )
    step_log_file.flush()

    eye_clf = EyeStateClassifier(EYE_CKPT, device=device)
    mouth_clf = MouthStateClassifier(MOUTH_CKPT, device=device)
    landmarker = _build_landmarker(str(resolve_asset(LANDMARKER_PATH)))
    pose_estimator = PoseEstimator(
        weights_path=str(resolve_asset(POSE_MODEL_PATH)),
        device=device if device != "mps" else None,
    )

    # Temporal model -----------------------------------------------------
    torch_device = torch.device(device)
    temporal_model = None
    norm_mean = None
    norm_std = None
    try:
        temporal_ckpt = resolve_asset(TEMPORAL_MODEL_PATH)
        norm_path = resolve_asset(NORMALIZATION_PATH)
    except FileNotFoundError:
        temporal_ckpt = norm_path = None
    if temporal_ckpt is not None and norm_path is not None:
        norm = load_normalization(norm_path)
        norm_mean = torch.from_numpy(norm["mean"]).to(torch_device)
        norm_std = torch.from_numpy(norm["std"]).to(torch_device)
        temporal_model = build_temporal_model().to(torch_device)
        state = torch.load(
            temporal_ckpt, map_location=torch_device, weights_only=True
        )
        temporal_model.load_state_dict(state)
        temporal_model.eval()
        print(f"[live_local] temporal model loaded - focus score active")
    else:
        print(
            "[live_local] temporal model or normalization missing - "
            "focus score will stay '--' (aggregator still runs)"
        )

    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {WEBCAM_INDEX}")

    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    ms_per_frame = MS_PER_SECOND / TARGET_FPS

    aggregator = FeatureAggregator(
        fps=TARGET_FPS,
        sub_window_s=DEMO_SUB_WINDOW_S,
        step_stride_s=DEMO_STEP_STRIDE_S,
    )
    cold_start_s = DEMO_SUB_WINDOW_S + WINDOW_STEPS * DEMO_STEP_STRIDE_S
    print(
        f"[live_local] target fps={TARGET_FPS:.0f}  "
        f"sub_window_s={DEMO_SUB_WINDOW_S}  step_stride_s={DEMO_STEP_STRIDE_S}  "
        f"window_steps={WINDOW_STEPS}"
    )
    print(
        f"[live_local] first aggregator step after ~{aggregator.sub_window_frames} "
        f"frames (~{DEMO_SUB_WINDOW_S:.0f}s). "
        f"BiGRU first emission after additional ~{WINDOW_STEPS}s "
        f"(total cold start ~{cold_start_s:.0f}s)."
    )

    frame_idx = 0
    last_step: FeatureStep | None = None
    feature_buffer: deque[np.ndarray] = deque(maxlen=WINDOW_STEPS)
    score_raw: float | None = None
    score_ema: float | None = None
    blink_total = 0
    # Counts emits since the buffer first filled; BiGRU forward only fires
    # when this counter is a multiple of BIGRU_INFERENCE_STRIDE.
    emits_since_full = 0
    yawn_total = 0

    inference_dt_avg = 0.0
    temporal_dt_avg = 0.0
    overall_dt_avg = 0.0
    inference_fps = 0.0
    temporal_fps = 0.0
    overall_fps = 0.0
    latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)

    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            # Inference frame is NOT flipped (training distribution).
            # Display copy below is mirrored for selfie feel.
            h, w = frame.shape[:2]

            # --- Inference timing zone -----------------------------------
            t_inf_start = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * ms_per_frame)
            frame_idx += 1
            result = landmarker.detect_for_video(mp_image, ts_ms)

            face_box = None
            le_box = re_box = mo_box = None
            p_left = p_right = p_mouth = None

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                face_box = _face_bbox(landmarks, w, h)
                le, re_, mo, le_box, re_box, mo_box = _crops_from_landmarks(
                    frame, landmarks
                )
                p_left, p_right = eye_clf.predict_probs_batch([le, re_])
                p_mouth = mouth_clf.predict_prob(mo)

                if (
                    p_left is not None
                    and p_right is not None
                    and EYE_LOG_EVERY_N > 0
                    and (frame_idx % EYE_LOG_EVERY_N) == 0
                ):
                    delta = p_left - p_right
                    flag = "  *ASYM*" if abs(delta) > 0.3 else ""
                    print(
                        f"[eye {frame_idx:6d}] "
                        f"P(L_closed)={p_left:.3f} "
                        f"P(R_closed)={p_right:.3f} "
                        f"delta(L-R)={delta:+.3f}{flag}"
                    )

                if (
                    EYE_DUMP_DIR
                    and le is not None
                    and re_ is not None
                    and (frame_idx % EYE_DUMP_EVERY_N) == 0
                ):
                    Path(EYE_DUMP_DIR).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        f"{EYE_DUMP_DIR}/L_{frame_idx:06d}_p{int(p_left * 1000):03d}.png",
                        le,
                    )
                    cv2.imwrite(
                        f"{EYE_DUMP_DIR}/R_{frame_idx:06d}_p{int(p_right * 1000):03d}.png",
                        re_,
                    )

            kpts = pose_estimator.predict(frame)
            if (
                POSE_LOG_EVERY_N > 0
                and kpts is not None
                and (frame_idx % POSE_LOG_EVERY_N) == 0
            ):
                conf_str = " ".join(
                    f"{name}={kpts[i, 2]:.2f}"
                    for i, name in enumerate(UPPER_BODY_KPT_NAMES)
                )
                print(f"[pose {frame_idx:6d}] {conf_str}")

            if (
                MOUTH_LOG_EVERY_N > 0
                and p_mouth is not None
                and (frame_idx % MOUTH_LOG_EVERY_N) == 0
            ):
                state = "ACTIVE" if aggregator.is_yawning else "      "
                print(
                    f"[mouth {frame_idx:6d}] {state}  P(mouth)={p_mouth:.3f}  "
                    f"yawn_total={aggregator.yawn_total}"
                )

            face_detected = (
                result.face_landmarks is not None and len(result.face_landmarks) > 0
            )
            probs = FrameProbs(
                frame_idx=frame_idx - 1,
                timestamp_s=(frame_idx - 1) / TARGET_FPS,
                face_detected=face_detected,
                p_eye_left_closed=p_left,
                p_eye_right_closed=p_right,
                p_mouth_open=p_mouth,
                keypoints=kpts,
            )
            step = aggregator.add(probs)
            is_yawning = aggregator.is_yawning
            is_blinking = aggregator.is_blinking
            t_inf_end = time.perf_counter()
            inference_dt = t_inf_end - t_inf_start
            inference_dt_avg = (
                inference_dt
                if inference_dt_avg == 0.0
                else 0.9 * inference_dt_avg + 0.1 * inference_dt
            )
            inference_fps = 1.0 / inference_dt_avg if inference_dt_avg > 0 else 0.0
            latencies.append(inference_dt * 1000.0)
            latency_mean = sum(latencies) / len(latencies)

            # --- Step + BiGRU timing zone --------------------------------
            if step is not None:
                last_step = step
                blink_total = step.blink_count_total
                yawn_total = step.yawn_count_total
                f = step.features

                if temporal_model is not None and f.valid:
                    feature_buffer.append(f.to_array())
                    if len(feature_buffer) == WINDOW_STEPS:
                        run_bigru = (emits_since_full % BIGRU_INFERENCE_STRIDE) == 0
                        emits_since_full += 1
                    else:
                        run_bigru = False

                    if run_bigru:
                        t_temp_start = time.perf_counter()
                        window = np.stack(feature_buffer, axis=0)
                        x = torch.from_numpy(window).to(torch_device)
                        x = (x - norm_mean) / norm_std
                        x = x.unsqueeze(0)
                        with torch.no_grad():
                            logit = temporal_model.forward_logits(x).squeeze().item()
                        score_raw = float(1.0 / (1.0 + np.exp(-logit)))
                        score_ema = (
                            score_raw
                            if score_ema is None
                            else EMA_ALPHA * score_raw + (1.0 - EMA_ALPHA) * score_ema
                        )
                        temporal_dt = time.perf_counter() - t_temp_start
                        temporal_dt_avg = (
                            temporal_dt
                            if temporal_dt_avg == 0.0
                            else 0.9 * temporal_dt_avg + 0.1 * temporal_dt
                        )
                        temporal_fps = (
                            1.0 / temporal_dt_avg if temporal_dt_avg > 0 else 0.0
                        )

                print(
                    f"[step {step.step_idx:5d} t={step.timestamp_s:6.2f}s "
                    f"{'OK ' if f.valid else 'BAD'}] "
                    f"PERCLOS={f.perclos:.3f} "
                    f"blink_bpm={f.blink_rate_bpm:5.2f} "
                    f"yawn_pm={f.yawn_rate_per_min:5.2f} "
                    f"pitch={f.head_pitch:+.2f} "
                    f"roll={f.head_roll:+.2f} "
                    f"vis={f.kpt_visibility:.2f} "
                    f"blinks={step.blink_count_total} "
                    f"yawns={step.yawn_count_total}"
                )
                if score_ema is None:
                    print(
                        f"[focus step {step.step_idx:5d} "
                        f"t={step.timestamp_s:6.2f}s] "
                        f"warmup {len(feature_buffer)}/{WINDOW_STEPS}"
                    )
                else:
                    label, _ = _status_label(score_ema)
                    print(
                        f"[focus step {step.step_idx:5d} "
                        f"t={step.timestamp_s:6.2f}s] "
                        f"raw={score_raw:.3f}  ema={score_ema:.3f}  ({label})"
                    )

                step_log_writer.writerow(
                    [
                        f"{step.timestamp_s:.3f}",
                        step.step_idx,
                        f.valid,
                        *(f"{getattr(f, n):.6f}" for n in FEATURE_NAMES),
                        step.blink_count_total,
                        step.yawn_count_total,
                        "" if score_raw is None else f"{score_raw:.6f}",
                        "" if score_ema is None else f"{score_ema:.6f}",
                    ]
                )
                step_log_file.flush()

            # --- Build display (flip + draw overlay) --------------------
            display = cv2.flip(frame, 1)
            disp_w = display.shape[1]
            _draw_geometry(
                display,
                _mirror_box(face_box, disp_w),
                _mirror_box(le_box, disp_w),
                _mirror_box(re_box, disp_w),
                _mirror_box(mo_box, disp_w),
                p_left,
                p_right,
                p_mouth,
                _mirror_kpts(kpts, disp_w),
                is_yawning=is_yawning,
                is_blinking=is_blinking,
            )
            preview = cv2.resize(display, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

            warmup_progress = min(frame_idx, aggregator.sub_window_frames)
            _draw_left_hud(
                preview,
                p_left,
                p_right,
                p_mouth,
                kpts,
                inference_fps,
                temporal_fps,
                overall_fps,
                latency_mean,
                last_step,
                warmup_progress,
                aggregator.sub_window_frames,
            )
            _draw_right_hud(
                preview,
                score_raw,
                score_ema,
                blink_total,
                yawn_total,
                buffer_fill=len(feature_buffer),
                buffer_target=WINDOW_STEPS,
            )

            cv2.imshow("live_inference_local (q to quit)", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            overall_dt = time.perf_counter() - t_loop
            overall_dt_avg = (
                overall_dt
                if overall_dt_avg == 0.0
                else 0.9 * overall_dt_avg + 0.1 * overall_dt
            )
            overall_fps = 1.0 / overall_dt_avg if overall_dt_avg > 0 else 0.0

            sleep_for = TARGET_DT - overall_dt
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        pose_estimator.close()
        step_log_file.close()


if __name__ == "__main__":
    main()
