"""Constants for the production inference pipeline.

Self-contained - the production pipeline does NOT import from vision_pipeline.
"""

from __future__ import annotations

# ---- MediaPipe FaceLandmarker indices (478-point mesh) ----
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]
MOUTH_INDICES = [61, 291, 81, 178, 13, 14, 402, 311, 308]

LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_OUTER_CORNER = 263


# ---- Crop output sizes (W, H) ----
EYE_CROP_SIZE = (64, 64)
MOUTH_CROP_SIZE = (64, 64)


# ---- Geometric box scaling (multiplied by inter-eye distance) ----
EYE_BOX_W_RATIO = 0.50
EYE_BOX_H_RATIO = 0.35

MOUTH_LANDMARK_W_SCALE = 1.6
MOUTH_LANDMARK_H_SCALE = 2.5
MOUTH_BOX_H_RATIO_HYBRID = 0.70
MOUTH_VERTICAL_OFFSET_RATIO_HYBRID = 0.20


# ---- Default video timing fallback ----
DEFAULT_FPS = 30.0
MS_PER_SECOND = 1000


# ---- Feature aggregation (Phase B -> temporal model input) ----
DEFAULT_SUB_WINDOW_S = 60.0
DEFAULT_STEP_STRIDE_S = 1.0

# BiGRU inference stride: how many feature steps between BiGRU forward passes
# at runtime. 1 = run BiGRU on every emitted step (densest, costliest);
# 2 = every other step, etc. Independent of the dataset-time
# DEFAULT_WINDOW_STRIDE used in training - this controls live inference
# only. Increase if BiGRU forward is the runtime bottleneck or if the
# focus score updates too jitterily for the UI.
DEFAULT_BIGRU_INFERENCE_STRIDE = 1

EYE_CLOSED_THRESH = 0.80
MOUTH_OPEN_THRESH = 0.7
MIN_VALID_FRACTION = 0.5

# Hysteresis bands for edge-triggered event counting.
# Event starts when prob crosses _RISING upward, ends when it falls below
# _FALLING. The dead-band kills near-threshold flicker that would otherwise
# inflate blink/yawn counts.
EYE_RISING_THRESH = 0.85
EYE_FALLING_THRESH = 0.7
MOUTH_RISING_THRESH = 0.8
# Falling threshold deliberately low (wide hysteresis gap). Mouth classifier
# output dips frame-to-frame mid-yawn even when mouth is clearly open; a
# tight 0.5 gap fragments one real yawn into multiple sub-events that fail
# the min-duration gate. 0.3 keeps Schmitt active across natural noise.
MOUTH_FALLING_THRESH = 0.5

# Reject ultra-short spikes that aren't real events.
MIN_BLINK_DURATION_S = 0.03
MIN_YAWN_DURATION_S = 1.50

SECONDS_PER_MINUTE = 60.0


# ---- Temporal model input layout ----
FEATURE_DIM = 17
FEATURE_NAMES = (
    # eye/mouth aggregates (Phase B)
    "perclos",
    "blink_rate_bpm",
    "mean_blink_duration",
    "eye_closure_variance",
    "yawn_rate_per_min",
    "mean_yawn_duration",
    "mouth_open_ratio",
    "mean_p_eye",
    "mean_p_mouth",
    # pose-derived (Phase A2 - yolo11n-pose upper body)
    "head_pitch",
    "head_roll",
    "shoulder_tilt",
    "head_size_ratio",
    "head_motion_energy",
    "head_drift_y",
    "posture_drift",
    "kpt_visibility",
)


# ---- Pose estimation (yolo11n-pose upper body subset) ----
# COCO-17 indices kept for seated/driver POV. Lower body is off-frame.
# Eyes (idx 1, 2) are dropped - no current pose feature uses them, and the
# Phase B eye classifier already covers eye-state behavior.
UPPER_BODY_KPT_INDICES = (0, 3, 4, 5, 6)
UPPER_BODY_KPT_NAMES = (
    "nose",
    "ear_left",
    "ear_right",
    "shoulder_left",
    "shoulder_right",
)
POSE_KPT_DIM = 3  # x, y, conf
POSE_NUM_KPTS = len(UPPER_BODY_KPT_INDICES)

POSE_MIN_KPT_CONF = 0.3  # below this a kpt is treated as missing (NaN)
POSE_MIN_VISIBILITY = 0.4  # window mean conf below this -> pose features = 0
POSE_PERSON_CONF = 0.5  # person detection threshold


# ---- CNN classifier I/O ----
EYE_MODEL_INPUT_HW = (64, 64)
MOUTH_MODEL_INPUT_HW = (64, 64)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---- Logit class indices (must match training label maps) ----
EYE_CLASS_CLOSED = 0
EYE_CLASS_OPEN = 1
MOUTH_CLASS_CLOSED = 0
MOUTH_CLASS_OPEN = 1
