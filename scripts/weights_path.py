"""Checkpoint and asset paths for FatigueSense inference scripts.

Each constant may be either:

- **Hugging Face:** ``hf://<repo_id>/<filename>`` — downloaded on first use via
  ``resolve_asset`` / ``fatigue_pipeline.cnn_predictors._resolve_checkpoint``.
- **Local:** a ``str`` or ``Path`` to an existing file on disk (no Hub call).

To use weights you trained or copied locally, uncomment the ``_LOCAL_*`` lines
below and comment out the matching ``hf://`` lines for that asset.

MediaPipe Face Landmarker is not on HF. Download the ``.task`` bundle:

  Guide: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
  Direct: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Default: ``face_landmarker.task`` in the repo root (or set ``LANDMARKER_PATH``).
"""

from __future__ import annotations

from pathlib import Path

from fatigue_pipeline.cnn_predictors import _resolve_checkpoint as resolve_asset

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- CNN (Phase B) ----
# Default: Hugging Face. For local training outputs, swap as noted above.
EYE_CKPT = "hf://FatigueSense/eye_classifier/best_eye_classifier.pt"
MOUTH_CKPT = "hf://FatigueSense/mouth_classifier/best_mouth_classifier.pt"
# EYE_CKPT = _REPO_ROOT / "runs/binary/eyes/best.pt"
# MOUTH_CKPT = _REPO_ROOT / "runs/binary/mouth/best.pt"

# ---- MediaPipe Face Landmarker (Phase A) — local .task file ----
LANDMARKER_PATH: str | Path = _REPO_ROOT / "face_landmarker.task"
# LANDMARKER_PATH = Path("face_landmarker.task")  # cwd-relative alternative

# ---- YOLO pose (Phase A2) ----
POSE_MODEL_PATH = "hf://FatigueSense/pose_model/best_pose.pt"
# POSE_MODEL_PATH = _REPO_ROOT / "runs/pose/best.pt"

# ---- BiGRU temporal (Phase C) ----
TEMPORAL_MODEL_PATH = "hf://FatigueSense/temporal_model/best_temporal.pt"
NORMALIZATION_PATH = "hf://FatigueSense/temporal_model/normalization.json"
# TEMPORAL_MODEL_PATH = _REPO_ROOT / "runs/temporal/best.pt"
# NORMALIZATION_PATH = _REPO_ROOT / "runs/temporal/normalization.json"
