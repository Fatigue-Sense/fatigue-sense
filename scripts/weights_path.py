"""Checkpoint and asset paths for FatigueSense inference scripts.

All model weights default to Hugging Face ``hf://FatigueSense/<repo>/<file>`` specs
(resolved at runtime via ``resolve_asset``). Eye/mouth classifiers use the same
helper inside ``fatigue_pipeline.cnn_predictors``.

MediaPipe Face Landmarker is not on HF — download the ``.task`` bundle locally:

  Guide: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
  Direct: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Place the file next to your working directory as ``face_landmarker.task``, or set
``LANDMARKER_PATH`` to an absolute path.
"""

from __future__ import annotations

from pathlib import Path

from fatigue_pipeline.cnn_predictors import _resolve_checkpoint as resolve_asset

# ---- CNN (Phase B) ----
EYE_CKPT = "hf://FatigueSense/eye_classifier/best_eye_classifier.pt"
MOUTH_CKPT = "hf://FatigueSense/mouth_classifier/best_mouth_classifier.pt"

# ---- MediaPipe Face Landmarker (Phase A) — local .task file ----
# See module docstring for download links (478-point mesh bundle).
LANDMARKER_PATH: str | Path = Path("face_landmarker.task")

# ---- YOLO pose (Phase A2) ----
POSE_MODEL_PATH = "hf://FatigueSense/pose_model/best_pose.pt"

# ---- BiGRU temporal (Phase C) ----
TEMPORAL_MODEL_PATH = "hf://FatigueSense/temporal_model/best_temporal.pt"
NORMALIZATION_PATH = "hf://FatigueSense/temporal_model/normalization.json"
