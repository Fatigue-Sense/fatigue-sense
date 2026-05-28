"""
Checkpoint and asset paths used by the inference scripts.

Paths can point to either local files or hf:// assets that are downloaded
when first used.
"""

from __future__ import annotations

from pathlib import Path

from fatigue_pipeline.cnn_predictors import _resolve_checkpoint as resolve_asset

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- MediaPipe Face Landmarker (Phase A) ----
LANDMARKER_PATH: str | Path = _REPO_ROOT / "face_landmarker.task"

# ---- YOLO pose (Phase A2) ----
POSE_MODEL_PATH = "hf://FatigueSense/pose_model/best_pose.pt"

# ---- CNN (Phase B) ----
EYE_CKPT = "hf://FatigueSense/eye_classifier/best_eye_classifier.pt"
MOUTH_CKPT = "hf://FatigueSense/mouth_classifier/best_mouth_classifier.pt"

# ---- BiGRU temporal (Phase C) ----
TEMPORAL_MODEL_PATH = "hf://FatigueSense/temporal_model/best_temporal.pt"
NORMALIZATION_PATH = "hf://FatigueSense/temporal_model/normalization.json"
