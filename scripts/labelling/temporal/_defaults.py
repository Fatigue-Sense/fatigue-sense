"""Shared defaults for temporal dataset preparation (run from repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.weights_path import (  # noqa: E402
    EYE_CKPT,
    LANDMARKER_PATH,
    MOUTH_CKPT,
    POSE_MODEL_PATH,
    resolve_asset,
)

DEFAULT_VIDEOS_DIR = _REPO_ROOT / "videos"
DEFAULT_RAW_PROBS_DIR = _REPO_ROOT / "data" / "temporal" / "raw_probs"
DEFAULT_FEATURES_DIR = _REPO_ROOT / "data" / "temporal" / "features"

FACE_MODEL_PATH = LANDMARKER_PATH
EYE_MODEL_PATH = EYE_CKPT
MOUTH_MODEL_PATH = MOUTH_CKPT
POSE_MODEL_PATH = POSE_MODEL_PATH


def resolve_weights() -> tuple[Path, Path, Path, Path]:
    """Resolve HF or local checkpoint paths before inference."""
    return (
        Path(resolve_asset(FACE_MODEL_PATH)),
        Path(resolve_asset(EYE_MODEL_PATH)),
        Path(resolve_asset(MOUTH_MODEL_PATH)),
        Path(resolve_asset(POSE_MODEL_PATH)),
    )
