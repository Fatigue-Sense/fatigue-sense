"""Shared defaults for CNN ROI dataset preparation (run from repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.weights_path import LANDMARKER_PATH  # noqa: E402

DEFAULT_VIDEOS_DIR = _REPO_ROOT / "videos"
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "dataset"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Batch labelling (__main__ in label_dataset.py)
DEFAULT_SAMPLE_EVERY_N_FRAMES = 15
DEFAULT_MAX_FRAMES: int | None = None
DEFAULT_FLIP_HORIZONTAL = False

# MediaPipe Face Landmarker landmark index sets
LEFT_EYE = [33, 133, 160, 144, 158, 153, 159, 145]
RIGHT_EYE = [362, 263, 387, 373, 385, 380, 386, 374]
MOUTH = [61, 291, 81, 178, 13, 14, 402, 311]

# EAR / MAR thresholds (Soukupová & Čech, 2016 style weak labels)
EAR_OPEN_THRESH = 0.13
EAR_CLOSE_THRESH = 0.10
MAR_OPEN_THRESH = 0.43
MAR_CLOSE_THRESH = 0.40

CROP_WIDTH = 64
CROP_HEIGHT = 64


def resolve_landmarker_path() -> Path:
    """Return repo-root MediaPipe model path."""
    return Path(LANDMARKER_PATH)
