"""Per-feature normalization fit / save / load.

The canonical feature-name order from `fatigue_pipeline.constants.FEATURE_NAMES`
is persisted alongside the stats so consumers can sanity-check that the
columns at inference time line up with the columns the model trained on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from fatigue_pipeline.constants import FEATURE_NAMES


class _StackableDataset(Protocol):
    """Anything that exposes the stacked raw feature rows it owns."""

    def stacked_features(self) -> np.ndarray:
        ...


def fit_normalization(dataset: _StackableDataset) -> dict[str, np.ndarray]:
    """Fit per-feature mean/std over a dataset's raw feature rows.

    Invalid steps (no face) carry NaN in some columns; nan-aware reductions
    skip them so the stats reflect only usable rows.
    """
    raw = dataset.stacked_features()
    mean = np.nanmean(raw, axis=0).astype(np.float32)
    std = np.nanstd(raw, axis=0).astype(np.float32)
    # Guard against zero-variance features (a flat column would divide by 0).
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def save_normalization(norm: dict[str, np.ndarray], path: Path) -> None:
    """Persist normalization stats with the canonical feature-name order."""
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "mean": norm["mean"].tolist(),
        "std": norm["std"].tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_normalization(path: Path) -> dict[str, np.ndarray]:
    """Reload stats produced by `save_normalization`."""
    payload = json.loads(path.read_text())
    return {
        "mean": np.asarray(payload["mean"], dtype=np.float32),
        "std": np.asarray(payload["std"], dtype=np.float32),
    }
