"""
Train / val split helpers.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

def split_videos(
    paths: list[Path],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """
    Split a list of file paths by VIDEO (never by window) to prevent leakage.

    Two windows from the same video share too much temporal state - splitting
    at window level would leak signal into the val set, split at the
    file-path level.
    """
    rng = np.random.default_rng(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    if not train:
        raise ValueError(f"Need at least 2 videos to split; got {len(paths)}.")
    return train, val
