"""NaN-tolerant signal fusion."""

from __future__ import annotations

import numpy as np


def bilateral_mean(left: float, right: float) -> float:
    """Mean of left + right, tolerating NaN on either side.

    Both NaN -> NaN. One NaN -> the other value. Used to fuse left/right
    eye-closed probabilities into one bilateral eye signal.
    """
    if np.isnan(left) and np.isnan(right):
        return float("nan")
    if np.isnan(left):
        return right
    if np.isnan(right):
        return left
    return 0.5 * (left + right)
