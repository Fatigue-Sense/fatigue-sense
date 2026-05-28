from __future__ import annotations
import numpy as np

def bilateral_mean(left: float, right: float) -> float:
    """
    Combine the left and right values into a single average.

    If one side is missing, use the other side instead. If both sides are
    missing, return NaN. This is used to turn the left/right eye-closed
    probabilities into one combined eye signal.
    """
    if np.isnan(left) and np.isnan(right):
        return float("nan")
    if np.isnan(left):
        return right
    if np.isnan(right):
        return left
    return 0.5 * (left + right)
