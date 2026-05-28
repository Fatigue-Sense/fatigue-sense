"""
Turns one window of per-frame probabilities into a single StepFeatures row.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
from fatigue_pipeline.aggregation.step_features import StepFeatures
from fatigue_pipeline.constants import (
    EYE_CLOSED_THRESH,
    MIN_VALID_FRACTION,
    MOUTH_OPEN_THRESH,
    SECONDS_PER_MINUTE,
)

@dataclass
class _SignalStats:
    """Generic stats for a 1-D probability signal over a sub-window"""
    valid_frac: float
    coverage_ratio: float
    event_rate_per_min: float
    mean_event_duration_s: float
    variance: float
    mean_value: float

def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Return the lengths of all True runs in a 1-D boolean array"""
    if mask.size == 0:
        return np.array([], dtype=np.int32)
    # Add False at both ends so runs at the start or end of the window are still counted cleanly
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return ends - starts

def _aggregate_signal(
    signal: np.ndarray,
    fps: float,
    sub_window_s: float,
    threshold: float,
) -> _SignalStats:
    valid_mask = ~np.isnan(signal)
    valid_frac = float(valid_mask.mean()) if valid_mask.size else 0.0

    if valid_frac < MIN_VALID_FRACTION or not valid_mask.any():
        nan = float("nan")
        return _SignalStats(
            valid_frac=valid_frac,
            coverage_ratio=nan,
            event_rate_per_min=nan,
            mean_event_duration_s=nan,
            variance=nan,
            mean_value=nan,
        )

    clean = signal[valid_mask]
    over_threshold = clean > threshold
    event_runs = _run_lengths(over_threshold)

    coverage_ratio = float(over_threshold.mean())
    event_rate_per_min = float(event_runs.size) * (SECONDS_PER_MINUTE / sub_window_s)
    mean_event_duration_s = float(event_runs.mean() / fps) if event_runs.size else 0.0

    return _SignalStats(
        valid_frac=valid_frac,
        coverage_ratio=coverage_ratio,
        event_rate_per_min=event_rate_per_min,
        mean_event_duration_s=mean_event_duration_s,
        variance=float(np.var(clean)),
        mean_value=float(clean.mean()),
    )


def compute_step_features(
    p_eye_left: np.ndarray,
    p_eye_right: np.ndarray,
    p_mouth_open: np.ndarray,
    fps: float,
) -> StepFeatures:
    """Aggregate one rolling sub-window of per-frame probs into one step"""
    n_frames = len(p_eye_left)
    sub_window_s = n_frames / fps if fps > 0 else 1.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        eye_signal = np.nanmean(np.vstack([p_eye_left, p_eye_right]), axis=0)

    eye_stats = _aggregate_signal(
        eye_signal, fps=fps, sub_window_s=sub_window_s, threshold=EYE_CLOSED_THRESH
    )
    mouth_stats = _aggregate_signal(
        np.asarray(p_mouth_open, dtype=np.float32),
        fps=fps,
        sub_window_s=sub_window_s,
        threshold=MOUTH_OPEN_THRESH,
    )

    # A step is only valid if both eye and mouth signals have enough real values
    valid = (
        eye_stats.valid_frac >= MIN_VALID_FRACTION
        and mouth_stats.valid_frac >= MIN_VALID_FRACTION
    )

    return StepFeatures(
        perclos=eye_stats.coverage_ratio,
        blink_rate_bpm=eye_stats.event_rate_per_min,
        mean_blink_duration=eye_stats.mean_event_duration_s,
        eye_closure_variance=eye_stats.variance,
        mean_p_eye=eye_stats.mean_value,
        yawn_rate_per_min=mouth_stats.event_rate_per_min,
        mean_yawn_duration=mouth_stats.mean_event_duration_s,
        mouth_open_ratio=mouth_stats.coverage_ratio,
        mean_p_mouth=mouth_stats.mean_value,
        valid=valid,
    )
