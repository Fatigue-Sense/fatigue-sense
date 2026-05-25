"""Windowed event bookkeeping for a single signal.

Sits on top of a ``SchmittDetector``. Owns:

- cumulative count snapshots aligned to the sub-window
- in-window event list (for mean duration)
- rate-per-minute computation
- pruning of events that fell out of the window

Why split from the detector: the detector is stateless w.r.t. windowing.
This class is where the "events ending in the last N frames" view lives.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from fatigue_pipeline.aggregation.event_detector import (
    CompletedEvent,
    SchmittDetector,
)
from fatigue_pipeline.constants import SECONDS_PER_MINUTE


class WindowEventTracker:
    def __init__(
        self,
        detector: SchmittDetector,
        sub_window_frames: int,
        sub_window_s: float,
    ) -> None:
        self.detector = detector
        self.sub_window_frames = sub_window_frames
        self.sub_window_s = sub_window_s

        self._count_history: deque[int] = deque(maxlen=sub_window_frames)
        self._events: deque[CompletedEvent] = deque()

    def reconfigure(self, sub_window_frames: int, sub_window_s: float) -> None:
        self.sub_window_frames = sub_window_frames
        self.sub_window_s = sub_window_s
        self._count_history = deque(maxlen=sub_window_frames)
        self._events = deque()

    def reset(self) -> None:
        self._count_history.clear()
        self._events.clear()

    def update(self, value: float, frame_idx: int) -> None:
        event = self.detector.update(value, frame_idx)
        if event is not None:
            self._events.append(event)
        self._count_history.append(self.detector.count)

    def _prune(self, frame_idx: int) -> None:
        window_start = frame_idx - self.sub_window_frames
        while self._events and self._events[0].end_frame < window_start:
            self._events.popleft()

    def rate_per_min(self, frame_idx: int) -> float:
        """Events that ended inside the current sub-window, scaled to bpm."""
        self._prune(frame_idx)
        if not self._count_history:
            return 0.0
        in_window = self.detector.count - self._count_history[0]
        return float(in_window * SECONDS_PER_MINUTE / self.sub_window_s)

    def mean_duration_s(self, fps: float) -> float:
        if not self._events:
            return 0.0
        durations = [e.duration_frames for e in self._events]
        return float(np.mean(durations) / fps)

    @property
    def total_count(self) -> int:
        return self.detector.count
