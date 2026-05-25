"""Schmitt-trigger edge detection for one probability signal.

Counts each "above-threshold" run exactly once. Hysteresis (separate
rising/falling thresholds) avoids flutter near the decision boundary;
a minimum duration filter discards spikes that aren't real events.

Used to count blinks (eye-closed signal) and yawns (mouth-open signal).
A new signal (e.g. head-pose) plugs in by instantiating one more
``SchmittDetector`` with its own thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompletedEvent:
    """One edge-detected event that finished on the latest update."""

    end_frame: int
    duration_frames: int


class SchmittDetector:
    """Frame-level edge detector with hysteresis + min-duration gate."""

    def __init__(
        self,
        rising: float,
        falling: float,
        min_duration_s: float,
        fps: float,
    ) -> None:
        if rising < falling:
            raise ValueError(
                f"rising ({rising}) must be >= falling ({falling})"
            )
        self.rising = rising
        self.falling = falling
        self.min_duration_s = min_duration_s
        self.fps = fps

        self._active = False
        self._run_start_frame = 0
        self._count = 0

    def reset(self, fps: float | None = None) -> None:
        if fps is not None and fps > 0:
            self.fps = fps
        self._active = False
        self._run_start_frame = 0
        self._count = 0

    def update(self, value: float, frame_idx: int) -> CompletedEvent | None:
        """Push one sample. Returns a CompletedEvent on falling edge."""
        if np.isnan(value):
            return None

        if not self._active and value > self.rising:
            self._active = True
            self._run_start_frame = frame_idx
            return None

        if self._active and value < self.falling:
            duration_frames = frame_idx - self._run_start_frame
            duration_s = duration_frames / self.fps
            self._active = False
            if duration_s >= self.min_duration_s:
                self._count += 1
                return CompletedEvent(
                    end_frame=frame_idx,
                    duration_frames=duration_frames,
                )

        return None

    @property
    def count(self) -> int:
        return self._count

    @property
    def active(self) -> bool:
        """True while a candidate event is currently in progress.

        Flips True on the rising crossing, flips False on the falling
        crossing (regardless of whether ``min_duration_s`` was met -
        gating happens at the falling edge inside ``update()``).
        """
        return self._active
