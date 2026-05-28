"""
Detects when a probability signal crosses a threshold and counts each event once.

This uses a Schmitt trigger, meaning the signal has to rise above one threshold
to start an event and fall below another threshold to end it. That small gap helps
avoid rapid on/off switching when the value is hovering near the boundary.

Very short events are ignored so brief spikes do not get counted.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CompletedEvent:
    end_frame: int
    duration_frames: int


"""Detects threshold-crossing events frame by frame."""
class SchmittDetector:
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

    # Add one new sample and return an event if one just ended
    def update(self, value: float, frame_idx: int) -> CompletedEvent | None:
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
        return self._active
