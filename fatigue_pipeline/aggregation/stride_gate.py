"""Per-frame emit gate.

State machine. Returns True from ``tick()`` only when:
    1. enough frames have been seen to fill the sub-window, AND
    2. the configured stride has elapsed since the last emit.
"""

from __future__ import annotations


class StrideGate:
    def __init__(self, sub_window_frames: int, stride_frames: int) -> None:
        self.sub_window_frames = sub_window_frames
        self.stride_frames = stride_frames
        self._frames_seen = 0
        self._frames_since_emit = 0

    def reconfigure(self, sub_window_frames: int, stride_frames: int) -> None:
        self.sub_window_frames = sub_window_frames
        self.stride_frames = stride_frames
        self.reset()

    def reset(self) -> None:
        self._frames_seen = 0
        self._frames_since_emit = 0

    def tick(self) -> bool:
        """Advance one frame. Returns True when an emit is due."""
        self._frames_seen += 1
        self._frames_since_emit += 1
        if (
            self._frames_seen < self.sub_window_frames
            or self._frames_since_emit < self.stride_frames
        ):
            return False
        self._frames_since_emit = 0
        return True

    @property
    def frames_seen(self) -> int:
        return self._frames_seen
