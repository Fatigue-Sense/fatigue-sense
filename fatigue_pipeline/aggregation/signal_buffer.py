"""Fixed-length ring buffer for one per-frame signal.

Stage-1 NaN policy: missing detection becomes NaN, never zero.
Snapshot returns a numpy view sized to the sub-window.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SignalBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque[float] = deque(maxlen=capacity)
        self._capacity = capacity

    def reconfigure(self, capacity: int) -> None:
        self._buf = deque(maxlen=capacity)
        self._capacity = capacity

    def clear(self) -> None:
        self._buf.clear()

    def push(self, value: float | None) -> float:
        v = float("nan") if value is None else float(value)
        self._buf.append(v)
        return v

    def snapshot(self) -> np.ndarray:
        return np.asarray(self._buf, dtype=np.float32)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buf)
