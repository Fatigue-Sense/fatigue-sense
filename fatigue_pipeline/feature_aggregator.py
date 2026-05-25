"""Stateful rolling-window feature aggregator (orchestrator).

Composes the building blocks in ``fatigue_pipeline.aggregation``:

    SignalBuffer        - ring buffer per per-frame channel
    SchmittDetector     - hysteresis edge detection per fused signal
    WindowEventTracker  - windowed event counts + durations
    StrideGate          - per-frame emit gating
    compute_step_features() - pure run-length aggregation (fallback)
    bilateral_mean()    - NaN-tolerant L/R eye fusion

Data types (``StepFeatures``, ``FeatureStep``) and the pure
``compute_step_features()`` live in the aggregation package and are
re-exported here for backward compatibility - existing imports of
``from fatigue_pipeline.feature_aggregator import ...`` keep working.
"""

from __future__ import annotations

from fatigue_pipeline.aggregation import (
    FeatureStep,
    PoseSignalBuffer,
    SchmittDetector,
    SignalBuffer,
    StepFeatures,
    StrideGate,
    WindowEventTracker,
    bilateral_mean,
    compute_pose_features,
    compute_step_features,
)
from fatigue_pipeline.constants import (
    DEFAULT_FPS,
    DEFAULT_STEP_STRIDE_S,
    DEFAULT_SUB_WINDOW_S,
    EYE_FALLING_THRESH,
    EYE_RISING_THRESH,
    FEATURE_DIM,
    MIN_BLINK_DURATION_S,
    MIN_YAWN_DURATION_S,
    MOUTH_FALLING_THRESH,
    MOUTH_RISING_THRESH,
)
from fatigue_pipeline.inference_pipeline import FrameProbs


__all__ = [
    "FeatureAggregator",
    "FeatureStep",
    "StepFeatures",
    "compute_step_features",
]


def _frames_for(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


class FeatureAggregator:
    """Live rolling-window feature builder.

    Usage::

        aggregator = FeatureAggregator(fps=video_fps)
        for frame in video:
            probs = pipeline.process_frame(frame)
            step = aggregator.add(probs)
            if step is not None and step.features.valid:
                window.append(step.features.to_array())
    """

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        sub_window_s: float = DEFAULT_SUB_WINDOW_S,
        step_stride_s: float = DEFAULT_STEP_STRIDE_S,
    ) -> None:
        self.sub_window_s = sub_window_s
        self.step_stride_s = step_stride_s
        self.fps = fps if fps and fps > 0 else DEFAULT_FPS

        sub_window_frames = _frames_for(self.sub_window_s, self.fps)
        stride_frames = _frames_for(self.step_stride_s, self.fps)

        self._eye_l = SignalBuffer(sub_window_frames)
        self._eye_r = SignalBuffer(sub_window_frames)
        self._mouth = SignalBuffer(sub_window_frames)
        self._pose = PoseSignalBuffer(sub_window_frames)
        self._timestamp: float = 0.0

        self._blink_detector = SchmittDetector(
            rising=EYE_RISING_THRESH,
            falling=EYE_FALLING_THRESH,
            min_duration_s=MIN_BLINK_DURATION_S,
            fps=self.fps,
        )
        self._yawn_detector = SchmittDetector(
            rising=MOUTH_RISING_THRESH,
            falling=MOUTH_FALLING_THRESH,
            min_duration_s=MIN_YAWN_DURATION_S,
            fps=self.fps,
        )
        self._blink_tracker = WindowEventTracker(
            detector=self._blink_detector,
            sub_window_frames=sub_window_frames,
            sub_window_s=self.sub_window_s,
        )
        self._yawn_tracker = WindowEventTracker(
            detector=self._yawn_detector,
            sub_window_frames=sub_window_frames,
            sub_window_s=self.sub_window_s,
        )
        self._gate = StrideGate(sub_window_frames, stride_frames)

        self._step_idx = 0

    # ---- lifecycle ----

    def reset(self, fps: float | None = None) -> None:
        """Clear buffers. Pass ``fps`` to also resize for a new clip."""
        if fps is not None and fps > 0:
            self.fps = fps
            self._reconfigure_for_fps()
        else:
            self._eye_l.clear()
            self._eye_r.clear()
            self._mouth.clear()
            self._pose.clear()
            self._blink_detector.reset()
            self._yawn_detector.reset()
            self._blink_tracker.reset()
            self._yawn_tracker.reset()
            self._gate.reset()

        self._step_idx = 0
        self._timestamp = 0.0

    # ---- per-frame ingest ----

    def add(self, probs: FrameProbs) -> FeatureStep | None:
        """Push one per-frame record. Returns a step on stride boundaries."""
        eye_l = self._eye_l.push(probs.p_eye_left_closed)
        eye_r = self._eye_r.push(probs.p_eye_right_closed)
        mouth = self._mouth.push(probs.p_mouth_open)
        self._pose.push(probs.keypoints)
        self._timestamp = float(probs.timestamp_s)

        frame_idx = self._gate.frames_seen + 1  # 1-indexed for event timing

        self._blink_tracker.update(bilateral_mean(eye_l, eye_r), frame_idx)
        self._yawn_tracker.update(mouth, frame_idx)

        if not self._gate.tick():
            return None

        features = compute_step_features(
            self._eye_l.snapshot(),
            self._eye_r.snapshot(),
            self._mouth.snapshot(),
            fps=self.fps,
        )

        # Override rate + mean-duration with edge-triggered numbers.
        # The pure compute_step_features() uses run-length counting which
        # double-counts events across overlapping sub-windows.
        features.blink_rate_bpm = self._blink_tracker.rate_per_min(frame_idx)
        features.yawn_rate_per_min = self._yawn_tracker.rate_per_min(frame_idx)
        features.mean_blink_duration = self._blink_tracker.mean_duration_s(self.fps)
        features.mean_yawn_duration = self._yawn_tracker.mean_duration_s(self.fps)

        pose = compute_pose_features(self._pose.snapshot())
        features.head_pitch = pose.head_pitch
        features.head_roll = pose.head_roll
        features.shoulder_tilt = pose.shoulder_tilt
        features.head_size_ratio = pose.head_size_ratio
        features.head_motion_energy = pose.head_motion_energy
        features.head_drift_y = pose.head_drift_y
        features.posture_drift = pose.posture_drift
        features.kpt_visibility = pose.kpt_visibility

        step = FeatureStep(
            step_idx=self._step_idx,
            timestamp_s=self._timestamp,
            features=features,
            blink_count_total=self._blink_tracker.total_count,
            yawn_count_total=self._yawn_tracker.total_count,
        )
        self._step_idx += 1
        return step

    # ---- properties ----

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def sub_window_frames(self) -> int:
        return self._gate.sub_window_frames

    @property
    def stride_frames(self) -> int:
        return self._gate.stride_frames

    # ---- live event state (uses the same edge-triggered detectors as the
    # per-second feature emission, so any consumer sees identical
    # behavior to what ``yawn_count_total`` / ``blink_count_total`` reflect).

    @property
    def is_yawning(self) -> bool:
        return self._yawn_detector.active

    @property
    def is_blinking(self) -> bool:
        return self._blink_detector.active

    @property
    def yawn_total(self) -> int:
        return self._yawn_tracker.total_count

    @property
    def blink_total(self) -> int:
        return self._blink_tracker.total_count

    # ---- private ----

    def _reconfigure_for_fps(self) -> None:
        sub_window_frames = _frames_for(self.sub_window_s, self.fps)
        stride_frames = _frames_for(self.step_stride_s, self.fps)

        self._eye_l.reconfigure(sub_window_frames)
        self._eye_r.reconfigure(sub_window_frames)
        self._mouth.reconfigure(sub_window_frames)
        self._pose.reconfigure(sub_window_frames)

        self._blink_detector.reset(fps=self.fps)
        self._yawn_detector.reset(fps=self.fps)

        self._blink_tracker.reconfigure(sub_window_frames, self.sub_window_s)
        self._yawn_tracker.reconfigure(sub_window_frames, self.sub_window_s)

        self._gate.reconfigure(sub_window_frames, stride_frames)
