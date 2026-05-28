"""
Composable building blocks for the per-step feature aggregator.

- ``event_detector`` : Schmitt-trigger edge detection for one signal
- ``event_tracker`` : windowed event counts + durations
- ``signal_buffer`` : fixed-length ring buffer with NaN policy
- ``fusion`` : NaN-tolerant bilateral fuser (left + right eye)
- ``pose_signal`` : upper-body kpt buffer + pose feature aggregation
- ``stride_gate`` : per-frame emit gate
- ``step_features`` : StepFeatures + FeatureStep data classes
- ``step_builder`` : pure compute_step_features() function
"""

from fatigue_pipeline.aggregation.event_detector import (CompletedEvent, SchmittDetector)
from fatigue_pipeline.aggregation.event_tracker import WindowEventTracker
from fatigue_pipeline.aggregation.fusion import bilateral_mean
from fatigue_pipeline.aggregation.pose_signal import (PoseFeatures, PoseSignalBuffer, compute_pose_features,)
from fatigue_pipeline.aggregation.signal_buffer import SignalBuffer
from fatigue_pipeline.aggregation.step_builder import compute_step_features
from fatigue_pipeline.aggregation.step_features import FeatureStep, StepFeatures
from fatigue_pipeline.aggregation.stride_gate import StrideGate

__all__ = [
    "CompletedEvent",
    "FeatureStep",
    "PoseFeatures",
    "PoseSignalBuffer",
    "SchmittDetector",
    "SignalBuffer",
    "StepFeatures",
    "StrideGate",
    "WindowEventTracker",
    "bilateral_mean",
    "compute_pose_features",
    "compute_step_features",
]
