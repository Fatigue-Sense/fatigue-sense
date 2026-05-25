"""FatigueSense production inference pipeline (Phase A + Phase B).

Public surface:

    FatiguePipeline   - end-to-end orchestrator (face detect -> ROI crop -> CNN)
    FrameProbs        - per-frame probability record returned by FatiguePipeline
    EyeStateClassifier, MouthStateClassifier - direct CNN inference wrappers
"""

from fatigue_pipeline.cnn_predictors import (
    EyeStateClassifier,
    MouthStateClassifier,
)
from fatigue_pipeline.feature_aggregator import (
    FeatureAggregator,
    FeatureStep,
    StepFeatures,
    compute_step_features,
)
from fatigue_pipeline.inference_pipeline import FatiguePipeline, FrameProbs

__all__ = [
    "FatiguePipeline",
    "FrameProbs",
    "EyeStateClassifier",
    "MouthStateClassifier",
    "FeatureAggregator",
    "FeatureStep",
    "StepFeatures",
    "compute_step_features",
]
