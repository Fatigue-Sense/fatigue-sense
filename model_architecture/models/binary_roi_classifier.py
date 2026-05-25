"""
Binary ROI Classifier — Model
Shared MobileNetV3-Small backbone for all four ROI binary classifiers
(Eyes, Mouth, Head, Torso). Returns raw logits (B, 2).

Use torch.softmax(logits, dim=-1) externally when needed (ThresholdPredictor does this).
Training uses CrossEntropyLoss directly on logits.

For threshold-based pseudo-labeling, see src/pipeline/threshold_predictor.py.
See docs/phase_b/binary_classifier_bootstrap.md for the full bootstrap pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class BinaryROIClassifier(nn.Module):
    BACKBONE_OUT_FEATURES = 576

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Apply weights if pretrained
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            self.freeze_backbone(toggle=True)

        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_OUT_FEATURES, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x)

    def freeze_backbone(self, toggle: bool, last_n_blocks: int = 2):
        if toggle:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            blocks = list(self.backbone.children())
            for block in blocks[-last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True


def build_binary_classifier(
    pretrained: bool = True, freeze_backbone: bool = True, dropout: float = 0.2
) -> BinaryROIClassifier:
    """Construct a fresh BinaryROIClassifier."""
    return BinaryROIClassifier(
        pretrained=pretrained, freeze_backbone=freeze_backbone, dropout=dropout
    )
