"""
Phase B — Multi-Head CNN Classifier
Shared MobileNetV3-Small backbone with four independent classification heads,
one per ROI (Eyes, Mouth, Head, Torso). All four crops are batched through
the backbone in a single forward pass; each head then produces a softmax
probability vector consumed by a temporal model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from model_architecture.config.roi_config import ROI_CONFIG

# ---------------------------------------------------------------------------
# Classification head (shared structure, different output dims)
# ---------------------------------------------------------------------------


class _ClassHead(nn.Module):
    """Lightweight MLP head applied after global average pooling."""

    def __init__(
        self, in_features: int, num_classes: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class MultiHeadCNN(nn.Module):
    """
    Shared MobileNetV3-Small backbone with four ROI-specific classification heads.

    Forward contract
    ----------------
    Input : dict[str, Tensor]  —  {"eyes": (B,3,H,W), "mouth": ..., "head": ..., "torso": ...}
                                   Missing keys are accepted; their output will be None.
    Output: dict[str, Tensor]  —  {"eyes": (B,4), "mouth": (B,4), "head": (B,6), "torso": (B,6)}
                                   Softmax probabilities. None for absent ROIs (occlusion passthrough).
    """

    BACKBONE_OUT_FEATURES = 576  # MobileNetV3-Small final feature dim before classifier

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        # Strip the original classifier; keep features + adaptive pool
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.heads = nn.ModuleDict(
            {
                roi: _ClassHead(
                    self.BACKBONE_OUT_FEATURES, cfg["num_classes"], head_dropout
                )
                for roi, cfg in ROI_CONFIG.items()
            }
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return x.flatten(1)

    def forward(self, crops: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        outputs: dict[str, torch.Tensor | None] = {}

        for roi in ROI_CONFIG:
            if roi not in crops or crops[roi] is None:
                # Null vector — signals occlusion to ACAMF downstream
                outputs[roi] = None
                continue

            feats = self._extract_features(crops[roi])
            logits = self.heads[roi](feats)
            outputs[roi] = torch.softmax(logits, dim=-1)

        return outputs
