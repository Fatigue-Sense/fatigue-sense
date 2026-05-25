"""
Model utility functions — build, save, load, and sanity-check MultiHeadCNN.
Checkpoints use safetensors format (no pickle, safe deserialization).
"""

from __future__ import annotations

import torch
from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save

from .multi_head_cnn import MultiHeadCNN, ROI_CONFIG


def build_model(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    head_dropout: float = 0.2,
) -> MultiHeadCNN:
    return MultiHeadCNN(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        head_dropout=head_dropout,
    )


def save_model(model: MultiHeadCNN, path: str) -> None:
    safetensors_save(model.state_dict(), path)


def load_model(checkpoint_path: str, device: str = "cpu") -> MultiHeadCNN:
    model = build_model(pretrained=False)
    state_dict = safetensors_load(checkpoint_path, device=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    import torchinfo

    model = build_model(pretrained=False)

    dummy = {
        "eyes":  torch.randn(2, 3, 64, 64),
        "mouth": torch.randn(2, 3, 64, 64),
        "head":  torch.randn(2, 3, 128, 128),
        "torso": torch.randn(2, 3, 128, 128),
    }

    out = model(dummy)
    for roi, prob in out.items():
        if prob is None:
            print(f"{roi:>6}: occluded (null vector)")
        else:
            print(f"{roi:>6}: shape={tuple(prob.shape)}  sum={prob.sum(-1).tolist()}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
