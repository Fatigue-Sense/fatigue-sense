"""CNN inference wrappers for Phase B classifiers.

Each predictor loads a trained `BinaryROIClassifier` checkpoint, applies the
training-time preprocessing, and returns calibrated probabilities for a
single ROI crop (numpy BGR image as produced by `RegionCropper`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torchvision import transforms

from fatigue_pipeline.constants import (
    EYE_CLASS_CLOSED,
    EYE_MODEL_INPUT_HW,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOUTH_CLASS_OPEN,
    MOUTH_MODEL_INPUT_HW,
)
from model_architecture.models.binary_roi_classifier import build_binary_classifier


def _resolve_checkpoint(path_or_spec: str | Path) -> Path:
    """Resolve local path or `hf://<repo_id>/<filename>` spec to a local Path.

    HF spec format: `hf://<user_or_org>/<repo>/<filename>`. The repo_id is
    everything before the last slash, the filename is the final segment.
    """
    if isinstance(path_or_spec, str) and path_or_spec.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        spec = path_or_spec[len("hf://") :]
        repo_id, filename = spec.rsplit("/", 1)
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))

    p = Path(path_or_spec)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint missing: {p}")
    return p


class _BinaryROIPredictor:
    """Shared loader / preprocessor for binary ROI classifiers."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        input_hw: tuple[int, int],
        target_class: int,
        device: str | torch.device | None = None,
        grayscale: bool = True,
    ) -> None:
        self.checkpoint_path = _resolve_checkpoint(checkpoint_path)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.target_class = target_class
        self.input_hw = input_hw

        self.model = build_binary_classifier(pretrained=False, freeze_backbone=False)
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(self.device)

        ops: list = []
        if grayscale:
            ops.append(transforms.Grayscale(num_output_channels=3))
        ops.extend(
            [
                transforms.Resize(input_hw),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.transform = transforms.Compose(ops)

    def _to_pil(self, crop_bgr: np.ndarray):
        from PIL import Image

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @torch.no_grad()
    def predict_prob(self, crop_bgr: np.ndarray | None) -> float | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        tensor = self.transform(self._to_pil(crop_bgr)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, self.target_class].item())

    @torch.no_grad()
    def predict_probs_batch(
        self, crops_bgr: Iterable[np.ndarray | None]
    ) -> list[float | None]:
        items = list(crops_bgr)
        valid_idx = [i for i, c in enumerate(items) if c is not None and c.size > 0]
        if not valid_idx:
            return [None] * len(items)

        batch = torch.stack(
            [self.transform(self._to_pil(items[i])) for i in valid_idx]
        ).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=-1)[:, self.target_class].cpu().numpy()

        out: list[float | None] = [None] * len(items)
        for slot, p in zip(valid_idx, probs):
            out[slot] = float(p)
        return out


class EyeStateClassifier(_BinaryROIPredictor):
    """Returns P(eye_closed)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            input_hw=EYE_MODEL_INPUT_HW,
            target_class=EYE_CLASS_CLOSED,
            device=device,
            grayscale=True,
        )


class MouthStateClassifier(_BinaryROIPredictor):
    """Returns P(mouth_open)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            input_hw=MOUTH_MODEL_INPUT_HW,
            target_class=MOUTH_CLASS_OPEN,
            device=device,
            grayscale=True,
        )
