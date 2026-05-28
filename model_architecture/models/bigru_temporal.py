"""
BiGRU model used to score a window of fatigue features

The model takes a sequence of feature rows with shape (B, T, F) and returns
one focus score per window.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from fatigue_pipeline.constants import FEATURE_DIM

PoolMode = Literal["last", "mean", "hidden"]

class BiGRUTemporalModel(nn.Module):
    """BiGRU + small MLP head, single sigmoid output."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 1,
        gru_dropout: float = 0.0,
        head_dim: int = 32,
        head_dropout: float = 0.1,
        sequence_dropout: float = 0.0,
        pool: PoolMode = "last",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pool: PoolMode = pool

        # PyTorch only applies inter-layer dropout when num_layers > 1.
        effective_gru_dropout = gru_dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=effective_gru_dropout,
        )

        pooled_dim = 2 * hidden_dim  # forward + backward
        self.sequence_dropout = nn.Dropout(sequence_dropout)
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, head_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, 1),
        )

    def _pool(self, gru_out: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        """
        Collapse the sequence output into one vector per window
        """
        if self.pool == "last":
            return gru_out[:, -1, :]
        if self.pool == "mean":
            return gru_out.mean(dim=1)
        if self.pool == "hidden":
            # h_n is grouped by layer and direction. Take the final layer
            h = h_n.view(self.num_layers, 2, -1, self.hidden_dim)[-1]
            forward, backward = h[0], h[1]
            return torch.cat([forward, backward], dim=-1)
        raise ValueError(f"Unknown pool mode: {self.pool!r}")

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the raw pre-sigmoid logit, shape (B,)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, F), got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Feature dim mismatch: model expects {self.input_dim}, "
                f"got {x.size(-1)}"
            )

        gru_out, h_n = self.gru(x)  # (B, T, 2*hidden), (2*L, B, hidden)
        pooled = self.sequence_dropout(self._pool(gru_out, h_n))  # (B, 2*hidden)
        logit = self.head(pooled)  # (B, 1)
        return logit.squeeze(-1)  # (B,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the focus score in (0, 1), shape (B,)
        """
        return torch.sigmoid(self.forward_logits(x))

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_temporal_model(
    input_dim: int = FEATURE_DIM,
    hidden_dim: int = 64,
    num_layers: int = 1,
    gru_dropout: float = 0.0,
    head_dim: int = 32,
    head_dropout: float = 0.1,
    sequence_dropout: float = 0.0,
    pool: PoolMode = "last",
) -> BiGRUTemporalModel:
    """
    Create a BiGRU temporal model with the default settings
    """
    return BiGRUTemporalModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gru_dropout=gru_dropout,
        head_dim=head_dim,
        head_dropout=head_dropout,
        sequence_dropout=sequence_dropout,
        pool=pool,
    )


def infer_temporal_kwargs_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    Infer the model size from checkpoint weights
    """
    layer_idxs: list[int] = []
    for key in state:
        match = re.match(r"gru\.weight_ih_l(\d+)$", key)
        if match:
            layer_idxs.append(int(match.group(1)))

    # GRU weight names include the layer index
    w_ih = state["gru.weight_ih_l0"]
    hidden_dim = int(w_ih.shape[0]) // 3
    head_dim = int(state["head.0.weight"].shape[0])

    return {
        "hidden_dim": hidden_dim,
        "num_layers": max(layer_idxs) + 1 if layer_idxs else 1,
        "head_dim": head_dim,
    }


def load_temporal_checkpoint(
    weights_path: str | Path,
    device: torch.device | str,
    *,
    config_path: str | Path | None = None,
) -> BiGRUTemporalModel:
    """
    Load a saved temporal model checkpoint
    """
    weights_path = Path(weights_path)
    state = torch.load(weights_path, map_location=device, weights_only=True)

    if config_path is None:
        candidate = weights_path.parent / "model_config.json"
        config_path = candidate if candidate.is_file() else None

    kwargs: dict[str, Any] = {}
    if config_path is not None:
        kwargs = json.loads(Path(config_path).read_text(encoding="utf-8"))
    else:
        inferred = infer_temporal_kwargs_from_state(state)
        kwargs.update(inferred)
        logger.info(
            "No model_config.json beside %s; inferred architecture %s",
            weights_path,
            inferred,
        )

    model = build_temporal_model(**kwargs)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
