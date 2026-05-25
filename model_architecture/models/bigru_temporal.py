"""BiGRU temporal model for FatigueSense (Phase C).

Consumes a sliding window of per-second feature vectors produced by
`fatigue_pipeline.feature_aggregator` and emits a continuous focus score
in `[0, 1]` per window.

Input shape:  (B, T, F)
    B = batch
    T = sequence length in feature steps (default 60 = 60 s of context)
    F = feature dim (9 by default; matches `fatigue_pipeline.constants.FEATURE_DIM`)

Output:
    forward()        -> focus_score in (0, 1), shape (B,)
    forward_logits() -> raw logit, shape (B,)  (use with BCEWithLogitsLoss)

Architecture:

    Input (B, T, 9)
      ↓ BiGRU(hidden, num_layers, dropout, bidirectional=True)
    Sequence (B, T, 2*hidden)
      ↓ pool (last-step | mean | last-hidden)
    Pooled (B, 2*hidden)
      ↓ Linear(2*hidden -> head_dim) + GELU + Dropout
      ↓ Linear(head_dim -> 1)
    Logit (B, 1)
      ↓ squeeze + sigmoid (in `forward`)
    Focus score (B,)

Default hyperparameters target the v1 design in
`docs/architecture/pipeline_overview.md` and `docs/phase_c/temporal_models.md`.
"""

from __future__ import annotations

from typing import Literal

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
        pool: PoolMode = "last",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pool: PoolMode = pool

        # PyTorch only applies inter-layer dropout when num_layers > 1; passing
        # a non-zero dropout with one layer is a no-op + a warning, so guard it.
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
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, head_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, 1),
        )

    def _pool(self, gru_out: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        """Reduce the BiGRU output to a single (B, 2*hidden) vector.

        - "last":   last timestep of the sequence (default; cheap and effective
                    for fixed-length windows where the most recent state
                    carries the strongest fatigue signal).
        - "mean":   mean across timesteps, robust to noisy spikes near the end.
        - "hidden": concat forward+backward last-layer hidden states from h_n.
        """
        if self.pool == "last":
            return gru_out[:, -1, :]
        if self.pool == "mean":
            return gru_out.mean(dim=1)
        if self.pool == "hidden":
            # h_n shape: (num_layers * 2, B, hidden_dim). Reshape so the
            # bidirection axis is explicit, then take the final layer.
            h = h_n.view(self.num_layers, 2, -1, self.hidden_dim)[-1]
            forward, backward = h[0], h[1]
            return torch.cat([forward, backward], dim=-1)
        raise ValueError(f"Unknown pool mode: {self.pool!r}")

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw pre-sigmoid logit, shape (B,)."""
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, F), got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Feature dim mismatch: model expects {self.input_dim}, "
                f"got {x.size(-1)}"
            )

        gru_out, h_n = self.gru(x)  # (B, T, 2*hidden), (2*L, B, hidden)
        pooled = self._pool(gru_out, h_n)  # (B, 2*hidden)
        logit = self.head(pooled)  # (B, 1)
        return logit.squeeze(-1)  # (B,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the focus score in (0, 1), shape (B,)."""
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
    pool: PoolMode = "last",
) -> BiGRUTemporalModel:
    """Construct a fresh BiGRUTemporalModel with the v1 defaults."""
    return BiGRUTemporalModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gru_dropout=gru_dropout,
        head_dim=head_dim,
        head_dropout=head_dropout,
        pool=pool,
    )
