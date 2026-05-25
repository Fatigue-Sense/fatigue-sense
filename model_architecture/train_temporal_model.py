"""Train the BiGRU temporal model on aggregated feature windows.

Pipeline:

    Stage 2 feature Parquets (per-second steps, one per video)
        ↓ TemporalWindowDataset (slides 60-step windows, drops invalid steps)
        ↓ split by VIDEO (not by window) into train / val
        ↓ per-feature normalization fit on train, applied to both
        ↓ BiGRUTemporalModel + AdamW + Cosine LR + MSE on focus score
        ↓ best checkpoint by val MSE -> runs/temporal/best.pt

Labels are bootstrapped from the features themselves so v1 needs no human
annotation. Replace the dataset's `label_fn` once real labels exist.

Run:
    python -m model_architecture.train_temporal_model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_architecture.dataset import (
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WINDOW_STEPS,
    DEFAULT_WINDOW_STRIDE,
    TemporalWindowDataset,
)
from model_architecture.models.bigru_temporal import build_temporal_model
from model_architecture.utils import (
    fit_normalization,
    save_normalization,
    split_videos,
)


# =============================================================================
# Constants
# =============================================================================

FEATURES_DIR = Path("data/temporal/features")
OUTPUT_DIR = Path("runs/temporal")

EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

WINDOW_STEPS = DEFAULT_WINDOW_STEPS
WINDOW_STRIDE = DEFAULT_WINDOW_STRIDE
VAL_FRACTION = DEFAULT_VAL_FRACTION
SEED = DEFAULT_SEED

NUM_WORKERS = 0  # 0 keeps everything in the main process; bump on Linux


# =============================================================================
# Train / eval loop
# =============================================================================


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """One pass over `loader`. Trains when `optimizer` is provided."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            score = model(x)
            loss = loss_fn(score, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * x.size(0)
            total_items += x.size(0)

    return total_loss / max(1, total_items)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(FEATURES_DIR.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No feature Parquet files in {FEATURES_DIR}")

    train_paths, val_paths = split_videos(
        paths, val_fraction=VAL_FRACTION, seed=SEED
    )
    print(f"Videos: train={len(train_paths)} val={len(val_paths)}")

    # ---- datasets ----

    train_ds = TemporalWindowDataset(
        paths=train_paths, window_steps=WINDOW_STEPS, stride=WINDOW_STRIDE
    )
    norm = fit_normalization(train_ds)
    train_ds.normalization = norm

    val_ds = TemporalWindowDataset(
        paths=val_paths,
        window_steps=WINDOW_STEPS,
        stride=WINDOW_STRIDE,
        normalization=norm,
    )
    print(f"Windows: train={len(train_ds)} val={len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Empty split. Check that feature Parquets contain enough valid steps."
        )

    save_normalization(norm, OUTPUT_DIR / "normalization.json")

    # ---- loaders ----

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    # ---- model ----

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_temporal_model().to(device)
    print(f"Model params: {model.count_parameters():,}  (device: {device})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    # ---- loop ----

    best_val = float("inf")
    history: list[dict] = []
    best_path = OUTPUT_DIR / "best.pt"

    for epoch in tqdm(range(1, EPOCHS + 1), desc="epochs", unit="ep"):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        tqdm.write(
            f"epoch {epoch:03d}  train {train_loss:.4f}  val {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"Best val MSE: {best_val:.4f}  -> {best_path}")


if __name__ == "__main__":
    main()
