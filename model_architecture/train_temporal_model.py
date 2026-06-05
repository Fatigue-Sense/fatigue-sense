"""
Train the BiGRU temporal model on aggregated feature windows.

Usage:
    python -m model_architecture.train_temporal_model
"""

from __future__ import annotations

import json
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

EPOCHS = 500
EARLY_STOP_PATIENCE = 50
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 5e-4
LR_MIN = 1e-6
GRAD_CLIP_NORM = 1.0

WINDOW_STEPS = DEFAULT_WINDOW_STEPS
TRAIN_WINDOW_STRIDE = 5  # fewer overlapping train windows
VAL_WINDOW_STRIDE = DEFAULT_WINDOW_STRIDE
VAL_FRACTION = DEFAULT_VAL_FRACTION
SEED = DEFAULT_SEED

# Input noise on normalized features (train only)
FEATURE_NOISE_STD = 0.05

# Model regularization
HIDDEN_DIM = 64
NUM_LAYERS = 2
GRU_DROPOUT = 0.25
HEAD_DIM = 32
HEAD_DROPOUT = 0.35
SEQUENCE_DROPOUT = 0.25
POOL = "mean"

NUM_WORKERS = 0

# =============================================================================
# Train / eval loop
# =============================================================================
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    feature_noise_std: float = 0.0,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_train and feature_noise_std > 0.0:
                x = x + torch.randn_like(x) * feature_noise_std

            score = model(x)
            loss = loss_fn(score, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += float(loss.item()) * x.size(0)
            total_items += x.size(0)

    return total_loss / max(1, total_items)

def _model_config() -> dict:
    return {
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "gru_dropout": GRU_DROPOUT,
        "head_dim": HEAD_DIM,
        "head_dropout": HEAD_DROPOUT,
        "sequence_dropout": SEQUENCE_DROPOUT,
        "pool": POOL,
    }

def _plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
    *,
    best_epoch: int,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", palette="Blues_r")

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses, label="Val loss")
    if 1 <= best_epoch <= len(val_losses):
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Temporal model loss")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved to {save_path}")

def _save_history(history: list[dict], path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(FEATURES_DIR.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No feature Parquet files in {FEATURES_DIR}. "
            "Run scripts.download_hf_datasets --temporal or build features with "
            "scripts.labelling.temporal (see docs/training.md)."
        )

    train_paths, val_paths = split_videos(
        paths, val_fraction=VAL_FRACTION, seed=SEED
    )
    print(f"Videos: train={len(train_paths)} val={len(val_paths)}")

    # ---- datasets ----

    train_ds = TemporalWindowDataset(
        paths=train_paths,
        window_steps=WINDOW_STEPS,
        stride=TRAIN_WINDOW_STRIDE,
    )
    norm = fit_normalization(train_ds)
    train_ds.normalization = norm

    val_ds = TemporalWindowDataset(
        paths=val_paths,
        window_steps=WINDOW_STEPS,
        stride=VAL_WINDOW_STRIDE,
        normalization=norm,
    )
    print(
        f"Windows: train={len(train_ds)} (stride={TRAIN_WINDOW_STRIDE}) "
        f"val={len(val_ds)} (stride={VAL_WINDOW_STRIDE})"
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Empty split. Check that feature Parquets contain enough valid steps."
        )

    save_normalization(norm, OUTPUT_DIR / "normalization.json")
    model_config_path = OUTPUT_DIR / "model_config.json"
    model_config_path.write_text(
        json.dumps(_model_config(), indent=2) + "\n", encoding="utf-8"
    )

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
    model = build_temporal_model(**_model_config()).to(device)
    print(f"Model params: {model.count_parameters():,}  (device: {device})")
    print(f"Regularization: {_model_config()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN
    )
    loss_fn = nn.MSELoss()

    # ---- loop ----

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_path = OUTPUT_DIR / "best.pt"
    history_path = OUTPUT_DIR / "training_history.csv"
    curves_path = OUTPUT_DIR / "training_curves.png"

    for epoch in tqdm(range(1, EPOCHS + 1), desc="epochs", unit="ep"):
        train_loss = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            feature_noise_std=FEATURE_NOISE_STD,
        )
        val_loss = run_epoch(
            model, val_loader, loss_fn, device, optimizer=None, feature_noise_std=0.0
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_so_far": min(best_val, val_loss),
            }
        )
        _save_history(history, history_path)
        tqdm.write(
            f"epoch {epoch:03d}  train {train_loss:.4f}  val {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                tqdm.write(
                    f"Early stop at epoch {epoch} "
                    f"(best epoch {best_epoch}, val MSE {best_val:.4f})"
                )
                break

    _plot_loss_curves(
        train_losses, val_losses, curves_path, best_epoch=best_epoch
    )
    print(f"Best val MSE: {best_val:.4f} at epoch {best_epoch}  -> {best_path}")
    print(f"Training history: {history_path}")
    print(f"Model config (match at inference): {model_config_path}")


if __name__ == "__main__":
    main()
