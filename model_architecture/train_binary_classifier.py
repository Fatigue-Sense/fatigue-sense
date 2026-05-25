"""
Binary ROI Classifier — Training Script

Stage 1 (epochs 1–10):  frozen backbone, head only, LR=1e-3
Stage 2 (epochs 11–25): unfreeze last 2 backbone blocks, differential LR,
                         backbone=1e-4 / head=1e-3, MixUp(alpha=0.4)

Usage:
    python -m model_architecture.train_binary_classifier --roi eyes
    python -m model_architecture.train_binary_classifier --roi mouth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from model_architecture.dataset.binary_classifier_dataset import build_dataloaders
from model_architecture.models.binary_roi_classifier import build_binary_classifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_1_EPOCHS = 10
STAGE_2_EPOCHS = 30
BATCH_SIZE = 64
STAGE_1_LR = 1e-3
STAGE_2_HEAD_LR = 1e-3
STAGE_2_BACKBONE_LR = 1e-5
DROPOUT = 0.2
MIXUP_ALPHA = 0.4
EARLY_STOP_PATIENCE = 0  # 0 = disabled

# Unzipped from FatigueSense/binary_classifier_dataset (see docs/training.md)
_DEFAULT_TRAIN_ROOT = Path(r"C:\Users\jlord\Downloads\dataset_split\train")

DATASET_ROOTS: dict[str, str] = {
    "eyes": str(_DEFAULT_TRAIN_ROOT / "eyes"),
    "mouth": str(_DEFAULT_TRAIN_ROOT / "mouth"),
}

CLASS_LABELS: dict[str, dict[str, int]] = {
    "eyes": {"closed": 0, "open": 1},
    "mouth": {"closed": 0, "open": 1},
}

# Mouth crops are heavily imbalanced (mostly closed); downsample majority class.
ROI_DATALOADER_OPTIONS: dict[str, dict] = {
    "eyes": {"downsample_majority": False},
    "mouth": {"downsample_majority": True},
}

CHECKPOINT_DIR = Path("runs/binary")


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()

    # Variables
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Get batch logits and loss
            logits = model(images)
            loss = criterion(logits, labels)

            # Get prediction
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Calculate validation loss, precision, recall and f1
    avg_loss = total_loss / len(loader.dataset)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, precision, recall, f1


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


MIXUP_WARMUP_EPOCHS = 3


def run_stage(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    start_epoch: int,
    use_mixup: bool,
    checkpoint_path: Path,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, list[float], list[float], list[float]]:
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = start_epoch

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1s: list[float] = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss = 0.0

        epoch_offset = epoch - start_epoch
        mixup_active = use_mixup and epoch_offset >= MIXUP_WARMUP_EPOCHS

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                if mixup_active:
                    mixed, labels_a, labels_b, lam = mixup_batch(
                        images, labels, MIXUP_ALPHA
                    )
                    logits = model(mixed)
                    loss = mixup_loss(criterion, logits, labels_a, labels_b, lam)
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            running_loss += loss.item() * images.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, precision, recall, f1 = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(f1)

        mixup_tag = " [mixup]" if mixup_active else ""
        print(
            f"Epoch {epoch:>3}{mixup_tag} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if EARLY_STOP_PATIENCE > 0 and patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch} (best={best_epoch})")
                break

    return best_epoch, train_losses, val_losses, val_f1s


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_training(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    stage_boundary: int,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", palette="Blues_r")

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax in (ax1, ax2):
        ax.axvline(
            stage_boundary + 0.5,
            color="#aaa",
            linestyle="--",
            linewidth=1,
            label="Stage 2 start",
        )

    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    sns.despine(ax=ax1)

    ax2.plot(epochs, val_f1s, label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.set_title("Validation F1")
    ax2.set_ylim(0, 1)
    ax2.legend()
    sns.despine(ax=ax2)

    plt.suptitle("Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_dataset_root(roi: str) -> Path:
    if roi not in DATASET_ROOTS:
        raise ValueError(
            f"No dataset registered for ROI '{roi}'. Add to DATASET_ROOTS."
        )
    root = Path(DATASET_ROOTS[roi]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root for '{roi}' not found: {root}\n"
            f"Update DATASET_ROOTS['{roi}'] or unzip dataset_split.zip under "
            f"C:\\Users\\jlord\\Downloads\\dataset_split\\train\\{roi}\\{{closed,open}}\\"
        )
    return root


def train(roi: str) -> None:
    root = _resolve_dataset_root(roi)
    class_to_label = CLASS_LABELS[roi]
    loader_opts = ROI_DATALOADER_OPTIONS.get(roi, {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {root}")
    if loader_opts.get("downsample_majority"):
        print("  downsample_majority=True (balance classes)")

    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    train_loader, val_loader = build_dataloaders(
        root=root,
        class_to_label=class_to_label,
        batch_size=BATCH_SIZE,
        num_workers=6,
        persistent_workers=True,
        **loader_opts,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = build_binary_classifier(
        pretrained=True, freeze_backbone=True, dropout=DROPOUT
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = CHECKPOINT_DIR / roi
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"

    # Stage 1 — frozen backbone
    print("\n--- Stage 1: frozen backbone ---")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE_1_LR,
    )
    _, s1_train_losses, s1_val_losses, s1_val_f1s = run_stage(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs=STAGE_1_EPOCHS,
        start_epoch=1,
        use_mixup=False,
        checkpoint_path=checkpoint_path,
        scaler=scaler,
    )

    # Stage 2 — partial unfreeze + differential LR
    print("\n--- Stage 2: unfreeze last 2 blocks ---")
    model.freeze_backbone(toggle=False, last_n_blocks=2)
    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": STAGE_2_BACKBONE_LR},
            {"params": model.head.parameters(), "lr": STAGE_2_HEAD_LR},
        ]
    )
    _, s2_train_losses, s2_val_losses, s2_val_f1s = run_stage(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        epochs=STAGE_2_EPOCHS,
        start_epoch=STAGE_1_EPOCHS + 1,
        use_mixup=True,
        checkpoint_path=checkpoint_path,
        scaler=scaler,
    )

    print(f"\nBest checkpoint saved to {checkpoint_path}")

    _plot_training(
        s1_train_losses + s2_train_losses,
        s1_val_losses + s2_val_losses,
        s1_val_f1s + s2_val_f1s,
        stage_boundary=len(s1_train_losses),
        save_path=checkpoint_dir / "training_curves.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", default="eyes", choices=list(DATASET_ROOTS.keys()))
    args = parser.parse_args()
    train(args.roi)
