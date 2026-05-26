"""YOLO11n-pose retraining script - 5-kpt upper-body head.

Trains a custom-head YOLO11n-pose model on the FatigueSense pose dataset
(see ``data/pose/dataset.yaml``). The pose head is rebuilt for the 5
upper-body keypoints we actually consume (nose, ears, shoulders); the
backbone is warm-started from the stock COCO ``yolo11n-pose.pt`` weights
via ``model.load()``, which silently skips the kpt-head layers because
their shapes no longer match.

Checkpointing:
  * Ultralytics writes ``last.pt`` (and ``best.pt``) every epoch.
  * ``SAVE_PERIOD`` additionally snapshots ``epoch<N>.pt`` every N epochs
    so an OOM / power loss mid-run can be rolled back to a named epoch.
  * After training, exports ``training_history.csv`` and ``training_curves.png``
    (train/val loss per epoch) from Ultralytics ``results.csv``.

Resume an interrupted run::

    python -m model_architecture.train_yolo_pose --resume

That reloads the newest ``last.pt`` under the run dir and continues with
the exact args saved in the checkpoint (epoch count, LR schedule, aug).
A fresh run (no flag) builds the 5-kpt head from scratch.

Usage::

    python -m model_architecture.train_yolo_pose            # fresh
    python -m model_architecture.train_yolo_pose --resume   # continue
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import pandas as pd

# ---- knobs ----
DATA_YAML = Path("data/pose/dataset.yaml")
ARCH_YAML = "yolo11n-pose.yaml"  # ultralytics-resolved, 5-kpt head from data yaml
PRETRAINED = "yolo11n-pose.pt"  # backbone warm-start (kpt head dropped)

EPOCHS = 100
BATCH = 8  # RTX 4050 6GB safe; bump if VRAM allows
IMGSZ = 640
WORKERS = 4
PATIENCE = 25
DEVICE = 0  # 0 = first CUDA GPU; set to "cpu" if no GPU
SAVE_PERIOD = 5  # snapshot epoch<N>.pt every N epochs; -1 disables

# Augmentation - heavier than YOLO defaults because our 25 videos lack
# subject/lighting diversity. Mosaic is the biggest gain for pose; keep
# fliplr active since we declared flip_idx in the dataset yaml.
AUG = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "degrees": 5.0,
}

PROJECT_DIR = "runs/pose"
RUN_NAME = "yolo11n_pose_upper5"

# Where the live pipeline expects the trained weights to land.
FINAL_WEIGHTS_DEST = Path("runs/best_pose_model.pt")


def _disable_remote_logging(settings) -> None:
    """Kill every Ultralytics auto-integration (ClearML, W&B, etc.).

    They auto-activate whenever the matching package is installed AND a
    global config exists; we want no remote experiment logging here.
    """
    settings.update(
        {
            "clearml": False,
            "comet": False,
            "dvc": False,
            "hub": False,
            "mlflow": False,
            "neptune": False,
            "raytune": False,
            "tensorboard": False,
            "wandb": False,
        }
    )


def _find_last_checkpoint() -> Path | None:
    """Newest ``last.pt`` under PROJECT_DIR belonging to this run name."""
    matches = glob.glob(f"{PROJECT_DIR}/**/weights/last.pt", recursive=True)
    matches = [m for m in matches if RUN_NAME in m]
    if not matches:
        return None
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def _parse_ultralytics_results(save_dir: Path) -> pd.DataFrame | None:
    """Build per-epoch train/val loss table from Ultralytics ``results.csv``."""
    results_csv = save_dir / "results.csv"
    if not results_csv.is_file():
        return None

    df = pd.read_csv(results_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if "epoch" not in df.columns:
        return None

    history = pd.DataFrame({"epoch": df["epoch"]})

    if "train/loss" in df.columns and "val/loss" in df.columns:
        history["train_loss"] = df["train/loss"]
        history["val_loss"] = df["val/loss"]
    else:
        train_cols = [
            c for c in df.columns if c.startswith("train/") and "loss" in c.lower()
        ]
        val_cols = [
            c for c in df.columns if c.startswith("val/") and "loss" in c.lower()
        ]
        if not train_cols or not val_cols:
            return None
        history["train_loss"] = df[train_cols].sum(axis=1)
        history["val_loss"] = df[val_cols].sum(axis=1)
        for col in train_cols + val_cols:
            history[col.replace("/", "_")] = df[col]

    return history


def _plot_loss_curves(
    history: pd.DataFrame,
    save_path: Path,
    *,
    best_epoch: int,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", palette="Blues_r")

    epochs = history["epoch"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"], label="Val loss")
    if best_epoch > 0:
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Pose model loss (YOLO)")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved to {save_path}")


def _export_training_artifacts(save_dir: Path) -> None:
    """Write training_history.csv + training_curves.png under the run directory."""
    history = _parse_ultralytics_results(save_dir)
    if history is None or history.empty:
        print(f"WARNING: could not parse training history from {save_dir / 'results.csv'}")
        return

    history_path = save_dir / "training_history.csv"
    curves_path = save_dir / "training_curves.png"
    history.to_csv(history_path, index=False)

    best_idx = history["val_loss"].astype(float).idxmin()
    best_epoch = int(history.loc[best_idx, "epoch"])
    _plot_loss_curves(history, curves_path, best_epoch=best_epoch)
    print(f"Training history saved to {history_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue the newest interrupted run from its last.pt.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    from ultralytics import YOLO, settings

    args = _build_arg_parser().parse_args(argv)
    _disable_remote_logging(settings)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found - run scripts/pose/pseudo_label.py first."
        )

    if args.resume:
        ckpt = _find_last_checkpoint()
        if ckpt is None:
            raise FileNotFoundError(
                f"--resume given but no last.pt found under {PROJECT_DIR} "
                f"for run '{RUN_NAME}'. Run a fresh train first."
            )
        print(f"Resuming from {ckpt}")
        # resume=True restores epoch count, LR schedule, and aug from the
        # checkpoint - do NOT re-pass the train args, they're ignored.
        model = YOLO(str(ckpt))
        results = model.train(resume=True)
    else:
        # Build a fresh 5-kpt head, warm-start everything else from COCO.
        model = YOLO(ARCH_YAML).load(PRETRAINED)
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            workers=WORKERS,
            patience=PATIENCE,
            device=DEVICE,
            save_period=SAVE_PERIOD,
            project=PROJECT_DIR,
            name=RUN_NAME,
            exist_ok=False,
            **AUG,
        )

    save_dir = Path(results.save_dir)
    _export_training_artifacts(save_dir)

    metrics = model.val(data=str(DATA_YAML), device=DEVICE)
    print(f"Validation metrics:\n{metrics}")

    best = save_dir / "weights" / "best.pt"
    if best.exists():
        FINAL_WEIGHTS_DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, FINAL_WEIGHTS_DEST)
        print(f"Copied {best} -> {FINAL_WEIGHTS_DEST}")
    else:
        print(f"WARNING: best.pt not found at {best}")


if __name__ == "__main__":
    main()
