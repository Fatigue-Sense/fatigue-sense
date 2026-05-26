"""Download FatigueSense training datasets from Hugging Face into ``data/``.

Layouts match ``model_architecture/train_*.py`` defaults:

  data/binary/   <- FatigueSense/binary_classifier_dataset (dataset_split.zip)
  data/pose/     <- FatigueSense/pose_dataset
  data/temporal/ <- FatigueSense/temporal_dataset

Public repos need no token. For private mirrors, set ``HF_TOKEN`` or run
``hf auth login``.

Usage:
    python -m scripts.download_hf_datasets
    python -m scripts.download_hf_datasets --binary --pose
    python -m scripts.download_hf_datasets --dry-run
    python -m scripts.download_hf_datasets --force
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

HF_ORG = "FatigueSense"

BINARY_REPO = f"{HF_ORG}/binary_classifier_dataset"
BINARY_ZIP = "dataset_split.zip"

POSE_REPO = f"{HF_ORG}/pose_dataset"
TEMPORAL_REPO = f"{HF_ORG}/temporal_dataset"

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.webp")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = _REPO_ROOT / "data"


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(sum(1 for _ in directory.glob(pattern)) for pattern in IMAGE_GLOBS)


def _binary_ready(dest: Path) -> bool:
    train_eyes = dest / "train" / "eyes" / "closed"
    train_mouth = dest / "train" / "mouth" / "closed"
    return _count_images(train_eyes) > 0 and _count_images(train_mouth) > 0


def _pose_ready(dest: Path) -> bool:
    return (dest / "dataset.yaml").is_file()


def _temporal_ready(dest: Path) -> bool:
    features = dest / "features"
    return features.is_dir() and any(features.glob("*.parquet"))


def _normalize_binary_layout(root: Path) -> None:
    """Ensure ``train/`` and ``test/`` sit directly under *root*."""
    if (root / "train").is_dir() and (root / "test").is_dir():
        return

    nested = root / "dataset_split"
    if not nested.is_dir():
        raise FileNotFoundError(
            f"Unrecognized layout under {root}. Expected train/ and test/ "
            f"(or dataset_split/train after unzip)."
        )

    for name in ("train", "test"):
        src = nested / name
        if not src.is_dir():
            raise FileNotFoundError(f"Missing {src} in archive")
        dst = root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    try:
        nested.rmdir()
    except OSError:
        pass


def download_binary(dest: Path, *, dry_run: bool, force: bool) -> None:
    print(f"[binary] {BINARY_REPO} -> {dest}")
    if not force and _binary_ready(dest):
        print("  skip (train/eyes and train/mouth already present)")
        return
    if dry_run:
        print(f"  would download {BINARY_ZIP} and extract")
        return

    dest.mkdir(parents=True, exist_ok=True)
    if force and dest.exists():
        for child in dest.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    zip_path = Path(
        hf_hub_download(
            repo_id=BINARY_REPO,
            filename=BINARY_ZIP,
            repo_type="dataset",
        )
    )
    print(f"  zip: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    _normalize_binary_layout(dest)
    if not _binary_ready(dest):
        raise RuntimeError(f"Binary dataset incomplete after extract: {dest}")
    print(
        f"  ready: eyes={_count_images(dest / 'train' / 'eyes' / 'closed')} "
        f"closed train crops (sample count)"
    )


def download_pose(dest: Path, *, dry_run: bool, force: bool) -> None:
    print(f"[pose] {POSE_REPO} -> {dest}")
    if not force and _pose_ready(dest):
        print("  skip (dataset.yaml present)")
        return
    if dry_run:
        print("  would snapshot_download")
        return

    if force and dest.exists():
        shutil.rmtree(dest)

    snapshot_download(
        repo_id=POSE_REPO,
        repo_type="dataset",
        local_dir=str(dest),
    )
    if not _pose_ready(dest):
        raise RuntimeError(f"Pose dataset missing dataset.yaml under {dest}")
    print(f"  ready: {dest / 'dataset.yaml'}")


def download_temporal(dest: Path, *, dry_run: bool, force: bool) -> None:
    print(f"[temporal] {TEMPORAL_REPO} -> {dest}")
    if not force and _temporal_ready(dest):
        n = len(list((dest / "features").glob("*.parquet")))
        print(f"  skip ({n} feature parquets present)")
        return
    if dry_run:
        print("  would snapshot_download")
        return

    if force and dest.exists():
        shutil.rmtree(dest)

    snapshot_download(
        repo_id=TEMPORAL_REPO,
        repo_type="dataset",
        local_dir=str(dest),
    )
    if not _temporal_ready(dest):
        raise RuntimeError(f"Temporal dataset missing features/*.parquet under {dest}")
    n = len(list((dest / "features").glob("*.parquet")))
    print(f"  ready: {n} feature parquets")


def _resolve_targets(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    want_binary = args.binary or args.all
    want_pose = args.pose or args.all
    want_temporal = args.temporal or args.all
    selective = want_binary or want_pose or want_temporal
    if selective:
        return want_binary, want_pose, want_temporal
    return True, True, True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", action="store_true", help="Eye + mouth ROI crops (zip)")
    p.add_argument("--pose", action="store_true", help="YOLO pose dataset")
    p.add_argument("--temporal", action="store_true", help="BiGRU feature parquets")
    p.add_argument("--all", action="store_true", help="Same as default (all three)")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output root (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if target layout already exists",
    )
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    do_binary, do_pose, do_temporal = _resolve_targets(args)

    print(f"Data directory: {data_dir}\n")

    if do_binary:
        download_binary(
            data_dir / "binary",
            dry_run=args.dry_run,
            force=args.force,
        )
    if do_pose:
        download_pose(
            data_dir / "pose",
            dry_run=args.dry_run,
            force=args.force,
        )
    if do_temporal:
        download_temporal(
            data_dir / "temporal",
            dry_run=args.dry_run,
            force=args.force,
        )

    if not args.dry_run:
        print("\nDone. See docs/training.md for train commands.")


if __name__ == "__main__":
    main()
