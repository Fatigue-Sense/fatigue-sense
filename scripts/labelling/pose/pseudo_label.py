"""Stage 1: pseudo-label sampled frames with the base yolo11n-pose model.

Reads JPEGs from the frames directory (from ``extract_frames.py``), runs
stock ``yolo11n-pose.pt``, and writes YOLO-pose labels for five upper-body
keypoints (nose, ears, shoulders). Also writes ``data/pose/dataset.yaml``
for ``model_architecture.train_yolo_pose``.

Output layout::

    <output-root>/
      images/train|val|test/<frame>.jpg
      labels/train|val|test/<frame>.txt
      review/<frame>.jpg
      dataset.yaml

Usage::

    python -m scripts.labelling.pose.pseudo_label

    python -m scripts.labelling.pose.pseudo_label \\
        --frames-dir data/pose/frames \\
        --output-root data/pose
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from scripts.labelling.pose._defaults import (
    DATASET_YAML_TEMPLATE,
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_FRAMES_DIR,
    DEFAULT_OUTPUT_ROOT,
    POSE_MIN_KPT_CONF,
    POSE_PERSON_CONF,
    UPPER_BODY_KPT_INDICES,
)

DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_SEED = 42
DEFAULT_BBOX_PAD_FRAC = 0.10


@dataclass
class LabelConfig:
    frames_dir: Path
    output_root: Path
    base_weights: str
    val_fraction: float
    test_fraction: float
    seed: int
    bbox_pad_frac: float
    person_conf: float
    kpt_min_conf: float


@dataclass
class LabelStats:
    total: int = 0
    labelled: int = 0
    no_person: int = 0
    visible_kpts: int = 0
    hidden_kpts: int = 0


def _video_stem_from_frame(frame_name: str) -> str:
    stem = Path(frame_name).stem
    return stem.rsplit("_", 1)[0]


def _split_by_video(
    frames: list[Path], val_fraction: float, test_fraction: float, seed: int
) -> dict[Path, str]:
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction ({val_fraction}) + test_fraction "
            f"({test_fraction}) must be < 1"
        )

    rng = random.Random(seed)
    stems = sorted({_video_stem_from_frame(p.name) for p in frames})
    rng.shuffle(stems)

    val_count = max(1, int(round(len(stems) * val_fraction)))
    test_count = max(1, int(round(len(stems) * test_fraction)))

    val_stems = set(stems[:val_count])
    test_stems = set(stems[val_count : val_count + test_count])

    def assign(stem: str) -> str:
        if stem in val_stems:
            return "val"
        if stem in test_stems:
            return "test"
        return "train"

    return {p: assign(_video_stem_from_frame(p.name)) for p in frames}


def _bbox_from_kpts(
    kpts_xy: np.ndarray, img_w: int, img_h: int, pad_frac: float
) -> tuple[float, float, float, float]:
    xs = kpts_xy[:, 0]
    ys = kpts_xy[:, 1]
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    pad_x = w * pad_frac
    pad_y = h * pad_frac
    x_min = max(0.0, x_min - pad_x)
    y_min = max(0.0, y_min - pad_y)
    x_max = min(float(img_w - 1), x_max + pad_x)
    y_max = min(float(img_h - 1), y_max + pad_y)

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    return cx, cy, bw, bh


def _format_yolo_pose_line(
    cx: float, cy: float, bw: float, bh: float, kpts_norm: np.ndarray
) -> str:
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for kx, ky, v in kpts_norm:
        parts.append(f"{kx:.6f} {ky:.6f} {int(v)}")
    return " ".join(parts)


def _prepare_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "images_train": root / "images" / "train",
        "images_val": root / "images" / "val",
        "images_test": root / "images" / "test",
        "labels_train": root / "labels" / "train",
        "labels_val": root / "labels" / "val",
        "labels_test": root / "labels" / "test",
        "review": root / "review",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_dataset_yaml(output_root: Path) -> Path:
    """Write Ultralytics dataset config with absolute ``path`` (required on Windows)."""
    yaml_path = output_root / "dataset.yaml"
    path_str = output_root.resolve().as_posix()
    yaml_path.write_text(
        DATASET_YAML_TEMPLATE.format(path=path_str),
        encoding="utf-8",
    )
    return yaml_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--base-weights",
        type=str,
        default=DEFAULT_BASE_WEIGHTS,
        help="Path or Ultralytics-resolvable name of the pose model.",
    )
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--bbox-pad-frac", type=float, default=DEFAULT_BBOX_PAD_FRAC)
    p.add_argument(
        "--person-conf",
        type=float,
        default=POSE_PERSON_CONF,
        help="Person detection confidence threshold.",
    )
    p.add_argument(
        "--kpt-min-conf",
        type=float,
        default=POSE_MIN_KPT_CONF,
        help="Below this a keypoint is marked not-visible in the label.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    from ultralytics import YOLO

    args = _build_arg_parser().parse_args(argv)
    cfg = LabelConfig(
        frames_dir=args.frames_dir,
        output_root=args.output_root,
        base_weights=args.base_weights,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        bbox_pad_frac=args.bbox_pad_frac,
        person_conf=args.person_conf,
        kpt_min_conf=args.kpt_min_conf,
    )

    if not cfg.frames_dir.exists():
        raise FileNotFoundError(
            f"{cfg.frames_dir} not found - run extract_frames.py first"
        )

    frames = sorted(cfg.frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames in {cfg.frames_dir}")

    dirs = _prepare_dirs(cfg.output_root)
    split = _split_by_video(
        frames, cfg.val_fraction, cfg.test_fraction, cfg.seed
    )

    print(f"Loading {cfg.base_weights}...")
    model = YOLO(cfg.base_weights)
    kpt_idx = list(UPPER_BODY_KPT_INDICES)
    stats = LabelStats()

    for frame_path in tqdm(frames, desc="pseudo-labelling"):
        stats.total += 1
        partition = split[frame_path]

        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        results = model(img, verbose=False, conf=cfg.person_conf)
        kpts_obj = results[0].keypoints if results else None
        boxes_obj = results[0].boxes if results else None

        no_detection = (
            kpts_obj is None
            or kpts_obj.data is None
            or kpts_obj.data.shape[0] == 0
        )
        if no_detection:
            stats.no_person += 1
            shutil.copy2(frame_path, dirs["review"] / frame_path.name)
            continue

        kpts_all = kpts_obj.data.detach().cpu().numpy()
        mean_conf = kpts_all[:, :, 2].mean(axis=1)
        best = int(np.argmax(mean_conf))

        upper = kpts_all[best, kpt_idx, :].astype(np.float32)
        visible_mask = upper[:, 2] >= cfg.kpt_min_conf

        if not visible_mask.any():
            stats.no_person += 1
            shutil.copy2(frame_path, dirs["review"] / frame_path.name)
            continue

        bbox_xyxy: np.ndarray | None = None
        if boxes_obj is not None and boxes_obj.xyxy is not None:
            boxes_np = boxes_obj.xyxy.detach().cpu().numpy()
            if best < boxes_np.shape[0]:
                bbox_xyxy = boxes_np[best]

        if bbox_xyxy is not None:
            x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
        else:
            cx, cy, bw, bh = _bbox_from_kpts(
                upper[visible_mask, :2], img_w, img_h, cfg.bbox_pad_frac
            )

        kpts_norm = np.zeros((upper.shape[0], 3), dtype=np.float32)
        kpts_norm[:, 0] = upper[:, 0] / img_w
        kpts_norm[:, 1] = upper[:, 1] / img_h
        kpts_norm[:, 2] = np.where(visible_mask, 2, 1)

        stats.visible_kpts += int(visible_mask.sum())
        stats.hidden_kpts += int((~visible_mask).sum())

        image_out = dirs[f"images_{partition}"] / frame_path.name
        label_out = dirs[f"labels_{partition}"] / (frame_path.stem + ".txt")
        shutil.copy2(frame_path, image_out)
        label_out.write_text(_format_yolo_pose_line(cx, cy, bw, bh, kpts_norm) + "\n")
        stats.labelled += 1

    yaml_path = write_dataset_yaml(cfg.output_root)
    print(
        f"Done. total={stats.total} labelled={stats.labelled} "
        f"no_person={stats.no_person} "
        f"visible_kpts={stats.visible_kpts} hidden_kpts={stats.hidden_kpts}"
    )
    print(f"Review skipped frames in {dirs['review']}")
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
