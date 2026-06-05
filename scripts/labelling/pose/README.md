# Pose model dataset preparation

Builds a YOLO11n-pose dataset for the five upper-body keypoints used at
runtime (nose, left/right ear, left/right shoulder). Draft labels come from
the stock COCO `yolo11n-pose.pt` model; fix errors in CVAT before training.

Alternative: download `data/pose/` from Hugging Face via
`scripts.download_hf_datasets --pose` (see [docs/training.md](../../../docs/training.md)).

## Pipeline

```text
videos/*.mp4
    |  extract_frames.py      (Stage 0 - sample + dedupe JPEGs)
    v
data/pose/frames/<stem>_<idx>.jpg
    |  pseudo_label.py        (Stage 1 - YOLO labels + dataset.yaml)
    v
data/pose/images/{train,val,test}/
data/pose/labels/{train,val,test}/
data/pose/dataset.yaml
    |  (optional) build_cvat_zip.py -> CVAT import -> manual fixes
    |  train_yolo_pose.py     (training - outside this folder)
    v
runs/pose/...
```

Frames with no person detection are copied to `data/pose/review/` for manual
labelling or discard.

## Files

| File | Role |
|------|------|
| `_defaults.py` | Repo-root paths, kpt indices, `dataset.yaml` template |
| `extract_frames.py` | Stage 0: sample frames from videos |
| `pseudo_label.py` | Stage 1: pseudo-label + train/val/test split + `dataset.yaml` |
| `build_cvat_zip.py` | Optional CVAT import zip per split |

Pose scripts only depend on `opencv-python`, `numpy`, `tqdm`, and `ultralytics`
(plus stdlib). They do not import `fatigue_pipeline`.

## Prerequisites

- Run from the **repository root**.
- Raw videos in `videos/` (see [labelling README](../README.md) for Drive link).
- Ultralytics will download `yolo11n-pose.pt` on first run if it is not cached.

## Stage 0 - extract frames

```powershell
python -m scripts.labelling.pose.extract_frames `
    --videos-dir videos `
    --output-dir data/pose/frames
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--videos-dir` | `videos/` | Directory of source videos |
| `--output-dir` | `data/pose/frames` | Flat JPEG output |
| `--sample-stride-s` | `1.0` | Minimum seconds between samples |
| `--dedupe-mse-threshold` | `25.0` | Skip near-duplicate thumbs |
| `--trim-lead-s` / `--trim-tail-s` | `0.5` | Skip start/end of each clip |

## Stage 1 - pseudo-label

```powershell
python -m scripts.labelling.pose.pseudo_label `
    --frames-dir data/pose/frames `
    --output-root data/pose
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--frames-dir` | `data/pose/frames` | Input JPEGs from Stage 0 |
| `--output-root` | `data/pose` | YOLO `images/` + `labels/` tree |
| `--base-weights` | `yolo11n-pose.pt` | Ultralytics pose checkpoint |
| `--val-fraction` / `--test-fraction` | `0.15` each | Group split by video stem |
| `--seed` | `42` | Split RNG |
| `--person-conf` | `0.5` | Detection threshold |
| `--kpt-min-conf` | `0.3` | Keypoint visibility threshold |

Writes `data/pose/dataset.yaml` with an absolute `path:` (required by Ultralytics on Windows).

## Optional - CVAT import

After pseudo-labelling, export one split for correction:

```powershell
python -m scripts.labelling.pose.build_cvat_zip --split train
python -m scripts.labelling.pose.build_cvat_zip --split val
python -m scripts.labelling.pose.build_cvat_zip --split test
```

Import `data/pose/cvat_labels_<split>.zip` in CVAT using **Ultralytics YOLO Pose 1.0**.
Re-export corrected labels back into `data/pose/labels/<split>/`.

## Train

```powershell
python -m model_architecture.train_yolo_pose
```

Reads `data/pose/dataset.yaml`. Copy or symlink the best weights to the path
used at inference (`scripts/weights_path.POSE_MODEL_PATH`).

## Output layout

```text
data/pose/
  frames/                 # Stage 0 staging (flat)
  images/train|val|test/
  labels/train|val|test/
  review/                 # no-detection frames
  dataset.yaml
  cvat_labels_train.zip   # optional
```

## Tuning tips

- **Too few frames:** lower `--sample-stride-s` in Stage 0.
- **Many files in `review/`:** lower `--person-conf` or re-shoot with better framing.
- **Leakage:** split is by video stem; keep one subject per filename prefix.
