# Training reproducibility

| Dataset | Repository | Training step | Notes |
|---------|------------|---------------|--------|
| Eyes + mouth ROI crops | [FatigueSense/binary_classifier_dataset](https://huggingface.co/datasets/FatigueSense/binary_classifier_dataset) | 1  - `--roi eyes` / `--roi mouth` | `data/binary/` - see [download section](#download-datasets-and-expected-layout) |
| Pose (YOLO) | [FatigueSense/pose_dataset](https://huggingface.co/datasets/FatigueSense/pose_dataset) | 2  - `train_yolo_pose` | `data/pose/` + `dataset.yaml` |
| Temporal features | [FatigueSense/temporal_dataset](https://huggingface.co/datasets/FatigueSense/temporal_dataset) | 3  - `train_temporal_model` | `data/temporal/features/*.parquet` |

Published **model weights**: [eye_classifier](https://huggingface.co/FatigueSense/eye_classifier), [mouth_classifier](https://huggingface.co/FatigueSense/mouth_classifier), [pose_model](https://huggingface.co/FatigueSense/pose_model), [temporal_model](https://huggingface.co/FatigueSense/temporal_model).

You can either **download** the published datasets below or **build** them from raw
`videos/` using [scripts/labelling/README.md](../scripts/labelling/README.md).

| Target | Labelling guide | Output path |
|--------|-----------------|-------------|
| CNN eyes/mouth | [cnn/README.md](../scripts/labelling/cnn/README.md) | `data/binary/train/` |
| Pose YOLO | [pose/README.md](../scripts/labelling/pose/README.md) | `data/pose/` + `dataset.yaml` |
| Temporal features | [temporal/README.md](../scripts/labelling/temporal/README.md) | `data/temporal/features/` |

Recommended build order: CNN (or HF binary zip) → pose → temporal (needs trained or HF
eye/mouth/pose weights for Stage 1 extraction).

### Download datasets and expected layout

Use [`scripts/download_hf_datasets.py`](../scripts/download_hf_datasets.py) from the repo root. It writes under `data/` in the shapes expected by each `train_*.py` script. Public HF repos need no token; for private mirrors set `HF_TOKEN` or run `hf auth login`.

**All three datasets** (binary zip + pose + temporal snapshots):

```bash
python -m scripts.download_hf_datasets
```

**One dataset at a time:**

```bash
python -m scripts.download_hf_datasets --binary
python -m scripts.download_hf_datasets --pose
python -m scripts.download_hf_datasets --temporal
```

Re-download after a bad extract: add `--force`. Preview paths only: `--dry-run`.

#### 1. Binary ROI crops (eyes + mouth classifiers)

| Item | Value |
|------|--------|
| HF dataset | [FatigueSense/binary_classifier_dataset](https://huggingface.co/datasets/FatigueSense/binary_classifier_dataset) |
| Archive | `dataset_split.zip` (train + test, both ROIs in one zip) |
| Local root | `data/binary/` |
| Training | `train_binary_classifier.py --roi eyes` or `--roi mouth` |

After download and unzip:

```
data/binary/
├── train/                    # used by default for training
│   ├── eyes/
│   │   ├── closed/           # *.png, …
│   │   └── open/
│   └── mouth/
│       ├── closed/
│       └── open/
└── test/                     # holdout for evaluation (optional at train time)
    ├── eyes/
    │   ├── closed/
    │   └── open/
    └── mouth/
        ├── closed/
        └── open/
```

Class folder names must be lowercase `closed` and `open`. Training reads `data/binary/train/eyes` and `data/binary/train/mouth` unless you override `DATASET_ROOTS` in `train_binary_classifier.py`.

To auto-label crops from videos instead of downloading the zip, run
`python -m scripts.labelling.cnn.label_dataset` (writes under `data/binary/train/` by
default). That produces a **train** tree only; add `data/binary/test/` manually or use
the HF archive for a held-out split.

Published weights (inference): [eye_classifier](https://huggingface.co/FatigueSense/eye_classifier), [mouth_classifier](https://huggingface.co/FatigueSense/mouth_classifier).

#### 2. Pose (YOLO upper-body keypoints)

| Item | Value |
|------|--------|
| HF dataset | [FatigueSense/pose_dataset](https://huggingface.co/datasets/FatigueSense/pose_dataset) |
| Local root | `data/pose/` |
| Training | `train_yolo_pose.py` (expects `data/pose/dataset.yaml`) |

Typical layout after `snapshot_download`:

```
data/pose/
├── dataset.yaml              # paths to train/val image and label dirs
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Exact split names follow `dataset.yaml` on the Hub. To build from videos:
`extract_frames` → `pseudo_label` (writes `dataset.yaml`); see
[pose/README.md](../scripts/labelling/pose/README.md). Published weights:
[pose_model](https://huggingface.co/FatigueSense/pose_model).

#### 3. Temporal features (BiGRU focus model)

| Item | Value |
|------|--------|
| HF dataset | [FatigueSense/temporal_dataset](https://huggingface.co/datasets/FatigueSense/temporal_dataset) |
| Local root | `data/temporal/` |
| Training | `train_temporal_model.py` (reads `data/temporal/features/*.parquet`) |

```
data/temporal/
├── features/
│   ├── <video_id>.parquet    # one file per video, 1 Hz aggregated features
│   └── …
├── raw_probs/                # optional per-frame probabilities (not required to train)
└── manifest.json             # optional dataset index from the Hub release
```

Published weights: [temporal_model](https://huggingface.co/FatigueSense/temporal_model).

To build from videos: `extract_probs` → `aggregate_features`; see
[temporal/README.md](../scripts/labelling/temporal/README.md). Window labels for
training are bootstrapped inside `TemporalWindowDataset` (no separate label export).

## 1. Binary ROI classifiers (eyes and mouth)

```bash
python -m model_architecture.train_binary_classifier --roi eyes
python -m model_architecture.train_binary_classifier --roi mouth
```

Requires `data/binary/` as in the [layout section](#1-binary-roi-crops-eyes--mouth-classifiers). Checkpoints go under `runs/binary/`.

## 2. Pose estimator

```bash
python -m model_architecture.train_yolo_pose
```

Requires `data/pose/dataset.yaml` - see [pose layout](#2-pose-yolo-upper-body-keypoints) or
`python -m scripts.labelling.pose.pseudo_label`.

Per-run outputs under `runs/pose/<run_name>/` (default run name `yolo11n_pose_upper5`):

- `weights/best.pt`, `weights/last.pt` - Ultralytics checkpoints
- `results.csv` - raw epoch metrics from Ultralytics
- `training_history.csv` - train/val loss per epoch (exported after training)
- `training_curves.png` - loss plot (best epoch marked at minimum val loss)

A copy of the best weights is also written to `runs/best_pose_model.pt`.

## 3. Temporal focus model

```bash
python -m model_architecture.train_temporal_model
```

Requires `data/temporal/features/*.parquet` - see [temporal layout](#3-temporal-features-bigru-focus-model). Outputs under `runs/temporal/`:

- `best.pt` - best checkpoint by validation MSE (early stopping, patience 50)
- `normalization.json` - per-feature mean/std fit on train windows
- `model_config.json` - BiGRU hyperparameters (use when loading `best.pt` locally)
- `training_history.csv` - train and val MSE per epoch (updated each epoch)
- `training_curves.png` - loss plot written at end of training

Training defaults: up to **500 epochs**, early stop if val MSE does not improve for
**50** epochs. Anti-overfit settings include train window stride 5, 2-layer BiGRU
with dropout, feature noise, gradient clipping, and stronger weight decay (see
`train_temporal_model.py` constants). Retraining changes `best.pt` layout vs older
HF weights; ship `model_config.json` with new checkpoints or point inference at
`runs/temporal/`.

To **build** feature Parquets from raw video, use
[scripts/labelling/temporal](../scripts/labelling/temporal/README.md) (`extract_probs`,
then `aggregate_features`), then run this training step.

## Manual download (optional)

If you prefer the Hub API directly instead of `download_hf_datasets.py`:

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Binary: single zip, then extract under data/binary/ (see script for layout normalization)
hf_hub_download("FatigueSense/binary_classifier_dataset", filename="dataset_split.zip", repo_type="dataset")

snapshot_download("FatigueSense/pose_dataset", repo_type="dataset", local_dir="data/pose")
snapshot_download("FatigueSense/temporal_dataset", repo_type="dataset", local_dir="data/temporal")
```

[← Back to README](../README.md)
