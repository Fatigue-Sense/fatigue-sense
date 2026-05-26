# Training reproducibility

Training scripts live in `model_architecture/`. **Datasets and pretrained weights**
are on Hugging Face under the [FatigueSense](https://huggingface.co/FatigueSense)
org (public -no token required). Run steps in order: per-frame models first, then
the temporal model that consumes their outputs.

| Step | Model | Training script | Dataset | Weights repo |
|------|-------|-----------------|--------|--------------|
| 1 | Eye / mouth ROI classifiers | `train_binary_classifier` | [binary_classifier_dataset](https://huggingface.co/datasets/FatigueSense/binary_classifier_dataset) (zip) | `eye_classifier`, `mouth_classifier` |
| 2 | Upper-body pose (YOLO11n, 5 kpts) | `train_yolo_pose` | `pose_dataset` | `pose_model` |
| 3 | Focus score (BiGRU) | `train_temporal_model` | `temporal_dataset` | `temporal_model` |

## Hugging Face datasets

| Dataset | Repository | Training step | Notes |
|---------|------------|---------------|--------|
| Eyes + mouth ROI crops | [FatigueSense/binary_classifier_dataset](https://huggingface.co/datasets/FatigueSense/binary_classifier_dataset) | 1  - `--roi eyes` / `--roi mouth` | Single `dataset_split.zip` (train + test, both ROIs); unzip locally (below) |
| Pose (YOLO) | [FatigueSense/pose_dataset](https://huggingface.co/datasets/FatigueSense/pose_dataset) | 2  - `train_yolo_pose` | YOLO images + labels; `dataset.yaml` |
| Temporal features | [FatigueSense/temporal_dataset](https://huggingface.co/datasets/FatigueSense/temporal_dataset) | 3  - `train_temporal_model` | `features/*.parquet`, optional `raw_probs/`; `manifest.json` |

Published **model weights**: [eye_classifier](https://huggingface.co/FatigueSense/eye_classifier), [mouth_classifier](https://huggingface.co/FatigueSense/mouth_classifier), [pose_model](https://huggingface.co/FatigueSense/pose_model), [temporal_model](https://huggingface.co/FatigueSense/temporal_model).

### Binary classifier crops (eyes + mouth)

Per-image uploads to separate `eye_dataset` / `mouth_dataset` repos are impractical
(thousands of PNGs, HF rate limits). Use the bundled archive instead:

**[FatigueSense/binary_classifier_dataset](https://huggingface.co/datasets/FatigueSense/binary_classifier_dataset)**  - contains `dataset_split.zip` with `train/` and `test/` splits for both eyes and mouth (`closed` / `open` class folders).

#### Download and unzip

From the repo root (downloads all three HF datasets into `data/`):

```bash
python -m scripts.download_hf_datasets
```

Or only the binary zip:

```bash
python -m scripts.download_hf_datasets --binary
```

Manual unzip (alternative):

```python
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

# From repository root
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust if running elsewhere
DATA_DIR = REPO_ROOT / "data" / "binary"

zip_path = hf_hub_download(
    repo_id="FatigueSense/binary_classifier_dataset",
    filename="dataset_split.zip",
    repo_type="dataset",
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(DATA_DIR)
```

If the archive has an extra top-level `dataset_split/` folder inside the zip, move
`train/` and `test/` into `data/binary/`.

Or download `dataset_split.zip` from the [dataset files tab](https://huggingface.co/datasets/FatigueSense/binary_classifier_dataset/tree/main) and extract it yourself.

#### Expected layout after unzip

```
data/binary/
├── train/
│   ├── eyes/
│   │   ├── closed/   (*.png, …)
│   │   └── open/
│   └── mouth/
│       ├── closed/
│       └── open/
└── test/
    ├── eyes/
    │   ├── closed/
    │   └── open/
    └── mouth/
        ├── closed/
        └── open/
```

`train_binary_classifier.py` defaults to `data/binary/train/{eyes,mouth}` (lowercase
`closed` / `open` class folders). Override `DATASET_ROOTS` only if you store crops elsewhere.

## 1. Binary ROI classifiers (eyes and mouth)

```bash
python -m model_architecture.train_binary_classifier --roi eyes
python -m model_architecture.train_binary_classifier --roi mouth
```

Requires the unzipped tree above. Checkpoints go under `runs/binary/`.

## 2. Pose estimator

```bash
# Expects data/pose/ (images, labels, dataset.yaml)
python -m model_architecture.train_yolo_pose
```

Download [pose_dataset](https://huggingface.co/datasets/FatigueSense/pose_dataset)
into `data/pose/`. Published weights: `FatigueSense/pose_model`.

## 3. Temporal focus model

```bash
# Expects data/temporal/features/*.parquet (one file per video)
python -m model_architecture.train_temporal_model
```

Download [temporal_dataset](https://huggingface.co/datasets/FatigueSense/temporal_dataset)
(`features/` and optionally `raw_probs/`) into `data/temporal/`. Outputs:
`runs/temporal/best.pt`, `runs/temporal/normalization.json`.

To **build** feature Parquets from raw video (not in this minimal release), use the
main FatigueSense dev repo: per-frame prob extraction, then 1 Hz aggregation, then
this training step.

## Download other datasets (example)

```python
from huggingface_hub import snapshot_download

snapshot_download("FatigueSense/pose_dataset", repo_type="dataset", local_dir="data/pose")
snapshot_download("FatigueSense/temporal_dataset", repo_type="dataset", local_dir="data/temporal_hf")
# Map downloaded layouts → paths expected by each train_*.py script
```

[← Back to README](../README.md)
