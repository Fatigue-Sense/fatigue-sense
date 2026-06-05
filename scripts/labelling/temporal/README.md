# Temporal model dataset preparation

Builds the offline inputs for the BiGRU focus-score model: per-frame CNN/pose
outputs, then per-second engineered features. Alternative: download
`data/temporal/` from Hugging Face via `scripts.download_hf_datasets --temporal`
(see [docs/training.md](../../../docs/training.md)).

**Window labels are not written here** - training uses bootstrap targets from
`model_architecture.dataset.temporal_window_dataset.default_label_from_window`
when you run `train_temporal_model`.

## Pipeline

```text
videos/*.mp4
    |  extract_probs.py          (Stage 1)
    v
data/temporal/raw_probs/<stem>.parquet
    |  aggregate_features.py     (Stage 2)
    v
data/temporal/features/<stem>.parquet
    |  train_temporal_model.py   (training - outside this folder)
    v
runs/temporal/...
```

Stage 1 uses the same `FatiguePipeline` + `PoseEstimator` stack as live
inference, so offline rows match runtime behaviour.

## Files

| File | Role |
|------|------|
| `_defaults.py` | Repo-root paths and weight resolution via `scripts/weights_path.py` |
| `extract_probs.py` | Stage 1: video(s) -> per-frame Parquet (+ CSV under `csv/`) |
| `aggregate_features.py` | Stage 2: Parquet -> 1 Hz feature steps (+ CSV under `csv/`) |

## Prerequisites

- Run from the **repository root**.
- Python: `opencv-python`, `pandas`, `pyarrow`, `tqdm`, `torch`, `mediapipe`, `ultralytics` (pose in Stage 1).
- `face_landmarker.task` at the repo root (see main [`README.md`](../../../README.md)).
- Trained or HF weights for eye, mouth, and pose (defaults in `scripts/weights_path.py`).
- Input videos in `videos/` (or pass `--video`).

## Stage 1 - extract per-frame probabilities

```powershell
python -m scripts.labelling.temporal.extract_probs `
    --video videos `
    --output data/temporal/raw_probs
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--video` | `videos/` | One file or directory of videos |
| `--output` | `data/temporal/raw_probs` | Output directory |
| `--frame-step` | `5` | Process every Nth frame |
| `--overwrite` | off | Re-run if Parquet already exists |

Single clip:

```powershell
python -m scripts.labelling.temporal.extract_probs `
    --video videos/vid_001.mp4 `
    --output data/temporal/raw_probs
```

## Stage 2 - aggregate to feature steps

```powershell
python -m scripts.labelling.temporal.aggregate_features `
    --probs data/temporal/raw_probs `
    --output data/temporal/features
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--probs` | `data/temporal/raw_probs` | One Parquet or a directory of them |
| `--output` | `data/temporal/features` | Output directory |
| `--sub-window-s` | pipeline default (60 s) | Sub-window for PERCLOS / events |
| `--step-stride-s` | pipeline default (1 s) | Emit stride |
| `--overwrite` | off | Re-run if output exists |

Rows with `valid=False` had too many missing detections in the sub-window;
`TemporalWindowDataset` drops windows that include invalid steps.

## Stage 3 - train (bootstrap labels)

```powershell
python -m model_architecture.train_temporal_model
```

Expects feature Parquets under `data/temporal/features/` (see
`FEATURES_DIR` in that script). Labels are computed on the fly from
PERCLOS, yawn rate, blink duration, and posture heuristics until human
focus scores exist.

## Output layout

```text
data/temporal/
  raw_probs/
    vid_001.parquet
    csv/vid_001.csv
  features/
    vid_001.parquet
    csv/vid_001.csv
```

## Tuning tips

- **Slow Stage 1:** increase `--frame-step` (fewer frames, coarser timeline).
- **Missing pose columns:** re-run Stage 1 with a working pose checkpoint.
- **Too many invalid steps:** check face/pose detection rates printed after Stage 1.
