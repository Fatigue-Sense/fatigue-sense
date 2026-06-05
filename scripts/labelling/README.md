# Labelling

Tools for building labelled training datasets from the driver videos we collected.

## Collected videos

All raw videos used for labelling live in this shared Google Drive folder:

https://drive.google.com/drive/folders/1FvOzc8nzJV51d3Yv8I-ciYk-OgWJiGCo?usp=sharing

Download the videos into a local `videos/` folder before running any labelling
script (`*.mp4` is gitignored, so videos are not tracked in the repo).

## Run from the repository root

All commands below assume your shell's current directory is the **fatigue-sense**
repo root (the folder that contains `fatigue_pipeline/`, `scripts/`, and
`face_landmarker.task`).

```powershell
cd path\to\fatigue-sense
```

Use `python -m scripts.labelling.<folder>.<script>` so imports resolve correctly.

## Subfolders

| Folder | Model / data | Guide |
|--------|----------------|-------|
| [`cnn/`](cnn/) | Eye and mouth CNN classifiers (MediaPipe crops + EAR/MAR weak labels) | [`cnn/README.md`](cnn/README.md) |
| [`pose/`](pose/) | YOLO11n-pose upper-body keypoints (frame sample + pseudo-label + optional CVAT) | [`pose/README.md`](pose/README.md) |
| [`temporal/`](temporal/) | BiGRU temporal model (per-frame probs + 1 Hz features; bootstrap labels at train time) | [`temporal/README.md`](temporal/README.md) |

## Recommended order

1. **CNN** (or download [`FatigueSense/binary_classifier_dataset`](https://huggingface.co/datasets/FatigueSense/binary_classifier_dataset)) so eye/mouth classifiers exist.
2. **Pose** if you are retraining the upper-body pose model.
3. **Temporal** after CNN + pose weights are available (defaults pull from Hugging Face via `scripts/weights_path.py`).

Training entry points (not in this folder):

- `python -m model_architecture.train_yolo_pose`
- `python -m model_architecture.train_temporal_model`
