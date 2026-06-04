# CNN Auto-Labelling

Turns raw driver videos into a **labelled image dataset** for the eye and mouth
CNN classifiers. MediaPipe face landmarks are used to crop the left eye, right
eye, and mouth from each sampled frame, and the crops are auto-labelled
`open` / `closed` from the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).

> **This is the Phase A dataset preparation method.** It documents how the CNN
> training data was built from the raw videos we collected (see the
> [labelling README](../README.md) for the Google Drive link to those videos).
> Assuming you have access to the raw videos, running this should reproduce the
> same dataset published as
> [`FatigueSense/binary_classifier_dataset`](https://huggingface.co/datasets/FatigueSense/binary_classifier_dataset)
> on Hugging Face. If you only need the prepared dataset (not the raw videos),
> download it directly via `scripts/download_hf_datasets.py` instead of running
> this pipeline.

## Files

| File | Role |
|------|------|
| `mediapipe_labelling.py` | `MediaPipeRegionExtractor` — detects landmarks, computes EAR/MAR, returns eye/mouth crops + states for one frame. Run directly for a live webcam preview. |
| `label_dataset.py` | Walks a folder of videos, runs the extractor frame by frame, and writes the labelled crop dataset to disk. |

## How labelling works

For every sampled frame the extractor:

1. Detects one face with MediaPipe `FaceLandmarker` (video mode).
2. Computes **EAR** per eye and **MAR** for the mouth from fixed landmark sets.
3. Assigns a state using hysteresis thresholds (see `mediapipe_labelling.py`):
   - Eye: `CLOSED` if `EAR <= 0.10`, `OPEN` if `EAR >= 0.13`, otherwise **ambiguous → frame skipped**.
   - Mouth: `OPEN` if `MAR > 0.43`, `CLOSED` if `MAR < 0.40` (last state held in between).
4. Crops each region to **64×64** and saves it into the matching class folder.

Frames are skipped (and counted in the stats) when no face is detected, an eye
state is ambiguous, or any crop is missing.

## Prerequisites

- Python deps: `opencv-python`, `mediapipe`, `numpy`.
- The MediaPipe model file `face_landmarker.task` (it lives at the
  **repo root** of `fatigue-sense`).
- A folder of input videos. Supported extensions: `.mp4 .avi .mov .mkv .webm .m4v`.

## Running

### Option A — batch a folder of videos (the `__main__` defaults)

Edit the constants at the bottom of `label_dataset.py`, then run it from the
folder that contains your `videos/` directory and the model file:

```powershell
python scripts/labelling/cnn/label_dataset.py
```

Defaults in `__main__`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `VIDEOS_DIR` | `videos` | Folder of input videos |
| `OUTPUT_ROOT` | `dataset` | Where crops are written |
| `MODEL_PATH` | `face_landmarker.task` | MediaPipe model |
| `sample_every_n_frames` | `15` | Process every 15th frame |
| `max_frames` | `None` | Cap processed frames per video (None = all) |
| `flip_horizontal` | `False` | Mirror frames before cropping |

### Option B — call from your own script

```python
from scripts.labelling.cnn.label_dataset import (
    process_video_to_region_dataset,
    process_videos_in_directory,
)

# Single video
stats = process_video_to_region_dataset(
    video_path="videos/clip01.mp4",
    model_path="face_landmarker.task",
    output_root="dataset",
    sample_every_n_frames=15,
    max_frames=None,
    flip_horizontal=False,
)

# Whole directory
all_stats = process_videos_in_directory(
    videos_dir="videos",
    model_path="face_landmarker.task",
    output_root="dataset",
)
```

### Preview the extractor live (sanity check)

```powershell
python scripts/labelling/cnn/mediapipe_labelling.py
```

Opens the webcam and shows the live eye/mouth crops. Press `q` or `Esc` to quit.

## Output layout

```
<output_root>/
├── eyes/
│   ├── open/      <video>_f000123_left.png, <video>_f000123_right.png ...
│   └── closed/
├── mouth/
│   ├── open/      <video>_f000123_mouth.png ...
│   └── closed/
└── ear_log.csv    per-sample EAR values and eye states
```

- Filenames are `<sanitized_video_stem>_f<frame_index>_{left,right,mouth}.png`,
  so samples are traceable back to the source video and frame.
- Images are saved as PNG (lossless, fast compression).
- `ear_log.csv` is **appended** across runs with columns:
  `sample_id, ear_left, ear_right, ear, left_eye_state, right_eye_state`.

## Tuning tips

- **Too few / too many samples:** adjust `sample_every_n_frames`.
- **Mislabelled eyes/mouth:** tune the `EAR_*` / `MAR_*` thresholds in
  `mediapipe_labelling.py`. Lighting and camera angle shift EAR/MAR, so verify a
  handful of saved crops per class before training.
- **Class imbalance:** check the printed per-video stats (`eyes_open`,
  `eyes_closed`, `mouth_open`, `mouth_closed`) to see how balanced the dataset is.