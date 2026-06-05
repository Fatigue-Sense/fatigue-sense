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
> download it directly via `scripts.download_hf_datasets.py --binary` instead of
> running this pipeline (see [docs/training.md](../../../docs/training.md)).

## Files

| File | Role |
|------|------|
| `_defaults.py` | Repo-root paths, landmarker path, EAR/MAR thresholds, crop size, batch sampling defaults |
| `mediapipe_labelling.py` | `MediaPipeRegionExtractor` — detects landmarks, computes EAR/MAR, returns eye/mouth crops + states for one frame. Run directly for a live webcam preview. |
| `label_dataset.py` | Walks a folder of videos, runs the extractor frame by frame, and writes the labelled crop dataset to disk. |

## How labelling works

For every sampled frame the extractor:

1. Detects one face with MediaPipe `FaceLandmarker` (video mode).
2. Computes **EAR** per eye and **MAR** for the mouth from fixed landmark sets.
3. Assigns a state using hysteresis thresholds (see `_defaults.py`):
   - Eye: `CLOSED` if `EAR <= EAR_CLOSE_THRESH`, `OPEN` if `EAR >= EAR_OPEN_THRESH`, otherwise **ambiguous → frame skipped**.
   - Mouth: `OPEN` if `MAR > MAR_OPEN_THRESH`, `CLOSED` if `MAR < MAR_CLOSE_THRESH` (last state held in between).
4. Crops each region to **64×64** (`CROP_WIDTH` × `CROP_HEIGHT`) and saves it into the matching class folder.

Frames are skipped (and counted in the stats) when no face is detected, an eye
state is ambiguous, or any crop is missing.

## Prerequisites

- Run from the **repository root** (same as `temporal/` and `pose/`).
- Python deps: `opencv-python`, `mediapipe`, `numpy`.
- `face_landmarker.task` at the repo root (see main [`README.md`](../../../README.md)).
- Input videos in `videos/` (or pass paths in your own script).

## Running

### Option A — batch all videos (`label_dataset.py`)

Uses defaults from `_defaults.py`:

```powershell
python -m scripts.labelling.cnn.label_dataset
```

| Constant (`_defaults.py`) | Default | Meaning |
|---------------------------|---------|---------|
| `DEFAULT_VIDEOS_DIR` | `<repo>/videos` | Folder of input videos |
| `DEFAULT_OUTPUT_ROOT` | `<repo>/data/binary/train` | Train-split crops (matches `train_binary_classifier.py`) |
| `LANDMARKER_PATH` | `<repo>/face_landmarker.task` | MediaPipe model (via `scripts/weights_path.py`) |
| `DEFAULT_SAMPLE_EVERY_N_FRAMES` | `15` | Process every 15th frame |
| `DEFAULT_MAX_FRAMES` | `None` | Cap processed frames per video (None = all) |
| `DEFAULT_FLIP_HORIZONTAL` | `False` | Mirror frames before cropping |

### Option B — call from your own script

```python
from scripts.labelling.cnn._defaults import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAMPLE_EVERY_N_FRAMES,
    DEFAULT_VIDEOS_DIR,
    resolve_landmarker_path,
)
from scripts.labelling.cnn.label_dataset import (
    process_video_to_region_dataset,
    process_videos_in_directory,
)

model_path = resolve_landmarker_path()

# Single video
stats = process_video_to_region_dataset(
    video_path=DEFAULT_VIDEOS_DIR / "clip01.mp4",
    model_path=model_path,
    output_root=DEFAULT_OUTPUT_ROOT,
    sample_every_n_frames=DEFAULT_SAMPLE_EVERY_N_FRAMES,
)

# Whole directory
all_stats = process_videos_in_directory(
    videos_dir=DEFAULT_VIDEOS_DIR,
    model_path=model_path,
    output_root=DEFAULT_OUTPUT_ROOT,
)
```

### Preview the extractor live (sanity check)

```powershell
python -m scripts.labelling.cnn.mediapipe_labelling
```

Opens the webcam and shows the live eye/mouth crops. Press `q` or `Esc` to quit.

## Output layout

Default `output_root` is `data/binary/train/` so crops land where training expects them:

```
data/binary/train/
├── eyes/
│   ├── open/      <video>_f000123_left.png, <video>_f000123_right.png ...
│   └── closed/
├── mouth/
│   ├── open/      <video>_f000123_mouth.png ...
│   └── closed/
└── ear_log.csv    per-sample EAR values and eye states
```

There is no automatic `test/` split. Use the HF dataset zip or copy a held-out subset
into `data/binary/test/` with the same folder names.

- Filenames are `<sanitized_video_stem>_f<frame_index>_{left,right,mouth}.png`,
  so samples are traceable back to the source video and frame.
- Images are saved as PNG (lossless, fast compression).
- `ear_log.csv` is **appended** across runs with columns:
  `sample_id, ear_left, ear_right, ear, left_eye_state, right_eye_state`.

## Tuning tips

- **Too few / too many samples:** adjust `sample_every_n_frames`.
- **Mislabelled eyes/mouth:** tune `EAR_*` / `MAR_*` in `_defaults.py`. Lighting
  and camera angle shift EAR/MAR, so verify a handful of saved crops per class
  before training.
- **Class imbalance:** check the printed per-video stats (`eyes_open`,
  `eyes_closed`, `mouth_open`, `mouth_closed`) to see how balanced the dataset is.