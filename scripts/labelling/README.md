# Labelling

Tools for building labelled training datasets from the driver videos we collected.

## Collected videos

All raw videos used for labelling live in this shared Google Drive folder:

https://drive.google.com/drive/folders/1FvOzc8nzJV51d3Yv8I-ciYk-OgWJiGCo?usp=sharing

Download the videos into a local `videos/` folder before running the labelling
scripts (note: `*.mp4` is gitignored, so videos are not tracked in the repo).

## Subfolders

| Folder | Purpose |
|--------|---------|
| [`cnn/`](cnn/) | Auto-labels eye/mouth crops (`open` / `closed`) from the videos using MediaPipe landmarks, for training the CNN classifiers. See [`cnn/README.md`](cnn/README.md). |
