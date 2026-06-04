from __future__ import annotations

from pathlib import Path
from typing import Optional
import cv2
import csv

from vision_pipeline.labelling.mediapipe_labelling import MediaPipeRegionExtractor


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def sanitize_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def save_crop(crop, output_path: Path) -> bool:
    if crop is None:
        return False
    
    # Check if we are saving a PNG
    if output_path.suffix.lower() == ".png":
        # PNG Compression level 1 (fast, lossless)
        return bool(cv2.imwrite(str(output_path), crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 1]))
    
    # Fallback for JPEG
    return bool(cv2.imwrite(str(output_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))


def process_video_to_region_dataset(
    video_path: str | Path,
    model_path: str | Path,
    output_root: str | Path = "cropped_images",
    sample_every_n_frames: int = 1,
    max_frames: Optional[int] = None,
    flip_horizontal: bool = False,
) -> dict[str, int]:
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    # Add this after output_root.mkdir
    ear_log_path = output_root / "ear_log.csv"
    ear_file = open(ear_log_path, "a", newline="")
    ear_writer = csv.writer(ear_file)

    # Write header only if file is empty
    if ear_log_path.stat().st_size == 0:
        ear_writer.writerow(["sample_id", "ear_left", "ear_right", "ear", "left_eye_state", "right_eye_state"])

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")

    output_root.mkdir(parents=True, exist_ok=True)

    eyes_open_dir = output_root / "eyes" / "open"
    eyes_closed_dir = output_root / "eyes" / "closed"
    mouth_open_dir = output_root / "mouth" / "open"
    mouth_closed_dir = output_root / "mouth" / "closed"

    eyes_open_dir.mkdir(parents=True, exist_ok=True)
    eyes_closed_dir.mkdir(parents=True, exist_ok=True)
    mouth_open_dir.mkdir(parents=True, exist_ok=True)
    mouth_closed_dir.mkdir(parents=True, exist_ok=True)

    extractor = MediaPipeRegionExtractor(model_path=model_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Could not open video: {video_path}")

    video_stem = sanitize_stem(video_path)
    frame_index = 0
    processed_frames = 0

    stats = {
        "frames_seen": 0,
        "frames_processed": 0,
        "frames_detected": 0,
        "eyes_open": 0,
        "eyes_closed": 0,
        "mouth_open": 0,
        "mouth_closed": 0,
        "left_eye_saved": 0,
        "right_eye_saved": 0,
        "mouth_saved": 0,
        "skipped_no_face": 0,
        "skipped_ambiguous_eye": 0,
        "skipped_missing_crop": 0,
    }

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            stats["frames_seen"] += 1

            if sample_every_n_frames > 1 and frame_index % sample_every_n_frames != 0:
                frame_index += 1
                continue

            if flip_horizontal:
                frame = cv2.flip(frame, 1)

            outputs = extractor.process_frame(frame)
            processed_frames += 1
            stats["frames_processed"] += 1

            if not outputs.detected:
                stats["skipped_no_face"] += 1
                frame_index += 1
                if max_frames is not None and processed_frames >= max_frames:
                    break
                continue

            stats["frames_detected"] += 1

            # Skip ambiguous eye states
            if outputs.left_eye_state is None or outputs.right_eye_state is None:
                stats["skipped_ambiguous_eye"] += 1
                frame_index += 1
                if max_frames is not None and processed_frames >= max_frames:
                    break
                continue

            # Skip if any crop is missing
            if (
                outputs.left_eye_crop is None
                or outputs.right_eye_crop is None
                or outputs.mouth_crop is None
            ):
                stats["skipped_missing_crop"] += 1
                frame_index += 1
                if max_frames is not None and processed_frames >= max_frames:
                    break
                continue

            left_eye_label = 1 if outputs.left_eye_state == "OPEN" else 0
            right_eye_label = 1 if outputs.right_eye_state == "OPEN" else 0

            mouth_state = (outputs.mouth_state or "closed").lower()
            mouth_label = 1 if mouth_state == "open" else 0

            stats["eyes_open"] += left_eye_label + right_eye_label
            stats["eyes_closed"] += (1 - left_eye_label) + (1 - right_eye_label)

            if mouth_label == 1:
                stats["mouth_open"] += 1
            else:
                stats["mouth_closed"] += 1

            sample_id = f"{video_stem}_f{frame_index:06d}"

            ear_writer.writerow([
                sample_id,
                f"{outputs.ear_left:.4f}",
                f"{outputs.ear_right:.4f}",
                f"{outputs.ear:.4f}",
                outputs.left_eye_state,
                outputs.right_eye_state,
            ])

            # Left eye
            left_eye_dir = eyes_open_dir if left_eye_label == 1 else eyes_closed_dir
            left_eye_path = left_eye_dir / f"{sample_id}_left.png"

            # Right eye
            right_eye_dir = eyes_open_dir if right_eye_label == 1 else eyes_closed_dir
            right_eye_path = right_eye_dir / f"{sample_id}_right.png"

            # Mouth
            mouth_dir = mouth_open_dir if mouth_label == 1 else mouth_closed_dir
            mouth_path = mouth_dir / f"{sample_id}_mouth.png"

            if save_crop(outputs.left_eye_crop, left_eye_path):
                stats["left_eye_saved"] += 1
            if save_crop(outputs.right_eye_crop, right_eye_path):
                stats["right_eye_saved"] += 1
            if save_crop(outputs.mouth_crop, mouth_path):
                stats["mouth_saved"] += 1

            frame_index += 1

            if max_frames is not None and processed_frames >= max_frames:
                break

    finally:
        cap.release()
        extractor.close()
        ear_file.close()

    return stats


def process_videos_in_directory(
    videos_dir: str | Path,
    model_path: str | Path,
    output_root: str | Path = "cropped_images_test",
    sample_every_n_frames: int = 1,
    max_frames: Optional[int] = None,
    flip_horizontal: bool = False,
) -> dict[str, dict[str, int]]:
    videos_dir = Path(videos_dir)
    output_root = Path(output_root)

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    video_files = sorted(
        [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    )

    print(f"\nFound {len(video_files)} video(s) to process:")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    print()

    if not video_files:
        raise FileNotFoundError(f"No supported video files found in: {videos_dir}")

    all_results = {}

    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        results = process_video_to_region_dataset(
            video_path=video_path,
            model_path=model_path,
            output_root=output_root,
            sample_every_n_frames=sample_every_n_frames,
            max_frames=max_frames,
            flip_horizontal=flip_horizontal,
        )
        all_results[video_path.name] = results

        print(f"Done: {video_path.name}")
        for key, value in results.items():
            print(f"  {key}: {value}")

    return all_results


if __name__ == "__main__":
    MODEL_PATH = Path("vision_pipeline/models/face_landmarker.task")
    VIDEOS_DIR = Path("videos")
    OUTPUT_ROOT = Path("dataset")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Missing videos directory: {VIDEOS_DIR}")

    all_results = process_videos_in_directory(
        videos_dir=VIDEOS_DIR,
        model_path=MODEL_PATH,
        output_root=OUTPUT_ROOT,
        sample_every_n_frames=15,
        max_frames=None,
        flip_horizontal=False,
    )

    print("\nAll videos done.")