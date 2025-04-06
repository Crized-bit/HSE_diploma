import json
import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
from typing import Dict, List, Any
from pathlib import Path
import shutil
import concurrent.futures
from tqdm import tqdm
from yolov5_motion.data.utils import create_control_image


def preprocess_videos(
    videos_dir, annotations_dir, output_dir, resize_to=(640, 640), pad_color=(114, 114, 114), num_workers=8, prev_frame_time_diff=1.0
):
    """
    Preprocess videos by extracting frames and computing control images, saving them as individual images.

    Args:
        videos_dir: Directory containing video files
        annotations_dir: Directory containing annotation files
        output_dir: Directory to save extracted frames
        resize_to: Target size (width, height) for resizing
        pad_color: Padding color (RGB)
        num_workers: Number of workers for parallel processing
        prev_frame_time_diff: Time difference in seconds between frames for motion computation
    """
    output_path = Path(output_dir)
    frames_path = output_path / "frames"
    control_path = output_path / "control_images"

    # Create output directories if they don't exist
    if output_path.exists():
        print(f"Output directory {output_path} already exists. Removing...")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)
    control_path.mkdir(parents=True, exist_ok=True)

    # Get all annotation files
    annotation_files = list(Path(annotations_dir).glob("*.json"))
    print(f"Found {len(annotation_files)} annotation files")

    # Save resolutions for each video for proper bbox adjustment
    video_resolutions = {}

    for ann_file in annotation_files:
        # Load annotation file
        with open(ann_file, "r") as f:
            annotation = json.load(f)

        # Get video file path
        video_filename = annotation["metadata"]["data_path"]
        video_path = Path(videos_dir) / video_filename

        # Skip if video file doesn't exist
        if not video_path.exists():
            print(f"Warning: Video file {video_path} not found, skipping annotations from {ann_file}")
            continue

        # Get video resolution
        resolution = (annotation["metadata"]["resolution"]["width"], annotation["metadata"]["resolution"]["height"])

        # Store resolution
        video_id = Path(video_filename).stem
        video_resolutions[video_id] = resolution

    # Create a list of frames to extract
    frames_to_extract = []
    frames_with_annotations = {}  # To track frames that have annotations

    for ann_file in annotation_files:
        # Load annotation file
        with open(ann_file, "r") as f:
            annotation = json.load(f)

        # Get video file path
        video_filename = annotation["metadata"]["data_path"]
        video_path = Path(videos_dir) / video_filename

        # Skip if video file doesn't exist
        if not video_path.exists():
            print(f"Warning: Video file {video_path} not found, skipping annotations from {ann_file}")
            continue

        # Get fps for time to frame conversion
        fps = annotation["metadata"]["fps"]

        # Extract unique frame indices and track which ones have annotations
        frame_indices = set()
        video_id = Path(video_filename).stem

        if video_id not in frames_with_annotations:
            frames_with_annotations[video_id] = set()

        for entity in annotation["entities"]:
            frame_idx = entity["blob"]["frame_idx"]
            frame_indices.add(frame_idx)
            frames_with_annotations[video_id].add(frame_idx)  # Track frames with annotations

            # Calculate frame time in seconds
            frame_time = frame_idx / fps

            # Calculate previous frame time based on time difference parameter
            prev_frame_time = max(0, frame_time - prev_frame_time_diff)

            # Calculate previous frame index
            prev_frame_idx = max(0, int(prev_frame_time * fps))
            frame_indices.add(prev_frame_idx)

        # Add frames to extraction list
        for frame_idx in frame_indices:
            frames_to_extract.append(
                {
                    "video_path": str(video_path),
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "has_annotation": frame_idx in frames_with_annotations[video_id],
                }
            )

    print(f"Extracting {len(frames_to_extract)} frames from videos...")

    # Define a function to extract and save frames from a batch of videos
    def process_video_batch(video_items):
        if not video_items:
            return

        # Group items by video path
        by_video = {}
        for item in video_items:
            video_path = item["video_path"]
            if video_path not in by_video:
                by_video[video_path] = []
            by_video[video_path].append(item)

        # Process each video
        for video_path, items in by_video.items():
            try:
                # Open video
                vr = VideoReader(video_path, ctx=cpu(0))

                # Get all frame indices for this video
                frame_indices = [item["frame_idx"] for item in items]

                # Create output directories if they don't exist
                video_id = items[0]["video_id"]
                video_output_dir = frames_path / video_id
                video_output_dir.mkdir(exist_ok=True)

                control_output_dir = control_path / video_id
                control_output_dir.mkdir(exist_ok=True)

                if not frame_indices:
                    continue

                # Get fps from annotation file
                fps = None
                for ann_file in annotation_files:
                    with open(ann_file, "r") as f:
                        annotation = json.load(f)
                    if Path(annotation["metadata"]["data_path"]).stem == video_id:
                        fps = annotation["metadata"]["fps"]
                        break

                if fps is None:
                    print(f"Warning: Could not find fps for video {video_id}, skipping")
                    continue

                # Compute control frame indices based on time difference
                control_frame_pairs = []
                for frame_idx in frame_indices:
                    frame_time = frame_idx / fps
                    prev_frame_time = max(0, frame_time - prev_frame_time_diff)
                    prev_frame_idx = max(0, int(prev_frame_time * fps))
                    control_frame_pairs.append((frame_idx, prev_frame_idx))

                # Get unique indices to read (includes both frame_indices and their corresponding control frames)
                all_indices = set(frame_indices)
                for _, prev_idx in control_frame_pairs:
                    all_indices.add(prev_idx)
                all_indices = sorted(all_indices)

                # Read all needed frames at once
                frames_dict = {}
                try:
                    batch_frames = vr.get_batch(all_indices).asnumpy()
                    for i, idx in enumerate(all_indices):
                        frames_dict[idx] = batch_frames[i]
                except Exception as e:
                    print(f"Error reading frames from {video_path}: {e}")
                    continue

                # Process each frame and its control pair
                for frame_idx, prev_frame_idx in control_frame_pairs:
                    if frame_idx not in frames_dict or prev_frame_idx not in frames_dict:
                        continue

                    frame = frames_dict[frame_idx]
                    prev_frame = frames_dict[prev_frame_idx]

                    # Resize and pad frame
                    h, w = frame.shape[:2]
                    target_w, target_h = resize_to

                    # Calculate scaling factor to maintain aspect ratio
                    scale = min(target_w / w, target_h / h)

                    # Calculate new size
                    new_w = int(w * scale)
                    new_h = int(h * scale)

                    # Resize images
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    resized_prev = cv2.resize(prev_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    # Create target image with padding color
                    target_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
                    target_prev = np.ones((target_h, target_w, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)

                    # Calculate padding
                    pad_w = (target_w - new_w) // 2
                    pad_h = (target_h - new_h) // 2

                    # Place resized image on target image
                    target_image[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
                    target_prev[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_prev

                    # Save frame as JPEG
                    frame_output_path = video_output_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_output_path), cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))

                    # Save previous frame if it has annotations (we might need it later)
                    if prev_frame_idx in frames_with_annotations.get(video_id, set()):
                        prev_frame_output_path = video_output_dir / f"frame_{prev_frame_idx:06d}.jpg"
                        cv2.imwrite(str(prev_frame_output_path), cv2.cvtColor(target_prev, cv2.COLOR_RGB2BGR))

                    # Generate and save control image
                    control_image = create_control_image(target_prev, target_image)
                    control_output_path = control_output_dir / f"control_{frame_idx:06d}_{prev_frame_idx:06d}.jpg"
                    cv2.imwrite(str(control_output_path), cv2.cvtColor(control_image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error processing batch for {video_path}: {e}")
                import traceback

                traceback.print_exc()

    # Process videos in batches
    batch_size = 100  # Number of frames to process at a time
    video_batches = [frames_to_extract[i : i + batch_size] for i in range(0, len(frames_to_extract), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_video_batch, video_batches), total=len(video_batches)))

    print(f"Preprocessing complete. Extracted frames saved to {frames_path}")
    print(f"Control images saved to {control_path}")

    # Save metadata about processed frames including control image pairs
    metadata = {"num_frames": len(frames_to_extract), "frame_size": resize_to, "prev_frame_time_diff": prev_frame_time_diff, "videos": {}}

    # Create a mapping of frame to previous frame for control images
    frame_to_prev_frame = {}

    # Group frames by video
    for item in frames_to_extract:
        video_id = item["video_id"]
        frame_idx = item["frame_idx"]
        has_annotation = item["has_annotation"]

        if video_id not in metadata["videos"]:
            metadata["videos"][video_id] = {
                "frames": [],
                "annotation_frames": [],
                "resolution": video_resolutions.get(video_id, (0, 0)),
                "frame_to_prev_frame": {},
            }

        metadata["videos"][video_id]["frames"].append(frame_idx)

        if has_annotation:
            metadata["videos"][video_id]["annotation_frames"].append(frame_idx)

            # Find fps from annotation files to compute prev_frame_idx
            for ann_file in annotation_files:
                with open(ann_file, "r") as f:
                    annotation = json.load(f)
                if Path(annotation["metadata"]["data_path"]).stem == video_id:
                    fps = annotation["metadata"]["fps"]
                    frame_time = frame_idx / fps
                    prev_frame_time = max(0, frame_time - prev_frame_time_diff)
                    prev_frame_idx = max(0, int(prev_frame_time * fps))

                    # Store the mapping
                    metadata["videos"][video_id]["frame_to_prev_frame"][str(frame_idx)] = prev_frame_idx
                    break

    # Save metadata to output directory
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path
