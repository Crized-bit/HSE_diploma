import json
import numpy as np
import cv2
from decord import VideoReader, cpu
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

    # Organize frames by video
    frames_by_video = {}
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

        if video_id not in frames_by_video:
            frames_by_video[video_id] = {"video_path": str(video_path), "frames": set(), "fps": fps}

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

        # Add frames to video's list
        frames_by_video[video_id]["frames"].update(frame_indices)

    # Count total frames to process
    total_frames = sum(len(video_info["frames"]) for video_info in frames_by_video.values())
    print(f"Will extract {total_frames} frames from {len(frames_by_video)} videos")

    # Create a list of videos to process in parallel
    videos_to_process = []
    for video_id, video_info in frames_by_video.items():
        videos_to_process.append(
            {
                "video_id": video_id,
                "video_path": video_info["video_path"],
                "frames": sorted(list(video_info["frames"])),
                "fps": video_info["fps"],
                "has_annotations": frames_with_annotations.get(video_id, set()),
            }
        )

    # Define a function to process an entire video
    def process_video(video_item):
        video_id = video_item["video_id"]
        video_path = video_item["video_path"]
        frames_to_extract = video_item["frames"]
        fps = video_item["fps"]
        has_annotations = video_item["has_annotations"]

        try:
            print(f"Processing video {video_id} with {len(frames_to_extract)} frames...")

            # Create output directories for this video
            video_output_dir = frames_path / video_id
            video_output_dir.mkdir(exist_ok=True)

            control_output_dir = control_path / video_id
            control_output_dir.mkdir(exist_ok=True)

            # Open video
            vr = VideoReader(video_path, ctx=cpu(0))

            # Compute control frame pairs
            control_frame_pairs = []
            for frame_idx in frames_to_extract:
                frame_time = frame_idx / fps
                prev_frame_time = max(0, frame_time - prev_frame_time_diff)
                prev_frame_idx = max(0, int(prev_frame_time * fps))
                control_frame_pairs.append((frame_idx, prev_frame_idx))

            # Get unique indices to read (includes both frames_to_extract and their control frames)
            all_indices = set(frames_to_extract)
            all_indices = sorted(all_indices)

            # Read all needed frames in smaller chunks to avoid memory issues
            chunk_size = 500  # Process 100 frames at a time
            frames_dict = {}

            for i in range(0, len(all_indices), chunk_size):
                chunk_indices = all_indices[i : i + chunk_size]
                try:
                    batch_frames = vr.get_batch(chunk_indices).asnumpy()
                    for j, idx in enumerate(chunk_indices):
                        frames_dict[idx] = batch_frames[j]
                except Exception as e:
                    print(f"Error reading frames chunk from {video_path}: {e}")

            # Process each frame and its control pair
            for frame_idx in tqdm(frames_to_extract, desc=f"Processing {video_id} frames"):
                if frame_idx not in frames_dict:
                    continue

                # Identify the corresponding control frame
                prev_frame_idx = None
                for curr, prev in control_frame_pairs:
                    if curr == frame_idx:
                        prev_frame_idx = prev
                        break

                if prev_frame_idx is None or prev_frame_idx not in frames_dict:
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

                # Save previous frame if it has annotations
                if prev_frame_idx in has_annotations:
                    prev_frame_output_path = video_output_dir / f"frame_{prev_frame_idx:06d}.jpg"
                    cv2.imwrite(str(prev_frame_output_path), cv2.cvtColor(target_prev, cv2.COLOR_RGB2BGR))

                # Generate and save control image
                control_image = create_control_image(target_prev, target_image)
                control_output_path = control_output_dir / f"control_{frame_idx:06d}_{prev_frame_idx:06d}.jpg"
                cv2.imwrite(str(control_output_path), cv2.cvtColor(control_image, cv2.COLOR_RGB2BGR))

            # Clear memory
            del frames_dict

            return video_id, len(frames_to_extract)

        except Exception as e:
            print(f"Error processing video {video_id} ({video_path}): {e}")
            import traceback

            traceback.print_exc()
            return video_id, 0

    # Process videos in parallel
    print(f"Processing {len(videos_to_process)} videos with {num_workers} parallel workers...")
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {executor.submit(process_video, video_item): video_item["video_id"] for video_item in videos_to_process}

        for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(videos_to_process), desc="Processing videos"):
            video_id = future_to_video[future]
            try:
                processed_video, processed_frames = future.result()
                results.append((processed_video, processed_frames))
            except Exception as e:
                print(f"Video {video_id} generated an exception: {e}")

    total_processed = sum(frames for _, frames in results)
    print(f"Preprocessing complete. Extracted {total_processed} frames across {len(results)} videos.")
    print(f"Frames saved to {frames_path}")
    print(f"Control images saved to {control_path}")

    # Create a mapping of frame to previous frame for control images
    print("Creating metadata...")
    metadata = {"num_frames": total_processed, "frame_size": resize_to, "prev_frame_time_diff": prev_frame_time_diff, "videos": {}}

    # Pre-process annotation files to create mappings for faster lookup
    video_fps_mapping = {}
    for ann_file in annotation_files:
        with open(ann_file, "r") as f:
            annotation = json.load(f)
        video_id = Path(annotation["metadata"]["data_path"]).stem
        fps = annotation["metadata"]["fps"]
        video_fps_mapping[video_id] = fps

    # Process each video's frames for metadata
    for video_id, video_info in tqdm(frames_by_video.items(), desc="Building metadata"):
        if video_id not in metadata["videos"]:
            metadata["videos"][video_id] = {
                "frames": sorted(list(video_info["frames"])),
                "annotation_frames": sorted(list(frames_with_annotations.get(video_id, set()))),
                "resolution": video_resolutions.get(video_id, (0, 0)),
                "frame_to_prev_frame": {},
            }

        # Get fps for this video
        fps = video_fps_mapping.get(video_id)
        if fps is None:
            print(f"Warning: Could not find fps for video {video_id}, skipping metadata")
            continue

        # Process annotation frames for this video to create frame_to_prev_frame mapping
        for frame_idx in frames_with_annotations.get(video_id, set()):
            # Compute previous frame index
            frame_time = frame_idx / fps
            prev_frame_time = max(0, frame_time - prev_frame_time_diff)
            prev_frame_idx = max(0, int(prev_frame_time * fps))

            # Store the mapping
            metadata["videos"][video_id]["frame_to_prev_frame"][str(frame_idx)] = prev_frame_idx

    # Save metadata to output directory
    print("Saving metadata...")
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path
