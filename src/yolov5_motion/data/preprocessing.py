import json
import cv2
from decord import VideoReader, cpu
from pathlib import Path
import shutil
import concurrent.futures
from tqdm import tqdm
from yolov5_motion.data.utils import create_control_image
from yolov5_motion.config import my_config

def preprocess_videos(
    videos_dir,
    annotations_dir,
    output_dir,
    control_mode: str = "flow",
):
    """
    Preprocess videos by extracting frames and computing control images, saving them as individual images.

    Args:
        annotations_dir: Directory containing annotation files
        output_dir: Directory to save extracted frames
        pad_color: Padding color (RGB)
        prev_frame_time_diff: Time difference in seconds between frames for motion computation
    """
    output_path = Path(output_dir)
    frames_path = output_path / "frames"
    control_path = output_path / "control_images"

    # Create output directories if they don't exist
    if output_path.exists():
        print(f"Output directory {output_path} already exists.")
        exit(1)

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

        try:
            print(f"Processing video {video_id} with {len(frames_to_extract)} frames...")
            print(f"Using {my_config.data.control_stack_length} control frames for each target frame")

            # Create output directories for this video
            video_output_dir = frames_path / video_id
            video_output_dir.mkdir(exist_ok=True)

            control_output_dir = control_path / video_id
            control_output_dir.mkdir(exist_ok=True)

            # Open video
            vr = VideoReader(video_path, ctx=cpu(0))

            # Compute multiple control frames for each target frame
            all_control_frames = []
            for frame_idx in frames_to_extract:
                frame_time = frame_idx / fps
                control_frames = set()

                for i in range(my_config.data.control_stack_length):
                    # Calculate time for each control frame going back in time
                    # The total time span will be prev_frame_time_diff * n_control_frames
                    control_time_diff = my_config.data.prev_frame_time_diff * (i + 1)
                    control_frame_time = max(0, frame_time - control_time_diff)
                    control_frame_idx = max(0, int(control_frame_time * fps))
                    control_frames.add(control_frame_idx)

                all_control_frames.append((frame_idx, sorted(list(control_frames))))

            # Get unique indices to read (includes both frames_to_extract and all their control frames)
            all_indices = set(frames_to_extract)
            for _, control_frames in all_control_frames:
                all_indices.update(control_frames)
            all_indices = sorted(all_indices)

            # Read all needed frames in smaller chunks to avoid memory issues
            chunk_size = 500  # Process frames in chunks to avoid memory issues
            frames_dict = {}

            for i in range(0, len(all_indices), chunk_size):
                chunk_indices = all_indices[i : i + chunk_size]
                try:
                    batch_frames = vr.get_batch(chunk_indices).asnumpy()
                    for j, idx in enumerate(chunk_indices):
                        frames_dict[idx] = batch_frames[j]
                except Exception as e:
                    print(f"Error reading frames chunk from {video_path}: {e}")

            # Process each frame and its control frames
            for frame_idx in tqdm(frames_to_extract, desc=f"Processing {video_id} frames"):
                if frame_idx not in frames_dict:
                    continue

                # Get the corresponding control frames
                control_frames_indices = None
                for curr, controls in all_control_frames:
                    if curr == frame_idx:
                        control_frames_indices = controls
                        break

                if control_frames_indices is None:
                    continue

                # Get the target frame
                frame = frames_dict[frame_idx]

                # Resize and prepare the target frame
                h, w = frame.shape[:2]
                target_w, target_h = my_config.model.img_size, my_config.model.img_size

                # Calculate scaling factor to maintain aspect ratio
                scale = min(target_w / w, target_h / h)

                # Calculate new size
                new_w = int(w * scale)
                new_h = int(h * scale)

                # Resize target frame
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Save frame as JPEG
                frame_output_path = video_output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_output_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

                # Process each control frame
                preprocessed_control_frames = []
                for i, control_frame_idx in enumerate(control_frames_indices):
                    if control_frame_idx not in frames_dict:
                        raise ValueError(f"Control frame {control_frame_idx} not found in frames_dict")
                        # continue

                    control_frame = frames_dict[control_frame_idx]
                    # Resize control frame
                    resized_control = cv2.resize(control_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    preprocessed_control_frames.append(resized_control)

                # Generate and save control image
                control_image = create_control_image(preprocessed_control_frames, resized, control_mode)
                control_image[..., :3] = control_image[..., 2::-1]
                # Save with index to differentiate between multiple control frames
                control_output_path = control_output_dir / f"control_{frame_idx:06d}.png"
                cv2.imwrite(str(control_output_path), control_image)

            # Clear memory
            del frames_dict

            return video_id, len(frames_to_extract)
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            return video_id, 0

        except Exception as e:
            print(f"Error processing video {video_id} ({video_path}): {e}")
            import traceback

            traceback.print_exc()
            return video_id, 0

    # Process videos in parallel
    print(f"Processing {len(videos_to_process)} videos with {my_config.training.workers} parallel workers...")
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=my_config.training.workers) as executor:
        future_to_video = {
            executor.submit(process_video, video_item): video_item["video_id"] for video_item in videos_to_process
        }

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
    metadata = {"num_frames": total_processed, "prev_frame_time_diff": my_config.data.prev_frame_time_diff, "videos": {}}

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
            }

    # Save metadata to output directory
    print("Saving metadata...")
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path
