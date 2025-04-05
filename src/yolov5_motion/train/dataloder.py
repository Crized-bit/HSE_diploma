import json
import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
from pathlib import Path
import shutil
import concurrent.futures
from tqdm import tqdm


def create_control_image(prev_image: np.ndarray, cur_image: np.ndarray) -> np.ndarray:
    """
    Create a control image from previous and current frames.

    Args:
        prev_image: Previous frame as a numpy array (H, W, C)
        cur_image: Current frame as a numpy array (H, W, C)

    Returns:
        Control image as a numpy array (H, W, C)
    """
    # Convert images to grayscale
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
    cur_gray = cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to RGB using HSV color wheel
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((prev_image.shape[0], prev_image.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow


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

    # Create a list of frames to extract
    frames_to_extract = []

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

    # Define a function to extract and save a single frame
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

                # Skip frames that already exist
                video_id = items[0]["video_id"]
                video_output_dir = frames_path / video_id
                video_output_dir.mkdir(exist_ok=True)

                control_output_dir = control_path / video_id
                control_output_dir.mkdir(exist_ok=True)

                if not frame_indices:
                    continue

                # Get fps from annotation file (same as in dataset)
                for ann_file in annotation_files:
                    with open(ann_file, "r") as f:
                        annotation = json.load(f)
                    if Path(annotation["metadata"]["data_path"]).stem == video_id:
                        fps = annotation["metadata"]["fps"]
                        break

                control_frame_indices = [max(0, int(frame_index - prev_frame_time_diff * fps)) for frame_index in frame_indices]

                # Read frames in batch
                frames = vr.get_batch(frame_indices).asnumpy()
                control_frames = vr.get_batch(control_frame_indices).asnumpy()

                for i in range(len(frames)):
                    frame = frames[i]
                    control_frame = control_frames[i]
                    frame_idx = frame_indices[i]
                    prev_frame_idx = control_frame_indices[i]

                    # Resize and pad frame
                    h, w = frame.shape[:2]
                    target_w, target_h = resize_to

                    # Calculate scaling factor to maintain aspect ratio
                    scale = min(target_w / w, target_h / h)

                    # Calculate new size
                    new_w = int(w * scale)
                    new_h = int(h * scale)

                    # Resize image
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    resized_control = cv2.resize(control_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    # Create target image with padding color
                    target_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
                    target_control = target_image.copy()  # Fixed typo in variable name

                    # Calculate padding
                    pad_w = (target_w - new_w) // 2
                    pad_h = (target_h - new_h) // 2

                    # Place resized image on target image
                    target_image[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
                    target_control[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_control

                    # Save frame as JPEG
                    frame_output_path = video_output_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_output_path), cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))

                    # Generate and save control image
                    control_image = create_control_image(target_control, target_image)
                    control_output_path = control_output_dir / f"control_{frame_idx:06d}_{prev_frame_idx:06d}.jpg"
                    cv2.imwrite(str(control_output_path), cv2.cvtColor(control_image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error processing batch for {video_path}: {e}")
                import traceback

                traceback.print_exc()  # Print full stack trace for better debugging

    # Process videos in batches
    batch_size = 100  # Number of frames to process at a time
    video_batches = [frames_to_extract[i : i + batch_size] for i in range(0, len(frames_to_extract), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_video_batch, video_batches), total=len(video_batches)))

    print(f"Preprocessing complete. Extracted frames saved to {frames_path}")
    print(f"Control images saved to {control_path}")

    # Save metadata about processed frames including control image pairs
    metadata = {"num_frames": len(frames_to_extract), "frame_size": resize_to, "prev_frame_time_diff": prev_frame_time_diff, "videos": {}}

    # Group frames by video
    for item in frames_to_extract:
        video_id = item["video_id"]
        frame_idx = item["frame_idx"]
        has_annotation = item["has_annotation"]

        if video_id not in metadata["videos"]:
            metadata["videos"][video_id] = {"frames": [], "annotation_frames": [], "resolution": video_resolutions.get(video_id, (0, 0))}

        metadata["videos"][video_id]["frames"].append(frame_idx)

        if has_annotation:
            metadata["videos"][video_id]["annotation_frames"].append(frame_idx)

    # Save metadata to output directory
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


class PreprocessedVideoDataset(Dataset):
    """
    Dataset for loading preprocessed video frames and precomputed control images from disk.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        annotations_dir: str = "./annotations",
        prev_frame_time_diff: float = None,  # Time difference in seconds, if None will use from metadata
        transform=None,
    ):
        """
        Initialize the dataset.

        Args:
            preprocessed_dir: Directory containing preprocessed frames
            annotations_dir: Directory containing annotation files
            prev_frame_time_diff: Time difference in seconds to get the previous frame
            transform: Optional transform to be applied on frames
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.frames_dir = self.preprocessed_dir / "frames"
        self.control_dir = self.preprocessed_dir / "control_images"
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        self.prev_frame_time_diff = prev_frame_time_diff

        # Load metadata
        metadata_path = self.preprocessed_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # If prev_frame_time_diff is None, use the one from metadata
        if prev_frame_time_diff is None and "prev_frame_time_diff" in self.metadata:
            self.prev_frame_time_diff = self.metadata["prev_frame_time_diff"]
            print(f"Using prev_frame_time_diff={self.prev_frame_time_diff} from metadata")

        # Get all annotation files
        self.annotation_files = list(self.annotations_dir.glob("*.json"))

        # Create a mapping of samples - each sample is a frame with annotations
        self.samples = self._create_samples()

    def _adjust_bbox_for_padding(self, bbox, original_size, padded_size=(640, 640)):
        """
        Adjust bounding box coordinates for padded image.

        Args:
            bbox: Original bounding box [center_x, center_y, width, height]
            original_size: Original image size (width, height)
            padded_size: Padded image size (width, height)

        Returns:
            Adjusted bounding box
        """
        orig_w, orig_h = original_size
        pad_w, pad_h = padded_size

        # Calculate scaling factor to maintain aspect ratio
        scale = min(pad_w / orig_w, pad_h / orig_h)

        # Calculate new size after scaling
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Calculate padding
        pad_left = (pad_w - new_w) // 2
        pad_top = (pad_h - new_h) // 2

        # Unpack bbox
        center_x, center_y, width, height = bbox

        # Scale coordinates
        center_x = center_x * scale + pad_left
        center_y = center_y * scale + pad_top
        width = width * scale
        height = height * scale

        return [center_x, center_y, width, height]

    def _create_samples(self) -> List[Dict]:
        """
        Create a list of samples from the annotation files.
        Each sample is a dictionary with video path, frame index, and annotations.
        """
        samples = []

        for ann_file in self.annotation_files:
            # Load annotation file
            with open(ann_file, "r") as f:
                annotation = json.load(f)

            # Get video file path
            video_filename = annotation["metadata"]["data_path"]
            video_id = Path(video_filename).stem

            # Skip if preprocessed directory for this video doesn't exist
            video_dir = self.frames_dir / video_id
            if not video_dir.exists():
                print(f"Warning: Preprocessed directory {video_dir} not found, skipping annotations from {ann_file}")
                continue

            # Get fps for time to frame conversion
            fps = annotation["metadata"]["fps"]

            # Get video resolution
            resolution = None
            if video_id in self.metadata["videos"]:
                resolution = self.metadata["videos"][video_id]["resolution"]
            else:
                # If not in metadata, try to get from annotation
                try:
                    resolution = (annotation["metadata"]["resolution"]["width"], annotation["metadata"]["resolution"]["height"])
                except (KeyError, TypeError):
                    print(f"Warning: Could not determine resolution for {video_id}")

            # Group annotations by frame index
            frame_annotations = {}

            for entity in annotation["entities"]:
                frame_idx = entity["blob"]["frame_idx"]

                if frame_idx not in frame_annotations:
                    frame_annotations[frame_idx] = []

                # Convert bbox from xywh (top-left) to xywh (center)
                x, y, w, h = entity["bb"]
                # Convert to center x, y
                center_x = x + w / 2
                center_y = y + h / 2

                # Store the bbox in xywh format where xy is center
                bbox = [center_x, center_y, w, h]

                # Adjust bbox for padding if resolution is available
                if resolution:
                    bbox = self._adjust_bbox_for_padding(bbox, resolution)

                frame_annotations[frame_idx].append(
                    {"bbox": bbox, "id": entity["id"], "labels": entity["labels"], "confidence": entity["confidence"]}
                )

            # Create a sample for each frame with annotations
            for frame_idx, annotations in frame_annotations.items():
                # Calculate frame time in seconds
                frame_time = frame_idx / fps

                # Calculate previous frame time
                prev_frame_time = max(0, frame_time - self.prev_frame_time_diff)

                # Calculate previous frame index
                prev_frame_idx = max(0, int(prev_frame_time * fps))

                # Check if frames exist
                current_frame_path = video_dir / f"frame_{frame_idx:06d}.jpg"
                prev_frame_path = video_dir / f"frame_{prev_frame_idx:06d}.jpg"

                # Check if precomputed control image exists
                control_image_path = self.control_dir / video_id / f"control_{frame_idx:06d}_{prev_frame_idx:06d}.jpg"

                if not current_frame_path.exists() or not prev_frame_path.exists():
                    print(f"Warning: Missing frames for {video_id}, frame {frame_idx} or {prev_frame_idx}")
                    continue

                samples.append(
                    {
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "prev_frame_idx": prev_frame_idx,
                        "annotations": annotations,
                        "current_frame_path": str(current_frame_path),
                        "prev_frame_path": str(prev_frame_path),
                        "control_image_path": str(control_image_path) if control_image_path.exists() else None,
                        "frame_time": frame_time,
                        "prev_frame_time": prev_frame_time,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Returns:
            Dictionary containing:
            - current_frame (torch.Tensor): Current frame as a tensor
            - prev_frame (torch.Tensor): Previous frame as a tensor
            - control_image (torch.Tensor): Control image generated from current and previous frames
            - annotations (List): List of annotations for the current frame
        """
        sample = self.samples[idx]

        # Load frames directly from disk
        current_frame = cv2.imread(sample["current_frame_path"])
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        prev_frame = cv2.imread(sample["prev_frame_path"])
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)

        # Load precomputed control image if available, otherwise create it
        if sample["control_image_path"] and Path(sample["control_image_path"]).exists():
            control_image = cv2.imread(sample["control_image_path"])
            control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)
            control_tensor = torch.from_numpy(control_image).permute(2, 0, 1).float() / 255.0
        else:
            # Fallback to computing control image on-the-fly if precomputed one doesn't exist
            control_image = create_control_image(prev_frame, current_frame)
            control_tensor = torch.from_numpy(control_image).permute(2, 0, 1).float() / 255.0

        # Convert frames to torch tensors (C, H, W format)
        current_frame_tensor = torch.from_numpy(current_frame).permute(2, 0, 1).float() / 255.0
        prev_frame_tensor = torch.from_numpy(prev_frame).permute(2, 0, 1).float() / 255.0

        # Apply transforms if specified
        if self.transform:
            current_frame_tensor = self.transform(current_frame_tensor)
            prev_frame_tensor = self.transform(prev_frame_tensor)

        return {
            "current_frame": current_frame_tensor,
            "prev_frame": prev_frame_tensor,
            "control_image": control_tensor,
            "annotations": sample["annotations"],
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of annotations per frame.
    """
    current_frames = torch.stack([item["current_frame"] for item in batch])
    prev_frames = torch.stack([item["prev_frame"] for item in batch])
    control_images = torch.stack([item["control_image"] for item in batch])
    annotations = [item["annotations"] for item in batch]

    return {"current_frames": current_frames, "prev_frames": prev_frames, "control_images": control_images, "annotations": annotations}


def get_dataloader(
    preprocessed_dir,
    annotations_dir="./annotations",
    batch_size=8,
    shuffle=True,
    num_workers=4,
    prev_frame_time_diff=1.0,  # Time difference in seconds
    transform=None,
):
    """
    Create a DataLoader for the preprocessed video dataset.

    Args:
        preprocessed_dir: Directory containing preprocessed frames
        annotations_dir: Directory containing annotation files
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for data loading
        prev_frame_time_diff: Time difference in seconds to get the previous frame
        transform: Optional transform to be applied on frames

    Returns:
        DataLoader instance
    """
    dataset = PreprocessedVideoDataset(
        preprocessed_dir=preprocessed_dir, annotations_dir=annotations_dir, prev_frame_time_diff=prev_frame_time_diff, transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    return dataloader


# Example usage
if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess videos and benchmark dataloader")
    parser.add_argument(
        "--videos_dir", type=str, default="/home/jovyan/p.kudrevatyh/yolov5_motion/data/videos", help="Directory containing video files"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations",
        help="Directory containing annotation files",
    )
    parser.add_argument("--output_dir", type=str, default="/home/jovyan/p.kudrevatyh/temp", help="Directory to save preprocessed frames")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--num_iterations", type=int, default=30, help="Number of iterations for benchmark")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--prev_frame_time_diff", type=float, default=1.0, help="Time difference in seconds to get the previous frame")
    args = parser.parse_args()

    if True:
        print("Preprocessing videos...")
        preprocess_videos(
            videos_dir=args.videos_dir,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            prev_frame_time_diff=args.prev_frame_time_diff,
        )

    print("\nBenchmarking dataloader...")

    # Create dataloader
    dataloader = get_dataloader(
        preprocessed_dir=args.output_dir,
        annotations_dir=args.annotations_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prev_frame_time_diff=args.prev_frame_time_diff,
    )

    # Warm-up
    print("Warming up...")
    for _ in range(min(2, len(dataloader))):
        next(iter(dataloader))

    # Time dataloader
    print(f"Benchmarking with {args.num_iterations} iterations...")
    start = time.time()
    num_batches = 0

    for batch in dataloader:
        num_batches += 1
        if num_batches == args.num_iterations:
            break

    end = time.time()

    print(f"\nBenchmark Results:")
    print(f"Time for {num_batches} batches: {end - start:.2f} seconds")
    print(f"Batches per second: {num_batches / (end - start):.2f}")
    print(f"Frames per second: {num_batches * args.batch_size / (end - start):.2f}")
