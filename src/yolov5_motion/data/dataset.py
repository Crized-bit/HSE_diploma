import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
from pathlib import Path


class PreprocessedVideoDataset(Dataset):
    """
    Dataset for loading preprocessed video frames and precomputed control images from disk.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        annotations_dir: str = "./annotations",
        prev_frame_time_diff: float = None,  # Time difference in seconds
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
                # Skip reflections
                if "labels" in entity and isinstance(entity["labels"], dict) and entity["labels"].get("reflection") == 1:
                    continue  # Skip this annotation

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
                # Look up the prev_frame_idx from metadata if available
                prev_frame_idx = None

                # Try to get from metadata
                if (
                    video_id in self.metadata["videos"]
                    and "frame_to_prev_frame" in self.metadata["videos"][video_id]
                    and str(frame_idx) in self.metadata["videos"][video_id]["frame_to_prev_frame"]
                ):
                    prev_frame_idx = self.metadata["videos"][video_id]["frame_to_prev_frame"][str(frame_idx)]
                else:
                    # Calculate frame time in seconds and previous frame index
                    frame_time = frame_idx / fps
                    prev_frame_time = max(0, frame_time - self.prev_frame_time_diff)
                    prev_frame_idx = max(0, int(prev_frame_time * fps))

                # Construct file paths
                current_frame_path = video_dir / f"frame_{frame_idx:06d}.jpg"
                control_image_path = self.control_dir / video_id / f"control_{frame_idx:06d}_{prev_frame_idx:06d}.jpg"

                # Skip if files don't exist
                if not current_frame_path.exists():
                    print(f"Warning: Missing current frame for {video_id}, frame {frame_idx}")
                    continue

                if not control_image_path.exists():
                    print(f"Warning: Missing control image for {video_id}, frame {frame_idx}, prev {prev_frame_idx}")
                    continue

                samples.append(
                    {
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "prev_frame_idx": prev_frame_idx,
                        "annotations": annotations,
                        "current_frame_path": str(current_frame_path),
                        "control_image_path": str(control_image_path),
                        "frame_time": frame_idx / fps,
                        "prev_frame_time": prev_frame_idx / fps if prev_frame_idx is not None else None,
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
            - control_image (torch.Tensor): Control image as a tensor
            - annotations (List): List of annotations for the current frame
        """
        sample = self.samples[idx]

        # Load current frame
        current_frame = cv2.imread(sample["current_frame_path"])
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # Load precomputed control image
        control_image = cv2.imread(sample["control_image_path"])
        control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)

        # Convert frames to torch tensors (C, H, W format)
        current_frame_tensor = torch.from_numpy(current_frame).permute(2, 0, 1).float() / 255.0
        control_tensor = torch.from_numpy(control_image).permute(2, 0, 1).float() / 255.0

        # Apply transforms if specified
        if self.transform:
            current_frame_tensor = self.transform(current_frame_tensor)
            control_tensor = self.transform(control_tensor)

        return {
            "current_frame": current_frame_tensor,
            "control_image": control_tensor,
            "annotations": sample["annotations"],
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of annotations per frame.
    """
    current_frames = torch.stack([item["current_frame"] for item in batch])
    control_images = torch.stack([item["control_image"] for item in batch])
    annotations = [item["annotations"] for item in batch]

    return {"current_frames": current_frames, "control_images": control_images, "annotations": annotations}


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
