import json
import numpy as np
import torch

# import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
from pathlib import Path
import albumentations as A
from yolov5_motion.config import my_config


class PreprocessedVideoDataset(Dataset):
    """
    Dataset for loading preprocessed video frames and precomputed control images from disk.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        annotations_dir: str = "./annotations",
        augment: bool = False,
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
        self.augment = augment
        # Setup augmentations if enabled
        if self.augment:
            self.aug_transform = A.Compose(
                [
                    # Spatial augmentations that apply to both image and control
                    A.HorizontalFlip(p=my_config.training.augment_prob),
                    # Noise and blur
                    A.OneOf(
                        [
                            A.GaussNoise(p=1.0),
                            A.GaussianBlur(p=1.0),
                        ],
                        p=my_config.training.augment_prob * 0.5,
                    ),  # Lower probability for these
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=1.0, ensure_safe_range=True),
                            A.HueSaturationValue(p=1.0),
                            A.RGBShift(p=1.0),
                        ],
                        p=my_config.training.augment_prob,
                    ),
                    A.RandomResizedCrop(size=(640, 640), scale=(0.6, 1.0), p=my_config.training.augment_prob),
                    A.Rotate(limit=15, p=my_config.training.augment_prob, fill=(114, 114, 114), fill_mask=(114, 114, 114)),
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    clip=True,
                    filter_invalid_bboxes=True,
                    min_visibility=my_config.data.bbox_skip_percentage,
                ),
            )
        # Load metadata
        metadata_path = self.preprocessed_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Get all annotation files
        self.annotation_files = list(self.annotations_dir.glob("*.json"))

        # Create a mapping of samples - each sample is a frame with annotations
        self.samples = self._create_samples()

    def _adjust_bbox(self, bbox, original_size):
        """
        Normalize bbox
        """
        orig_w, orig_h = original_size

        # Unpack and scale bbox
        center_x, center_y, width, height = bbox
        center_x = center_x / orig_w
        center_y = center_y / orig_h
        width = width / orig_w
        height = height / orig_h

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

            frames_to_skip = set()
            for entity in annotation["entities"]:
                # skip, but mark crowd annotations
                frame_idx = entity["blob"]["frame_idx"]

                if "labels" in entity and isinstance(entity["labels"], dict):
                    if entity["labels"].get("crowd") == 1:
                        frames_to_skip.add(frame_idx)
                        continue

                if (
                    entity["labels"].get("fully_occluded") == 1
                    or entity["labels"].get("reflection") == 1
                    or entity["labels"].get("severly_occluded_person") == 1
                ):
                    continue

                # Get rid of frames w.o. normal control images
                if frame_idx <= fps * my_config.data.prev_frame_time_diff * my_config.data.control_stack_length:
                    # print(f"Skipping frame {frame_idx} for video {video_id}")
                    frames_to_skip.add(frame_idx)
                    continue

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
                    bbox = self._adjust_bbox(bbox, resolution)
                    if not bbox:
                        continue

                frame_annotations[frame_idx].append(
                    {
                        "bbox": bbox,
                        "id": entity["id"],
                        "labels": entity["labels"],
                        "confidence": entity.get("confidence", 1),
                    }
                )

            # Create a sample for each frame with annotations
            for frame_idx, annotations in frame_annotations.items():
                if frame_idx in frames_to_skip:
                    continue
                # Construct file paths
                current_frame_path = video_dir / f"frame_{frame_idx:06d}.png"
                control_image_path = self.control_dir / video_id / f"control_{frame_idx:06d}.png"

                # Skip if files don't exist
                if not current_frame_path.exists():
                    print(f"Warning: Missing current frame for {video_id}, frame {frame_idx}")
                    continue

                if not control_image_path.exists():
                    print(f"Warning: Missing control image for {video_id}, frame {frame_idx}")
                    continue

                samples.append(
                    {
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "annotations": annotations,
                        "current_frame_path": str(current_frame_path),
                        "control_image_path": str(control_image_path),
                        "frame_time": frame_idx / fps,
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
        control_image = cv2.imread(sample["control_image_path"], cv2.IMREAD_UNCHANGED)
        control_image[..., :3] = control_image[..., 2::-1]
        # control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)

        # Extract bboxes for augmentation
        bboxes = []
        for ann in sample["annotations"]:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, w, h])

        # Apply augmentations if enabled
        if self.augment and bboxes:
            augmented = self.aug_transform(
                image=current_frame, masks=[control_image], bboxes=bboxes, class_labels=[0] * len(sample["annotations"])
            )

            current_frame = augmented["image"]
            control_image = augmented["masks"][0]

            # Update annotations with augmented bboxes
            for i, bbox in enumerate(augmented["bboxes"]):
                x, y, w, h = bbox
                bboxes[i] = [x, y, w, h]

        # Create target image with padding color
        current_frame_padded = np.ones((my_config.model.img_size, my_config.model.img_size, 3), dtype=np.uint8) * np.array(
            114, dtype=np.uint8
        )
        control_image_padded = np.ones((my_config.model.img_size, my_config.model.img_size, control_image.shape[2]), dtype=np.uint8) * np.array(
            114, dtype=np.uint8
        )

        # Calculate padding
        pad_w = (my_config.model.img_size - current_frame.shape[1]) // 2
        pad_h = (my_config.model.img_size - current_frame.shape[0]) // 2

        # Place resized image on target image
        current_frame_padded[pad_h : pad_h + current_frame.shape[0], pad_w : pad_w + current_frame.shape[1]] = current_frame
        control_image_padded[pad_h : pad_h + control_image.shape[0], pad_w : pad_w + control_image.shape[1]] = control_image

        final_boxes = []
        for i, box in enumerate(bboxes):
            x, y, w, h = box

            initial_space = w * h

            top_left_x = max(x - w / 2, 0)
            top_left_y = max(y - h / 2, 0)
            bottom_right_x = min(x + w / 2, 1)
            bottom_right_y = min(y + h / 2, 1)

            x = (top_left_x + bottom_right_x) / 2
            y = (top_left_y + bottom_right_y) / 2
            w = bottom_right_x - top_left_x
            h = bottom_right_y - top_left_y

            result_space = w * h
            ratio = result_space / initial_space

            if ratio <= my_config.data.bbox_skip_percentage:
                continue

            x = (x * current_frame.shape[1] + pad_w) / my_config.model.img_size
            y = (y * current_frame.shape[0] + pad_h) / my_config.model.img_size
            w = w * current_frame.shape[1] / my_config.model.img_size
            h = h * current_frame.shape[0] / my_config.model.img_size
            # Convert back to center format
            final_boxes.append([x, y, w, h])

        # Convert frames to torch tensors (C, H, W format)
        current_frame_tensor = torch.from_numpy(current_frame_padded).permute(2, 0, 1).float() / 255.0
        control_tensor = torch.from_numpy(control_image_padded).permute(2, 0, 1).float() / 255.0

        return {
            "current_frame": current_frame_tensor,
            "control_image": control_tensor,
            "annotations": final_boxes,
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of annotations per frame.
    """
    current_frames = torch.stack([item["current_frame"] for item in batch])
    control_images = torch.stack([item["control_image"] for item in batch])
    annotations = [item["annotations"] for item in batch]

    return {"current_frames": current_frames, "control_images": control_images, "annotations": annotations}
