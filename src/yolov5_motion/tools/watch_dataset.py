import cv2
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader

# Import our dataset class
from yolov5_motion.train.dataloder import PreprocessedVideoDataset, collate_fn


def visualize_bboxes(dataset, output_dir="./debug_images", num_samples=10):
    """
    Generate debug images with visualized bounding boxes.

    Args:
        dataset: Dataset containing images and annotations
        output_dir: Directory to save debug images
        num_samples: Number of samples to visualize
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get random sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        # Get sample
        sample = dataset[idx]

        # Get current frame and annotations
        frame = sample["current_frame"].permute(1, 2, 0).numpy() * 255
        frame = frame.astype(np.uint8)
        annotations = sample["annotations"]

        # Create a copy of the frame for drawing
        frame_with_boxes = frame.copy()

        # Draw bounding boxes
        for ann in annotations:
            # Get bbox in xywh format (center)
            center_x, center_y, width, height = ann["bbox"]

            # Convert to top-left coordinates for OpenCV rectangle
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            # Draw rectangle
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label if available
            if "labels" in ann:
                label_text = str(ann["labels"])
                cv2.putText(frame_with_boxes, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save image
        output_file = output_path / f"debug_bbox_{i:03d}.jpg"
        cv2.imwrite(str(output_file), cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR))

        print(f"Saved debug image {i+1}/{len(indices)}: {output_file}")

    print(f"Debug visualization complete. Images saved to {output_path}")


def visualize_dataloader_batch(dataloader, output_dir="./debug_images"):
    """
    Visualize a batch from the dataloader.

    Args:
        dataloader: DataLoader to get a batch from
        output_dir: Directory to save debug images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get a batch
    batch = next(iter(dataloader))

    current_frames = batch["current_frames"]
    annotations = batch["annotations"]

    # Visualize each frame in the batch
    for i, (frame, anns) in enumerate(zip(current_frames, annotations)):
        # Convert frame to numpy
        frame = frame.permute(1, 2, 0).numpy() * 255
        frame = frame.astype(np.uint8)

        # Create a copy of the frame for drawing
        frame_with_boxes = frame.copy()

        # Draw bounding boxes
        for ann in anns:
            # Get bbox in xywh format (center)
            center_x, center_y, width, height = ann["bbox"]

            # Convert to top-left coordinates for OpenCV rectangle
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            # Draw rectangle
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label if available
            if "labels" in ann:
                label_text = str(ann["labels"])
                cv2.putText(frame_with_boxes, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save image
        output_file = output_path / f"batch_bbox_{i:03d}.jpg"
        cv2.imwrite(str(output_file), cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR))

    print(f"Batch visualization complete. Images saved to {output_path}")


def visualize_before_after_preprocessing(videos_dir, annotations_dir, preprocessed_dir, output_dir="./debug_images", num_samples=5):
    """
    Compare original frames with preprocessed frames and visualize bounding boxes on both.

    Args:
        videos_dir: Directory containing original videos
        annotations_dir: Directory containing annotations
        preprocessed_dir: Directory containing preprocessed frames
        output_dir: Directory to save debug images
        num_samples: Number of samples to visualize
    """
    from decord import VideoReader, cpu
    import json

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get annotation files
    annotation_files = list(Path(annotations_dir).glob("*.json"))

    # Load metadata
    metadata_path = Path(preprocessed_dir) / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Pick random annotation files
    selected_ann_files = random.sample(annotation_files, min(num_samples, len(annotation_files)))

    for ann_file in selected_ann_files:
        # Load annotation file
        with open(ann_file, "r") as f:
            annotation = json.load(f)

        # Get video file path
        video_filename = annotation["metadata"]["data_path"]
        video_path = Path(videos_dir) / video_filename
        video_id = Path(video_filename).stem

        # Skip if video file doesn't exist
        if not video_path.exists():
            print(f"Warning: Video file {video_path} not found, skipping")
            continue

        # Get a random frame with annotations
        frame_indices = []
        for entity in annotation["entities"]:
            frame_idx = entity["blob"]["frame_idx"]
            frame_indices.append(frame_idx)

        if not frame_indices:
            print(f"No frames with annotations in {ann_file}")
            continue

        # Pick a random frame
        frame_idx = random.choice(frame_indices)

        # Get annotations for this frame
        frame_annotations = []
        for entity in annotation["entities"]:
            if entity["blob"]["frame_idx"] == frame_idx:
                x, y, w, h = entity["bb"]
                center_x = x + w / 2
                center_y = y + h / 2
                frame_annotations.append({"bbox": [center_x, center_y, w, h], "id": entity["id"], "labels": entity.get("labels", {})})

        # Load original frame from video
        vr = VideoReader(str(video_path), ctx=cpu(0))
        original_frame = vr[frame_idx].asnumpy()

        # Draw bounding boxes on original frame
        original_with_boxes = original_frame.copy()
        for ann in frame_annotations:
            center_x, center_y, width, height = ann["bbox"]

            # Convert to top-left coordinates for OpenCV rectangle
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            # Draw rectangle
            cv2.rectangle(original_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Load preprocessed frame
        preprocessed_frame_path = Path(preprocessed_dir) / video_id / f"frame_{frame_idx:06d}.jpg"
        if not preprocessed_frame_path.exists():
            print(f"Preprocessed frame {preprocessed_frame_path} not found, skipping")
            continue

        preprocessed_frame = cv2.imread(str(preprocessed_frame_path))
        preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2RGB)

        # Get original resolution
        original_resolution = None
        if video_id in metadata["videos"]:
            original_resolution = metadata["videos"][video_id]["resolution"]
        else:
            try:
                original_resolution = (annotation["metadata"]["resolution"]["width"], annotation["metadata"]["resolution"]["height"])
            except (KeyError, TypeError):
                print(f"Warning: Could not determine resolution for {video_id}")
                continue

        # Draw bounding boxes on preprocessed frame
        preprocessed_with_boxes = preprocessed_frame.copy()
        target_size = metadata["frame_size"]

        for ann in frame_annotations:
            # Get original bbox
            center_x, center_y, width, height = ann["bbox"]

            # Create a temporary dataset instance for bbox adjustment
            temp_dataset = PreprocessedVideoDataset(preprocessed_dir, annotations_dir)

            # Adjust bbox for padding
            bbox_adjusted = temp_dataset._adjust_bbox_for_padding([center_x, center_y, width, height], original_resolution, target_size)

            # Unpack adjusted bbox
            center_x, center_y, width, height = bbox_adjusted

            # Convert to top-left coordinates for OpenCV rectangle
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            # Draw rectangle
            cv2.rectangle(preprocessed_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create side-by-side comparison
        # Resize original to match height of preprocessed
        h_orig, w_orig = original_with_boxes.shape[:2]
        h_prep, w_prep = preprocessed_with_boxes.shape[:2]

        # Resize original to match height of preprocessed
        scale = h_prep / h_orig
        new_w = int(w_orig * scale)
        resized_original = cv2.resize(original_with_boxes, (new_w, h_prep))

        # Create comparison image
        comparison = np.hstack([resized_original, preprocessed_with_boxes])

        # Save comparison
        output_file = output_path / f"compare_{video_id}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_file), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        # Also save individual images
        cv2.imwrite(str(output_path / f"original_{video_id}_frame_{frame_idx:06d}.jpg"), cv2.cvtColor(resized_original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(output_path / f"preprocessed_{video_id}_frame_{frame_idx:06d}.jpg"),
            cv2.cvtColor(preprocessed_with_boxes, cv2.COLOR_RGB2BGR),
        )

        print(f"Saved comparison for {video_id}, frame {frame_idx}")

    print(f"Before/after visualization complete. Images saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug visualization for bounding boxes")
    parser.add_argument(
        "--preprocessed_dir", type=str, default="/home/jovyan/p.kudrevatyh/temp", help="Directory containing preprocessed frames"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations",
        help="Directory containing annotation files",
    )
    parser.add_argument(
        "--videos_dir", type=str, default="/home/jovyan/p.kudrevatyh/yolov5_motion/data/videos", help="Directory containing video files"
    )
    parser.add_argument("--output_dir", type=str, default="/home/jovyan/p.kudrevatyh/debug", help="Directory to save debug images")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for dataloader visualization")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dataset", "batch", "compare"],
        default="compare",
        help="Visualization mode: dataset, batch, or compare before/after preprocessing",
    )

    args = parser.parse_args()

    if args.mode == "dataset":
        # Visualize individual samples from the dataset
        dataset = PreprocessedVideoDataset(preprocessed_dir=args.preprocessed_dir, annotations_dir=args.annotations_dir)

        visualize_bboxes(dataset, args.output_dir, args.num_samples)

    elif args.mode == "batch":
        # Visualize a batch from the dataloader
        dataset = PreprocessedVideoDataset(preprocessed_dir=args.preprocessed_dir, annotations_dir=args.annotations_dir)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        visualize_dataloader_batch(dataloader, args.output_dir)

    elif args.mode == "compare":
        # Compare before and after preprocessing
        visualize_before_after_preprocessing(
            args.videos_dir, args.annotations_dir, args.preprocessed_dir, args.output_dir, args.num_samples
        )
