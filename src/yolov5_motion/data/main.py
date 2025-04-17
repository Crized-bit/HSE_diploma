import argparse
import time
from pathlib import Path

from yolov5_motion.data.preprocessing import preprocess_videos
from yolov5_motion.data.dataset_splits import create_dataset_splits, get_dataloaders
from yolov5_motion.data.utils import draw_bounding_boxes

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Video Motion Analysis Pipeline")

    # Directory paths
    parser.add_argument(
        "--videos_dir", type=str, default="/home/jovyan/p.kudrevatyh/yolov5_motion/data/videos", help="Directory containing video files"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations",
        help="Directory containing annotation files",
    )
    parser.add_argument("--output_dir", type=str, default="/home/jovyan/p.kudrevatyh/difference", help="Directory to save preprocessed frames")
    parser.add_argument(
        "--splits_file", type=str, default="/home/jovyan/p.kudrevatyh/yolov5_motion/data/splits.json", help="Path to splits JSON file"
    )

    # Preprocessing options
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--prev_frame_time_diff", type=float, default=1.0, help="Time difference in seconds to get the previous frame")
    parser.add_argument("--resize_to", type=int, nargs=2, default=[640, 640], help="Target size (width, height) for resizing frames")
    parser.add_argument("--control_mode", type=str, default="difference", help="Control mode for preprocessing")

    # Dataset and dataloader options
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloaders")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of training data to use for validation")

    # Visualization options
    parser.add_argument("--visualize", action="store_true", help="Visualize samples from the dataset")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize from each split")

    args = parser.parse_args()

    # Step 1: Preprocess videos if not skipped
    if True:
        print("\n===== Preprocessing Videos =====")
        start_time = time.time()
        preprocess_videos(
            videos_dir=args.videos_dir,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir,
            resize_to=tuple(args.resize_to),
            num_workers=args.num_workers,
            prev_frame_time_diff=args.prev_frame_time_diff,
            control_mode=args.control_mode,
        )
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

    # Step 2: Create dataset splits
    print("\n===== Creating Dataset Splits =====")
    start_time = time.time()
    datasets = create_dataset_splits(
        preprocessed_dir=args.output_dir,
        annotations_dir=args.annotations_dir,
        splits_file=args.splits_file,
        val_ratio=args.val_ratio,
    )
    print(f"Dataset splits created in {time.time() - start_time:.2f} seconds")

    # Step 3: Create dataloaders
    print("\n===== Creating DataLoaders =====")
    start_time = time.time()
    dataloaders = get_dataloaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"DataLoaders created in {time.time() - start_time:.2f} seconds")

    # Print dataloader info
    print("\nDataLoader Info:")
    for split, dataloader in dataloaders.items():
        print(f"  {split}: {len(dataloader)} batches")

    # Step 4: Benchmark dataloaders
    print("\n===== Benchmarking DataLoaders =====")
    for split, dataloader in dataloaders.items():
        start_time = time.time()
        num_batches = min(10, len(dataloader))  # Process up to 10 batches

        print(f"Processing {num_batches} batches from {split} dataloader...")
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

        elapsed = time.time() - start_time
        print(f"  {split}: {num_batches} batches in {elapsed:.2f} seconds")
        print(f"  {num_batches / elapsed:.2f} batches/second")
        print(f"  {num_batches * args.batch_size / elapsed:.2f} samples/second")

    # Step 5: Visualize samples if requested
    if True:
        print("\n===== Visualizing Samples =====")
        visualization_dir = Path(args.output_dir) / "visualizations"
        visualization_dir.mkdir(exist_ok=True)

        for split, dataset in datasets.items():
            if split == "full":
                continue

            print(f"Visualizing {args.num_samples} samples from {split} set...")
            split_dir = visualization_dir / split
            split_dir.mkdir(exist_ok=True)

            # Create indices list for visualization
            if args.num_samples >= len(dataset):
                indices = list(range(len(dataset)))
            else:
                indices = np.random.choice(len(dataset), args.num_samples, replace=False)

            for idx in indices:
                sample = dataset[idx]

                # Get current frame and control image
                current_frame = sample["current_frame"].permute(1, 2, 0).numpy() * 255
                current_frame = current_frame.astype(np.uint8)
                control_image = sample["control_image"].permute(1, 2, 0).numpy() * 255
                control_image = control_image.astype(np.uint8)

                # Draw bounding boxes on current frame
                annotated_frame = draw_bounding_boxes(current_frame, sample["annotations"])

                # Create a combined visualization
                # Convert from RGB to BGR for cv2
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                control_image_bgr = cv2.cvtColor(control_image, cv2.COLOR_RGB2BGR)

                # Stack horizontally
                combined = np.hstack((annotated_frame_bgr, control_image_bgr))

                # Save the visualization
                viz_path = split_dir / f"sample_{idx}.jpg"
                cv2.imwrite(str(viz_path), combined)

            print(f"Visualizations saved to {split_dir}")


if __name__ == "__main__":
    main()
