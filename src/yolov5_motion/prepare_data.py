import time
from pathlib import Path

from yolov5_motion.config import my_config
from yolov5_motion.data.preprocessing import preprocess_videos
from yolov5_motion.data.dataset_splits import create_dataset_splits, get_dataloaders
from yolov5_motion.data.utils import draw_bounding_boxes

import cv2
import numpy as np

NUM_SAMPLES = 100
OUTPUT_DIR = "/home/jovyan/p.kudrevatyh/bg_subtraction"
CONTROL_MODE = "bg_subtraction"


def main():
    if False:
        print("\n===== Preprocessing Videos =====")
        start_time = time.time()
        preprocess_videos(
            videos_dir="/home/jovyan/p.kudrevatyh/yolov5_motion/data/videos",
            annotations_dir="/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations",
            output_dir=OUTPUT_DIR,
            control_mode=CONTROL_MODE,
        )
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

    # Step 2: Create dataset splits
    print("\n===== Creating Dataset Splits =====")
    start_time = time.time()
    datasets = create_dataset_splits(
        preprocessed_dir=OUTPUT_DIR,
        annotations_dir=my_config.data.annotations_dir,
        augment=my_config.training.augment,
    )

    # Step 3: Create dataloaders
    print("\n===== Creating DataLoaders =====")
    start_time = time.time()
    dataloaders = get_dataloaders(datasets=datasets)

    # Print dataloader info
    print("\nDataLoader Info:")
    for split, dataloader in dataloaders.items():
        print(f"  {split}: {len(dataloader)} batches")

    print("\n===== Visualizing Samples =====")
    visualization_dir = Path(OUTPUT_DIR) / "visualizations"
    visualization_dir.mkdir(exist_ok=True)

    for split, dataset in datasets.items():
        if split == "full":
            continue

        print(f"Visualizing {NUM_SAMPLES} samples from {split} set...")
        split_dir = visualization_dir / split
        split_dir.mkdir(exist_ok=True)

        # Create indices list for visualization
        if NUM_SAMPLES >= len(dataset):
            indices = list(range(len(dataset)))
        else:
            indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)

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
