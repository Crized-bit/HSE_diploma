import time
from pathlib import Path

from yolov5_motion.data.preprocessing import preprocess_videos
from yolov5_motion.data.dataset_splits import create_dataset_splits, get_dataloaders
from yolov5_motion.data.utils import draw_bounding_boxes
from yolov5_motion.config import my_config
import cv2
import numpy as np

NUM_SAMPLES = 10
OUTPUT_DIR = "/home/jovyan/p.kudrevatyh/difference"
CONTROL_MODE = "difference"


def main():
    if True:
        print("\n===== Preprocessing Videos =====")
        start_time = time.time()
        preprocess_videos(
            videos_dir="/home/jovyan/p.kudrevatyh/yolov5_motion/data/videos",
            annotations_dir="/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations",
            output_dir=OUTPUT_DIR,
            resize_to=tuple(640, 640),
            num_workers=my_config.training.workers,
            prev_frame_time_diff=my_config.data.prev_frame_time_diff,
            control_mode=CONTROL_MODE,
            control_stack_length=my_config.data.control_stack_length,
        )
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

    # Step 2: Create dataset splits
    print("\n===== Creating Dataset Splits =====")
    start_time = time.time()
    datasets = create_dataset_splits(
        preprocessed_dir=OUTPUT_DIR,
        annotations_dir="/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations",
        splits_file="/home/jovyan/p.kudrevatyh/yolov5_motion/data/splits.json",
        val_ratio=my_config.training.val_ratio,
        control_stack_length=my_config.data.control_stack_length,
    )

    # Step 3: Create dataloaders
    print("\n===== Creating DataLoaders =====")
    start_time = time.time()
    dataloaders = get_dataloaders(
        datasets=datasets,
        batch_size=my_config.training.batch_size,
        num_workers=my_config.training.workers,
    )

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
