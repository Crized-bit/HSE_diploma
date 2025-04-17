import json
import random
import torch
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import numpy as np
from yolov5_motion.data.dataset import PreprocessedVideoDataset, collate_fn


def create_dataset_splits(
    preprocessed_dir, annotations_dir, splits_file, prev_frame_time_diff=1.0, val_ratio=0.1, seed=42, augment=True, augment_prob=0.5
):
    """
    Create train, test, and validation dataset splits using the provided splits file.
    Validation set is created by taking a portion of the training set.

    Args:
        preprocessed_dir: Directory containing preprocessed frames
        annotations_dir: Directory containing annotation files
        splits_file: Path to splits JSON file
        val_ratio: Ratio of training data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing train, val, and test datasets
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the dataset
    base_dataset = PreprocessedVideoDataset(
        preprocessed_dir=preprocessed_dir,
        annotations_dir=annotations_dir,
        prev_frame_time_diff=prev_frame_time_diff,
        augment=augment,
        augment_prob=augment_prob,
    )

    # Load the splits file
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # Create mappings for each split
    test_videos = set([Path(video_path).stem for video_path in splits["test"]])
    train_videos = list(set([Path(video_path).stem for video_path in splits["train"]]))

    val_size = int(len(train_videos) * val_ratio)
    val_videos = train_videos[:val_size]
    train_videos = train_videos[val_size:]

    # Create indices for each split
    train_indices = []
    test_indices = []
    val_indices = []

    for idx, sample in enumerate(base_dataset.samples):
        video_id = sample["video_id"]
        if video_id in test_videos:
            test_indices.append(idx)
        elif video_id in train_videos:
            train_indices.append(idx)
        elif video_id in val_videos:
            val_indices.append(idx)

    print(f"Dataset split complete:")
    print(f"  Total samples: {len(base_dataset)}")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples: {len(val_indices)}")
    print(f"  Test samples: {len(test_indices)}")

    # Create three separate datasets with appropriate augmentation settings
    train_dataset = PreprocessedVideoDataset(
        preprocessed_dir=preprocessed_dir, annotations_dir=annotations_dir, augment=True  # Enable augmentation only for training
    )

    val_dataset = PreprocessedVideoDataset(
        preprocessed_dir=preprocessed_dir, annotations_dir=annotations_dir, augment=False  # No augmentation for validation
    )

    test_dataset = PreprocessedVideoDataset(
        preprocessed_dir=preprocessed_dir, annotations_dir=annotations_dir, augment=False  # No augmentation for testing
    )

    # Use the indices with Subset to get the right samples for each dataset
    return {
        "train": Subset(train_dataset, train_indices),
        "val": Subset(val_dataset, val_indices),
        "test": Subset(test_dataset, test_indices),
        "full": base_dataset,
    }


def get_dataloaders(datasets, batch_size=8, num_workers=4):
    """
    Create dataloaders for train, validation, and test datasets.

    Args:
        datasets: Dictionary containing train, val, and test datasets
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading

    Returns:
        Dictionary containing train, val, and test dataloaders
    """
    dataloaders = {}

    # Create train dataloader with shuffling
    dataloaders["train"] = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # Create validation and test dataloaders without shuffling
    for split in ["val", "test"]:
        dataloaders[split] = DataLoader(
            datasets[split], batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )

    return dataloaders
