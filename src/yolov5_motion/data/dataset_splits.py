import json
import random
import torch
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import numpy as np
from yolov5_motion.data.dataset import PreprocessedVideoDataset, collate_fn
from yolov5_motion.config import my_config

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def create_dataset_splits(
    preprocessed_dir,
    annotations_dir,
    augment=True,
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

    # Load the dataset
    base_dataset = PreprocessedVideoDataset(
        preprocessed_dir=preprocessed_dir,
        annotations_dir=annotations_dir,
        augment=augment,
    )

    # Load the splits file
    with open(my_config.data.splits_file, "r") as f:
        splits = json.load(f)

    # Create mappings for each split
    test_videos = [Path(video_path).stem for video_path in splits["test"]]
    train_videos = [Path(video_path).stem for video_path in splits["train"]]

    val_size = int(len(test_videos) * my_config.training.val_ratio)
    val_videos = test_videos[:val_size]
    test_videos = test_videos[val_size:]

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
        preprocessed_dir=preprocessed_dir,
        annotations_dir=annotations_dir,
        augment=my_config.training.augment,
    )

    val_dataset = PreprocessedVideoDataset(
        preprocessed_dir=preprocessed_dir,
        annotations_dir=annotations_dir,
        augment=False,
    )

    test_dataset = PreprocessedVideoDataset(
        preprocessed_dir=preprocessed_dir,
        annotations_dir=annotations_dir,
        augment=False,
    )

    # Use the indices with Subset to get the right samples for each dataset
    return {
        "train": Subset(train_dataset, train_indices),
        "val": Subset(val_dataset, val_indices),
        "test": Subset(test_dataset, test_indices),
        "full": base_dataset,
    }


def get_dataloaders(datasets):
    """
    Create dataloaders for train, validation, and test datasets.

    Args:
        datasets: Dictionary containing train, val

    Returns:
        Dictionary containing train, val dataloaders
    """
    dataloaders = {}

    # Create train dataloader with shuffling
    dataloaders["train"] = DataLoader(
        datasets["train"],
        batch_size=my_config.training.batch_size,
        persistent_workers = True,
        shuffle=True,
        num_workers=my_config.training.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Create validation and test dataloaders without shuffling
    dataloaders["val"] = DataLoader(
        datasets["val"],
        batch_size=my_config.training.val_batch_size,
        persistent_workers = True,
        shuffle=False,
        num_workers=my_config.training.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    return dataloaders
