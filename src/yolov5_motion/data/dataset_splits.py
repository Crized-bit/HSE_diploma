import json
import random
import torch
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import numpy as np
from yolov5_motion.data.dataset import PreprocessedVideoDataset, collate_fn


def create_dataset_splits(preprocessed_dir, annotations_dir, splits_file, val_ratio=0.1, seed=42):
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
    full_dataset = PreprocessedVideoDataset(preprocessed_dir=preprocessed_dir, annotations_dir=annotations_dir)

    # Load the splits file
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # Get video IDs from the dataset samples
    video_ids = {sample["video_id"] for sample in full_dataset.samples}

    # Create mappings for each split
    test_videos = set([Path(video_path).stem for video_path in splits["test"]])
    train_videos = set([Path(video_path).stem for video_path in splits["train"]])

    # Create indices for each split
    train_indices = []
    test_indices = []

    for idx, sample in enumerate(full_dataset.samples):
        video_id = sample["video_id"]
        if video_id in test_videos:
            test_indices.append(idx)
        elif video_id in train_videos:
            train_indices.append(idx)

    # Shuffle train indices
    random.shuffle(train_indices)

    # Split train into train and validation
    val_size = int(len(train_indices) * val_ratio)
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]

    print(f"Dataset split complete:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples: {len(val_indices)}")
    print(f"  Test samples: {len(test_indices)}")

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset, "full": full_dataset}


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
