#!/usr/bin/env python3
"""
Simple runner script for the YOLOv5 ControlNet training.
Creates and runs example configuration.
"""

import os
import sys
import yaml
from pathlib import Path

# Define default config path
DEFAULT_CONFIG_PATH = "configs/train_config.yaml"


def ensure_config_dir():
    """Create configs directory if it doesn't exist"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    return config_dir


def create_example_config():
    """Create an example configuration file"""
    config_dir = ensure_config_dir()
    config_path = config_dir / "train_config.yaml"

    # Skip if config already exists
    if config_path.exists():
        print(f"Config file already exists at {config_path}")
        return config_path

    # Create example configuration
    example_config = {
        "data": {
            "preprocessed_dir": "/path/to/preprocessed/data",
            "annotations_dir": "/path/to/annotations",
            "splits_file": "/path/to/splits.json",
            "output_dir": "./training_outputs",
        },
        "model": {
            "yolo_weights": "/path/to/yolov5m.pt",
            "controlnet_weights": None,
            "yolo_cfg": "yolov5m.yaml",
            "img_size": 640,
            "num_classes": 80,
            "train_controlnet_only": True,
        },
        "training": {
            "epochs": 100,
            "batch_size": 16,
            "val_batch_size": 32,
            "workers": 8,
            "val_ratio": 0.1,
            "optimizer": "prodigy",
            "lr": 0.001,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "loss": {
                "box_weight": 0.05,
                "obj_weight": 1.0,
                "cls_weight": 0.5,
            },
            "precision": "bf16",
            "save_interval": 10,
            "log_interval": 10,
            "eval_interval": 5,
            "resume": None,
        },
    }

    # Write configuration to file
    with open(config_path, "w") as f:
        yaml.dump(example_config, f, default_flow_style=False)

    print(f"Created example configuration at {config_path}")
    return config_path


def run_training(config_path, additional_args=None):
    """Run the training script with the given configuration"""
    from yolov5_motion.train import main

    # Build command
    cmd = [sys.executable, "-m", "yolov5_motion.train", "--config", str(config_path)]

    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)

    # Print command
    print(f"Running: {' '.join(cmd)}")

    # Execute command
    os.system(" ".join(cmd))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLOv5 ControlNet training")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file (creates example if not provided)")
    parser.add_argument("--create-config-only", action="store_true", help="Only create example config, don't run training")

    # Add passthrough arguments for overriding config
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")

    args, unknown_args = parser.parse_known_args()

    # Create example config if none provided
    if not args.config:
        config_path = create_example_config()
    else:
        config_path = args.config

    # Exit if only creating config
    if args.create_config_only:
        print("Created configuration file. Edit it before running training.")
        sys.exit(0)

    # Collect additional arguments to pass through
    additional_args = []
    if args.output_dir:
        additional_args.extend(["--output_dir", args.output_dir])
    if args.epochs:
        additional_args.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        additional_args.extend(["--batch_size", str(args.batch_size)])
    if args.lr:
        additional_args.extend(["--lr", str(args.lr)])

    # Add any unknown args
    additional_args.extend(unknown_args)

    # Run training
    run_training(config_path, additional_args)
