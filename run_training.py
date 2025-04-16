import os
import sys
from pathlib import Path



def ensure_config_dir():
    """Create configs directory if it doesn't exist"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    return config_dir


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
        raise ValueError("No configuration file provided. Please provide a path to a YAML configuration file.")
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
