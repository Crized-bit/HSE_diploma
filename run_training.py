import sys
from yolov5_motion.train import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLOv5 ControlNet training")
    parser.add_argument("--config", type=str, default="/home/jovyan/p.kudrevatyh/yolov5_motion/configs/a100-train-config.yaml", help="Path to configuration file")

    args, unknown_args = parser.parse_known_args()

    # Create example config if none provided
    if not args.config:
        raise ValueError("No configuration file provided. Please provide a path to a YAML configuration file.")
    else:
        config_path = args.config


    # Run training
    main(config_path)
