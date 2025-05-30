import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import pickle


@contextmanager
def suppress_output():
    """Suppress all output using contextlib"""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


# Add YOLOv5 directory to path
yolov5_dir = "/home/jovyan/p.kudrevatyh/yolov5"
if os.path.exists(yolov5_dir):
    sys.path.append(yolov5_dir)

from utils.general import non_max_suppression  # type: ignore

# Import your modules
sys.path.append("/home/jovyan/p.kudrevatyh/yolov5_motion/src")
from yolov5_motion.models.yolov5_controlnet import create_combined_model
from yolov5_motion.data.dataset_splits import create_dataset_splits
from yolov5_motion.utils.metrics import calculate_precision_recall, calculate_map
from yolov5_motion.data.dataset import collate_fn
from yolov5_motion.config import my_config


# Define the models to test (n series only)
MODEL_CONFIGS = {
    "bg_subtraction": {
        "name": "Background Subtraction",
        "path": "yolov5n/0.2/bg_sub/control_lora + yolo_lora",
        "control_type": "bg_sub",
        "data_path": "/home/jovyan/p.kudrevatyh/bg_subtraction",
    },
    "difference": {
        "name": "Frame Difference",
        "path": "yolov5n/0.2/difference/control_lora + yolo_lora",
        "control_type": "difference",
        "data_path": "/home/jovyan/p.kudrevatyh/difference",
    },
    "mixed": {
        "name": "Mixed",
        "path": "yolov5n/0.2/mixed/control_lora + yolo_lora",
        "control_type": "mixed",
        "data_path": "/home/jovyan/p.kudrevatyh/mixed_bg_difference",
    },
    # "canny": {"name": "Canny Edge", "path": "yolov5n/0.2/canny/control_lora + yolo_lora", "control_type": "canny"},
    # "flow": {"name": "Optical Flow", "path": "yolov5n/0.2/flow/control_lora + yolo_lora", "control_type": "flow"},
}


# Base paths
BASE_PATH = "/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs"
OUTPUT_DIR = Path("/home/jovyan/p.kudrevatyh/yolov5_motion/control_scale_analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_existing_results():
    """Load existing results if available"""
    pickle_path = OUTPUT_DIR / "control_scale_results.pkl"
    json_path = OUTPUT_DIR / "control_scale_results.json"

    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing results from {pickle_path}")
        return results
    elif json_path.exists():
        with open(json_path, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results from {json_path}")
        return results
    else:
        print("No existing results found, starting fresh")
        return {}


def evaluate_model_with_control_scale(model, test_dataloader, control_scale, device):
    """
    Evaluate a model with a specific control scale value

    Args:
        model_path: Path to the model checkpoint
        control_scale: Control scale value to use
        device: Device to run on
        test_dataloader: Pre-loaded test dataloader  # ADD THIS PARAMETER

    Returns:
        Dictionary with evaluation metrics
    """

    # Evaluate
    all_pred_boxes = []
    all_true_boxes = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Evaluating (scale={control_scale:.1f})", leave=False):
            current_frames = batch["current_frames"].to(device)
            control_images = batch["control_images"].to(device)
            targets = batch["annotations"]

            # Forward pass
            predictions = model(current_frames, control_images)

            # NMS
            detections = non_max_suppression(
                predictions if isinstance(predictions, torch.Tensor) else predictions[0],
                conf_thres=my_config.training.detection.conf_thres,
                iou_thres=my_config.training.detection.iou_thres,
                max_det=my_config.training.detection.max_det,
            )

            # Process each image
            for i, det in enumerate(detections):
                pred_boxes = []
                if det is not None and len(det):
                    for *xyxy, conf, cls_id in det:
                        if cls_id == 0:  # Only people
                            pred_boxes.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item(), cls_id.item()])

                # Process ground truth
                true_boxes = []
                if i < len(targets):
                    for ann in targets[i]:
                        bbox = ann
                        x1 = (bbox[0] - bbox[2] / 2) * my_config.model.img_size
                        y1 = (bbox[1] - bbox[3] / 2) * my_config.model.img_size
                        x2 = (bbox[0] + bbox[2] / 2) * my_config.model.img_size
                        y2 = (bbox[1] + bbox[3] / 2) * my_config.model.img_size
                        true_boxes.append([x1, y1, x2, y2, 0])

                all_pred_boxes.append(pred_boxes)
                all_true_boxes.append(true_boxes)

    # Calculate mAP
    map50, map_all = calculate_map(all_pred_boxes, all_true_boxes)

    return {"mAP@0.5": map50, "mAP@0.5:0.95": map_all}


def create_test_dataloader(preprocessed_path: str):
    """Create test dataloader once to be reused"""
    datasets = create_dataset_splits(
        preprocessed_dir=preprocessed_path,
        annotations_dir=my_config.data.annotations_dir,
        augment=False,
    )

    test_dataloader = DataLoader(
        datasets["test"],
        batch_size=128,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return test_dataloader


def analyze_control_scale(resume=True, additional_scales=None):
    """
    Main function to analyze control scale impact on mAP

    Args:
        resume: Whether to resume from existing results
        additional_scales: Additional control scale values to test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Results storage
    # Load existing results if resuming
    if resume:
        results = load_existing_results()
    else:
        results = {}

    # Test each model
    for model_key, model_config in MODEL_CONFIGS.items():
        if model_key == "mixed":
            my_config.model.num_input_channels = 4
        elif model_key == "difference":
            results.pop(model_key)
        else:
            my_config.model.num_input_channels = 3
        print(f"\n{'='*50}")
        print(f"Testing model: {model_config['name']}")
        print(f"Path: {model_config['path']}")
        print(f"{'='*50}")

        # Create test dataloader once
        print("Loading test dataset...")
        with suppress_output():
            test_dataloader = create_test_dataloader(model_config["data_path"])
        print(f"Test dataset loaded: {len(test_dataloader)} batches")

        # Initialize results for this model if not exists
        if model_key not in results:
            results[model_key] = {
                "control_scales": [],
                "mAP@0.5": [],
                "mAP@0.5:0.95": [],
            }

        # Get existing control scales for this model
        existing_scales = set(np.round(np.array(results[model_key]["control_scales"]), 2).tolist())

        model = create_combined_model(
            cfg=my_config.model.yolo_cfg,
            yolo_weights=None,
            controlnet_weights=None,
            lora_weights=None,
            img_size=my_config.model.img_size,
            nc=my_config.model.num_classes,
            control_scale=1.0,  # Set the control scale
        )

        checkpoint_path = Path(BASE_PATH) / model_config["path"] / "checkpoints/best_model.pt"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            try:
                model.load_state_dict(checkpoint)
            except:
                if "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])

        model = model.to(device)
        model.eval()

        # Test each control scale value
        for control_scale in additional_scales:
            if control_scale in existing_scales:
                continue
            print(f"\nTesting control_scale = {control_scale:.2f}")
            model.control_scale = control_scale


            metrics = evaluate_model_with_control_scale(model, test_dataloader, control_scale, device)

            if metrics:
                # Insert in sorted order
                insert_idx = 0
                for i, existing_scale in enumerate(results[model_key]["control_scales"]):
                    if control_scale > existing_scale:
                        insert_idx = i + 1
                    else:
                        break

                # Insert at the correct position to maintain sorted order
                results[model_key]["control_scales"].insert(insert_idx, control_scale)
                results[model_key]["mAP@0.5"].insert(insert_idx, metrics["mAP@0.5"])
                results[model_key]["mAP@0.5:0.95"].insert(insert_idx, metrics["mAP@0.5:0.95"])

                print(f"  mAP@0.5: {metrics['mAP@0.5']:.4f}")
                print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
            else:
                print(f"  Failed to evaluate model")

        # Clean up
    del model
    torch.cuda.empty_cache()

    # Save results
    save_results(results)

    # Plot results
    plot_results(results)

    return results


def save_results(results):
    """Save results to multiple formats for backup"""
    # Save as JSON
    json_path = OUTPUT_DIR / "control_scale_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {json_path}")

    # Save as pickle for exact Python object preservation
    pickle_path = OUTPUT_DIR / "control_scale_results.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {pickle_path}")

    # Save as CSV for easy viewing
    import csv

    csv_path = OUTPUT_DIR / "control_scale_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["Model", "Control_Scale", "mAP@0.5", "mAP@0.5:0.95"])
        # Data
        for model_key, model_results in results.items():
            model_name = MODEL_CONFIGS[model_key]["name"]
            for i in range(len(model_results["control_scales"])):
                writer.writerow(
                    [
                        model_name,
                        model_results["control_scales"][i],
                        model_results["mAP@0.5"][i],
                        model_results["mAP@0.5:0.95"][i],
                    ]
                )
    print(f"Results saved to {csv_path}")


def plot_results(results):
    """Create plots for control scale analysis"""
    # Set up the plot style
    plt.style.use("ggplot")

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Create main plot for mAP@0.5
    plt.figure(figsize=(12, 8))

    for i, (model_key, model_results) in enumerate(results.items()):
        if model_results["control_scales"]:  # Check if we have data
            model_name = MODEL_CONFIGS[model_key]["name"]
            plt.plot(
                model_results["control_scales"],
                model_results["mAP@0.5"],
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="o",
                markersize=6,
                alpha=0.8,
            )

    plt.xlabel("Control Scale", fontsize=14)
    plt.ylabel("mAP@0.5", fontsize=14)
    plt.title("Impact of Control Scale on mAP@0.5 for Different Control Types", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.5)
    # plt.ylim(0, 1.0)

    # Add vertical line at control_scale = 1.0 (default)
    plt.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="Default (1.0)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "control_scale_map50.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "control_scale_map50.pdf")
    plt.close()

    # Create plot for mAP@0.5:0.95
    plt.figure(figsize=(12, 8))

    for i, (model_key, model_results) in enumerate(results.items()):
        if model_results["control_scales"]:
            model_name = MODEL_CONFIGS[model_key]["name"]
            plt.plot(
                model_results["control_scales"],
                model_results["mAP@0.5:0.95"],
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="s",
                markersize=6,
                alpha=0.8,
            )

    plt.xlabel("Control Scale", fontsize=14)
    plt.ylabel("mAP@0.5:0.95", fontsize=14)
    plt.title("Impact of Control Scale on mAP@0.5:0.95 for Different Control Types", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.5)
    # plt.ylim(0, 1.0)

    # Add vertical line at control_scale = 1.0
    plt.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="Default (1.0)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "control_scale_map_all.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "control_scale_map_all.pdf")
    plt.close()

    # Create combined plot with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # mAP@0.5 subplot
    for i, (model_key, model_results) in enumerate(results.items()):
        if model_results["control_scales"]:
            model_name = MODEL_CONFIGS[model_key]["name"]
            ax1.plot(
                model_results["control_scales"],
                model_results["mAP@0.5"],
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="o",
                markersize=6,
                alpha=0.8,
            )

    ax1.set_xlabel("Control Scale", fontsize=14)
    ax1.set_ylabel("mAP@0.5", fontsize=14)
    ax1.set_title("Impact of Control Scale on mAP@0.5", fontsize=16, fontweight="bold")
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.5)
    # ax1.set_ylim(0, 1.0)
    ax1.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

    # mAP@0.5:0.95 subplot
    for i, (model_key, model_results) in enumerate(results.items()):
        if model_results["control_scales"]:
            model_name = MODEL_CONFIGS[model_key]["name"]
            ax2.plot(
                model_results["control_scales"],
                model_results["mAP@0.5:0.95"],
                label=model_name,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="s",
                markersize=6,
                alpha=0.8,
            )

    ax2.set_xlabel("Control Scale", fontsize=14)
    ax2.set_ylabel("mAP@0.5:0.95", fontsize=14)
    ax2.set_title("Impact of Control Scale on mAP@0.5:0.95", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.5)
    # ax2.set_ylim(0, 1.0)
    ax2.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "control_scale_combined.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "control_scale_combined.pdf")
    plt.close()

    print(f"\nPlots saved to {OUTPUT_DIR}")


def load_and_plot_saved_results():
    """Load saved results and re-plot (useful if you need to modify plots later)"""
    # Try loading from pickle first (most accurate)
    pickle_path = OUTPUT_DIR / "control_scale_results.pkl"
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded results from {pickle_path}")
        plot_results(results)
        return results

    # Fallback to JSON
    json_path = OUTPUT_DIR / "control_scale_results.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            results = json.load(f)
        print(f"Loaded results from {json_path}")
        plot_results(results)
        return results

    print("No saved results found!")
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze control scale impact on mAP")
    parser.add_argument("--replot", action="store_true", help="Just reload and replot saved results without re-evaluating")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh without loading existing results")
    parser.add_argument("--scale-min", type=float, default=0, help="Minimum control scale value (default: 0.0)")
    parser.add_argument("--scale-max", type=float, default=1.5, help="Maximum control scale value (default: 2.0)")
    parser.add_argument("--scale-steps", type=int, default=31, help="Number of scale steps (default: 21)")
    parser.add_argument("--add-scales", nargs="+", type=float, help="Specific additional scale values to test")

    args = parser.parse_args()

    if args.replot:
        # Just reload and plot
        load_and_plot_saved_results()
    else:
        # Determine scales to test
        if args.add_scales:
            # Test only specific additional scales
            additional_scales = args.add_scales
            print(f"Testing additional control scales: {additional_scales}")
        else:
            # Use the range
            additional_scales = np.round(np.linspace(args.scale_min, args.scale_max, args.scale_steps).tolist(), 2)
            print(additional_scales)

        # Run the analysis
        results = analyze_control_scale(resume=not args.no_resume, additional_scales=additional_scales)

        print("\n" + "=" * 50)
        print("Analysis complete!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print("=" * 50)
