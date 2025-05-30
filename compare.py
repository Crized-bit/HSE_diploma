import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Base path to your data
BASE_PATH = "/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs"

# Model names to analyze
MODEL_NAMES = [
    # "yolov5n/base_model",
    "yolov5n/lora",
    "yolov5n/0.2/bg_sub/control_lora + yolo_lora",
    "yolov5n/0.2/difference/control_lora + yolo_lora",
    "yolov5n/0.2/mixed/control_lora + yolo_lora",
    # "yolov5n/0.2/canny/control_lora + yolo_lora",
    # "yolov5n/0.2/flow/control_lora + yolo_lora",
]

COLOR_PALETTE = [
    "#1D7B92",  # Teal Blue
    "#F3863F",  # Bright Orange
    # "#5D4A98",  # Royal Purple
    "#2E7D32",  # Forest Green
    "#C62828",  # Deep Red
]

# For positive/negative indicators
POSITIVE_CHANGE_COLOR = "#2E7D32"  # Green
NEGATIVE_CHANGE_COLOR = "#C62828"  # Red

# Background colors
PLOT_BACKGROUND_COLOR = "#F8F9FA"  # Light gray background
GRID_COLOR = "#E0E0E0"  # Light grid lines

# Font settings for academic presentation
FONT_SIZE_TITLE = 18
FONT_SIZE_LABELS = 14
FONT_SIZE_TICKS = 16
FONT_SIZE_ANNOTATIONS = 14


def load_metrics(model_name):
    """Loads metrics from a JSON file for the specified model."""
    json_path = os.path.join(BASE_PATH, model_name, "test_metrics", "test_results.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        # Return the metrics object from JSON
        if "metrics" in data:
            return data["metrics"]
        else:
            print(f"Error: 'metrics' object not found in {json_path}")
            return None
    except FileNotFoundError:
        print(f"Error: File {json_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Cannot decode JSON from {json_path}")
        return None


def plot_metrics():
    """Creates a comparative histogram of metrics for specified models with percentage change."""
    # Metrics we want to display
    metrics = ["mAP@0.5", "mAP@0.5:0.95"]

    # Number of models and metrics
    n_models = len(MODEL_NAMES)
    n_metrics = len(metrics)

    if n_models < 1:
        print("At least one model must be specified")
        return

    # Prepare data for each model
    all_data = {}
    for model in MODEL_NAMES:
        data = load_metrics(model)
        if data:
            all_data[model] = data

    if not all_data:
        print("No data available to display")
        return

    # Define base model (first in the list)
    base_model = MODEL_NAMES[0]
    if base_model not in all_data:
        print(f"Error: Base model {base_model} not found or data unavailable")
        return

    # Configure figure and grid
    plt.figure(figsize=(14, 8))
    plt.rcParams.update({"font.size": FONT_SIZE_LABELS})

    # Bar width
    bar_width = 0.8 / n_models

    # Create positions for bar groups
    index = np.arange(n_metrics)

    # Use the global color palette
    colors = COLOR_PALETTE[:n_models] if n_models <= len(COLOR_PALETTE) else plt.cm.viridis(np.linspace(0, 0.8, n_models))

    # Create bars for each model
    for i, (model, data) in enumerate(all_data.items()):
        # Get metric values for this model
        values = []
        for metric in metrics:
            if metric in data:
                values.append(data[metric])
            elif metric.lower() in data:
                values.append(data[metric.lower()])
            elif metric.replace("@", "_") in data:
                values.append(data[metric.replace("@", "_")])
            else:
                print(f"Metric {metric} not found for model {model}")
                values.append(0)

        # Create bars with offset for each model
        position = index + (i - n_models / 2 + 0.5) * bar_width
        bars = plt.bar(position, values, bar_width, label=model, color=colors[i], alpha=0.8)

        # Add values above bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()

            # Calculate percentage change relative to base model
            if model == base_model:
                # For base model, just display the value
                percentage_text = ""
            else:
                # Get corresponding metric value for base model
                base_val = 0
                if metrics[j] in all_data[base_model]:
                    base_val = all_data[base_model][metrics[j]]
                elif metrics[j].lower() in all_data[base_model]:
                    base_val = all_data[base_model][metrics[j].lower()]
                elif metrics[j].replace("@", "_") in all_data[base_model]:
                    base_val = all_data[base_model][metrics[j].replace("@", "_")]

                if base_val > 0:
                    percentage_change = ((val - base_val) / base_val) * 100
                    # Determine color for change (green for positive, red for negative)
                    percentage_text = f"\n{percentage_change:+.1f}%"
                else:
                    percentage_text = ""

            # Main value
            val_text = f"{val:.4f}"

            # Text color for percentage change
            if model != base_model and percentage_text:
                # Create text with value
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    val_text,
                    ha="center",
                    va="bottom",
                    rotation=0,
                    fontsize=FONT_SIZE_TICKS,
                )

                # Add text with percentage change slightly higher
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    percentage_text.strip(),
                    ha="center",
                    va="bottom",
                    rotation=0,
                    fontsize=FONT_SIZE_ANNOTATIONS - 1,
                    color=POSITIVE_CHANGE_COLOR if "+" in percentage_text else NEGATIVE_CHANGE_COLOR,
                    fontweight="bold",
                )
            else:
                # For base model, just show the value
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    val_text,
                    ha="center",
                    va="bottom",
                    rotation=0,
                    fontsize=FONT_SIZE_TICKS,
                )

    # Configure the plot
    plt.ylabel("Value", fontweight="bold", fontsize=FONT_SIZE_LABELS)
    plt.title("Test Detection Metrics", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    plt.xticks(index, metrics, fontsize=FONT_SIZE_LABELS)
    plt.ylim(0, 1.0)  # Set Y-axis limit from 0 to 1

    # Add grid
    plt.grid(axis="y", linestyle="-", alpha=0.2, color=GRID_COLOR)
    plt.grid(axis="x", linestyle="-", alpha=0.2, color=GRID_COLOR)

    # Add legend with base model highlighting
    if n_models > 1:
        # Create legend elements manually (one for each model)
        custom_handles = []
        custom_labels = []

        for i, model_name in enumerate(MODEL_NAMES):
            if model_name in all_data:  # Only models with data
                patch = plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.8)
                custom_handles.append(patch)
                # Add "(base)" to base model name
                if i == 0:
                    custom_labels.append(f"{model_name} (base)")
                else:
                    custom_labels.append(model_name)

        # Add custom legend
        plt.legend(custom_handles, custom_labels, loc="upper left", title="Models", fontsize=FONT_SIZE_TICKS)

    # Configure background and borders
    ax = plt.gca()
    ax.set_facecolor(PLOT_BACKGROUND_COLOR)

    # Save and display
    output_path = os.path.join(BASE_PATH, "metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved: {output_path}")
    plt.show()


def plot_simple_histograms():
    """Create simple overlapping histograms with proper handling of log-scale bins."""
    # Histogram metrics to visualize
    hist_metrics = ["bin_mAP@0.5", "bin_mAP@0.5:0.95"]

    # Load data for each model
    all_data = {}
    for model in MODEL_NAMES:
        data = load_metrics(model)
        if data:
            all_data[model] = data

    if not all_data:
        print("No data available for histograms")
        return

    # Get edge values (representing bounding box space)
    base_model = MODEL_NAMES[0]
    if base_model not in all_data or "edges" not in all_data[base_model]:
        print("Error: Edge values not found in base model")
        return

    edges = all_data[base_model]["edges"]

    # Create plots for each metric
    for metric in hist_metrics:
        # Create figure
        plt.figure(figsize=(14, 8))

        # Use global color palette
        colors = (
            COLOR_PALETTE[: len(MODEL_NAMES)]
            if len(MODEL_NAMES) <= len(COLOR_PALETTE)
            else plt.cm.viridis(np.linspace(0, 0.8, len(MODEL_NAMES)))
        )

        # Plot histograms for each model
        for i, model_name in enumerate(MODEL_NAMES):
            if model_name in all_data and metric in all_data[model_name]:
                values = all_data[model_name][metric]

                # Skip models with invalid data
                if len(values) + 1 != len(edges):
                    print(f"Error: Length mismatch between values and edges for {model_name}")
                    continue

                # Use step function to plot histogram properly
                plt.bar(
                    edges[:-1],  # Bar positions (left edge)
                    values,  # Bar heights
                    width=np.diff(edges),  # Bar widths
                    align="edge",  # Align bars with left edge
                    alpha=0.6,  # Transparency
                    color=colors[i],  # Bar color
                    label=model_name,  # Model name for legend
                )

        # Configure plot
        plt.xlabel("Box normalized area", fontweight="bold", fontsize=FONT_SIZE_LABELS)
        plt.ylabel(f"{metric}", fontweight="bold", fontsize=FONT_SIZE_LABELS)
        plt.title(f"{metric} Distribution", fontsize=FONT_SIZE_TITLE, fontweight="bold")

        # Set x-axis to log scale if the edges appear to be logarithmically spaced
        if np.min(edges) > 0 and np.max(edges) / np.min(edges) > 100:
            plt.xscale("log")
            # Format x-axis with cleaner tick labels in log scale
            plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))

        # Add grid (appropriate for log scale too)
        plt.grid(True, which="both", ls="-", alpha=0.2, color=GRID_COLOR)

        # Set background color
        ax = plt.gca()
        ax.set_facecolor(PLOT_BACKGROUND_COLOR)

        # Add legend
        plt.legend(title="Models", fontsize=FONT_SIZE_TICKS)

        # Save and display
        output_path = os.path.join(BASE_PATH, f"{metric.replace('@', '_')}_histogram.png")
        plt.tight_layout()
        plt.rc('xtick', labelsize=FONT_SIZE_TICKS)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=FONT_SIZE_TICKS)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Histogram saved: {output_path}")
        plt.show()


if __name__ == "__main__":
    # plot_metrics()
    plot_simple_histograms()
