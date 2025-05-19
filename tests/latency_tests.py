import os
import time
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import necessary modules from the project
from yolov5_motion.data.utils import create_control_image
from yolov5_motion.models.yolov5_controlnet import create_combined_model
from yolov5_motion.config import my_config


def test_preprocessing_fps(video_path, num_frames=32):
    """
    Test the FPS of different preprocessing methods

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to process
        batch_size: Number of frames to process at once to create control images
        warm_up_frames: Number of frames to warm up before timing

    Returns:
        Dict with average FPS for each method
    """
    print(f"\n===== Testing Preprocessing FPS on {os.path.basename(video_path)} =====")

    # Open video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Limit num_frames to total available frames
    num_frames = min(num_frames, total_frames - 1)

    # Initialize results dictionary
    results = {"flow": [], "bg_subtraction": [], "difference": []}

    # Process video
    frames = []

    # First we collect frames
    print(f"Collecting {num_frames} frames...")
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        ratio = my_config.model.img_size / max(height, width)
        frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
        print(frame.shape)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    substractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    # Process frames in batches
    for method in ["flow", "bg_subtraction", "difference"]:
        print(f"\nTesting {method} method...")

        # Process in batches to simulate the control stack length
        batch_timings = []

        for _ in tqdm(range(30)):
            # Process each frame in the batch
            start_time = time.time()
            prev_frame = frames[0]
            for frame in frames[1:]:
                if method == "bg_subtraction":
                    _ = substractor.apply(frame)
                elif method == "flow":
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                    # Calculate optical flow using Farneback method
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # Convert flow to RGB using HSV color wheel
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv = np.zeros((cur_gray.shape[0], cur_gray.shape[1], 3), dtype=np.uint8)
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 1] = 255
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                elif method == "difference":
                    cur_image = frame.astype(np.float32)
                    prev_image = prev_frame.astype(np.float32)

                    control_image = np.zeros((cur_image.shape[0], cur_image.shape[1], 3), dtype=np.float32)
                    control_image[..., 0] = cur_image[..., 0] - prev_image[..., 0]
                    control_image[..., 1] = cur_image[..., 1] - prev_image[..., 1]
                    control_image[..., 2] = cur_image[..., 2] - prev_image[..., 2]

                    mask = np.abs(control_image) < 15

                    control_image = ((control_image + 255) / 2).astype(np.uint8)

                    control_image[mask] = 0
                
                elif method == "canny":
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    # Создание первого канала - Canny
                    edges = cv2.Canny(cur_gray, threshold1=50, threshold2=150).astype(np.uint8)

                    # Вычисление градиентов по X и Y с помощью оператора Собеля
                    sobelx = cv2.Sobel(cur_gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(cur_gray, cv2.CV_64F, 0, 1, ksize=3)

                    blur = cv2.GaussianBlur(cur_gray, (5, 5), 0)
                    log = cv2.Laplacian(blur, cv2.CV_64F)
                    log_normalized = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    # Создание третьего канала - величина градиента
                    magnitude = cv2.magnitude(sobelx, sobely)
                    # Нормализация величины градиента
                    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                    # Объединение трех каналов в одно RGB изображение
                    combined_image = cv2.merge([edges, log_normalized, magnitude])

                prev_frame = frame

            batch_size_actual = len(frames) - 1

            elapsed = time.time() - start_time
            batch_fps = batch_size_actual / elapsed
            batch_timings.append(batch_fps)

        # Calculate average FPS
        if batch_timings:
            avg_fps = np.mean(batch_timings)
            std_fps = np.std(batch_timings)
            results[method] = batch_timings
            print(f"Average FPS for {method}: {avg_fps:.2f} ± {std_fps:.2f}")
        else:
            print(f"No valid batches processed for {method}")

    return results


def test_model_inference_fps(model_configs, input_size=640, num_iterations=100, warm_up=100):
    """
    Test the inference FPS of different model configurations

    Args:
        model_configs: List of model configuration dictionaries
        input_size: Input image size
        num_iterations: Number of inference iterations
        warm_up: Number of warm-up iterations

    Returns:
        Dict with average FPS for each model configuration
    """
    print(f"\n===== Testing Model Inference FPS =====")

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}

    for config in model_configs:
        model_name = config["name"]
        print(f"\nTesting {model_name}...")

        # Create model
        model = create_combined_model(
            cfg=config.get("cfg", my_config.model.yolo_cfg),
            yolo_weights=config.get("yolo_weights"),
            controlnet_weights=config.get("controlnet_weights"),
            lora_weights=config.get("lora_weights"),
            img_size=input_size,
            nc=my_config.model.num_classes,
            control_scale=config.get("control_scale", 1.0),
            lora_scale=0.0 if config.get("disable_lora", False) else 1.0,
        )

        # Apply any model-specific settings
        if config.get("disable_controlnet", False):
            model.use_controlnet = False
            print("ControlNet disabled")

        if config.get("disable_lora", False):
            model.disable_lora()
            print("LoRA disabled")

        # if not config.get("disable_lora", False):
        model.yolo.model = model.yolo.model.merge_and_unload()
        if config.get("fuse_controlnet", False):
            model.controlnet.nodes = model.controlnet.nodes.merge_and_unload()
        model.eval()

        # Move model to device
        # model = torch.compile(model)
        model = model.to(device)

        # Create random input tensors
        x = torch.randn(1, 3, input_size, input_size).to(device)
        control = torch.randn(1, 3, input_size, input_size).to(device)

        # Warm-up
        print("Warming up...")
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(warm_up):
                    _ = model(x, control)

        # Measure inference time
        timings = []
        print(f"Running {num_iterations} iterations...")
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in tqdm(range(num_iterations)):
                    start_time = time.time()
                    predictions = model(x, control)

                    # Synchronize CUDA for accurate timing
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    elapsed = time.time() - start_time
                    timings.append(elapsed)

        # Calculate FPS
        timings = np.array(timings)
        fps = 1 / timings

        # print(f"Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        # print(f"Average FPS: {fps:.2f}")

        results[model_name] = {"timings": timings, "fps": fps}

        # Free up memory
        del model
        torch.cuda.empty_cache()

    return results


def plot_preprocessing_results(results, output_dir, fontsize_base=16):
    """
    Plot preprocessing FPS results with larger fonts for scientific publications
    
    Args:
        results: Dict with FPS results for each method
        output_dir: Directory to save plots
        fontsize_base: Base font size (other sizes will scale accordingly)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate font sizes based on the base size
    title_size = fontsize_base + 4
    label_size = fontsize_base + 2
    tick_size = fontsize_base
    
    # Set global font parameters
    plt.rcParams.update({
        'font.size': fontsize_base,
        'axes.titlesize': title_size,
        'axes.labelsize': label_size,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'legend.fontsize': fontsize_base
    })

    # Plot FPS distribution
    plt.figure(figsize=(12, 8))

    data = []
    labels = []

    for method, fps_values in results.items():
        if fps_values:
            data.append(fps_values)
            labels.append(method)

    if data:
        plt.boxplot(data, tick_labels=labels, showmeans=True, showfliers=False)
        plt.title("Preprocessing FPS Comparison", fontsize=title_size, fontweight="bold")
        plt.ylabel("Frames Per Second (FPS)", fontsize=label_size)
        plt.grid(axis="y", alpha=0.3)

        # # Uncommented mean values with larger font
        # for i, d in enumerate(data):
        #     mean_val = np.mean(d)
        #     plt.text(i + 1, mean_val, f"{mean_val:.2f}", 
        #             horizontalalignment="center", 
        #             size=fontsize_base, 
        #             color="red", 
        #             weight="semibold")

        plt.tight_layout()
        plt.savefig(output_path / "preprocessing_fps_comparison.png", dpi=300)
        plt.savefig(output_path / "preprocessing_fps_comparison.pdf")
        plt.close()

        # Save raw data
        with open(output_path / "preprocessing_fps_data.txt", "w") as f:
            for method, fps_values in results.items():
                if fps_values:
                    f.write(f"{method}:\n")
                    f.write(f"  Mean FPS: {np.mean(fps_values):.2f}\n")
                    f.write(f"  Std Dev: {np.std(fps_values):.2f}\n")
                    f.write(f"  Min FPS: {min(fps_values):.2f}\n")
                    f.write(f"  Max FPS: {max(fps_values):.2f}\n")
                    f.write(f"  FPS Values: {','.join([f'{v:.2f}' for v in fps_values])}\n\n")


def plot_inference_results(results, output_dir, fontsize_base=16):
    """
    Plot model inference FPS results with larger fonts for scientific publications
    
    Args:
        results: Dict with inference results for each model
        output_dir: Directory to save plots
        fontsize_base: Base font size (other sizes will scale accordingly)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate font sizes based on the base size
    title_size = fontsize_base + 4
    label_size = fontsize_base + 2
    tick_size = fontsize_base
    
    # Set global font parameters
    plt.rcParams.update({
        'font.size': fontsize_base,
        'axes.titlesize': title_size,
        'axes.labelsize': label_size,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'legend.fontsize': fontsize_base
    })

    # Extract FPS values and model names
    model_names = list(results.keys())
    fps_values = [results[model]["fps"] for model in model_names]
    
    # Create bar chart for FPS comparison
    plt.figure(figsize=(12, 8))
    plt.boxplot(fps_values, tick_labels=model_names, showmeans=True, showfliers=False)
    # bars = plt.bar(range(len(model_names)), fps_values, width=0.6, yerr=fps_std, capsize=5)

    plt.xlabel("Model Configuration", fontsize=label_size)
    plt.ylabel("Frames Per Second (FPS)", fontsize=label_size)
    plt.title("Model Inference Speed Comparison", fontsize=title_size, fontweight="bold")
    # plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right", fontsize=tick_size)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path / "inference_fps_comparison.png", dpi=300)
    plt.savefig(output_path / "inference_fps_comparison.pdf")
    plt.close()

    # Create latency plot (milliseconds)
    plt.figure(figsize=(12, 8))
    latency_ms = [results[model]["timings"] * 1000 for model in model_names]
    # latency_std = [results[model]["std_time"] * 1000 for model in model_names]

    plt.boxplot(latency_ms, tick_labels=model_names, showmeans=True, showfliers=False)

    plt.xlabel("Model Configuration", fontsize=label_size)
    plt.ylabel("Inference Latency (ms)", fontsize=label_size)
    plt.title("Model Inference Latency Comparison", fontsize=title_size, fontweight="bold")
    # plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right", fontsize=tick_size)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path / "inference_latency_comparison.png", dpi=300)
    plt.savefig(output_path / "inference_latency_comparison.pdf")
    plt.close()


def main():
    output_dir = Path("/home/jovyan/p.kudrevatyh/yolov5_motion/tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_weights = "/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs/yolov5n/0.2/difference/control_lora + yolo_lora/checkpoints/best_lora"
    yolo_weights = "/home/jovyan/p.kudrevatyh/yolov5n.pt"
    control_weights = "/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs/yolov5n/0.2/difference/control_lora + yolo_lora/checkpoints/best_controlnet.pt"

    # Define model configurations
    model_configs = [
        {"name": "YOLOv5 Base", "yolo_weights": yolo_weights, "disable_controlnet": True, "disable_lora": True},
        {"name": "YOLOv5 + LoRA", "yolo_weights": yolo_weights, "lora_weights": lora_weights, "disable_controlnet": True},
        {
            "name": "YOLOv5 + LoRA\n + ControlNet Fused",
            "yolo_weights": yolo_weights,
            "controlnet_weights": control_weights,
            "lora_weights": lora_weights,
            "fuse_controlnet": True,
        },
        {
            "name": "YOLOv5 + LoRA\n + ControlNet Shared",
            "yolo_weights": yolo_weights,
            "controlnet_weights": control_weights,
            "lora_weights": lora_weights,
        },
    ]

    preprocessing_results = test_preprocessing_fps("/home/jovyan/p.kudrevatyh/yolov5_motion/data/videos/uid_vid_00006.mp4", num_frames=32)
    plot_preprocessing_results(preprocessing_results, output_dir / "preprocessing")

    # inference_results = test_model_inference_fps(model_configs, num_iterations=300*32)
    # plot_inference_results(inference_results, output_dir / "inference")

    print(f"\nAll tests completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
