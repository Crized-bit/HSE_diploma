import torch
import matplotlib.pyplot as plt
from pathlib import Path

from yolov5_motion.models.yolov5_controlnet import create_combined_model
from yolov5_motion.config import my_config

# Set up output directory
output_dir = Path("/home/jovyan/p.kudrevatyh/yolov5_motion/tests/model_params_analysis")
output_dir.mkdir(exist_ok=True, parents=True)


def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    return {
        "total": total_params,
        "trainable": trainable_params,
        "percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
    }


def analyze_model_configs():
    """Analyze different model configurations and visualize parameter counts"""
    model_configs = [
        ("YOLOv5 Base", create_base_model),
        ("YOLOv5 + LoRA", create_lora_model),
        ("YOLOv5 + LoRA + ControlNet", create_full_model),
    ]

    infer_configs = [
        ("YOLOv5 Base", create_base_model),
        ("YOLOv5 + LoRA + ControlNet Shared", create_full_model_infer_shared),
        ("YOLOv5 + LoRA + ControlNet Fused", create_full_model_infer_fused),
    ]
    results = []

    print("Analyzing model configurations...")
    for name, creator_func in model_configs:
        model = creator_func()
        params = count_parameters(model)
        params["name"] = name
        results.append(params)

        print(f"\n{name}:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Trainable percentage: {params['percentage']:.2f}%")

        # Free up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    visualize_results(results)

    results = []

    print("\nAnalyzing inference configurations...")
    for name, creator_func in infer_configs:
        model = creator_func()
        params = count_parameters(model)
        params["name"] = name
        results.append(params)

        print(f"\n{name}:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Trainable percentage: {params['percentage']:.2f}%")

        # Free up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    visualize_infer(results)
    return results


def create_base_model():
    """Create base YOLOv5 model without LoRA or ControlNet"""
    # Use the same YOLO config as your project
    cfg = my_config.model.yolo_cfg
    weights = my_config.model.yolo_weights

    # Create model without ControlNet or LoRA
    model = create_combined_model(
        cfg=cfg,
        yolo_weights=weights,
        controlnet_weights=None,
        lora_weights=None,
        img_size=my_config.model.img_size,
        nc=my_config.model.num_classes,
        control_scale=0.0,  # Disable controlnet influence
    )

    model.controlnet = None
    model.yolo.model = model.yolo.model.merge_and_unload()
    # Make sure all parameters are not trainable
    for param in model.parameters():
        param.requires_grad = False

    return model


def create_lora_model():
    """Create YOLOv5 model with LoRA but without ControlNet"""
    # Use the same config as your project
    cfg = my_config.model.yolo_cfg
    weights = my_config.model.yolo_weights

    # Create model without ControlNet
    model = create_combined_model(
        cfg=cfg,
        yolo_weights=weights,
        controlnet_weights=None,
        lora_weights=None,
        img_size=my_config.model.img_size,
        nc=my_config.model.num_classes,
        control_scale=0.0,  # Disable controlnet influence
    )

    model.controlnet = None
    # Make sure all parameters are not trainable
    for param in model.parameters():
        param.requires_grad = False

    # Enable LoRA training
    model.use_controlnet = False
    model.train_lora()

    return model


def create_full_model():
    """Create YOLOv5 model with both LoRA and ControlNet"""
    # Use the same config as your project
    cfg = my_config.model.yolo_cfg
    weights = my_config.model.yolo_weights

    # Create full model
    model = create_combined_model(
        cfg=cfg,
        yolo_weights=weights,
        controlnet_weights=None,
        lora_weights=None,
        img_size=my_config.model.img_size,
        nc=my_config.model.num_classes,
    )

    # Make sure all parameters are initially not trainable
    for param in model.parameters():
        param.requires_grad = False

    # Enable LoRA and ControlNet training
    model.train_lora()
    model.train_controlnet()

    return model


def create_full_model_infer_shared():
    """Create YOLOv5 model with both LoRA and ControlNet"""
    # Use the same config as your project
    cfg = my_config.model.yolo_cfg
    weights = my_config.model.yolo_weights

    # Create full model
    model = create_combined_model(
        cfg=cfg,
        yolo_weights=weights,
        controlnet_weights=None,
        lora_weights=None,
        img_size=my_config.model.img_size,
        nc=my_config.model.num_classes,
    )

    # Make sure all parameters are initially not trainable
    for param in model.parameters():
        param.requires_grad = False

    model.yolo.model = model.yolo.model.merge_and_unload()

    return model


def create_full_model_infer_fused():
    """Create YOLOv5 model with both LoRA and ControlNet"""
    # Use the same config as your project
    cfg = my_config.model.yolo_cfg
    weights = my_config.model.yolo_weights

    # Create full model
    model = create_combined_model(
        cfg=cfg,
        yolo_weights=weights,
        controlnet_weights="/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs/yolov5n/0.2/bg_sub/control_lora + yolo_lora/checkpoints/best_controlnet.pt",
        lora_weights=None,
        img_size=my_config.model.img_size,
        nc=my_config.model.num_classes,
        should_share_weights=False,
    )

    # Make sure all parameters are initially not trainable
    for param in model.parameters():
        param.requires_grad = False

    model.yolo.model = model.yolo.model.merge_and_unload()
    model.controlnet.nodes = model.controlnet.nodes.merge_and_unload()
    return model


def visualize_results(results):
    """Create visualizations for the analysis results"""

    # Pie charts for trainable vs non-trainable parameters
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))

    for i, (result, ax) in enumerate(zip(results, axes)):
        name = result["name"]
        trainable = result["trainable"]
        non_trainable = result["total"] - trainable

        sizes = [trainable, non_trainable]
        labels = [
            f'Trainable\n{trainable/1e6:.1f}M ({result["percentage"]:.1f}%)',
            f'Non-trainable\n{non_trainable/1e6:.1f}M ({100-result["percentage"]:.1f}%)',
        ]
        colors = ["#ff9999", "#66b3ff"]

        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distribution.png", dpi=300)
    plt.savefig(output_dir / "parameter_distribution.pdf")

    print(f"\nVisualizations saved to {output_dir}")


def visualize_infer(results):
    """Create visualizations for the analysis results"""
    base_model = results[0]

    # Pie charts for trainable vs non-trainable parameters
    fig, axes = plt.subplots(1, len(results) - 1, figsize=(15, 5))

    for i, (result, ax) in enumerate(zip(results[1:], axes)):
        name = result["name"]
        base = base_model["total"]
        additonal = result["total"] - base

        sizes = [base, additonal]
        labels = [
            f'Base\n{base/1e6:.1f}M ({result["percentage"]:.1f}%)',
            f'Added\n{additonal/1e6:.1f}M ({100-result["percentage"]:.1f}%)',
        ]
        colors = ["#66b3ff", "#ff9999"]

        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distribution_infer.png", dpi=300)
    plt.savefig(output_dir / "parameter_distribution_infer.pdf")

    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    print("Starting model parameter analysis...")
    analyze_model_configs()
    print("\nAnalysis completed successfully!")
