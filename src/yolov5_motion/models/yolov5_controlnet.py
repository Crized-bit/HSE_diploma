import peft
from sympy import im
import torch
import torch.nn as nn
import os
import sys
from matplotlib import pyplot as plt
from pathlib import Path

sys.path.append("/home/jovyan/p.kudrevatyh/yolov5")

from models.yolo import Detect, Segment, Model as YOLOv5Model  # type: ignore
from utils.general import check_img_size  # type: ignore
from utils.torch_utils import model_info  # type: ignore

from yolov5_motion.models.blocks import ControlNetModel
from yolov5_motion.config import my_config


class CustomYOLOwithPEFT(YOLOv5Model):
    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        if hasattr(self.model, "base_model"):
            m = self.model.base_model.model[-1]
        else:
            m = self.model[-1]
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return torch.nn.Module._apply(self, fn)


class YOLOv5WithControlNet(nn.Module):
    def __init__(
        self,
        cfg="yolov5m.yaml",
        ch=3,
        nc=80,
        anchors=None,
        yolo_weights=None,
        controlnet_weights=None,
        lora_weights=None,
        lora_scale: float = 1.0,
        control_scale: float = 1.0,
        lora_alpha: float = 16.0,
        lora_rank: int = 32,
    ):
        """
        Initializes YOLOv5 model with ControlNet integration

        Args:
            cfg: YOLOv5 model configuration file
            ch: input channels
            nc: number of classes
            anchors: detection anchors
        """
        super().__init__()
        # Initialize YOLOv5 base model
        self.yolo = CustomYOLOwithPEFT(cfg, ch=ch, nc=nc, anchors=anchors)

        if yolo_weights is not None:
            current_model_dict = self.yolo.state_dict()
            new_state_dict = {
                k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
                for k, v in zip(current_model_dict.keys(), yolo_weights.values())
            }
            missing_keys, unexpected = self.yolo.load_state_dict(new_state_dict, strict=False)
            print("missing_keys:", missing_keys)
            print("unexpected:", unexpected)
            print(f"Loaded YOLOv5 weights from file")
        config = peft.LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=r"1[7-9].*\.conv|2[0-3].*\.conv|24\.m\.[0-2]", bias="none"
        )
        # Initialize ControlNet with the YOLOv5 model
        self.controlnet = my_config.model.model_cls(self.yolo)
        if controlnet_weights is not None:
            self.controlnet.load_state_dict(controlnet_weights)
            print(f"Loaded ControlNet weights from file")

        if not lora_weights:
            self.yolo.model = peft.get_peft_model(self.yolo.model, config)
        else:
            self.yolo.model = peft.PeftModel.from_pretrained(
                self.yolo.model,
                lora_weights,
                adapter_name="default",
                is_trainable=False,
                adapter_scale=lora_scale,  # Установка масштаба при загрузке
            )

        self.control_scale = control_scale
        # Flag to enable/disable ControlNet during inference
        self.use_controlnet = True

        # Initially we don't train anything
        for param in self.parameters():
            param.requires_grad = False

        print(model_info(self, verbose=True))

        total_params = 0
        trainable_params = 0
        for name, param in self.yolo.model.base_model.named_parameters():
            total_params += param.numel()
            if "lora" in name:
                trainable_params += param.numel()

        # Вычисление процента
        percentage = (trainable_params / total_params) * 100
        print(f"LoRA добавляет {percentage:.2f}% к общему количеству параметров")

    def forward(self, x, condition_img=None):
        """
        Forward pass through the combined model

        Args:
            x: input image tensor
            condition_img: conditioning image for ControlNet (same size as x)

        Returns:
            YOLOv5 detection output with ControlNet influence
        """

        # Forward pass through ControlNet
        if self.use_controlnet and condition_img is not None:
            control_outputs = self.controlnet(x, condition_img)

        # Store intermediate feature maps that will be modified by ControlNet
        control_indices = [17, 18, 19, 20, 22, 23]
        # Initial processing through YOLO backbone
        y = []  # outputs
        i = 0
        if isinstance(self.yolo.model, peft.PeftModel):
            model = self.yolo.model.base_model.model
        else:
            model = self.yolo.model
        for idx, m in enumerate(model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # Run layer
            x = m(x)

            # Store feature map before modifications if it's a control point
            if self.use_controlnet and condition_img is not None and idx in control_indices:
                x = x + self.control_scale * control_outputs[i]
                i += 1
            # Store output
            y.append(x if m.i in self.yolo.save else None)

        return x  # Return final detection output

    def train_controlnet(self):
        """Set model to train only ControlNet parameters"""
        self.controlnet.train()

    def train_lora(self):
        """Set model to train only LoRA parameters"""
        # Unfreeze LoRA parameters
        for name, param in self.yolo.model.base_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

    def save_controlnet(self, path):
        """Save only the ControlNet part of the model

        Args:
            path: path to save the ControlNet state dict
        """
        torch.save(self.controlnet.state_dict(), path)

    def save_lora(self, path):
        """Save only the LoRA part of the model

        Args:
            path: path to save the LoRA state dict
        """
        # TODO: save
        self.yolo.model.save_pretrained(path)

    def disable_lora(self):
        self.yolo.model.disable_adapter_layers()

    def enable_lora(self):
        self.yolo.model.enable_adapter_layers()


# Helper function to create the combined model
def create_combined_model(
    cfg, yolo_weights=None, controlnet_weights=None, lora_weights=None, img_size=640, nc=80, lora_scale=1.0, control_scale=1.0
):
    """
    Create YOLOv5 with ControlNet integration

    Args:
        yolo_weights: path to YOLOv5 weights (if None, initialized with random weights)
        controlnet_weights: path to ControlNet weights (if None, initialized with zeros)
        cfg: YOLOv5 model configuration
        img_size: input image size
        nc: number of classes

    Returns:
        Combined YOLOv5+ControlNet model
    """

    # Load YOLOv5 weights if provided
    if yolo_weights:
        if yolo_weights.endswith(".pt"):
            if os.path.exists(yolo_weights):
                state_dict = torch.load(yolo_weights, map_location="cpu")
                # Handle different YOLOv5 weight formats
                if isinstance(state_dict, dict):
                    if "model" in state_dict:
                        state_dict = state_dict["model"]
                    if hasattr(state_dict, "float"):
                        state_dict = state_dict.float()
                    if hasattr(state_dict, "state_dict"):
                        state_dict = state_dict.state_dict()
    else:
        print(f"Warning: YOLOv5 weights file {yolo_weights} not found")
        state_dict = None

    # Load YOLOv5 weights if provided
    if controlnet_weights:
        if controlnet_weights.endswith(".pt"):
            if os.path.exists(controlnet_weights):
                control_dict = torch.load(controlnet_weights, map_location="cpu")
                # Handle different YOLOv5 weight formats
                if isinstance(control_dict, dict):
                    if "model" in control_dict:
                        control_dict = control_dict["model"]
                    if hasattr(control_dict, "float"):
                        control_dict = control_dict.float()
                    if hasattr(control_dict, "state_dict"):
                        control_dict = control_dict.state_dict()
    else:
        print(f"Warning: YOLOv5 weights file {yolo_weights} not found")
        control_dict = None

    # Create model
    model = YOLOv5WithControlNet(
        cfg=cfg,
        nc=nc,
        yolo_weights=state_dict,
        controlnet_weights=control_dict,
        lora_weights=lora_weights,
        lora_scale=lora_scale,
        control_scale=control_scale,
    )

    # Make sure img_size is divisible by stride
    gs = max(int(model.yolo.stride.max()), 32)
    img_size = check_img_size(img_size, gs)

    return model


class GradientTracker:
    def __init__(self, trainer):
        self.trainer = trainer
        self.gradient_norms = {}
        self.hook_handles = []
        self.module_names = {}

    def register_hooks(self):
        """Register hooks to track gradient norms for ControlNetModel modules"""
        # Create a mapping of modules to their names for easier identification
        for name, module in self.trainer.model.controlnet.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Sequential)) and name != "":
                self.module_names[module] = name
                handle = module.register_full_backward_hook(self._backward_hook)
                self.hook_handles.append(handle)

        print(f"Registered gradient hooks for {len(self.hook_handles)} modules in ControlNetModel")

    def _backward_hook(self, module, grad_input, grad_output):
        """Hook function to record gradient norms during backpropagation"""
        module_name = self.module_names.get(module, "unknown")
        # Use the norm of the gradient output for visualization
        if grad_output and isinstance(grad_output, tuple) and grad_output[0] is not None:
            norm = grad_output[0].norm().item()
            if module_name not in self.gradient_norms:
                self.gradient_norms[module_name] = []
            self.gradient_norms[module_name].append(norm)

    def visualize_gradient_norms(self, epoch):
        """Visualize gradient norms for ControlNetModel layers"""
        if not self.gradient_norms:
            print("No gradient norms collected yet")
            return

        # Group parameters by layer type
        layer_norms = {}
        for name, norms in self.gradient_norms.items():
            # Extract layer type from name (e.g., 'convs.0.0' -> 'convs_0')
            parts = name.split(".")
            if len(parts) >= 2:
                layer_name = f"{parts[0]}_{parts[1]}"
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                # Append the mean norm for this parameter
                if norms:
                    layer_norms[layer_name].append(sum(norms) / len(norms))

        # Calculate mean norm for each layer type
        mean_layer_norms = {layer: sum(norms) / len(norms) for layer, norms in layer_norms.items()}

        # Sort layers by name for consistent ordering
        sorted_layers = sorted(mean_layer_norms.keys())

        # Create visualization
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(sorted_layers)), [mean_layer_norms[layer] for layer in sorted_layers])

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}", ha="center", va="bottom", rotation=0)

        plt.xticks(range(len(sorted_layers)), sorted_layers, rotation=45)
        plt.xlabel("ControlNet Layer")
        plt.ylabel("Mean Gradient Norm")
        plt.title(f"Mean Gradient Norms for ControlNet Layers (Epoch {epoch+1})")
        plt.tight_layout()

        # Save figure
        grad_viz_dir = self.trainer.output_dir / "gradient_visualizations"
        grad_viz_dir.mkdir(exist_ok=True)
        plt.savefig(str(grad_viz_dir / f"gradient_norms_epoch_{epoch+1}.png"))
        plt.close()

        # Also log to tensorboard
        for layer in sorted_layers:
            self.trainer.writer.add_scalar(f"gradients/{layer}", mean_layer_norms[layer], epoch)

        # Reset gradient norms for next epoch
        self.gradient_norms = {}

    def visualize_scaling_factors(self, epoch):
        """Visualize learned scaling factors for ControlNetModel"""
        if hasattr(self.trainer.model.controlnet, "scale_factors") and self.trainer.model.controlnet.scale_factors is not None:
            scale_factors = self.trainer.model.controlnet.scale_factors.detach().cpu().numpy()

            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(scale_factors)), scale_factors)

            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}", ha="center", va="bottom", rotation=0)

            plt.xticks(range(len(scale_factors)), [f"Layer {i+1}" for i in range(len(scale_factors))])
            plt.xlabel("Control Layer")
            plt.ylabel("Scaling Factor")
            plt.title(f"Learned Scaling Factors for Control Layers (Epoch {epoch+1})")
            plt.ylim(0, max(2.0, max(scale_factors) * 1.1))  # Set reasonable y-axis limits
            plt.tight_layout()

            # Save figure
            scale_viz_dir = self.trainer.output_dir / "scaling_visualizations"
            scale_viz_dir.mkdir(exist_ok=True)
            plt.savefig(str(scale_viz_dir / f"scaling_factors_epoch_{epoch+1}.png"))
            plt.close()

            # Log to tensorboard
            for i, factor in enumerate(scale_factors):
                self.trainer.writer.add_scalar(f"scaling_factors/layer_{i+1}", factor, epoch)

    def remove_hooks(self):
        """Remove all hooks to prevent memory leaks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


# Example usage
if __name__ == "__main__":

    # Create model with pretrained weights
    weights = "/home/jovyan/p.kudrevatyh/yolov5s.pt"
    model = create_combined_model(
        yolo_weights=weights,  # Load pretrained YOLOv5 weights
        controlnet_weights=None,  # Initialize ControlNet from model encoder
        cfg="/home/jovyan/p.kudrevatyh/yolov5/models/yolov5s.yaml",  # Use YOLOv5m configuration
        img_size=640,
        nc=80,
    )
    # print(model)
    # Example of saving and loading just the ControlNet portion
    # After training
    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "global_step": None,
        "best_val_loss": None,
    }

    # # Save regular checkpoint
    checkpoint_path = Path("/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs/yolov5s/base_model/checkpoints") / f"best_model.pt"
    torch.save(checkpoint, checkpoint_path)
    # model.save_lora("/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs/lora/checkpoints/lora.pt")

    # Example inputs (batch size 1, RGB images)
    # input_img = torch.randn(1, 3, 640, 640)
    # condition_img = torch.randn(1, 3, 640, 640)

    # # Run inference
    # # with torch.no_grad():
    # outputs = model(input_img, condition_img)
    # # make_dot(tuple(outputs), params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    # print(f"Output shape: {[output.shape for output in outputs]}")

    # # Training example
    # print("Training only ControlNet parameters")
    # model.train_controlnet()  # Train only ControlNet
