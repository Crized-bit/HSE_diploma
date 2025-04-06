import torch
import torch.nn as nn
import os

# Import Ultralytics YOLOv5
import sys

sys.path.append("/home/jovyan/p.kudrevatyh/yolov5")

from models.yolo import Model as YOLOv5Model
from utils.torch_utils import model_info
from utils.general import check_img_size

from yolov5_motion.models.blocks import ControlNetModel


class YOLOv5WithControlNet(nn.Module):
    def __init__(self, cfg="yolov5m.yaml", ch=3, nc=80, anchors=None):
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
        self.yolo = YOLOv5Model(cfg, ch=ch, nc=nc, anchors=anchors)

        # self.yolo.nc = nc  # attach number of classes to model

        # Initialize ControlNet with the YOLOv5 model
        self.controlnet = ControlNetModel(self.yolo)

        # Flag to enable/disable ControlNet during inference
        self.use_controlnet = True

        print(model_info(self, verbose=True))

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
            control_outputs = self.controlnet(condition_img)

        # Store intermediate feature maps that will be modified by ControlNet
        control_indices = [17, 19, 22]
        # Initial processing through YOLO backbone
        y = []  # outputs
        for i, m in enumerate(self.yolo.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # Run layer
            x = m(x)

            # Store feature map before modifications if it's a control point
            if self.use_controlnet and condition_img is not None and i in control_indices:
                x += control_outputs[control_indices.index(i)]

            # Store output
            y.append(x if m.i in self.yolo.save else None)

        return x  # Return final detection output

    def train_controlnet(self):
        """Set model to train only ControlNet parameters"""
        # Freeze YOLOv5 parameters
        for param in self.yolo.parameters():
            param.requires_grad = False

        # Unfreeze ControlNet parameters
        for param in self.controlnet.parameters():
            param.requires_grad = True

    def train_all(self):
        """Set model to train all parameters"""
        for param in self.parameters():
            param.requires_grad = True

    def save_controlnet(self, path):
        """Save only the ControlNet part of the model

        Args:
            path: path to save the ControlNet state dict
        """
        torch.save(self.controlnet.state_dict(), path)

    def load_controlnet(self, path):
        """Load only the ControlNet part of the model

        Args:
            path: path to the ControlNet state dict
        """
        self.controlnet.load_state_dict(torch.load(path, map_location="cpu"))


# Helper function to create the combined model
def create_combined_model(cfg, yolo_weights=None, controlnet_weights=None, img_size=640, nc=80):
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
    # Create model
    model = YOLOv5WithControlNet(cfg=cfg, nc=nc)

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
                model.yolo.load_state_dict(state_dict, strict=False)
                print(f"Loaded YOLOv5 weights from {yolo_weights}")
            else:
                print(f"Warning: YOLOv5 weights file {yolo_weights} not found")

    # Load ControlNet weights if provided
    if controlnet_weights:
        if controlnet_weights.endswith(".pt"):
            if os.path.exists(controlnet_weights):
                model.load_controlnet(controlnet_weights)
                print(f"Loaded ControlNet weights from {controlnet_weights}")
            else:
                print(f"Warning: ControlNet weights file {controlnet_weights} not found")

    # Make sure img_size is divisible by stride
    gs = max(int(model.yolo.stride.max()), 32)
    img_size = check_img_size(img_size, gs)

    return model


# Example usage
if __name__ == "__main__":
    # Create model with pretrained weights
    weights = "/home/jovyan/p.kudrevatyh/yolov5m.pt"
    model = create_combined_model(
        yolo_weights=weights,  # Load pretrained YOLOv5 weights
        controlnet_weights=None,  # Initialize ControlNet with zeros
        cfg="/home/jovyan/p.kudrevatyh/yolov5/models/yolov5m.yaml",  # Use YOLOv5m configuration
        img_size=640,
    )

    print(model)
    # Example of saving and loading just the ControlNet portion
    # After training
    model.save_controlnet("controlnet_weights.pt")

    # Later, to load a pretrained ControlNet
    new_model = create_combined_model(
        cfg="/home/jovyan/p.kudrevatyh/yolov5/models/yolov5m.yaml", yolo_weights=weights, controlnet_weights="controlnet_weights.pt"
    )

    # Example inputs (batch size 1, RGB images)
    input_img = torch.randn(1, 3, 640, 640)
    condition_img = torch.randn(1, 3, 640, 640)

    # Run inference
    with torch.no_grad():
        outputs = model(input_img, condition_img)
        print(f"Output shape: {[output.shape for output in outputs]}")

    # Training example
    print("Training only ControlNet parameters")
    model.train_controlnet()  # Train only ControlNet

    # After some training, switch to training the full model
    print("Training all parameters")
    model.train_all()
