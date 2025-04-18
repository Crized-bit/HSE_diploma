from pathlib import Path
from yolov5_motion.models.yolov5_controlnet import create_combined_model
import torch

weights = "/home/jovyan/p.kudrevatyh/yolov5m.pt"
control_weights = "/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs/fixed_bg_sub_200/checkpoints/best_controlnet.pt"
model = create_combined_model(
    yolo_weights=weights,  # Load pretrained YOLOv5 weights
    controlnet_weights=control_weights,  # Initialize ControlNet from model encoder
    cfg="/home/jovyan/p.kudrevatyh/yolov5/models/yolov5m.yaml",  # Use YOLOv5m configuration
    img_size=640,
    nc=80,
)

checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "global_step": None,
        "best_val_loss": None,
    }

    # Save regular checkpoint
checkpoint_path = Path("/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs/base_model_w_controlnet/checkpoints") / f"best_model.pt"
torch.save(checkpoint, checkpoint_path)