from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    preprocessed_dir: str = "/home/jovyan/p.kudrevatyh/bg_subtraction"
    annotations_dir: str = "/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations"
    splits_file: str = "/home/jovyan/p.kudrevatyh/yolov5_motion/data/splits.json"
    output_dir: str = "/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs"
    control_stack_length: int = 50
    prev_frame_time_diff: float = 0.2


@dataclass
class ModelConfig:
    yolo_weights: str = "/home/jovyan/p.kudrevatyh/yolov5m.pt"
    controlnet_weights: Optional[str] = None
    yolo_cfg: str = "/home/jovyan/p.kudrevatyh/yolov5/models/yolov5m.yaml"
    img_size: int = 640
    num_classes: int = 80
    train_lora: bool = True
    train_controlnet: bool = False


@dataclass
class LossConfig:
    box_weight: float = 0.0539
    obj_weight: float = 0.632
    cls_weight: float = 0.299


@dataclass
class DetectionConfig:
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 1000


@dataclass
class TrainingConfig:
    checkpoint_name: str = "test_chkp"
    epochs: int = 200
    batch_size: int = 64
    val_batch_size: int = 128
    workers: int = 12
    val_ratio: float = 0.2

    optimizer: str = "prodigy"
    lr: float = 1.0
    weight_decay: float = 0.01
    momentum: float = 0.937

    loss: LossConfig = field(default_factory=LossConfig)

    precision: str = "bf16"

    save_interval: int = 10
    log_interval: int = 50
    eval_interval: int = 1

    resume: Optional[str] = None

    detection: DetectionConfig = field(default_factory=DetectionConfig)

    augment: bool = True
    augment_prob: float = 0.5

    disable_controlnet: bool = True
    disable_lora: bool = False


@dataclass
class A100OptimizedYOLOv5Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    test_epoch: int = 0
    control_scale: float = 1.0


my_config = A100OptimizedYOLOv5Config()
