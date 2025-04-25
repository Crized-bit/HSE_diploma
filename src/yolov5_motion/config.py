import json
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Optional
from yolov5_motion.models.blocks import ControlNetModel, ControlNetModelLora


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: v for k, v in asdict(obj).items()}
        if isinstance(obj, type):
            return obj.__name__  # Convert type objects to their string name
        return super().default(obj)


@dataclass
class DataConfig:
    preprocessed_dir: str = "/home/jovyan/p.kudrevatyh/bg_subtraction"
    annotations_dir: str = "/home/jovyan/p.kudrevatyh/yolov5_motion/data/annotations"
    splits_file: str = "/home/jovyan/p.kudrevatyh/yolov5_motion/data/splits.json"
    output_dir: str = "/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs/yolov5s"
    control_stack_length: int = 50
    prev_frame_time_diff: float = 0.2
    bbox_skip_percentage: float = 0.2


@dataclass
class ModelConfig:
    model_cls: type = ControlNetModelLora
    yolo_weights: str = "/home/jovyan/p.kudrevatyh/yolov5s.pt"
    yolo_cfg: str = "/home/jovyan/p.kudrevatyh/yolov5/models/yolov5s.yaml"
    controlnet_weights: Optional[str] = None
    img_size: int = 640
    num_classes: int = 80
    train_lora: bool = True
    train_controlnet: bool = True


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
    checkpoint_name: str = "0.2/bg_sub/control_lora + yolo_lora"
    epochs: int = 200
    batch_size: int = 64
    val_batch_size: int = 128
    workers: int = 8
    val_ratio: float = 0.2

    optimizer: str = "prodigy"
    lr: float = 1.
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

    disable_controlnet: bool = False
    disable_lora: bool = False


@dataclass
class A100OptimizedYOLOv5Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    test_epoch: int = 0
    control_scale: float = 1.0


my_config = A100OptimizedYOLOv5Config()
