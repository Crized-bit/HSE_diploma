import argparse
import time
import os
import sys
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from datetime import datetime
from collections import defaultdict
import matplotlib.cm as cm

# Add YOLOv5 directory to path to import YOLOv5 modules
yolov5_dir = "/home/jovyan/p.kudrevatyh/yolov5"
if os.path.exists(yolov5_dir):
    sys.path.append(yolov5_dir)
    print(f"Added YOLOv5 directory to path: {yolov5_dir}")
else:
    print(f"Warning: YOLOv5 directory not found at {yolov5_dir}")

# Import YOLOv5 modules for loss computation and bounding box processing
try:
    from utils.loss import ComputeLoss  # type: ignore
    from utils.general import non_max_suppression, scale_boxes  # type: ignore

    print("Successfully imported YOLOv5 loss computation modules")
except ImportError as e:
    print(f"Warning: Could not import YOLOv5 modules: {e}")
    # print("Will use placeholder loss function instead")

# Import custom modules
from yolov5_motion.models.yolov5_controlnet import create_combined_model, GradientTracker
from yolov5_motion.data.dataset_splits import create_dataset_splits, get_dataloaders
from yolov5_motion.utils.metrics import calculate_precision_recall, calculate_map

# Try to import Prodigy optimizer if available
try:
    from prodigyopt import Prodigy

    PRODIGY_AVAILABLE = True
except ImportError:
    PRODIGY_AVAILABLE = False
    print("Prodigy optimizer not available. Will use Adam instead.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv5 with ControlNet for Motion")

    # Input directories
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Directory containing preprocessed frames")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory containing annotation files")
    parser.add_argument("--splits_file", type=str, required=True, help="Path to splits JSON file")

    # Output directories
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model checkpoints and logs")

    # Model configuration
    parser.add_argument("--yolo_weights", type=str, default=None, help="Path to YOLOv5 weights (.pt file)")
    parser.add_argument("--controlnet_weights", type=str, default=None, help="Path to ControlNet weights (.pt file)")
    parser.add_argument("--yolo_cfg", type=str, default="yolov5m.yaml", help="Path to YOLOv5 model configuration (.yaml file)")
    parser.add_argument("--img_size", type=int, default=640, help="Input image size")
    parser.add_argument("--num_classes", type=int, default=80, help="Number of classes in the dataset")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of training data to use for validation")

    # Training mode
    parser.add_argument("--train_controlnet", action="store_true", help="Train only the ControlNet")
    parser.add_argument("--train_head", action="store_true", help="Train head of YOLOv5")
    parser.add_argument("--train_all", action="store_true", help="Train all layers of YOLOv5 and ControlNet model")

    # Augmentation parameters
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--augment_prob", type=float, default=0.5, help="Probability of applying data augmentation")
    # Optimizer selection
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "prodigy"], help="Optimizer to use for training")

    # Detection metrics
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max_det", type=int, default=100, help="Maximum number of detections per image")

    # Saving and logging

    parser.add_argument("--save_interval", type=int, default=10, help="Interval (in epochs) to save model checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval (in iterations) to log training progress")
    parser.add_argument("--eval_interval", type=int, default=5, help="Interval (in epochs) to evaluate on validation set")

    # Resume training
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    # Mixed precision training
    parser.add_argument(
        "--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Precision for training (fp32, fp16, or bf16)"
    )

    # Loss weights
    parser.add_argument("--box_weight", type=float, default=0.05, help="Weight for box loss")
    parser.add_argument("--obj_weight", type=float, default=1.0, help="Weight for objectness loss")
    parser.add_argument("--cls_weight", type=float, default=0.5, help="Weight for class loss")

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir) / timestamp
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.viz_dir = self.output_dir / "visualizations"
        self.metrics_dir = self.output_dir / "metrics"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)  # Create metrics directory

        # Track detection metrics for plotting
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        self.val_map50 = []
        self.val_map = []
        # Track metrics for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_box_losses = []
        self.train_obj_losses = []
        self.train_cls_losses = []
        self.val_box_losses = []
        self.val_obj_losses = []
        self.val_cls_losses = []
        self.lr_history = []

        self.gradient_tracker = GradientTracker(self)
        # Save the arguments
        with open(self.output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))

        # Save the arguments
        with open(self.output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

        # Load model
        print("Creating model...")
        self.model = create_combined_model(
            cfg=args.yolo_cfg,
            yolo_weights=args.yolo_weights,
            controlnet_weights=args.controlnet_weights,
            img_size=args.img_size,
            nc=args.num_classes,
        )
        self.model = self.model.to(self.device)

        # Initialize loss function if YOLOv5 loss is available
        try:
            if hasattr(self.model.yolo, "hyp"):
                self.compute_loss_fn = ComputeLoss(self.model.yolo)
            else:
                # Create default hyperparameters for loss function
                default_hyp = {
                    "box": args.box_weight,  # box loss gain
                    "cls": args.cls_weight,  # cls loss gain
                    "cls_pw": 0.0825,  # cls BCELoss positive_weight
                    "obj": args.obj_weight,  # obj loss gain
                    "obj_pw": 1.0,  # obj BCELoss positive_weight
                    "fl_gamma": 0.0,  # focal loss gamma
                    "anchor_t": 3.44,  # anchor-multiple threshold
                }
                # Set hyperparameters on the model
                self.model.yolo.hyp = default_hyp
                self.compute_loss_fn = ComputeLoss(self.model.yolo)
        except NameError:
            self.compute_loss_fn = None
            print("Using placeholder loss function (YOLOv5 loss not available)")

        # Set training mode
        if args.train_head:
            print("Training YOLOv5 head")
            self.model.train_head()

        if args.train_controlnet:
            print("Training ControlNet")
            self.model.train_controlnet()

        if args.train_all:
            print("Training all parameters")
            self.model.train_all()

        # Create dataloaders
        print("Creating datasets and dataloaders...")
        self.datasets = create_dataset_splits(
            preprocessed_dir=args.preprocessed_dir,
            annotations_dir=args.annotations_dir,
            splits_file=args.splits_file,
            val_ratio=args.val_ratio,
            augment=args.augment,  # Enable augmentation
            augment_prob=args.augment_prob,
        )

        self.dataloaders = get_dataloaders(datasets=self.datasets, batch_size=args.batch_size, num_workers=args.workers)

        # Adjust validation batch size separately if specified
        if args.val_batch_size != args.batch_size:
            from torch.utils.data import DataLoader
            from yolov5_motion.data.dataset import collate_fn

            self.dataloaders["val"] = DataLoader(
                self.datasets["val"],
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs)
        # Resume training if specified
        self.start_epoch = 0
        if args.resume:
            self._resume_checkpoint(args.resume)

        # Initialize training variables
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Setup precision mode
        self.precision = args.precision
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("Warning: BF16 not supported on this device. Falling back to FP32.")
            self.precision = "fp32"

        print(f"Training with {self.precision} precision")

    def _create_optimizer(self):
        """Create the optimizer based on the command line arguments"""
        # Get parameters that require gradients
        parameters = [p for p in self.model.parameters() if p.requires_grad]

        if self.args.optimizer.lower() == "adam":
            return optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == "sgd":
            return optim.SGD(parameters, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == "prodigy":
            if not PRODIGY_AVAILABLE:
                print("Prodigy optimizer requested but not available. Using Adam instead.")
                return optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return Prodigy(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay, safeguard_warmup=False, use_bias_correction=True)
        else:
            print(f"Unknown optimizer {self.args.optimizer}, using Adam")
            return optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _resume_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint"""
        print(f"Resuming from checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load other training state
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Resumed from epoch {checkpoint['epoch']}")

    def save_checkpoint(self, epoch, is_best=False):
        """Save a checkpoint of the model and training state"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save ControlNet weights separately
        controlnet_path = self.checkpoints_dir / f"controlnet_epoch_{epoch}.pt"
        self.model.save_controlnet(controlnet_path)

        # Save best model if this is the best
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

            best_controlnet_path = self.checkpoints_dir / "best_controlnet.pt"
            self.model.save_controlnet(best_controlnet_path)

    def compute_loss(self, outputs, targets):
        """
        Compute YOLOv5 loss for detection outputs and ground truth targets

        Args:
            outputs: Model outputs from forward pass
            targets: Ground truth target annotations

        Returns:
            (torch.Tensor): Total loss
            (dict): Loss components (box, obj, cls)
        """
        # If YOLOv5 loss is available, use it
        if self.compute_loss_fn is not None:
            # Convert targets from our format to YOLOv5 format
            yolo_targets = self._convert_targets_to_yolo_format(targets).to(self.device)

            # Compute loss
            loss, loss_items = self.compute_loss_fn(outputs, yolo_targets)

            # Extract individual loss components
            if isinstance(loss_items, torch.Tensor) and len(loss_items) >= 3:
                box_loss = loss_items[0]
                obj_loss = loss_items[1]
                cls_loss = loss_items[2]
            else:
                # Fallback if loss_items is not as expected
                box_loss = torch.tensor(0.0, device=self.device)
                obj_loss = torch.tensor(0.0, device=self.device)
                cls_loss = torch.tensor(0.0, device=self.device)

            # Create metrics dictionary
            metrics = {"box_loss": box_loss.item(), "obj_loss": obj_loss.item(), "cls_loss": cls_loss.item(), "total_loss": loss.item()}

            return loss, metrics

    def _convert_targets_to_yolo_format(self, targets):
        """
        Convert targets from our dataset format to YOLOv5 format

        Args:
            targets: List of annotation dictionaries

        Returns:
            torch.Tensor: Tensor with shape [num_targets, 6] where each row is
                          [batch_idx, class_idx, x, y, w, h]
        """
        # Initialize list to hold all target rows
        all_targets = []
        img_size = self.args.img_size

        # Process each batch item
        for batch_idx, annotations in enumerate(targets):
            for ann in annotations:
                # Extract bounding box
                bbox = torch.tensor(ann["bbox"], dtype=torch.float32)

                # Normalize coordinates to [0,1]
                normalized_bbox = torch.zeros_like(bbox)
                normalized_bbox[0] = bbox[0] / img_size  # normalize center_x
                normalized_bbox[1] = bbox[1] / img_size  # normalize center_y
                normalized_bbox[2] = bbox[2] / img_size  # normalize width
                normalized_bbox[3] = bbox[3] / img_size  # normalize height

                # Для модели с единственным классом (человек), используем class_idx = 0
                class_idx = 0

                # Create target row [batch_idx, class_idx, x, y, w, h]
                target_row = torch.tensor([batch_idx, class_idx, *normalized_bbox], dtype=torch.float32)
                all_targets.append(target_row)

        # Combine all target rows
        if all_targets:
            return torch.stack(all_targets)
        else:
            # Return empty tensor with correct shape if no targets
            return torch.zeros((0, 6), dtype=torch.float32, device=self.device)

    def train_epoch(self, epoch):
        """Train the model for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {"box_loss": 0, "obj_loss": 0, "cls_loss": 0}

        pbar = tqdm(self.dataloaders["train"], desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for i, batch in enumerate(pbar):
            # Get data and move to device
            current_frames = batch["current_frames"].to(self.device)
            control_images = batch["control_images"].to(self.device)
            targets = batch["annotations"]  # List of annotation dictionaries

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with selected precision
            if self.precision == "fp16" or self.precision == "bf16":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictions = self.model(current_frames, control_images)
                    loss, metrics = self.compute_loss(predictions, targets)
            else:
                # Regular FP32 forward pass
                predictions = self.model(current_frames, control_images)
                loss, metrics = self.compute_loss(predictions, targets)

            # Backward pass (no scaler needed for bf16)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.controlnet.parameters(), max_norm=10.0)
            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "box_loss": f"{metrics['box_loss']}",
                    "obj_loss": f"{metrics['obj_loss']}",
                    "cls_loss": f"{metrics['cls_loss']}",
                }
            )

            # Log metrics at regular intervals
            if i % self.args.log_interval == 0:
                # Log to tensorboard
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/box_loss", metrics["box_loss"], self.global_step)
                self.writer.add_scalar("train/obj_loss", metrics["obj_loss"], self.global_step)
                self.writer.add_scalar("train/cls_loss", metrics["cls_loss"], self.global_step)

                # You can also log learning rate
                for param_group in self.optimizer.param_groups:
                    self.writer.add_scalar("train/lr", param_group["lr"], self.global_step)

            self.global_step += 1
        self.scheduler.step()
        # Compute average metrics for the epoch
        num_batches = len(self.dataloaders["train"])
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        # Log epoch metrics
        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"train/epoch_{k}", v, epoch)

        # Store metrics for plotting
        self.train_losses.append(avg_loss)
        self.train_box_losses.append(avg_metrics.get("box_loss", 0))
        self.train_obj_losses.append(avg_metrics.get("obj_loss", 0))
        self.train_cls_losses.append(avg_metrics.get("cls_loss", 0))
        self.lr_history.append(self.optimizer.param_groups[0]["lr"])

        return avg_loss, avg_metrics

    def validate(self, epoch):
        """Validate the model on the validation set"""
        self.model.eval()
        val_loss = 0
        val_metrics = {"box_loss": 0, "obj_loss": 0, "cls_loss": 0}

        # Lists to store predictions and ground truth for calculating metrics
        all_pred_boxes = []
        all_true_boxes = []

        # Initialize metrics collector
        precision_by_class = defaultdict(list)
        recall_by_class = defaultdict(list)
        f1_by_class = defaultdict(list)

        with torch.no_grad():
            pbar = tqdm(self.dataloaders["val"], desc=f"Validating epoch {epoch+1}")

            for batch in pbar:
                # Get data and move to device
                current_frames = batch["current_frames"].to(self.device)
                control_images = batch["control_images"].to(self.device)
                targets = batch["annotations"]

                # Forward pass
                predictions = self.model(current_frames, control_images)

                # Get raw detection outputs
                detections = non_max_suppression(
                    predictions[0], conf_thres=self.args.conf_thres, iou_thres=self.args.iou_thres, max_det=self.args.max_det
                )

                # Process each image in the batch
                for i, det in enumerate(detections):
                    pred_boxes = []
                    if det is not None and len(det):
                        # Rescale boxes to original image size
                        det[:, :4] = scale_boxes(current_frames.shape[2:], det[:, :4], (640, 640)).round()

                        # Convert to list format expected by metrics functions
                        for *xyxy, conf, cls_id in det:
                            pred_boxes.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item(), cls_id.item()])

                    # Process ground truth
                    true_boxes = []
                    if i < len(targets):
                        for ann in targets[i]:
                            # Convert from center format to corner format
                            bbox = ann["bbox"]
                            x1 = bbox[0] - bbox[2] / 2
                            y1 = bbox[1] - bbox[3] / 2
                            x2 = bbox[0] + bbox[2] / 2
                            y2 = bbox[1] + bbox[3] / 2

                            # Get class ID (assumes single class in this case)
                            cls_id = 0  # Default to first class

                            true_boxes.append([x1, y1, x2, y2, cls_id])

                    # Calculate precision and recall for this image
                    if len(true_boxes) > 0 or len(pred_boxes) > 0:
                        precision, recall, f1 = calculate_precision_recall(
                            pred_boxes, true_boxes, iou_threshold=0.5, conf_threshold=self.args.conf_thres
                        )

                        # Store by class (for multi-class detection)
                        class_ids = set([box[5] for box in pred_boxes if len(box) > 5] + [box[4] for box in true_boxes if len(box) > 4])

                        for cls_id in class_ids:
                            precision_by_class[cls_id].append(precision)
                            recall_by_class[cls_id].append(recall)
                            f1_by_class[cls_id].append(f1)

                    # Store for mAP calculation
                    all_pred_boxes.append(pred_boxes)
                    all_true_boxes.append(true_boxes)

                # Calculate loss for training metrics
                _, train_output = predictions
                loss, metrics = self.compute_loss(train_output, targets)

                # Update metrics
                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v

                # Update progress bar
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        # Compute average metrics
        num_batches = len(self.dataloaders["val"])
        avg_val_loss = val_loss / num_batches
        avg_val_metrics = {k: v / num_batches for k, v in val_metrics.items()}

        # Calculate mAP using our utility function
        map50, map = calculate_map(all_pred_boxes, all_true_boxes)

        # Calculate average precision and recall across all images
        avg_precision = np.mean([np.mean(precisions) for cls_id, precisions in precision_by_class.items()]) if precision_by_class else 0
        avg_recall = np.mean([np.mean(recalls) for cls_id, recalls in recall_by_class.items()]) if recall_by_class else 0
        avg_f1 = np.mean([np.mean(f1s) for cls_id, f1s in f1_by_class.items()]) if f1_by_class else 0

        # Add detection metrics to validation metrics
        avg_val_metrics.update({"precision": avg_precision, "recall": avg_recall, "f1": avg_f1, "mAP@0.5": map50, "mAP@0.5:0.95": map})

        # Log validation metrics
        self.writer.add_scalar("val/epoch_loss", avg_val_loss, epoch)
        for k, v in avg_val_metrics.items():
            self.writer.add_scalar(f"val/epoch_{k}", v, epoch)

        # Store metrics for plotting
        self.val_losses.append(avg_val_loss)
        self.val_box_losses.append(avg_val_metrics.get("box_loss", 0))
        self.val_obj_losses.append(avg_val_metrics.get("obj_loss", 0))
        self.val_cls_losses.append(avg_val_metrics.get("cls_loss", 0))

        # Store detection metrics for plotting
        if not hasattr(self, "val_precision"):
            self.val_precision = []
            self.val_recall = []
            self.val_f1 = []
            self.val_map50 = []
            self.val_map = []

        self.val_precision.append(avg_precision)
        self.val_recall.append(avg_recall)
        self.val_f1.append(avg_f1)
        self.val_map50.append(map50)
        self.val_map.append(map)

        # Visualize some validation examples
        if len(self.dataloaders["val"]) > 0:
            self.visualize_predictions(epoch)

        return avg_val_loss, avg_val_metrics

    def visualize_predictions(self, epoch):
        """
        Visualize model predictions on validation data

        Creates visualizations comparing ground truth annotations
        with model predictions and saves them to the visualization directory.
        """
        # Get a batch from validation set
        self.model.eval()
        batch = next(iter(self.dataloaders["val"]))

        current_frames = batch["current_frames"].to(self.device)
        control_images = batch["control_images"].to(self.device)
        annotations = batch["annotations"]

        # Limit to 4 images for visualization
        n_samples = min(4, current_frames.size(0))

        with torch.no_grad():
            # Get predictions
            predictions = self.model(current_frames[:n_samples], control_images[:n_samples])

            # Process predictions using YOLOv5's non_max_suppression if available
            try:
                # Apply non-max suppression to get detections
                detections = non_max_suppression(
                    predictions,
                    conf_thres=self.args.conf_thres,
                    iou_thres=self.args.iou_thres,
                    max_det=self.args.max_det,  # Confidence threshold  # IoU threshold
                )  # Maximum detections

                # Now create visualizations with both ground truth and predictions
                for i in range(n_samples):
                    # Convert tensors to numpy arrays
                    frame = current_frames[i].cpu().permute(1, 2, 0).numpy() * 255
                    frame = frame.astype(np.uint8)

                    control = control_images[i].cpu().permute(1, 2, 0).numpy() * 255
                    control = control.astype(np.uint8)

                    # Draw ground truth annotations in green
                    gt_frame = frame.copy()
                    if i < len(annotations):
                        from yolov5_motion.data.utils import draw_bounding_boxes

                        gt_frame = draw_bounding_boxes(gt_frame, annotations[i], color=(0, 255, 0))

                    # Draw predicted bounding boxes in red
                    pred_frame = frame.copy()
                    if i < len(detections) and detections[i] is not None:
                        # Scale coordinates to image size
                        det = detections[i].clone()
                        det[:, :4] = scale_boxes(current_frames.shape[2:], det[:, :4], pred_frame.shape).round()

                        # Draw each detection
                        for *xyxy, conf, cls in det:
                            # Convert to integers
                            xyxy = [int(x.item()) for x in xyxy]

                            # Draw bounding box
                            cv2.rectangle(pred_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)

                            # Add label with confidence
                            label = f"{int(cls.item())}: {conf:.2f}"
                            cv2.putText(pred_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Create a side-by-side comparison: GT, Predictions, Control Image
                    comparison = np.hstack((gt_frame, pred_frame, control))

                    # Save the visualization
                    cv2.imwrite(str(self.viz_dir / f"val_epoch{epoch}_sample{i}.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

                # Create a panel of all samples
                if n_samples > 0:
                    # Load saved images
                    panel_images = []
                    for i in range(n_samples):
                        img_path = self.viz_dir / f"val_epoch{epoch}_sample{i}.jpg"
                        if img_path.exists():
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                panel_images.append(img)

                    # Create panel if images are available
                    if panel_images:
                        max_width = max(img.shape[1] for img in panel_images)
                        # Create vertical stack with padding to same width
                        panel = []
                        for img in panel_images:
                            if img.shape[1] < max_width:
                                pad_width = max_width - img.shape[1]
                                pad = np.ones((img.shape[0], pad_width, 3), dtype=np.uint8) * 255
                                padded_img = np.hstack((img, pad))
                                panel.append(padded_img)
                            else:
                                panel.append(img)

                        if panel:
                            panel = np.vstack(panel)
                            cv2.imwrite(str(self.viz_dir / f"val_panel_epoch{epoch}.jpg"), panel)

            except (NameError, Exception) as e:
                print(f"Could not generate prediction visualizations: {e}")

                # Fallback: just visualize ground truth and control images
                for i in range(n_samples):
                    # Convert tensors to numpy arrays
                    frame = current_frames[i].cpu().permute(1, 2, 0).numpy() * 255
                    frame = frame.astype(np.uint8)

                    control = control_images[i].cpu().permute(1, 2, 0).numpy() * 255
                    control = control.astype(np.uint8)

                    # Draw ground truth annotations
                    gt_frame = frame.copy()
                    if i < len(annotations):
                        from yolov5_motion.data.utils import draw_bounding_boxes

                        gt_frame = draw_bounding_boxes(gt_frame, annotations[i])

                    # Create a side-by-side comparison
                    comparison = np.hstack((gt_frame, control))

                    # Save the visualization
                    cv2.imwrite(str(self.viz_dir / f"val_epoch{epoch}_sample{i}.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    def plot_metrics(self):
        """Plot and save training metrics with enhanced visualizations"""
        # Create figure with subplots in a 3x2 grid
        plt.style.use("ggplot")  # Use ggplot style for better aesthetics
        fig, axs = plt.subplots(3, 2, figsize=(16, 18))

        # Prepare data
        epochs = range(1, len(self.train_losses) + 1)
        val_epochs = [e * self.args.eval_interval for e in range(1, len(self.val_losses) + 1)]

        # Plot total loss with markers for each point
        axs[0, 0].plot(epochs, self.train_losses, "b-", label="Training Loss", marker="o", markersize=4)
        if self.val_losses:
            axs[0, 0].plot(val_epochs, self.val_losses, "r-", label="Validation Loss", marker="s", markersize=4)
        axs[0, 0].set_title("Total Loss", fontsize=14, fontweight="bold")
        axs[0, 0].set_xlabel("Epochs", fontsize=12)
        axs[0, 0].set_ylabel("Loss", fontsize=12)
        axs[0, 0].legend(fontsize=10)
        axs[0, 0].grid(True, alpha=0.3)

        # Plot component losses with markers
        axs[0, 1].plot(epochs, self.train_box_losses, "g-", label="Box Loss", marker="o", markersize=4)
        axs[0, 1].plot(epochs, self.train_obj_losses, "b-", label="Obj Loss", marker="s", markersize=4)
        axs[0, 1].plot(epochs, self.train_cls_losses, "r-", label="Cls Loss", marker="^", markersize=4)
        axs[0, 1].set_title("Training Component Losses", fontsize=14, fontweight="bold")
        axs[0, 1].set_xlabel("Epochs", fontsize=12)
        axs[0, 1].set_ylabel("Loss", fontsize=12)
        axs[0, 1].legend(fontsize=10)
        axs[0, 1].grid(True, alpha=0.3)

        # Plot validation component losses with markers
        if self.val_losses:
            axs[1, 0].plot(val_epochs, self.val_box_losses, "g-", label="Box Loss", marker="o", markersize=4)
            axs[1, 0].plot(val_epochs, self.val_obj_losses, "b-", label="Obj Loss", marker="s", markersize=4)
            axs[1, 0].plot(val_epochs, self.val_cls_losses, "r-", label="Cls Loss", marker="^", markersize=4)
            axs[1, 0].set_title("Validation Component Losses", fontsize=14, fontweight="bold")
            axs[1, 0].set_xlabel("Epochs", fontsize=12)
            axs[1, 0].set_ylabel("Loss", fontsize=12)
            axs[1, 0].legend(fontsize=10)
            axs[1, 0].grid(True, alpha=0.3)

        # Plot Precision and Recall with markers
        if hasattr(self, "val_precision") and self.val_precision:
            axs[1, 1].plot(val_epochs, self.val_precision, "b-", label="Precision", marker="o", markersize=5)
            axs[1, 1].plot(val_epochs, self.val_recall, "r-", label="Recall", marker="s", markersize=5)
            axs[1, 1].plot(val_epochs, self.val_f1, "g-", label="F1 Score", marker="^", markersize=5)
            axs[1, 1].set_title("Precision and Recall Metrics", fontsize=14, fontweight="bold")
            axs[1, 1].set_xlabel("Epochs", fontsize=12)
            axs[1, 1].set_ylabel("Value", fontsize=12)
            axs[1, 1].set_ylim(0, 1.05)  # Precision and recall are between 0 and 1
            axs[1, 1].legend(fontsize=10)
            axs[1, 1].grid(True, alpha=0.3)

        # Plot mAP with markers
        if hasattr(self, "val_map50") and self.val_map50:
            axs[2, 0].plot(val_epochs, self.val_map50, "b-", label="mAP@0.5", marker="o", markersize=5)
            axs[2, 0].plot(val_epochs, self.val_map, "r-", label="mAP@0.5:0.95", marker="s", markersize=5)
            axs[2, 0].set_title("Mean Average Precision (mAP)", fontsize=14, fontweight="bold")
            axs[2, 0].set_xlabel("Epochs", fontsize=12)
            axs[2, 0].set_ylabel("mAP", fontsize=12)
            axs[2, 0].set_ylim(0, 1.05)  # mAP is between 0 and 1
            axs[2, 0].legend(fontsize=10)
            axs[2, 0].grid(True, alpha=0.3)

        # Plot learning rate with markers
        axs[2, 1].plot(epochs, self.lr_history, "k-", marker="o", markersize=4)
        axs[2, 1].set_title("Learning Rate", fontsize=14, fontweight="bold")
        axs[2, 1].set_xlabel("Epochs", fontsize=12)
        axs[2, 1].set_ylabel("Learning Rate", fontsize=12)
        axs[2, 1].set_yscale("log")  # Log scale for better visualization of LR changes
        axs[2, 1].grid(True, alpha=0.3)

        # Add plot annotations - values at start, end, min and max
        for ax in axs.flat:
            if ax.lines:
                for line in ax.lines:
                    y_data = line.get_ydata()
                    if len(y_data) > 0:
                        # Add value at the last point
                        ax.annotate(
                            f"{y_data[-1]:.4f}",
                            xy=(line.get_xdata()[-1], y_data[-1]),
                            xytext=(5, 0),
                            textcoords="offset points",
                            fontsize=8,
                            color=line.get_color(),
                        )

                        # Add value at min/max if it's not at the end
                        max_idx = np.argmax(y_data)
                        if max_idx != len(y_data) - 1:
                            ax.annotate(
                                f"{y_data[max_idx]:.4f}",
                                xy=(line.get_xdata()[max_idx], y_data[max_idx]),
                                xytext=(0, 5),
                                textcoords="offset points",
                                fontsize=8,
                                color=line.get_color(),
                            )

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "training_metrics.png"), dpi=300)
        plt.savefig(str(self.output_dir / "training_metrics.pdf"))
        plt.close(fig)

        # Create individual plots with more details
        self._plot_individual_metric("total_loss", epochs, self.train_losses, val_epochs, self.val_losses)
        self._plot_individual_metric("box_loss", epochs, self.train_box_losses, val_epochs, self.val_box_losses)
        self._plot_individual_metric("obj_loss", epochs, self.train_obj_losses, val_epochs, self.val_obj_losses)
        self._plot_individual_metric("cls_loss", epochs, self.train_cls_losses, val_epochs, self.val_cls_losses)

        # Plot detection metrics
        if hasattr(self, "val_precision") and self.val_precision:
            self._plot_detection_metrics(
                "precision_recall", val_epochs, {"Precision": self.val_precision, "Recall": self.val_recall, "F1 Score": self.val_f1}
            )

            self._plot_detection_metrics("map", val_epochs, {"mAP@0.5": self.val_map50, "mAP@0.5:0.95": self.val_map})

    def _plot_individual_metric(self, name, train_epochs, train_values, val_epochs=None, val_values=None):
        """Plot individual metric and save to file with enhanced visualization"""
        plt.figure(figsize=(10, 6))

        # Set style
        plt.style.use("ggplot")

        # Plot training metrics with markers
        plt.plot(train_epochs, train_values, "b-", label=f"Training {name}", marker="o", markersize=5, alpha=0.7)

        # Add trend line (smoothed)
        if len(train_epochs) > 5:
            try:
                from scipy.ndimage import gaussian_filter1d

                smoothed = gaussian_filter1d(train_values, sigma=2)
                plt.plot(train_epochs, smoothed, "b--", alpha=0.5, linewidth=1.5)
            except ImportError:
                pass

        # Plot validation metrics if available
        if val_values and val_epochs:
            plt.plot(val_epochs, val_values, "r-", label=f"Validation {name}", marker="s", markersize=5, alpha=0.7)

            # Add trend line for validation
            if len(val_epochs) > 5:
                try:
                    from scipy.ndimage import gaussian_filter1d

                    smoothed = gaussian_filter1d(val_values, sigma=2)
                    plt.plot(val_epochs, smoothed, "r--", alpha=0.5, linewidth=1.5)
                except ImportError:
                    pass

        # Add title and labels
        plt.title(f'{name.replace("_", " ").title()}', fontsize=16, fontweight="bold")
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Value", fontsize=12)

        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # Annotate min/max points
        train_min_idx = np.argmin(train_values)
        train_max_idx = np.argmax(train_values)

        # Add annotation for start point (first epoch)
        plt.annotate(
            f"Start: {train_values[0]:.4f}",
            xy=(train_epochs[0], train_values[0]),
            xytext=(-10, -15),  # Смещение влево и вниз
            textcoords="offset points",
            fontsize=9,
            ha="right",  # Выравнивание по правому краю относительно точки xy
            color="blue",
        )

        plt.annotate(
            f"Min: {train_values[train_min_idx]:.4f}",
            xy=(train_epochs[train_min_idx], train_values[train_min_idx]),
            xytext=(0, -15),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.7),
        )

        plt.annotate(
            f"Max: {train_values[train_max_idx]:.4f}",
            xy=(train_epochs[train_max_idx], train_values[train_max_idx]),
            xytext=(0, 15),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.7),
        )

        # Add value at last epoch
        plt.annotate(
            f"Last: {train_values[-1]:.4f}",
            xy=(train_epochs[-1], train_values[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=9,
            color="blue",
        )

        # Add values for validation if available
        if val_values and val_epochs:
            val_min_idx = np.argmin(val_values)
            val_max_idx = np.argmax(val_values)

            # Add annotation for start point of validation
            plt.annotate(
                f"Start: {val_values[0]:.4f}",
                xy=(val_epochs[0], val_values[0]),
                xytext=(-10, 15),  # Смещение влево и вверх
                textcoords="offset points",
                fontsize=9,
                ha="right",  # Выравнивание по правому краю
                color="red",
            )

            plt.annotate(
                f"Min: {val_values[val_min_idx]:.4f}",
                xy=(val_epochs[val_min_idx], val_values[val_min_idx]),
                xytext=(0, -15),
                textcoords="offset points",
                fontsize=9,
                ha="center",
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            )

            # Add max annotation for validation
            plt.annotate(
                f"Max: {val_values[val_max_idx]:.4f}",
                xy=(val_epochs[val_max_idx], val_values[val_max_idx]),
                xytext=(0, 15),
                textcoords="offset points",
                fontsize=9,
                ha="center",
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            )

            plt.annotate(
                f"Last: {val_values[-1]:.4f}",
                xy=(val_epochs[-1], val_values[-1]),
                xytext=(10, 15),  # Смещение вправо и вверх
                textcoords="offset points",
                fontsize=9,
                color="red",
            )

        # Save the figure
        plt.tight_layout()
        plt.savefig(str(self.metrics_dir / f"{name}.png"), dpi=300)
        plt.savefig(str(self.metrics_dir / f"{name}.pdf"))
        plt.close()

    def _plot_detection_metrics(self, name, epochs, metrics_dict):
        """Plot detection metrics like precision/recall or mAP"""
        plt.figure(figsize=(12, 8))

        # Set style
        plt.style.use("ggplot")

        # Use different colors for each metric
        colors = ["blue", "red", "green", "purple", "orange"]
        markers = ["o", "s", "^", "d", "x"]

        # Plot each metric
        for i, (metric_name, values) in enumerate(metrics_dict.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            plt.plot(epochs, values, f"-", color=color, label=metric_name, marker=marker, markersize=6, alpha=0.8)

            # Add last value annotation
            plt.annotate(
                f"{values[-1]:.4f}", xy=(epochs[-1], values[-1]), xytext=(5, 0), textcoords="offset points", fontsize=9, color=color
            )

        # Set limits for metrics between 0 and 1
        if name in ["precision_recall", "map"]:
            plt.ylim(0, 1.05)

        # Add title and labels
        if name == "precision_recall":
            title = "Precision, Recall and F1 Score"
        elif name == "map":
            title = "Mean Average Precision (mAP)"
        else:
            title = name.replace("_", " ").title()

        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Value", fontsize=12)

        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc="lower right")

        # Save the figure
        plt.tight_layout()
        plt.savefig(str(self.metrics_dir / f"{name}_metrics.png"), dpi=300)
        plt.savefig(str(self.metrics_dir / f"{name}_metrics.pdf"))
        plt.close()

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.args.epochs} epochs")

        self.gradient_tracker.register_hooks()

        # Record start time for total training time calculation
        start_time = time.time()

        for epoch in range(self.start_epoch, self.args.epochs):
            # Record epoch start time
            epoch_start = time.time()

            print(f"\n{'='*20} Epoch {epoch+1}/{self.args.epochs} {'='*20}")

            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(epoch)

            self.gradient_tracker.visualize_gradient_norms(epoch)

            # Print epoch summary
            print(f"Training: loss={train_loss:.4f}, " + ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()]))

            # Validate if this is a validation epoch
            if (epoch + 1) % self.args.eval_interval == 0:
                val_loss, val_metrics = self.validate(epoch)

                # Print validation summary
                print(f"Validation: loss={val_loss:.4f}, " + ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))

                # Check if this is the best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"New best model with val_loss={val_loss:.4f}")
            else:
                is_best = False

            # Save checkpoint if this is a save epoch
            if (epoch + 1) % self.args.save_interval == 0 or epoch == self.args.epochs - 1:
                self.save_checkpoint(epoch, is_best=is_best)
                print(f"Saved checkpoint at epoch {epoch+1}")

            # Plot current metrics every 10 epochs
            if (epoch + 1) % 1 == 0:
                self.plot_metrics()

            # Calculate and print epoch time
            epoch_time = time.time() - epoch_start
            print(f"Epoch completed in {epoch_time:.2f} seconds")

            # Estimate remaining time
            elapsed_time = time.time() - start_time
            epochs_done = epoch - self.start_epoch + 1
            epochs_left = self.args.epochs - epoch - 1
            estimated_time_left = (elapsed_time / epochs_done) * epochs_left if epochs_done > 0 else 0

            print(f"Elapsed time: {elapsed_time/3600:.2f} hours")
            print(f"Estimated time remaining: {estimated_time_left/3600:.2f} hours")

        # Calculate total training time
        total_time = time.time() - start_time
        print(f"Total training time: {total_time/3600:.2f} hours")

        # Save final model
        self.save_checkpoint(self.args.epochs - 1, is_best=False)
        print("Training completed!")

        self.gradient_tracker.remove_hooks()

        # Plot and save final metrics graphs
        self.plot_metrics()

        # Close the tensorboard writer
        self.writer.close()


def main():
    """
    Main training function that can be called directly or imported.
    Handles configuration loading and training initialization.
    """
    import argparse
    import yaml
    import os
    import torch
    import numpy as np
    from pathlib import Path

    # Parse command line arguments - define this inline to avoid needing parse_args()
    parser = argparse.ArgumentParser(description="Train YOLOv5 with ControlNet for Motion")

    # Only require config path for YAML configuration
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    # Allow command-line overrides for key parameters
    parser.add_argument("--output_dir", type=str, help="Override output directory from config")
    parser.add_argument("--epochs", type=int, help="Override number of epochs from config")
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    parser.add_argument("--lr", type=float, help="Override learning rate from config")
    parser.add_argument("--resume", type=str, help="Override resume checkpoint path from config")

    cmd_args = parser.parse_args()

    # Load configuration from YAML file - inline implementation
    with open(cmd_args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Apply command-line overrides if provided
    if cmd_args.output_dir:
        config_dict["data"]["output_dir"] = cmd_args.output_dir

    if cmd_args.epochs:
        config_dict["training"]["epochs"] = cmd_args.epochs

    if cmd_args.batch_size:
        config_dict["training"]["batch_size"] = cmd_args.batch_size

    if cmd_args.lr:
        config_dict["training"]["lr"] = cmd_args.lr

    if cmd_args.resume:
        config_dict["training"]["resume"] = cmd_args.resume

    # Create a flat namespace for backwards compatibility - inline implementation
    args = argparse.Namespace()

    # Data paths
    args.preprocessed_dir = config_dict["data"]["preprocessed_dir"]
    args.annotations_dir = config_dict["data"]["annotations_dir"]
    args.splits_file = config_dict["data"]["splits_file"]
    args.output_dir = config_dict["data"]["output_dir"]

    # Model configuration
    args.yolo_weights = config_dict["model"]["yolo_weights"]
    args.controlnet_weights = config_dict["model"]["controlnet_weights"]
    args.yolo_cfg = config_dict["model"]["yolo_cfg"]
    args.img_size = config_dict["model"]["img_size"]
    args.num_classes = config_dict["model"]["num_classes"]
    args.train_controlnet = config_dict["model"]["train_controlnet"]
    args.train_head = config_dict["model"]["train_head"]
    args.train_all = config_dict["model"]["train_all"]

    # Training parameters
    args.epochs = config_dict["training"]["epochs"]
    args.batch_size = config_dict["training"]["batch_size"]
    args.val_batch_size = config_dict["training"]["val_batch_size"]
    args.workers = config_dict["training"]["workers"]
    args.val_ratio = config_dict["training"]["val_ratio"]

    # Augmentation settings
    args.augment = config_dict["training"]["augment"]
    args.augment_prob = config_dict["training"]["augment_prob"]
    
    # Optimizer settings
    args.optimizer = config_dict["training"]["optimizer"]
    args.lr = config_dict["training"]["lr"]
    args.weight_decay = config_dict["training"]["weight_decay"]
    args.momentum = config_dict["training"]["momentum"]

    # Loss weights
    args.box_weight = config_dict["training"]["loss"]["box_weight"]
    args.obj_weight = config_dict["training"]["loss"]["obj_weight"]
    args.cls_weight = config_dict["training"]["loss"]["cls_weight"]

    # Precision
    args.precision = config_dict["training"]["precision"]

    # Checkpointing
    args.save_interval = config_dict["training"]["save_interval"]
    args.log_interval = config_dict["training"]["log_interval"]
    args.eval_interval = config_dict["training"]["eval_interval"]

    # Detection
    args.conf_thres = config_dict["training"]["detection"].get("conf_thres")
    args.iou_thres = config_dict["training"]["detection"].get("iou_thres")
    args.max_det = config_dict["training"]["detection"].get("max_det")

    # Resume
    args.resume = config_dict["training"]["resume"]

    # Save the resolved configuration
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
