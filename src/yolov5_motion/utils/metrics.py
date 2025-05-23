import numpy as np
from yolov5_motion.config import my_config


def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5, conf_threshold=0.25, mode="validation"):
    """
    Calculate precision and recall metrics for object detection.

    Args:
        pred_boxes: List of predicted bounding boxes in [x1, y1, x2, y2, conf, class_id] format
        true_boxes: List of ground truth bounding boxes in [x1, y1, x2, y2, class_id] format
        iou_threshold: IoU threshold for considering a prediction as true positive
        conf_threshold: Confidence threshold for predictions

    Returns:
        precision: Precision value
        recall: Recall value
        f1: F1 score
    """

    tp_area = []
    fp_area = []
    fn_area = []

    # Filter predictions by confidence
    pred_boxes = [box for box in pred_boxes if box[4] >= conf_threshold]

    # Sort predictions by confidence (highest first)
    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

    # Initialize counters
    true_positives = 0
    false_positives = 0

    # Mark ground truth boxes as matched or not
    gt_matched = [False] * len(true_boxes)

    for pred_box in pred_boxes:
        # Find ground truth box with highest IoU
        best_iou = 0
        best_idx = -1

        for i, gt_box in enumerate(true_boxes):
            # Skip already matched ground truth boxes
            if gt_matched[i]:
                continue

            # Skip if class doesn't match (for multi-class detection)
            if len(pred_box) > 5 and len(gt_box) > 4 and pred_box[5] != gt_box[4]:
                continue

            # Calculate IoU
            iou = calculate_box_iou(pred_box[:4], gt_box[:4])

            if iou > best_iou:
                best_iou = iou
                best_idx = i

        # Check if we found a match
        if best_idx >= 0 and best_iou >= iou_threshold:
            gt_matched[best_idx] = True
            true_positives += 1
        else:
            false_positives += 1
            fp_area.append(calculate_box_area(pred_box[:4]))

    tp_area.extend([calculate_box_area(box[:4]) for box, is_matched in zip(true_boxes, gt_matched) if is_matched])
    fn_area.extend([calculate_box_area(box[:4]) for box, is_matched in zip(true_boxes, gt_matched) if not is_matched])

    # Calculate metrics
    num_predictions = len(pred_boxes)
    num_ground_truth = len(true_boxes)

    precision = true_positives / max(num_predictions, 1) if num_predictions else 1
    recall = true_positives / max(num_ground_truth, 1) if num_ground_truth else 1

    # Calculate F1 score
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    if mode == "validation":
        return precision, recall, f1
    else:
        return precision, recall, f1, tp_area, fp_area, fn_area


def calculate_box_area(box: list[int, int, int, int]) -> float:
    """
    Calculate the area of a bounding box, normalized to the image size.

    Args:
        box: Bounding box in [x1, y1, x2, y2] format

    Returns:
        area: Area of the bounding box
    """
    return (box[2] - box[0]) * (box[3] - box[1]) / my_config.model.img_size**2


def calculate_box_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1: First box in [x1, y1, x2, y2] format
        box2: Second box in [x1, y1, x2, y2] format

    Returns:
        iou: IoU value
    """
    # Get intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / max(union_area, 1e-6)

    return iou


def calculate_map(pred_boxes_by_image, true_boxes_by_image, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    """
    Calculate mAP at different IoU thresholds using proper AP calculation.

    Args:
        pred_boxes_by_image: List containing predicted boxes for each image
        true_boxes_by_image: List containing ground truth boxes for each image
        iou_thresholds: List of IoU thresholds to calculate mAP

    Returns:
        map50: mAP at IoU threshold 0.5
        map: mAP averaged over IoU thresholds from 0.5 to 0.95
    """
    if len(pred_boxes_by_image) != len(true_boxes_by_image):
        raise ValueError("Number of predicted and ground truth image sets must match")

    # Calculate AP for each IoU threshold
    aps = []

    for threshold in iou_thresholds:
        # For each threshold, collect all predictions across all images
        all_predictions = []
        total_gt = 0

        # Process and combine data from all images
        for img_idx, (pred_boxes, true_boxes) in enumerate(zip(pred_boxes_by_image, true_boxes_by_image)):
            # Count total ground truths
            total_gt += len(true_boxes)

            # Filter predictions by confidence threshold
            filtered_preds = [box for box in pred_boxes if box[4] >= 0.25]  # Using default 0.25 threshold

            # Skip if no predictions
            if not filtered_preds:
                continue

            # For each prediction, calculate IoU with all ground truths in the same image
            for pred in filtered_preds:
                pred_info = {"confidence": pred[4], "matched": False, "img_idx": img_idx, "iou": 0}

                # Skip if no ground truths for this image
                if not true_boxes:
                    all_predictions.append(pred_info)
                    continue

                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(true_boxes):
                    # Skip if class doesn't match (for multi-class detection)
                    if len(pred) > 5 and len(gt) > 4 and pred[5] != gt[4]:
                        continue

                    iou = calculate_box_iou(pred[:4], gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                pred_info["iou"] = best_iou

                # Check if it's a match at this threshold
                if best_gt_idx >= 0 and best_iou >= threshold:
                    pred_info["matched"] = True
                    pred_info["gt_idx"] = best_gt_idx

                all_predictions.append(pred_info)

        # Sort all predictions by confidence (highest first)
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        # Initialize counters for precision-recall calculation
        true_positives = 0
        false_positives = 0

        # Track which ground truths have been matched to avoid double counting
        gt_matched = {}  # (img_idx, gt_idx) -> bool

        # Calculate precision and recall at each detection
        precisions = []
        recalls = []

        for pred in all_predictions:
            # If it's a match and the GT hasn't been matched yet
            if pred["matched"] and (pred["img_idx"], pred["gt_idx"]) not in gt_matched:
                true_positives += 1
                gt_matched[(pred["img_idx"], pred["gt_idx"])] = True
            else:
                false_positives += 1

            # Calculate precision and recall at this point
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / max(total_gt, 1)

            precisions.append(precision)
            recalls.append(recall)

        # Calculate AP using precision-recall points
        if not precisions:
            ap = 0
        else:
            # Use all points for AP calculation
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([0.0], precisions, [0.0]))

            # Compute the precision envelope (interpolation)
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # Calculate area under PR curve
            indices = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

        aps.append(ap)

    # Calculate mAP
    map50 = aps[0]  # AP at IoU threshold 0.5
    map = np.mean(aps)  # mAP averaged over IoU thresholds

    return map50, map
