import torch
import numpy as np


def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5, conf_threshold=0.25):
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
    if not pred_boxes or not true_boxes:
        if not pred_boxes and not true_boxes:  # No objects predicted and none present
            return 1.0, 1.0, 1.0
        if not pred_boxes:  # No objects predicted but some present
            return 0.0, 0.0, 0.0
        if not true_boxes:  # Objects predicted but none present
            return 0.0, 0.0, 0.0
    
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
    
    # Calculate metrics
    num_predictions = len(pred_boxes)
    num_ground_truth = len(true_boxes)
    
    precision = true_positives / max(num_predictions, 1)
    recall = true_positives / max(num_ground_truth, 1)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return precision, recall, f1


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
    Calculate mAP at different IoU thresholds.
    
    Args:
        pred_boxes_by_image: List of lists containing predicted boxes for each image
        true_boxes_by_image: List of lists containing ground truth boxes for each image
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
        precisions = []
        recalls = []
        
        for pred_boxes, true_boxes in zip(pred_boxes_by_image, true_boxes_by_image):
            precision, recall, _ = calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=threshold)
            precisions.append(precision)
            recalls.append(recall)
        
        # Average precision and recall across images
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        
        # Calculate AP as area under PR curve (simplified as precision * recall)
        ap = avg_precision * avg_recall
        aps.append(ap)
    
    # Calculate mAP
    map50 = aps[0]  # AP at IoU threshold 0.5
    map = np.mean(aps)  # mAP averaged over IoU thresholds
    
    return map50, map