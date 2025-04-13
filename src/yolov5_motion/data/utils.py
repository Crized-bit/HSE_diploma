import numpy as np
import cv2


def create_control_image(prev_image: np.ndarray, cur_image: np.ndarray) -> np.ndarray:
    """
    Create a control image from previous and current frames.

    Args:
        prev_image: Previous frame as a numpy array (H, W, C)
        cur_image: Current frame as a numpy array (H, W, C)

    Returns:
        Control image as a numpy array (H, W, C)
    """
    # Convert images to grayscale
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
    cur_gray = cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to RGB using HSV color wheel
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((prev_image.shape[0], prev_image.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow


def cxcywh_to_xyxy(bbox, img_width=640, img_height=640):
    """
    Convert bbox from [center_x, center_y, width, height] to [x1, y1, x2, y2] format.

    Args:
        bbox: Bounding box in [center_x, center_y, width, height] format
        img_width: Image width (for normalization if needed)
        img_height: Image height (for normalization if needed)

    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def xyxy_to_cxcywh(bbox, img_width=640, img_height=640):
    """
    Convert bbox from [x1, y1, x2, y2] to [center_x, center_y, width, height] format.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        img_width: Image width (for normalization if needed)
        img_height: Image height (for normalization if needed)

    Returns:
        Bounding box in [center_x, center_y, width, height] format
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def draw_bounding_boxes(image, annotations, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image.

    Args:
        image: Image as a numpy array (H, W, C)
        annotations: List of annotations with bounding boxes
        color: Color of the bounding box in BGR format
        thickness: Thickness of the bounding box lines

    Returns:
        Image with bounding boxes drawn
    """
    img_with_boxes = image.copy()

    for ann in annotations:
        bbox = ann["bbox"]
        # Convert from center format to corner format
        x1, y1, x2, y2 = cxcywh_to_xyxy(bbox)

        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

        # If available, draw label
        if "labels" in ann and ann["labels"]:  # Check if labels exist and are not empty
            label_text = ""

            # Handle different label formats
            if isinstance(ann["labels"], list) and len(ann["labels"]) > 0:
                label_item = ann["labels"][0]

                if isinstance(label_item, dict):
                    # Try different possible keys for label name
                    if "name" in label_item:
                        label_text = label_item["name"]
                    elif "id" in label_item:
                        label_text = f"Class {label_item['id']}"
                    elif "class" in label_item:
                        label_text = str(label_item["class"])
                elif isinstance(label_item, (str, int, float)):
                    # If the label is a simple value
                    label_text = str(label_item)

            # Only draw label if we have valid text
            if label_text:
                cv2.putText(img_with_boxes, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return img_with_boxes
