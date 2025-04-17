import numpy as np
import cv2


def create_control_image(frames_stack: list, cur_image: np.ndarray, mode: str = "flow") -> np.ndarray:
    """
    Create a control image from a stack of previous frames and current frame.

    Args:
        frames_stack: List of previous frames as numpy arrays [(H, W, C)], ordered from oldest to newest
        cur_image: Current frame as a numpy array (H, W, C)
        mode: Motion representation mode:
            - "flow": Optical flow visualization (Farneback method)
            - "difference": Simple frame difference with the most recent previous frame
            - "diff_color": Colored frame difference with the most recent previous frame
            - "bg_subtraction": Background subtraction using multiple frames history

    Returns:
        Control image as a numpy array (H, W, C)
    """
    # Get the most recent previous frame
    prev_image = frames_stack[-1] if frames_stack else None

    if prev_image is None:
        # If no previous frames available, return grayscale version of current
        return cv2.cvtColor(cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

    # Convert images to grayscale
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
    cur_gray = cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY)

    if mode == "flow":
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Convert flow to RGB using HSV color wheel
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((cur_image.shape[0], cur_image.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    elif mode == "difference":
        # Calculate difference in each color channel
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cur_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Вычисление разницы со знаком (нормализация для лучшего обучения нейросети)
        raw_diff = cv2.absdiff(cur_gray, prev_gray)

        # Создание 3-канального изображения
        result = np.zeros((cur_image.shape[0], cur_image.shape[1], 3), dtype=np.uint8)

        # Канал 0 (Red): текущее изображение в оттенках серого - дает контекст
        result[:, :, 0] = cur_gray.astype(np.uint8)

        # Канал 1 (Green): разница в изображениях
        diff = np.clip(raw_diff, 0, 255).astype(np.uint8)
        result[:, :, 1] = diff

        # Create background model from frame stack
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=len(frames_stack) + 1, varThreshold=16, detectShadows=True)

        # Add all previous frames to the model
        for frame in frames_stack:
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            bg_subtractor.apply(frame)

        # Канал 2 (Blue): маска от BG
        fg_mask = bg_subtractor.apply(cur_image)

        result[:, :, 2] = fg_mask
        return result

    elif mode == "bg_subtraction":
        # Use proper background subtraction with multiple frames

        # Create background model from frame stack
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=len(frames_stack) + 1, varThreshold=16, detectShadows=True)

        # Add all previous frames to the model
        for frame in frames_stack:
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            bg_subtractor.apply(frame)

        # Get foreground mask for current frame
        fg_mask = bg_subtractor.apply(cur_image)

        # Create control image from foreground mask
        result = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2RGB)
        return result

    else:
        raise ValueError(f"Unknown mode: {mode}. Valid modes are 'flow', 'difference', 'diff_color', 'bg_subtraction'")


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
