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

    if mode == "flow":
        # Convert images to grayscale
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        cur_gray = cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY)

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
        cur_image = cur_image.astype(np.float32)
        prev_image = prev_image.astype(np.float32)

        control_image = np.zeros((cur_image.shape[0], cur_image.shape[1], 3), dtype=np.float32)
        control_image[..., 0] = cur_image[..., 0] - prev_image[..., 0]
        control_image[..., 1] = cur_image[..., 1] - prev_image[..., 1]
        control_image[..., 2] = cur_image[..., 2] - prev_image[..., 2]

        mask = np.abs(control_image) < 15

        control_image = ((control_image + 255) / 2).astype(np.uint8)

        control_image[mask] = 0
        return control_image

    elif mode == "bg_subtraction":
        # Use proper background subtraction with multiple frames

        # Create background model from frame stack
        bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True, history=len(frames_stack) + 1)

        # Add all previous frames to the model
        for frame in frames_stack:
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            bg_subtractor.apply(frame)

        # Get foreground mask for current frame
        fg_mask = bg_subtractor.apply(cur_image)

        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 1,)
        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations = 1,)

        # Create control image from foreground mask
        result = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2RGB)

        return result

    elif mode == "mixed":
        # Use proper background subtraction with multiple frames

        # Create background model from frame stack
        bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True, history=len(frames_stack) + 1)

        # Add all previous frames to the model
        for frame in frames_stack:
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            bg_subtractor.apply(frame)

        # Get foreground mask for current frame
        fg_mask = bg_subtractor.apply(cur_image)

        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations = 1,)
        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations = 1,)

        # Create control image from foreground mask
        cur_image = cur_image.astype(np.float32)
        prev_image = prev_image.astype(np.float32)

        control_image = np.zeros((cur_image.shape[0], cur_image.shape[1], 4), dtype=np.float32)
        control_image[..., 0] = cur_image[..., 0] - prev_image[..., 0]
        control_image[..., 1] = cur_image[..., 1] - prev_image[..., 1]
        control_image[..., 2] = cur_image[..., 2] - prev_image[..., 2]

        mask = np.abs(control_image) < 15

        control_image[..., :3] = (control_image[..., :3] + 255) / 2
        control_image[mask] = 0
        control_image[..., 3] = fg_mask

        control_image = control_image.astype(np.uint8)

        return control_image

    else:
        raise ValueError(f"Unknown mode: {mode}. Valid modes are 'flow', 'difference', 'mixed', 'bg_subtraction'")


def cxcywh_to_xyxy(bbox):
    """
    Convert bbox from [center_x, center_y, width, height] to [x1, y1, x2, y2] format.

    Args:
        bbox: Bounding box in [center_x, center_y, width, height] format

    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


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
    h, w = image.shape[:2]
    for bbox in annotations:
        # Convert from center format to corner format
        x1, y1, x2, y2 = cxcywh_to_xyxy(bbox)

        # Convert to integers
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

    return img_with_boxes
