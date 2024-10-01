import cv2
import numpy as np
import tempfile
from PIL import Image
import os
from typing import Optional
from src.detector import Detector
from src.constants import EARLOBE_CLF_PATH, EAR_CLF_PATH


def ear_ring_place(model_path: str) -> dict[str, tuple[int, int, int, int]]:
    """
    Detect the position of the left and right ears in an image using a pre-trained ear detection model.

    Args:
        model_path (str): Path to the input image file.

    Returns:
        dict: A dictionary with keys "left" and "right", each containing a tuple (x, y, w, h)
              representing the bounding box coordinates and dimensions of the detected ear.

    Raises:
        FileNotFoundError: If the image is not found at the specified path.
        Exception: If no ears are detected in the image.
    """
    ear_dict = {}
    model_image = cv2.imread(model_path)
    if model_image is None:
        raise FileNotFoundError(f"Image not found at {model_path}")

    detector = Detector(EAR_CLF_PATH)
    detected_locations, detected_keys = detector.detect(model_path)

    for i in range(min(len(detected_locations), 2)):
        x, y, w, h = detected_locations[i]
        if detected_keys[i] == "leftears":
            ear_dict["left"] = (x, y, w, h)
        else:
            ear_dict["right"] = (x, y, w, h)

    if not ear_dict:
        raise Exception("No ear detected in the image")

    return ear_dict


def jewel_center(jewel_path: str) -> Optional[tuple[int, int]]:
    """
    Calculate the center coordinates of the visible object in a transparent jewel image.

    Args:
        jewel_path (str): Path to the jewel image file with an alpha channel.

    Returns:
        tuple: A tuple (cX, cY) representing the center coordinates of the object in the image.
        None: If no object is detected or if the image lacks transparency.
    """
    jewel_center_image = cv2.imread(jewel_path, cv2.IMREAD_UNCHANGED)
    if jewel_center_image.shape[2] == 4:
        jewel_alpha_channel = jewel_center_image[:, :, 3]
        _, jewel_binary_mask = cv2.threshold(
            jewel_alpha_channel, 1, 255, cv2.THRESH_BINARY
        )
        jewel_M = cv2.moments(jewel_binary_mask)

        if jewel_M["m00"] != 0:
            cX = int(jewel_M["m10"] / jewel_M["m00"])
            cY = int(jewel_M["m01"] / jewel_M["m00"])
            return cX, cY
        else:
            print("No object detected.")
    else:
        print("Image does not have a transparency channel.")
    return None


def jewel_anchor_point(jewel_path: str) -> Optional[tuple[int, int]]:
    """
    Calculate a specific anchor point on a visible object in a transparent jewel image.

    Args:
        jewel_path (str): Path to the jewel image file with an alpha channel.

    Returns:
        tuple: A tuple (cX, target_y) representing the anchor point coordinates.
        None: If no object is detected or if the image lacks transparency.
    """
    jewel_image = cv2.imread(jewel_path, cv2.IMREAD_UNCHANGED)
    if jewel_image.shape[2] == 4:
        jewel_alpha_channel = jewel_image[:, :, 3]
        _, jewel_binary_mask = cv2.threshold(
            jewel_alpha_channel, 1, 255, cv2.THRESH_BINARY
        )
        jewel_M = cv2.moments(jewel_binary_mask)

        if jewel_M["m00"] != 0:
            cX = int(jewel_M["m10"] / jewel_M["m00"])
            jewel_ys = np.where(jewel_binary_mask > 0)[0]
            if len(jewel_ys) == 0:
                print("No object detected.")
                return None

            highest_y = np.min(jewel_ys)
            lowest_y = np.max(jewel_ys)
            target_y = highest_y + (lowest_y - highest_y) // 11
            return cX, target_y
        else:
            print("No object detected.")
            return None
    else:
        print("Image does not have a transparency channel.")
        return None


def adjust_crop_area(
    x: int, y: int, w: int, h: int, min_size: int = 224
) -> tuple[int, int, int, int]:
    """
    Adjust the cropping area to ensure it meets the minimum size requirements.

    Args:
        x (int): X-coordinate of the top-left corner.
        y (int): Y-coordinate of the top-left corner.
        w (int): Width of the detected region.
        h (int): Height of the detected region.
        min_size (int): Minimum size for width and height. Defaults to 224.

    Returns:
        tuple: Adjusted (x, y) coordinates and (x_end, y_end) coordinates.
    """
    x_end = x + w
    y_end = y + h

    def adjust(start: int, end: int, size: int, min_size: int) -> tuple[int, int]:
        if size < min_size:
            extra = min_size - size
            return start - extra // 2, end + (extra - extra // 2)
        elif size > min_size:
            extra = size - min_size
            return start + extra // 2, end - (extra - extra // 2)
        return start, end

    x, x_end = adjust(x, x_end, w, min_size)
    y, y_end = adjust(y, y_end, h, min_size)

    return x, y, x_end, y_end


def model_anchor_point(
    model_path: str, x: int, y: int, w: int, h: int
) -> tuple[int, int]:
    """
    Detect the anchor point of the earlobe in the cropped image of an ear.

    Args:
        model_path (str): Path to the model image.
        x (int): X-coordinate of the top-left corner of the ear.
        y (int): Y-coordinate of the top-left corner of the ear.
        w (int): Width of the detected ear region.
        h (int): Height of the detected ear region.

    Returns:
        tuple: A tuple (anchor_x, anchor_y) representing the anchor point's coordinates.
    """
    detector_ring = Detector(EARLOBE_CLF_PATH)
    model_image = Image.open(model_path)
    x, y, x_end, y_end = adjust_crop_area(x, y, w, h)
    model_crop_image = model_image.crop((x, y, x_end, y_end))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as model_temp_file:
        model_temp_path = model_temp_file.name
        model_crop_image.save(model_temp_path)

    detected_loc, _ = detector_ring.detect(model_temp_path)
    assert len(detected_loc) != 0
    box_x, box_y, box_w, box_h = detected_loc[0]
    box_x_end = int(box_x + box_w / 2)
    box_y_end = int(box_y + box_h / 2)

    os.remove(model_temp_path)

    return x + box_x_end, y + box_y_end
