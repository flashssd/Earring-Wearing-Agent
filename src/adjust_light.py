import cv2
import numpy as np


def calculate_brightness(image: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Calculate the brightness of an image or a masked region of an image.

    Args:
        image (np.ndarray): Input image in BGR format.
        mask (np.ndarray, optional): Binary mask to specify the region for which brightness
                                     should be calculated. Defaults to None.

    Returns:
        float: The average brightness value in the grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        mean_val = cv2.mean(gray, mask=mask)[0]
    else:
        mean_val = np.mean(gray)
    return mean_val


def adjust_brightness(
    image: np.ndarray, target_brightness: float, mask: np.ndarray = None
) -> np.ndarray:
    """
    Adjust the brightness of an image to match a target brightness. Optionally,
    apply adjustments only to regions specified by a mask.

    Args:
        image (np.ndarray): Input image in BGR format.
        target_brightness (float): The target brightness to adjust the image to.
        mask (np.ndarray, optional): Binary mask that defines the region where brightness
                                     adjustment should be applied. Defaults to None.

    Returns:
        np.ndarray: The brightness-adjusted image.
    """
    current_brightness = calculate_brightness(image, mask)
    brightness_ratio = target_brightness / current_brightness * 2
    adjusted_image = np.zeros_like(image)

    if mask is not None:
        for i in range(3):
            adjusted_image[:, :, i] = cv2.convertScaleAbs(
                image[:, :, i], alpha=brightness_ratio, beta=0
            )
        adjusted_image = cv2.bitwise_and(
            adjusted_image, adjusted_image, mask=mask
        ) + cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    else:
        adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_ratio, beta=0)

    return adjusted_image


def main(model_path: str, earring_path: str, x: int, y: int, w: int, h: int) -> None:
    """
    Adjust the brightness of an earring image to match the brightness of a specific
    area on a model image, and save the adjusted earring image.

    Args:
        model_path (str): Path to the model image file.
        earring_path (str): Path to the earring image file (with alpha channel).
        x (int): X-coordinate of the top-left corner of the target area on the model image.
        y (int): Y-coordinate of the top-left corner of the target area on the model image.
        w (int): Width of the target area on the model image.
        h (int): Height of the target area on the model image.

    Returns:
        None: The function saves the brightness-adjusted earring image at the same location.
    """
    earring_image = cv2.imread(earring_path, cv2.IMREAD_UNCHANGED)
    earring_b, earring_g, earring_r, earring_a = cv2.split(earring_image)
    earring_image_bgr = cv2.merge((earring_b, earring_g, earring_r))

    model_image = cv2.imread(model_path)
    model_mask = np.zeros(model_image.shape[:2], dtype=np.uint8)
    model_mask[y : y + h, x : x + w] = 255
    model_brightness_mean = calculate_brightness(model_image, mask=model_mask)

    earring_adjusted = adjust_brightness(
        earring_image_bgr, model_brightness_mean, mask=earring_a
    )

    earring_adjusted_with_alpha = cv2.merge((earring_adjusted, earring_a))

    cv2.imwrite(earring_path, earring_adjusted_with_alpha)
