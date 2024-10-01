import cv2
import numpy as np


def resize_ear_ring(
    jewel_path: str,
    box_w: int,
    box_h: int,
) -> None:
    """
    Resizes an earring image to fit within a bounding box area, maintaining the aspect ratio.
    The resized earring will occupy 1/8th of the area of the given bounding box.

    Args:
        jewel_path (str): Path to the input image of the earring (must have an alpha channel).
        box_w (int): Width of the bounding box.
        box_h (int): Height of the bounding box.

    Raises:
        ValueError: If the image does not have an alpha channel.
    """
    jewel_image = cv2.imread(jewel_path, cv2.IMREAD_UNCHANGED)

    if jewel_image.shape[2] != 4:
        raise ValueError("Image does not have an alpha channel")

    jewel_a = jewel_image[:, :, 3]

    jewel_non_transparent_indices = np.where(jewel_a > 0)
    jewel_min_y, jewel_max_y = np.min(jewel_non_transparent_indices[0]), np.max(
        jewel_non_transparent_indices[0]
    )
    jewel_min_x, jewel_max_x = np.min(jewel_non_transparent_indices[1]), np.max(
        jewel_non_transparent_indices[1]
    )

    jewel_area = (jewel_max_x - jewel_min_x) * (jewel_max_y - jewel_min_y)
    ear_box_area = box_w * box_h

    scale_factor = (ear_box_area / (20 * jewel_area)) ** 0.5
    jewel_resized_w = int(jewel_image.shape[1] * scale_factor)
    jewel_resized_h = int(jewel_image.shape[0] * scale_factor)

    jewel_resized_image = cv2.resize(
        jewel_image, (jewel_resized_w, jewel_resized_h), interpolation=cv2.INTER_AREA
    )

    cv2.imwrite(jewel_path, jewel_resized_image)
