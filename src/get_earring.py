import random
from scipy.ndimage import binary_fill_holes
from PIL import Image
import numpy as np
import os
from src.compute_points import jewel_center
from src.bgremove import main as bg_remover
from src.constants import JEWEL_PATH, JEWEL_NO_BG_PATH, JEWEL_LEFT_PATH, JEWEL_RIGHT_PATH


def find_nonzero_pixel(jewel_image: np.ndarray) -> tuple[int, int] | None:
    """
    Find a random non-zero pixel in the image.

    Args:
        jewel_image (np.ndarray): The input image array.

    Returns:
        tuple: Coordinates of a randomly selected non-zero pixel or None if no such pixel exists.
    """
    jewel_nonzero_indices = np.argwhere(jewel_image[:, :, 0] > 0)
    if jewel_nonzero_indices.size == 0:
        return None
    return tuple(random.choice(jewel_nonzero_indices))


def flood_fill(
    jewel_nobg_image_array: np.ndarray,
    start: tuple[int, int],
    visited_pixels: set = None,
) -> set[tuple[int, int]]:
    """
    Recursively add all connected non-zero pixels to the cluster.

    Args:
        jewel_nobg_image_array (np.ndarray): The input image array.
        start (tuple[int, int]): The starting pixel coordinates.
        visited_pixels (set): Set of already visited pixels.

    Returns:
        set: A set of all connected non-zero pixels.
    """
    if visited_pixels is None:
        visited_pixels = set()
    rows, cols = jewel_nobg_image_array.shape[:2]
    stack = [start]

    while stack:
        pixel = stack.pop()
        if pixel in visited_pixels:
            continue
        visited_pixels.add(pixel)
        x, y = pixel
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < rows
                    and 0 <= ny < cols
                    and jewel_nobg_image_array[nx, ny][0] > 10
                ):
                    stack.append((nx, ny))
    return visited_pixels


def get_contour(
    visited_pixels: set[tuple[int, int]], image_shape: tuple[int, int]
) -> set[tuple[int, int]]:
    """
    Extract the contour of the visited pixels set.

    Args:
        visited_pixels (set): Set of visited pixels.
        image_shape (tuple): Shape of the input image.

    Returns:
        set: A set of contour pixels.
    """
    contour = set()
    rows, cols = image_shape
    visited_list = list(visited_pixels)
    mask = np.zeros(image_shape, dtype=bool)
    for pixel in visited_list:
        mask[pixel] = True

    for x, y in visited_list:
        is_edge = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not mask[nx, ny]:
                    is_edge = True
                    break
            if is_edge:
                contour.add((x, y))
                break

    return contour


def fill_contour(
    contour: set[tuple[int, int]], image_shape: tuple[int, int]
) -> np.ndarray:
    """
    Fill inside the contour to create a mask for the interior region.

    Args:
        contour (set): Set of contour pixels.
        image_shape (tuple): Shape of the image.

    Returns:
        np.ndarray: Binary mask where the interior region is filled.
    """
    mask = np.zeros(image_shape[:2], dtype=bool)
    for x, y in contour:
        mask[x, y] = True
    filled_mask = binary_fill_holes(mask).astype(bool)
    return filled_mask


def extract_one_earring(
    jewel_nobg_image_array: np.ndarray,
    filled_mask: np.ndarray,
    earring_save_path: str = None,
) -> None:
    """
    Apply the filled mask to the original image, make the background transparent, and save the result.

    Args:
        jewel_nobg_image_array (np.ndarray): Image array with no background.
        filled_mask (np.ndarray): Binary mask to apply to the image.
        earring_save_path (str, optional): Path to save the resulting image.
    """
    masked_image = jewel_nobg_image_array * filled_mask[:, :, np.newaxis]
    masked_image_rgb = Image.fromarray(masked_image).convert("RGBA")
    transparent_image = Image.new("RGBA", masked_image_rgb.size, (0, 0, 0, 0))
    background_color = masked_image_rgb.getpixel((0, 0))
    w, h = masked_image_rgb.size
    for x in range(w):
        for y in range(h):
            current_color = masked_image_rgb.getpixel((x, y))
            if current_color != background_color:
                transparent_image.putpixel((x, y), current_color)
    if earring_save_path:
        transparent_image.save(earring_save_path)


def extract_the_other(
    jewel_nobg_image_array: np.ndarray,
    filled_mask: np.ndarray,
    earring_save_path: str = None,
) -> None:
    """
    Extract the second earring by applying the inverted mask and extracting its contour.

    Args:
        jewel_nobg_image_array (np.ndarray): Image array with no background.
        filled_mask (np.ndarray): Binary mask of the first earring.
        earring_save_path (str, optional): Path to save the second earring image.
    """
    masked_image = np.zeros_like(jewel_nobg_image_array)
    for i in range(3):
        masked_image[:, :, i] = jewel_nobg_image_array[:, :, i] * ~filled_mask

    start_pixel = find_nonzero_pixel(masked_image)
    if start_pixel:
        visited_pixels = flood_fill(masked_image, start_pixel)
    else:
        visited_pixels = []

    contour_pixels = get_contour(visited_pixels, masked_image.shape[:2])
    filled_mask = fill_contour(contour_pixels, masked_image.shape)
    extract_one_earring(masked_image, filled_mask, earring_save_path)


def left_and_right(earring1_path: str, earring2_path: str) -> None:
    """
    Determine which earring is for the left ear and which is for the right based on their center x-coordinates.

    Args:
        earring1_path (str): Path to the first earring image.
        earring2_path (str): Path to the second earring image.
    """
    cX1, cX2 = jewel_center(earring1_path)[0], jewel_center(earring2_path)[0]
    left_earring_path, right_earring_path = (
        (earring1_path, earring2_path) if cX1 < cX2 else (earring2_path, earring1_path)
    )
    os.rename(left_earring_path, JEWEL_LEFT_PATH)
    os.rename(right_earring_path, JEWEL_RIGHT_PATH)


def main() -> None:
    """
    Process and extract two earrings from the jewelry image, save them separately, and determine which is left and right.
    """
    bg_remover("remove.bg", JEWEL_PATH, JEWEL_NO_BG_PATH)
    jewel_nobg_image = Image.open(JEWEL_NO_BG_PATH)
    jewel_nobg_image_rgb = jewel_nobg_image.convert("RGB")
    jewel_nobg_image_array = np.array(jewel_nobg_image_rgb)

    start_pixel = find_nonzero_pixel(jewel_nobg_image_array)
    if start_pixel:
        visited_pixels = flood_fill(jewel_nobg_image_array, start_pixel)
    else:
        visited_pixels = []

    contour_pixels = get_contour(visited_pixels, jewel_nobg_image_array.shape[:2])
    filled_mask = fill_contour(contour_pixels, jewel_nobg_image_array.shape)
    extract_one_earring(jewel_nobg_image_array, filled_mask, JEWEL_LEFT_PATH)
    extract_the_other(jewel_nobg_image_array, filled_mask, JEWEL_RIGHT_PATH)

    left_and_right(JEWEL_LEFT_PATH, JEWEL_RIGHT_PATH)
    print("earring prepared")
