from PIL import Image
from typing import Optional


def add_colored_blur(
    jewel_path: str,
    light_direction: str = "right",
    shadow_length: int = 3,
    shadow_opacity: int = 64,
) -> Optional[None]:
    """
    Adds a colored shadow to the edge of an object in an image with a transparent background.

    Args:
        jewel_path (str): Path to the input image with a transparent background.
        light_direction (str): Direction of the light source ('left' or 'right').
        shadow_length (int): The length of the shadow in pixels.
        shadow_opacity (int): The opacity of the shadow (0-255).

    Returns:
        None: The processed image is saved at the same path.
    """
    jewel_image = Image.open(jewel_path).convert("RGBA")
    pixels = jewel_image.load()
    w, h = jewel_image.size

    y_coords = [y for y in range(h) if any(pixels[x, y][3] > 0 for x in range(w))]

    if not y_coords:
        print("No non-transparent pixels found in the image.")
        return

    lowest_y, highest_y = min(y_coords), max(y_coords)

    jewel_output_image = Image.new(
        "RGBA", (w + shadow_length, h) if light_direction == "right" else (w, h)
    )
    jewel_output_pixels = jewel_output_image.load()

    x_offset = shadow_length if light_direction == "right" else 0
    for y in range(h):
        for x in range(w):
            jewel_output_pixels[x + x_offset, y] = pixels[x, y]

    for y in range(lowest_y, highest_y + 1):
        edge_x = next(
            (
                x
                for x in (
                    range(w) if light_direction == "right" else reversed(range(w))
                )
                if pixels[x, y][3] > 0
            ),
            None,
        )
        if edge_x is not None:
            edge_color = pixels[edge_x, y][:3]
            for i in range(1, shadow_length + 1):
                shadow_x = (
                    edge_x - i + shadow_length
                    if light_direction == "right"
                    else edge_x + i
                )
                jewel_output_pixels[shadow_x, y] = (*edge_color, shadow_opacity)

    jewel_output_image.save(jewel_path)
