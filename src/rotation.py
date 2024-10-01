import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image by the specified angle.

    Args:
        image (np.ndarray): The input image to rotate.
        angle (float): The angle (in degrees) by which to rotate the image.

    Returns:
        np.ndarray: The rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def main(jewel_path: str) -> None:
    """
    Detects the jewelry in an image, calculates the angle to make it vertical, and saves the rotated image.

    Args:
        jewel_path (str): Path to the input jewelry image.
    """
    jewel_image = cv2.imread(jewel_path, cv2.IMREAD_UNCHANGED)
    if jewel_image is None:
        print("Error: Could not open or find the image.")
        return

    gray_image = cv2.cvtColor(jewel_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    height, width = blurred_image.shape

    corner_values = [
        blurred_image[0, 0],
        blurred_image[0, width - 1],
        blurred_image[height - 1, 0],
        blurred_image[height - 1, width - 1],
    ]
    average_gray_value = np.mean(corner_values)

    mask = (average_gray_value - 3 <= blurred_image) & (
        blurred_image <= average_gray_value + 3
    )
    blurred_image[mask] = 0
    blurred_image[~mask] = 255

    contours, _ = cv2.findContours(
        blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rotation_angles = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area / (height * width) < 0.008:
            continue

        rect = cv2.minAreaRect(contour)
        box_points = cv2.boxPoints(rect)
        box_points = np.intp(box_points)

        top_two_points = sorted(box_points, key=lambda point: point[1])[:2]
        vector = top_two_points[1] - top_two_points[0]
        cos_theta = vector[0] / np.linalg.norm(vector)
        angle = rect[-1] - 90 if cos_theta > 0 and cos_theta != 1 else rect[-1]

        rotation_angles.append(angle)

    average_rotation_angle = -np.mean(rotation_angles)
    print(f"Detected angle for rotation: {average_rotation_angle:.2f} degrees")

    if not (
        -10 < average_rotation_angle < 10
        or 80 <= average_rotation_angle <= 90
        or -90 <= average_rotation_angle <= -80
    ):
        print(f"Rotating the jewelry by {average_rotation_angle:.2f} degrees.")
        rotated_image = rotate_image(jewel_image, average_rotation_angle)
        cv2.imwrite(jewel_path, rotated_image)
    else:
        print("Jewelry is already vertical. No rotation needed.")


if __name__ == "__main__":
    image_path = "jewelry/left_earring.png"
    main(image_path)
