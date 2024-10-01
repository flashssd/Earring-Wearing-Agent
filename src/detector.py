import os
import sys
import cv2
from ultralytics.models import YOLO


class Detector:
    """
    A class for detecting objects in images using a pre-trained YOLO model.

    Args:
        weights_path (str): The path to the YOLO model weights file.
    """

    def __init__(self, weights_path: str):
        self.weights = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "", weights_path
        )
        self.model = YOLO(self.weights)

    def normalize_to_absolute(
        self, x_norm: float, y_norm: float, w_norm: float, h_norm: float, 
        img_width: int, img_height: int
    ) -> tuple[int, int, int, int]:
        """
        Convert normalized bounding box coordinates to absolute pixel values.

        Args:
            x_norm (float): Normalized x-coordinate of the center of the bounding box.
            y_norm (float): Normalized y-coordinate of the center of the bounding box.
            w_norm (float): Normalized width of the bounding box.
            h_norm (float): Normalized height of the bounding box.
            img_width (int): Width of the input image in pixels.
            img_height (int): Height of the input image in pixels.

        Returns:
            tuple: (x, y, w, h) representing the top-left corner and size of the bounding box in pixels.
        """
        x_left = x_norm - 0.5 * w_norm
        y_up = y_norm - 0.5 * h_norm
        x = int(x_left * img_width)
        y = int(y_up * img_height)
        w = int(w_norm * img_width)
        h = int(h_norm * img_height)
        return x, y, w, h

    def detect(self, image_name: str) -> tuple[list[tuple[int, int, int, int]], list[str]]:
        """
        Detect objects in an image using the YOLO model and return their bounding boxes and labels.

        Args:
            image_name (str): The path to the image file.

        Returns:
            tuple: A list of bounding boxes as (x, y, w, h) and a list of object labels.
        """
        detections = []
        keys = []
        img = cv2.imread(image_name)
        img_height, img_width = img.shape[:2]

        results = self.model(image_name, imgsz=img.shape[:2])
        for result in results:
            x_norm, y_norm, w_norm, h_norm = result.boxes.xywhn[0].numpy()
            x, y, w, h = self.normalize_to_absolute(
                x_norm, y_norm, w_norm, h_norm, img_width, img_height
            )
            detections.append((x, y, w, h))

            names = result.names
            last_key = list(names.keys())[-1]
            keys.append(names[last_key])

        return detections, keys


if __name__ == "__main__":
    """
    Main execution block to run the object detection on the given image and save the result with bounding boxes.
    
    Args:
        sys.argv[1] (str): The path to the image file.
    
    Saves:
        An image with bounding boxes drawn around detected objects in the same directory as the input image.
    """
    file_name = sys.argv[1]
    img = cv2.imread(file_name)
    detector = Detector(weights_path="yolov8_weights.pt")  # You need to provide the weights file here
    detected_loc, _ = detector.detect(file_name)
    for x, y, w, h in detected_loc:
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
    cv2.imwrite(file_name + "_detected.jpg", img)
