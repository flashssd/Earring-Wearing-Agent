import os
from PIL import Image
import random, yaml
import warnings
from src.get_earring import main as prepare_earrings
from src.compute_points import (
    jewel_anchor_point,
    model_anchor_point,
    ear_ring_place,
)
from src.resize import resize_ear_ring
from src.rotation import main as find_and_rotate_jewelry
from src.util import download_file, add_watermark, upload_image, delete_all_files
from src.effect import add_colored_blur
from src.constants import (
    MODEL_DIR,
    JEWEL_DIR,
    MODEL_PATH,
    JEWEL_PATH,
    JEWEL_LEFT_PATH,
    JEWEL_RIGHT_PATH,
    JEWEL_NO_BG_PATH,
    MODEL_SALT_PATH,
    WATERMARK_TEXT,
    ROBOTO,
    FONT_SIZE,
)
from src.adjust_light import main as adjust_light


def put_on_by_ear(ear_dict, model_path):
    """
    Places the resized earring images onto the corresponding ear locations on the background image.

    Args:
        ear_dict (dict): A dictionary with keys 'left' and 'right', containing the coordinates and dimensions of the ears.
        background_path (str): Path to the background image (e.g., the model's image).

    Returns:
        None. The function modifies the background image by pasting the earring images onto the ears.
    """
    # Open the background image (the model's image)
    model_image = Image.open(model_path)

    # Iterate through each ear in the ear dictionary (left and/or right ear)
    for ear, (x, y, w, h) in ear_dict.items():
        # Path to the resized earring image
        earring_path = JEWEL_LEFT_PATH if ear == "left" else JEWEL_RIGHT_PATH

        # Compute the point on the earring image that should align with the ear on the background image
        jewel_anchor_x, jewel_anchor_y = jewel_anchor_point(earring_path)

        # Calculate the final position of the earring on the background image
        model_anchor_x, model_anchor_y = model_anchor_point(model_path, x, y, w, h)

        # Open the foreground (earring) image
        jewel_resized_image = Image.open(earring_path)

        # Calculate the offset for placing the earring relative to the ear location
        match_x = int(model_anchor_x - jewel_anchor_x)
        match_y = int(model_anchor_y - jewel_anchor_y)

        # Paste the earring image onto the background image at the computed offset, using the earring image's alpha channel for transparency
        model_image.paste(jewel_resized_image, (match_x, match_y), jewel_resized_image)

    # Display the final composite image (model with earrings)
    model_image.show()

    # Save the modified image to the background path
    model_image.save(model_path)


def main(
    model_path_url,
    jewel_path_url,
    jewelry_type=1,
    test=False,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(JEWEL_DIR, exist_ok=True)

    try:
        # download the file from the cloud link
        download_file(model_path_url, MODEL_PATH)
        download_file(jewel_path_url, JEWEL_PATH)
        # extract earring dictionary
        ear_dict = ear_ring_place(MODEL_PATH)

        # extract the left and right earrings
        if (not test) or (not os.path.exists(JEWEL_NO_BG_PATH)):
            prepare_earrings()

        for ear in ear_dict.keys():
            earring_path = JEWEL_LEFT_PATH if ear == "left" else JEWEL_RIGHT_PATH
            x, y, w, h = ear_dict[ear]
            adjust_light(MODEL_PATH, earring_path, x, y, w, h)

            find_and_rotate_jewelry(earring_path)

            add_colored_blur(earring_path)

            resize_ear_ring(
                earring_path,
                int(w),
                int(h),
            )
        # put on the earrings
        put_on_by_ear(ear_dict, MODEL_PATH)
        # add watermark
        add_watermark(
            input_image_path=MODEL_PATH,
            output_image_path=MODEL_SALT_PATH,
            watermark_text=WATERMARK_TEXT,
            font_path=ROBOTO,  # Optional: specify path to a .ttf font file
            font_size=FONT_SIZE,  # Adjust the font size as needed
        )
        img_link = upload_image(MODEL_PATH)
        saltImg_link = upload_image(MODEL_SALT_PATH)
    finally:
        # clear all files to save space
        delete_all_files(MODEL_DIR)
        delete_all_files(JEWEL_DIR)
    return img_link, saltImg_link


if __name__ == "__main__":
    warnings.simplefilter("error")

    with open("jewels_and_models_links.yaml", "r") as file:
        data = yaml.safe_load(file)

    model_path_url = random.choice(list(data["models"].values()))
    jewel_path_url = random.choice(list(data["jewels"].values()))
    print(f"Chosen Model is: {model_path_url}")
    print(f"Chosen Jewel is: {jewel_path_url}")

    jewelry_type = 1
    main(model_path_url, jewel_path_url, jewelry_type, test=False)
