from typing import Literal
from src.constants import CLIP_API_URL, CLIP_API_KEY, RMBG_API_KEY


def bg_remove_clip(
    API_URL: str, API_KEY: str, jewel_path: str, jewel_nobg_path: str
) -> None:
    """
    Remove the background of an image using the Clipdrop API and save the output image.

    Args:
        API_URL (str): The URL of the Clipdrop API endpoint.
        API_KEY (str): The API key for authenticating the Clipdrop request.
        jewel_path (str): The file path of the input image with background.
        jewel_nobg_path (str): The file path where the output image without background will be saved.
    """
    import requests

    # Read the image file in binary mode
    with open(jewel_path, "rb") as jewel_image_file:
        jewel_image_data = jewel_image_file.read()

    # Prepare the headers and the image payload for the API request
    headers = {"x-api-key": API_KEY}
    files = {"image_file": ("image.jpg", jewel_image_data, "image/jpeg")}

    # Send a POST request to the API with the image data
    response = requests.post(API_URL, headers=headers, files=files)

    # If the request is successful, save the output image
    if response.status_code == 200:
        with open(jewel_nobg_path, "wb") as jewel_output_file:
            jewel_output_file.write(response.content)
        print(f"Background removed successfully. Output saved to {jewel_nobg_path}")
    else:
        print(
            f"Error: {response.status_code}, {response.json().get('error', 'Unknown error')}"
        )


def bg_remove_removebg(API_KEY: str, jewel_path: str, jewel_nobg_path: str) -> None:
    """
    Remove the background of an image using the remove.bg API and save the output image.

    Args:
        API_KEY (str): The API key for authenticating the remove.bg request.
        jewel_path (str): The file path of the input image with background.
        jewel_nobg_path (str): The file path where the output image without background will be saved.
    """
    from removebg import RemoveBg
    import os

    # Initialize the RemoveBg API object with the provided API key
    rmbg = RemoveBg(API_KEY, "error.log")

    # Remove the background and save the result in a temporary file
    rmbg.remove_background_from_img_file(jewel_path)

    # Rename the temporary output file to the desired output path
    os.rename(jewel_path + "_no_bg.png", jewel_nobg_path)


def main(
    remover: Literal["clip", "remove.bg", "rembg"],
    jewel_path: str,
    jewel_nobg_path: str,
) -> None:
    """
    Remove the background of an image using the specified background remover service.

    Args:
        remover (Literal["clip", "remove.bg", "rembg"]): The background remover to use. Options are "clip", "remove.bg", or "rembg".
        jewel_path (str): The file path of the input image with background.
        jewel_nobg_path (str): The file path where the output image without background will be saved.
    """
    if remover == "clip":
        # Use Clipdrop API to remove background
        bg_remove_clip(CLIP_API_URL, CLIP_API_KEY, jewel_path, jewel_nobg_path)
    elif remover == "remove.bg":
        # Use remove.bg API to remove background
        bg_remove_removebg(RMBG_API_KEY, jewel_path, jewel_nobg_path)
    elif remover == "rembg":
        from rembg import remove

        # Remove the background using rembg library
        with open(jewel_path, "rb") as jewel_input_file:
            jewel_input_data = jewel_input_file.read()
            result = remove(jewel_input_data)

        # Save the result to the specified output file
        with open(jewel_nobg_path, "wb") as jewel_output_file:
            jewel_output_file.write(result)
