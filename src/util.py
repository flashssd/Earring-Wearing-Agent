import os
import io
import shutil
import requests
from PIL import Image, ImageDraw, ImageFont


def upload_image(
    result_path: str, url: str = "http://94.74.75.247:8005/fs/upload"
) -> str | None:
    """
    Uploads an image to the specified URL.

    Args:
        result_path (str): Path to the image file to be uploaded.
        url (str): The URL to which the image will be uploaded. Defaults to a preset URL.

    Returns:
        str | None: The URL of the uploaded image if successful, otherwise None.
    """
    with open(result_path, "rb") as img_file:
        files = {"file": img_file}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Image uploaded successfully!")
        print("Response:", response.json())
        return response.json()["data"]
    else:
        print("Failed to upload image")
        return None


def download_file(url: str, local_filename: str) -> None:
    """
    Downloads an image from a URL and saves it in JPEG format.

    Args:
        url (str): The URL to download the image from.
        local_filename (str): The filename where the image will be saved.
    """
    response = requests.get(url)
    if response.status_code == 200:
        image_data = io.BytesIO(response.content)
        image = Image.open(image_data)

        if image.format == "PNG":
            image = image.convert("RGB")
            image.save(local_filename, "JPEG")
            print(f"PNG image successfully converted and saved as {local_filename}")
        elif image.format == "JPEG":
            with open(local_filename, "wb") as file:
                file.write(response.content)
            print(f"JPEG image successfully saved as {local_filename}")
        else:
            print(f"Unsupported image format: {image.format}")
    else:
        print(f"Failed to retrieve the file. Status code: {response.status_code}")


def delete_all_files(directory: str) -> None:
    """
    Deletes all files and directories in the specified directory.

    Args:
        directory (str): The directory to clear of all files and subdirectories.
    """
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    files = os.listdir(directory)

    for filename in files:
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def add_watermark(
    input_image_path: str,
    output_image_path: str,
    watermark_text: str,
    font_path: str = None,
    font_size: int = 200,
) -> None:
    """
    Adds a diagonal watermark to an image and saves the result.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the watermarked image.
        watermark_text (str): The text to use as a watermark.
        font_path (str, optional): Path to the font file to be used. Defaults to None, which uses the default font.
        font_size (int, optional): Font size for the watermark text. Defaults to 200.
    """
    original = Image.open(input_image_path).convert("RGBA")
    txt = Image.new("RGBA", original.size, (255, 255, 255, 0))

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(txt)
    text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    width, height = original.size

    x = (width - text_width) / 2
    y = (height - text_height) / 2

    draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 128))

    rotated_txt = txt.rotate(45, expand=1)
    new_txt = Image.new("RGBA", original.size, (255, 255, 255, 0))
    new_txt.paste(
        rotated_txt,
        (
            int((new_txt.width - rotated_txt.width) / 2),
            int((new_txt.height - rotated_txt.height) / 2),
        ),
        rotated_txt,
    )

    watermarked = Image.alpha_composite(original, new_txt)
    watermarked.convert("RGB").save(output_image_path)
