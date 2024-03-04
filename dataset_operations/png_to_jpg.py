import os

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

from config.const import DATASET_PATH


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ C O N V E R T   A N D   S A V E -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def convert_and_save(src_directory: str, dst_directory: str, jpg_file: str) -> None:
    """
    Convert a PNG image to JPG format and save it to the destination directory.

    Args:
        src_directory (str): The source directory containing the PNG image.
        dst_directory (str): The destination directory to save the converted JPG image.
        jpg_file (str): The filename of the PNG image to be converted and saved.

    Returns:
        None
    """

    png_path = os.path.join(src_directory, jpg_file)
    jpg_path = os.path.join(dst_directory, os.path.splitext(jpg_file)[0] + '.jpg')

    with Image.open(png_path) as img:
        img = img.convert('RGB')
        img.save(jpg_path)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- C O N V E R T   P N G   T O   J P G -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def convert_png_to_jpg(src_directory: str, dst_directory: str, num_threads: int = 4) -> None:
    """
    Convert PNG images to JPG format concurrently using multiple threads.

    Args:
        src_directory (str): The source directory containing PNG images.
        dst_directory (str): The destination directory to save the converted JPG images.
        num_threads (int): The number of threads to use for concurrent conversion.

    Returns:
        None
    """

    os.makedirs(dst_directory, exist_ok=True)

    png_files = [file for file in os.listdir(src_directory) if file.lower().endswith('.png')]

    with (ThreadPoolExecutor(max_workers=num_threads) as executor):
        futures = \
        {executor.submit(convert_and_save, src_directory, dst_directory, jpg_file): jpg_file for jpg_file in png_files}

        for future in tqdm(futures):
            future.result()


if __name__ == "__main__":
    input_directory = DATASET_PATH.get_data_path("ogyei_reference_mask_images")
    output_directory = DATASET_PATH.get_data_path("ogyei_reference_mask_images")
    convert_png_to_jpg(src_directory=input_directory,
                       dst_directory=output_directory,
                       num_threads=6)
