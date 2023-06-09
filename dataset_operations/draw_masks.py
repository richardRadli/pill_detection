"""
File: draw_mask.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program based on YOLO format segmentation annotations, creates binary masks of the corresponding
images.
"""

import concurrent.futures
import logging
import numpy as np
import os

from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict

from config.const import DATASET_PATH, IMAGES_PATH
from config.logger_setup import setup_logger


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- P A T H   S E L E C T O R -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def path_selector(operation):
    """
    Selects the correct directory paths based on the given operation string.

    :param operation: A string indicating the operation mode (train or test).
    :return: A dictionary containing directory paths for images, masks, and other related files.
    :raises ValueError: If the operation string is not "train", "valid" or "test".
    """

    if operation.lower() == "train":
        path_to_images = {
            "images": DATASET_PATH.get_data_path("ogyi_v2_splitted_train_images"),
            "labels": DATASET_PATH.get_data_path("ogyi_v2_splitted_train_labels"),
            "masks": IMAGES_PATH.get_data_path("train_masks")
        }
    elif operation.lower() == "valid":
        path_to_images = {
            "images": DATASET_PATH.get_data_path("ogyi_v2_splitted_valid_images"),
            "labels": DATASET_PATH.get_data_path("ogyi_v2_splitted_valid_labels"),
            "masks": IMAGES_PATH.get_data_path("valid_masks")
        }
    elif operation.lower() == "test":
        path_to_images = {
            "images": DATASET_PATH.get_data_path("ogyi_v2_splitted_test_images"),
            "labels": DATASET_PATH.get_data_path("ogyi_v2_splitted_test_labels"),
            "masks": IMAGES_PATH.get_data_path("test_masks")
        }
    else:
        raise ValueError("Wrong operation!")

    return path_to_images


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- L O A D   F I L E S --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def load_files(images_dir: str, labels_dir: str) -> Tuple[List[str], List[str]]:
    """
    This function loads the image and label files from two directories: train_dir and labels_dir.
    :param images_dir: it is the path to the directory containing the image files.
    :param labels_dir: it is the path to the directory containing the corresponding label files for each image
    :return: two lists of file paths: image_files and text_files.
    """

    if not os.path.isdir(images_dir):
        raise ValueError(f"Invalid path: {images_dir} is not a directory")

    if not os.path.isdir(labels_dir):
        raise ValueError(f"Invalid path: {labels_dir} is not a directory")

    image_files = sorted([str(file) for file in Path(images_dir).glob("*.jpg")] +
                         [str(file) for file in Path(images_dir).glob("*.png")])

    text_files = sorted([str(file) for file in Path(labels_dir).glob("*.txt")])

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    if not text_files:
        raise ValueError(f"No text files found in {labels_dir}")

    return image_files, text_files


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- P R O C E S S   D A T A ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def process_data(img_files: str, txt_files: str):
    """
    Given the file paths to an image file and a corresponding text file with object coordinates in YOLO format,
    loads the image, extracts the object coordinates, and creates a binary mask indicating where the object is.

    :param img_files: A string specifying the path to an image file.
    :param txt_files: A string specifying the path to a text file with YOLO object coordinates.

    :return: A tuple of (img, mask), where img is a PIL Image object representing the loaded image,
             and mask is a numpy array representing a binary mask indicating the object location.
             Returns (None, None) if either file path is invalid.
    """

    try:
        img = Image.open(img_files)
        img_width, img_height = img.size
    except FileNotFoundError:
        logging.error(f"{img_files} is not a valid image file.")
        return None, None

    try:
        with open(txt_files, "r") as file:
            line = file.readline().strip()
            yolo_coords = line.split()[1:]
            yolo_coords = [float(x.strip('\'')) for x in yolo_coords]
    except FileNotFoundError:
        logging.error(f"{txt_files} is not a valid text file.")
        return None, None

    coords = [int(coord * img_width if i % 2 == 0 else coord * img_height) for i, coord in enumerate(yolo_coords)]

    mask = Image.new('1', (img_width, img_height), 0)
    xy = list(zip(coords[::2], coords[1::2]))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask)

    return img, mask


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- S A V E   M A S K S --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def save_masks(mask: np.ndarray, img_file: str, path_to_files: Dict[str, str]) -> None:
    """
    This function saves the mask to a given path.

    :param mask: Mask image.
    :param img_file: path of the image file.
    :param path_to_files: path to the files.
    :return: None
    """

    name = os.path.basename(img_file)
    save_path = (os.path.join(path_to_files.get("masks"), name))
    mask_pil = Image.fromarray(mask)
    mask_pil.save(save_path)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- M A I N --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main(operation: str = "Train") -> None:
    """
    Runs the main processing pipeline.
    :return: None
    """

    setup_logger()
    path_to_files = path_selector(operation)

    img_files, txt_files = load_files(images_dir=path_to_files.get("images"), labels_dir=path_to_files.get("labels"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for img_file, txt_file in zip(img_files, txt_files):
            futures.append(executor.submit(process_data, img_file, txt_file))

        for future, (img_file, _) in tqdm(zip(futures, zip(img_files, txt_files)), total=len(img_files),
                                          desc="Processing data"):
            try:
                img, mask = future.result()
                save_masks(mask=mask, img_file=img_file, path_to_files=path_to_files)
            except Exception as e:
                logging.error(f"Error processing {img_file}: {e}")


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- __M A I N__ ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(operation="Train")
