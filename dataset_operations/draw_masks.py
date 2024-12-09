"""
File: draw_mask.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program based on YOLO format segmentation annotations, creates binary masks of the corresponding
images.
"""

import concurrent.futures
import cv2
import logging
import numpy as np
import os

from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Tuple, List, Dict

from config.dataset_paths_selector import dataset_images_path_selector
from config.json_config import json_config_selector
from utils.utils import file_reader, setup_logger, load_config_json

cfg = (
    load_config_json(
        json_schema_filename=json_config_selector("stream_images").get("schema"),
        json_filename=json_config_selector("stream_images").get("config")
    )
)


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- P A T H   S E L E C T O R -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def path_selector(operation: str) -> dict:
    """
    Selects the correct directory paths based on the given operation string.

    Args:
        operation: A string indicating the operation mode (train or test).

    Returns:
        A dictionary containing directory paths for images, masks, and other related files.

    Raises ValueError:
        If the operation string is not "train", "valid", "test" or "whole".
    """

    dataset_type = cfg.get("dataset_type")

    if operation.lower() == "customer":
        path_to_images = {
            "images": dataset_images_path_selector(dataset_type).get(operation).get("customer_images"),
            "labels": dataset_images_path_selector(dataset_type).get(operation).get("customer_segmentation_labels"),
            "masks": dataset_images_path_selector(dataset_type).get(operation).get("customer_mask_images")
        }
    elif operation.lower() == "reference":
        path_to_images = {
            "images": dataset_images_path_selector(dataset_type).get(operation).get("reference_images"),
            "labels": dataset_images_path_selector(dataset_type).get(operation).get("reference_segmentation_labels"),
            "masks": dataset_images_path_selector(dataset_type).get(operation).get("reference_mask_images")
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
    Args:
        images_dir: it is the path to the directory containing the image files.
        labels_dir: it is the path to the directory containing the corresponding label files for each image

    Returns:
         two lists of file paths: image_files and text_files.
    """

    if not os.path.isdir(images_dir):
        raise ValueError(f"Invalid path: {images_dir} is not a directory")

    if not os.path.isdir(labels_dir):
        raise ValueError(f"Invalid path: {labels_dir} is not a directory")

    image_files = file_reader(images_dir, "jpg")
    text_files = file_reader(labels_dir, "txt")

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    if not text_files:
        raise ValueError(f"No text files found in {labels_dir}")

    assert len(image_files) == len(text_files)

    return image_files, text_files


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- P R O C E S S   D A T A ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def process_data(img_files: str, txt_files: str):
    """
    Given the file paths to an image file and a corresponding text file with object coordinates in YOLO format,
    loads the image, extracts the object coordinates, and creates a binary mask indicating where the object is.

    Args:
        img_files: A string specifying the path to an image file.
        txt_files: A string specifying the path to a text file with YOLO object coordinates.

    Returns:
        A tuple of (img, mask), where img is a PIL Image object representing the loaded image,
        and mask is a numpy array representing a binary mask indicating the object location.
        Returns (None, None) if either file path is invalid.
    """

    try:
        img = Image.open(img_files)
        img_width, img_height = img.size
        img.close()
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

    coords = []
    for i, coord in enumerate(yolo_coords):
        if i % 2 == 0:
            coords.append(int(coord * img_width))
        else:
            coords.append(int(coord * img_height))

    mask = Image.new('1', (img_width, img_height), 0)
    ImageDraw.Draw(mask).polygon(xy=coords, outline=1, fill=1)
    mask = np.array(mask)

    return mask


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- S A V E   M A S K S --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def save_masks(mask: np.ndarray, img_file: str, path_to_files: Dict[str, str]) -> None:
    """
    This function saves the mask to a given path.

    Args:
        mask: Mask image.
        img_file: path of the image file.
        path_to_files: path to the files.

    Returns:
        None
    """

    name = os.path.basename(img_file)
    save_path = (os.path.join(path_to_files.get("masks"), name))
    mask_pil = mask.astype(np.uint8) * 255
    cv2.imwrite(save_path, mask_pil)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- M A I N --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main(operation: str = "train", batch_size: int = 10) -> None:
    """
    Runs the main processing pipeline.

    Returns:
        None
    """

    setup_logger()
    path_to_files = path_selector(operation)

    img_files, txt_files = load_files(images_dir=path_to_files.get("images"), labels_dir=path_to_files.get("labels"))

    total_files = len(img_files)
    num_batches = (total_files + batch_size - 1) // batch_size

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.get("max_workers")) as executor:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_files)
            batch_img_files = img_files[start_idx:end_idx]
            batch_txt_files = txt_files[start_idx:end_idx]

            futures = []
            for img_file, txt_file in zip(batch_img_files, batch_txt_files):
                futures.append(executor.submit(process_data, img_file, txt_file))

            for future, (img_file, _) in tqdm(zip(futures, zip(batch_img_files, batch_txt_files)),
                                              total=len(batch_img_files),
                                              desc=f"Processing batch {i+1}/{num_batches}"):
                try:
                    mask = future.result()
                    save_masks(mask=mask, img_file=img_file, path_to_files=path_to_files)
                except Exception as e:
                    logging.error(f"Error processing {img_file}: {e}")


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- __M A I N__ ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        operations = ["reference", "customer"]
        for op in operations:
            main(operation=op)
    except KeyboardInterrupt as kie:
        logging.error(f"The following error has occurred: {kie}")