"""
File: check_images_and_annotations.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: May 03, 2023

Description: This program reads in images and their corresponding YOLO format annotations, and projects the segmentation
annotations on the images in order to check the correctness of the annotations.
"""

import cv2
import logging
import numpy as np
import os

from glob import glob
from tqdm import tqdm
from typing import List

from config.const import DATASET_PATH
from convert_segmentation_to_yolo import convert_yolo_format_to_pixels, read_yolo_annotations_to_list
from utils.utils import setup_logger


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- R E A D   I M A G E  T O   L I S T -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def read_image_to_list(dir_train_images: str) -> List[str]:
    """
    Reads image files from a directory and its subdirectories.

    :param dir_train_images: The directory path containing the images.
    :return: A list of image file paths.
    """

    img_files = sorted(glob(os.path.join(dir_train_images, "*.png")))
    if len(img_files) == 0:
        raise ValueError("Image folder is empty!")
    file_names = []

    for _, img_file in tqdm(enumerate(img_files), total=len(img_files), desc="Collecting image file names"):
        file_names.append(img_file)

    return file_names


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main():
    setup_logger()
    original_imgs_file_names = read_image_to_list(DATASET_PATH.get_data_path("ogyei_v2_single_splitted_train_images"))
    yolo_annotations = (
        read_yolo_annotations_to_list(DATASET_PATH.get_data_path("ogyei_v2_single_splitted_train_labels")))

    for i, (img, txt) in enumerate(zip(original_imgs_file_names, yolo_annotations)):
        logging.info(f'Image name: {os.path.basename(img)}')
        logging.info(f'txt name: {os.path.basename(txt)}')
        image = cv2.imread(img)

        with open(txt, "r") as file:
            annotation_text = file.readline().strip()

        annotation_list = list(map(float, annotation_text.split()))
        annotation_list = annotation_list[1:]
        annotation_points = convert_yolo_format_to_pixels(image=image, annotation=annotation_list)

        annotation_points = np.array(annotation_points, dtype=np.int32)
        annotation_points = annotation_points.reshape((-1, 1, 2))
        cv2.polylines(image, [annotation_points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("", cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2)))
        cv2.waitKey()

        # Print a separator line between iterations
        if i < len(original_imgs_file_names) - 1:
            logging.info('-' * 80)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        logging.error(kie)
