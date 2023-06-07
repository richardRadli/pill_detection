"""
File: change_background.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: May 22, 2023

Description: The program  applies bitwise operations to subtract the foreground from the background and combines them to
create an image with the pill(s) on a various backgrounds. The main() function uses multithreading to process multiple
images in parallel using the change_background() function.
"""

import cv2
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tqdm import tqdm

from config.const import DATASET_PATH, IMAGES_PATH
from convert_yolo import convert_yolo_format_to_pixels
from utils.utils import measure_execution_time


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ C H A N G E   B A C K G R O U N D -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def change_background(image_path: str, annotations_path: str, background_path: str) -> None:
    """
    The function reads in the path of an image and a corresponding annotation file. Converts the YOLO annotation values
     to pixel coordinates, after that it applies bitwise and operation in order to subtract the fore and background.
     Finally, it adds together the images. Results should be an image with the pill(s) on it, and a homogeneous
     background.

    :param image_path: Path to where the images are located.
    :param annotations_path: Path to where the annotations are located.
    :param background_path: Color of the background.
    :return: None
    """

    image = cv2.imread(image_path)
    background = cv2.imread(background_path)

    with open(annotations_path, 'r') as file:
        annotation_text = file.readlines()

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    foreground = None

    for anno_text in annotation_text:
        annotation_list = list(map(float, anno_text.split()))
        annotation_list = annotation_list[1:]
        annotation_points = convert_yolo_format_to_pixels(image=image, annotation=annotation_list)

        annotation_points = np.array(annotation_points, dtype=np.int32)
        annotation_points = annotation_points.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [annotation_points], color=(255, 255, 255))

        foreground = cv2.bitwise_and(image, image, mask=mask)
        background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

    output_image = cv2.add(foreground, background)

    output_file_name = os.path.join(IMAGES_PATH.get_data_path("wo_background"), os.path.basename(image_path))
    cv2.imwrite(output_file_name, output_image)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
@measure_execution_time
def main() -> None:
    """
    Main function of the file. Collects the files and in a parallelized way, it executes the change_background()
    function. As to demonstrate technical capability, the execution time is measured with the measure_execution_time()
    decorator function.

    :return: None
    """

    images = sorted(glob(DATASET_PATH.get_data_path("ogyi_v2_splitted_train_images") + "/*.png"))
    annotations = sorted(glob(DATASET_PATH.get_data_path("ogyi_v2_splitted_train_labels") + "/*.txt"))
    backgrounds = sorted(glob(DATASET_PATH.get_data_path("dtd_images") + "/*.jpg"))

    with ThreadPoolExecutor() as executor:
        futures = []
        for img, txt, bg in tqdm(zip(images, annotations, backgrounds), total=len(images), desc="Processing images"):
            future = executor.submit(change_background, img, txt, bg)
            futures.append(future)

        for future in tqdm(futures, total=len(futures), desc='Saving images'):
            future.result()


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
