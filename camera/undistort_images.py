"""
File: undistort_images.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program takes a directory of test images as input, undistorts the images using camera calibration data,
and saves the undistorted images to an output directory. It performs the undistortion process using multithreading for
faster processing.
"""

import cv2
import os
import logging
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from config.const import DATA_PATH


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++ U N D I S T O R T   T E S T   I M A G E S ++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class UnDistortTestImages:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        cam_mtx_np_file = os.path.join(DATA_PATH.get_data_path("cam_data"),
                                       os.listdir(DATA_PATH.get_data_path("cam_data"))[3])
        logging.info(f"The loaded camera matrix: {os.path.basename(cam_mtx_np_file)}")
        data = np.load(cam_mtx_np_file, allow_pickle=True)

        self.matrix = data.item()['matrix']
        self.dist_coeff = data.item()['dist_coeff']
        self.undst_matrix = data.item()['undst_matrix']
        self.roi = data.item()['roi']

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- P R O C E S S   I M A G E -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def process_image(self, img_path: str, output_path: str) -> None:
        """
        Process an image by undistorting it, using camera calibration data and save the undistorted image.

        :param img_path: The path of the input image.
        :param output_path: The path to save the undistorted image.
        :return: None
        """

        src_img = cv2.imread(img_path)

        undistorted_image = cv2.undistort(src_img, self.matrix, self.dist_coeff, None, self.undst_matrix)
        x, y, w, h = self.roi
        undistorted_image = undistorted_image[y:y + h, x:x + w]

        cv2.imwrite(output_path, undistorted_image)
        logging.info(f'Image {os.path.basename(output_path)} has been saved!')

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- U N D I S T O R T   I M A G E S ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def undistort_images(self) -> None:
        """
        Undistort the images in the input directory and save them to the output directory.

        :return: None
        """

        input_dir = "C:/Users/ricsi/Desktop/multitask"
        output_dir = "C:/Users/ricsi/Desktop/multitask_undistorted"

        os.makedirs(output_dir, exist_ok=True)

        with ThreadPoolExecutor() as executor:
            for subdir in tqdm(os.listdir(input_dir)):
                sub_input_dir = os.path.join(input_dir, subdir)
                sub_output_dir = os.path.join(output_dir, subdir)

                if not os.path.exists(sub_output_dir):
                    os.makedirs(sub_output_dir)

                image_paths = [os.path.join(sub_input_dir, filename) for filename in os.listdir(sub_input_dir)]
                output_paths = [os.path.join(sub_output_dir, os.path.basename(path)) for path in image_paths]
                executor.map(self.process_image, image_paths, output_paths)


if __name__ == "__main__":
    undistort_test_images = UnDistortTestImages()
    undistort_test_images.undistort_images()
