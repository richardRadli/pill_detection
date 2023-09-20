"""
File: create_stream_images.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program creates the different images (contour, lbp, rgb, texture) for the substreams.
"""

import cv2
import logging
import numpy as np
import pandas as pd
import re
import os
import shutil

from concurrent.futures import ThreadPoolExecutor, wait
from glob import glob
from pathlib import Path
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from typing import Tuple

from config.config import ConfigStreamNetwork
from config.const import IMAGES_PATH
from config.network_configs import dataset_images_path_selector, sub_stream_network_configs
from utils.utils import measure_execution_time, setup_logger, numerical_sort


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++ C R E A T E   S T R E A M   I M A G E S +++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CreateStreamImages:
    def __init__(self):
        setup_logger()

        self.cfg = ConfigStreamNetwork().parse()

        path_to_stream_images = sub_stream_network_configs(self.cfg)
        self.rgb_images_path = (
            path_to_stream_images.get("RGB").get(self.cfg.dataset_operation).get(self.cfg.dataset_type))
        self.contour_images_path =\
            path_to_stream_images.get("Contour").get(self.cfg.dataset_operation).get(self.cfg.dataset_type)
        self.texture_images_path = (
            path_to_stream_images.get("Texture").get(self.cfg.dataset_operation).get(self.cfg.dataset_type))
        self.lbp_images_path = (
            path_to_stream_images.get("LBP").get(self.cfg.dataset_operation).get(self.cfg.dataset_type))

        df = pd.DataFrame.from_dict(vars(self.cfg), orient='index', columns=['value'])
        logging.info(df)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- D R A W   B O U N D I N G   B O X ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def draw_bounding_box(self, in_img: np.ndarray, seg_map: np.ndarray, output_path: str) -> None:
        """
        Draws bounding box over medicines. It draws only the biggest bounding box, small ones are terminated.
        After that it crops out the bounding box's content.

        :param in_img: input testing image
        :param seg_map: output of the unet for the input testing image
        :param output_path: where the file should be saved
        :return: None
        """

        ret, thresh = cv2.threshold(seg_map, 0, 255, cv2.THRESH_BINARY)
        n_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S)

        max_area = 0
        max_x, max_y, max_w, max_h = None, None, None, None

        for i in range(1, n_objects):
            x, y, w, h, area = stats[i]
            if area > self.cfg.threshold_area and area > max_area:
                max_x, max_y, max_w, max_h = x, y, w, h
                max_area = area

        if max_area > 0:
            center_x = max_x + max_w / 2
            center_y = max_y + max_h / 2
            side_length = max(max_w, max_h)
            square_x = int(center_x - side_length / 2)
            square_y = int(center_y - side_length / 2)
            obj = in_img[square_y:square_y + side_length, square_x:square_x + side_length]
            if obj.size != 0:
                cv2.imwrite(output_path, obj)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- P R O C E S S   I M A G E -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def process_image(self, image_paths: Tuple[str, str]) -> None:
        """
        Processes a single image by reading in the color and mask images, drawing the bounding box, and saving the
        result.
        :param image_paths: tuple of paths to the color and mask images
        :return: None
        """

        color_path, mask_path = image_paths
        output_name = os.path.basename(color_path)
        output_file = (os.path.join(self.rgb_images_path, output_name))
        color_imgs = cv2.imread(str(color_path), 1)
        mask_imgs = cv2.imread(str(mask_path), 0)
        self.draw_bounding_box(color_imgs, mask_imgs, output_file)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------- S A V E   B O U N D I N G   B O X   I M G S ----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_rgb_images(self) -> None:
        """
        Reads in the images, draws the bounding box, and saves the images in parallel.
        :return: None
        """

        path_to_images = dataset_images_path_selector()

        color_images_dir = Path(path_to_images.get(self.cfg.dataset_type).get(self.cfg.dataset_operation).get("images"))
        color_images = sorted(color_images_dir.glob("*.png"))

        mask_images_dir = Path(path_to_images.get(self.cfg.dataset_type).get(self.cfg.dataset_operation).get("masks"))
        mask_images = sorted(mask_images_dir.glob("*.png"))

        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(self.process_image, zip(color_images, mask_images)), total=len(color_images),
                      desc="RGB images"))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ C R E A T E   C O N T O U R   I M A G E S -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_contour_images(args) -> None:
        """
        Applies edge detection on an input image and saves the resulting edge map.

        :param args: Tuple containing:
            - cropped_image (numpy.ndarray): The input image to apply edge detection on.
            - output_path (str): The path to save the output edge map.
            - kernel_size (int): The size of the kernel used in median blur.
            - canny_low_thr (int): The lower threshold for Canny edge detection.
            - canny_high_thr (int): The higher threshold for Canny edge detection.
        :return: None
        """

        cropped_image, output_path, kernel_size, canny_low_thr, canny_high_thr = args
        blured_images = cv2.medianBlur(cropped_image, kernel_size, 0)
        edges = cv2.Canny(blured_images, canny_low_thr, canny_high_thr)
        cv2.imwrite(output_path, edges)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- S A V E   C O N T O U R   I M G S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_contour_images(self) -> None:
        """
        Loads RGB images from the designated directory and applies Canny edge detection algorithm to extract contours
        for each image. The resulting images are saved in a new directory designated for contour images.

        :return: None
        """

        rgb_images = sorted(glob(str(self.rgb_images_path) + "/*.png"), key=numerical_sort)
        args_list = []

        for img_path in tqdm(rgb_images, desc="Contour images"):
            output_name = "contour_" + os.path.basename(img_path)
            output_file = os.path.join(self.contour_images_path, output_name)
            bbox_imgs = cv2.imread(img_path, 0)
            args_list.append((bbox_imgs, output_file, self.cfg.kernel_median_contour, self.cfg.canny_low_thr,
                              self.cfg.canny_high_thr))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.create_contour_images, args) for args in args_list]
            wait(futures)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ C R E A T E   T E X T U R E   I M A G E S -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_texture_images(args) -> None:
        """
        Given a cropped input image, this method applies a Gaussian blur, subtracts the blurred image from the original,
        normalizes the resulting image intensities between 0 and 255, and saves the resulting texture image to a
        given path.
        :param args: A tuple containing the following elements:
            - cropped_image: A numpy array representing the cropped input image
            - output_path: A string representing the path to save the output texture image
            - kernel_size: An integer representing the size of the Gaussian blur kernel
        :return: None
        """

        cropped_image, output_path, kernel_size = args
        blured_image = cv2.GaussianBlur(cropped_image, kernel_size, 0)
        sub_img = cv2.subtract(cropped_image, blured_image)
        min_val = sub_img.min()
        max_val = sub_img.max()
        sub_img = (sub_img - min_val) * (255.0 / (max_val - min_val))
        sub_img = sub_img.astype(np.uint8)
        cv2.imwrite(output_path, sub_img)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- S A V E   T E X T U R E   I M G S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_texture_images(self) -> None:
        """
        Save texture images using Gaussian Blur filter.

        :return: None
        """

        rgb_images = sorted(glob(self.rgb_images_path + "/*.png"), key=numerical_sort)
        args_list = []

        for img_path in tqdm(rgb_images, desc="Texture images"):
            output_name = "texture_" + os.path.basename(img_path)
            output_file = os.path.join(self.texture_images_path, output_name)
            bbox_imgs = cv2.imread(img_path, 0)
            args_list.append((bbox_imgs, output_file,
                              (self.cfg.kernel_gaussian_texture, self.cfg.kernel_gaussian_texture)))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.create_texture_images, args) for args in args_list]
            wait(futures)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- P R O C E S S   L B P   I M A G E S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def process_lbp_image(img_gray: np.ndarray, dest_image_path: str) -> None:
        """
        Process the LBP image for a given image file and save it to the destination directory.

        :param img_gray: The BGR image as a numpy array.
        :param dest_image_path: The destination path to save the LBP image.
        :return: None.
        """

        lbp_image = local_binary_pattern(image=img_gray, P=8 * 1, R=1, method="default")
        cv2.imwrite(dest_image_path, lbp_image)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- S A V E   L B P   I M A G E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_lbp_images(self) -> None:
        """
        Saves LBP images in parallel.
        :return: None
        """

        rgb_images = sorted(glob(self.rgb_images_path + "/*.png"), key=numerical_sort)

        with ThreadPoolExecutor() as executor:
            futures = []
            for img_path in tqdm(rgb_images, desc="LBP images"):
                output_name = "lbp_" + os.path.basename(img_path)
                output_file = os.path.join(self.lbp_images_path, output_name)
                bbox_imgs = cv2.imread(img_path, 0)
                future = executor.submit(self.process_lbp_image, bbox_imgs, output_file)
                futures.append(future)

            for future in futures:
                future.result()

    @staticmethod
    def copy_query_images():
        source_root = IMAGES_PATH.get_data_path("test_ref_ogyei")
        destination_root = IMAGES_PATH.get_data_path("test_query_ogyei")

        class_label_dirs = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]

        for class_label_dir in class_label_dirs:
            class_label_path = os.path.join(source_root, class_label_dir)

            subdirs = os.listdir(class_label_path)

            for subdir in subdirs:
                subdir_path = os.path.join(class_label_path, subdir)

                image_files = [f for f in os.listdir(subdir_path) if
                               f.endswith('.jpg') or f.endswith('.png')]

                destination_subdir = os.path.join(destination_root, class_label_dir, subdir)
                os.makedirs(destination_subdir, exist_ok=True)

                for i, image_file in enumerate(image_files):
                    if (i + 1) % 3 == 0:
                        source_image_path = os.path.join(subdir_path, image_file)
                        destination_image_path = os.path.join(destination_subdir, image_file)
                        shutil.move(source_image_path, destination_image_path)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ C R E A T E   L A B E L   D I R S -------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_label_dirs(self, rgb_path: str, contour_path: str, texture_path: str, lbp_path: str) -> None:
        """
        Create labeled directories for image files based on the given dataset.

        :param rgb_path: str, path to the directory containing RGB images.
        :param contour_path: str, path to the directory containing contour images.
        :param texture_path: str, path to the directory containing texture images.
        :param lbp_path: str, path to the directory containing LBP images.
        :return: None
        """

        files_rgb = os.listdir(rgb_path)
        files_contour = os.listdir(contour_path)
        files_texture = os.listdir(texture_path)
        files_lbp = os.listdir(lbp_path)
        value = None

        for idx, (file_rgb, file_contour, file_texture, file_lbp) in \
                tqdm(enumerate(zip(files_rgb, files_contour, files_texture, files_lbp)), desc="Copying image files"):
            if file_rgb.endswith(".png"):
                if self.cfg.dataset_type == 'ogyei':
                    # match = re.search(r'^id_\d{3}_([a-zA-Z0-9_]+)_\d{3}\.png$', file_rgb)
                    if "s_" in file_rgb:
                        match = re.search(r'^(.*?)_[s]_\d{3}\.png$', file_rgb)
                    elif "u_" in file_rgb:
                        match = re.search(r'^(.*?)_[u]_\d{3}\.png$', file_rgb)
                    else:
                        match = None

                    if match:
                        value = match.group(1)
                elif self.cfg.dataset_type == 'cure':
                    value = os.path.basename(file_rgb).split("_")[0]
                else:
                    raise ValueError("wrong dataset type has given!")

                out_path_rgb = os.path.join(rgb_path, value)
                out_path_contour = os.path.join(contour_path, value)
                out_path_texture = os.path.join(texture_path, value)
                out_path_lbp = os.path.join(lbp_path, value)

                os.makedirs(out_path_rgb, exist_ok=True)
                os.makedirs(out_path_contour, exist_ok=True)
                os.makedirs(out_path_texture, exist_ok=True)
                os.makedirs(out_path_lbp, exist_ok=True)

                try:
                    shutil.move(os.path.join(rgb_path, file_rgb), out_path_rgb)
                    shutil.move(os.path.join(contour_path, file_contour), out_path_contour)
                    shutil.move(os.path.join(texture_path, file_texture), out_path_texture)
                    shutil.move(os.path.join(lbp_path, file_lbp), out_path_lbp)
                except shutil.Error as se:
                    logging.error(f"Error moving file: {se.args[0]}")

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def main(self) -> None:
        """
        Executes the functions to create the images.

        :return: None
        """

        if self.cfg.dataset_type == "ogyei":
            self.save_rgb_images()
        self.save_contour_images()
        self.save_texture_images()
        self.save_lbp_images()

        self.create_label_dirs(
            rgb_path=self.rgb_images_path,
            contour_path=self.contour_images_path,
            texture_path=self.texture_images_path,
            lbp_path=self.lbp_images_path
        )

        if self.cfg.dataset_operation == "test":
            self.copy_query_images()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        proc_unet_imgs = CreateStreamImages()
        proc_unet_imgs.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
