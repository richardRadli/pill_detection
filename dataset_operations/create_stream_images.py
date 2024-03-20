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
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from typing import Tuple

from config.config import ConfigStreamImages
from config.config_selector import dataset_images_path_selector
from utils.utils import file_reader, measure_execution_time, setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++ C R E A T E   S T R E A M   I M A G E S +++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CreateStreamImages:
    def __init__(self) -> None:
        """
        Initialize the CreateStreamImages class, and paths.
        """

        setup_logger()

        self.cfg = ConfigStreamImages().parse()

        # Output paths
        self.contour_images_path = (
        dataset_images_path_selector(
            self.cfg.dataset_type).get("src_stream_images").get(self.cfg.operation).get("stream_images_contour")
        )
        self.lbp_images_path = (
            dataset_images_path_selector(
                self.cfg.dataset_type).get("src_stream_images").get(self.cfg.operation).get("stream_images_lbp")
        )
        self.rgb_images_path = (
            dataset_images_path_selector(
                self.cfg.dataset_type).get("src_stream_images").get(self.cfg.operation).get("stream_images_rgb")
        )
        self.texture_images_path = (
        dataset_images_path_selector(
            self.cfg.dataset_type).get("src_stream_images").get(self.cfg.operation).get("stream_images_texture")
        )

        df = pd.DataFrame.from_dict(vars(self.cfg), orient='index', columns=['value'])
        logging.info(df)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- D R A W   B O U N D I N G   B O X ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def draw_bounding_box(self, in_img: np.ndarray, seg_map: np.ndarray, output_path: str) -> None:
        """
         Draws bounding box over medicines. It draws only the biggest bounding box, small ones are terminated.
         After that it crops out the bounding box's content.

         Args:
             in_img: Input testing image.
             seg_map: Output of the UNet for the input testing image.
             output_path: Where the file should be saved.

         Returns:
             None
         """

        n_objects, _, stats, _ = cv2.connectedComponentsWithStats(seg_map, connectivity=8, ltype=cv2.CV_32S)

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

            # Calculate square coordinates ensuring it fits within image boundaries
            square_x = max(0, int(center_x - side_length / 2))
            square_y = max(0, int(center_y - side_length / 2))
            square_x_end = min(in_img.shape[1], square_x + side_length)
            square_y_end = min(in_img.shape[0], square_y + side_length)

            obj = in_img[square_y:square_y_end, square_x:square_x_end]

            if obj.size != 0:
                cv2.imwrite(output_path, obj)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- P R O C E S S   I M A G E -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def process_image(self, image_paths: Tuple[str, str]) -> None:
        """
        Processes a single image by reading in the color and mask images, drawing the bounding box, and saving the
        result.

        Args:
            image_paths: Tuple of paths to the color and mask images.

        Returns:
            None
        """

        color_path, mask_path = image_paths
        output_name = os.path.basename(color_path)
        output_file = (os.path.join(self.rgb_images_path, output_name))
        color_images = cv2.imread(str(color_path), 1)
        mask_images = cv2.imread(str(mask_path), 0)
        self.draw_bounding_box(color_images, mask_images, output_file)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ S A V E   R G B   I M A G E S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_rgb_images(self) -> None:
        """
        Reads in the images, draws the bounding box, and saves the images in parallel.

        Returns:
            None
        """

        images = (
            "customer_images" if self.cfg.operation == "customer"
            else ("reference_images" if self.cfg.operation == "reference"
                  else None)
        )

        if images is None:
            raise ValueError("Invalid value for self.cfg.operation")

        masks = (
            "customer_mask_images" if self.cfg.operation == "customer"
            else ("reference_mask_images" if self.cfg.operation == "reference"
                  else None)
        )

        color_images_dir = dataset_images_path_selector(self.cfg.dataset_type).get(self.cfg.operation).get(images)
        mask_images_dir = dataset_images_path_selector(self.cfg.dataset_type).get(self.cfg.operation).get(masks)

        color_images = file_reader(color_images_dir, "jpg")
        mask_images = file_reader(mask_images_dir, "jpg")

        with ThreadPoolExecutor(max_workers=self.cfg.max_worker) as executor:
            list(tqdm(executor.map(self.process_image, zip(color_images, mask_images)), total=len(color_images),
                      desc="RGB images"))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ C R E A T E   C O N T O U R   I M A G E S -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_contour_images(args) -> None:
        """
        Applies edge detection on an input image and saves the resulting edge map.

        Args:
            args (Tuple[numpy.ndarray, str, int, int, int]): A tuple containing:
                - cropped_image (numpy.ndarray): The input image to apply edge detection on.
                - output_path (str): The path to save the output edge map.
                - kernel_size (int): The size of the kernel used in Gaussian blur.
                - canny_low_thr (int): The lower threshold for Canny edge detection.
                - canny_high_thr (int): The higher threshold for Canny edge detection.

        Returns:
            None
        """

        cropped_image, output_path, kernel_size, canny_low_thr, canny_high_thr = args
        blured_images = cv2.GaussianBlur(cropped_image, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(blured_images, canny_low_thr, canny_high_thr)
        cv2.imwrite(output_path, edges)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- S A V E   C O N T O U R   I M G S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_contour_images(self) -> None:
        """
        Loads RGB images from the designated directory and applies Canny edge detection algorithm to extract contours
        for each image. The resulting images are saved in a new directory designated for contour images.

        Returns:
             None
        """

        rgb_images = file_reader(self.rgb_images_path, "jpg")
        args_list = []

        for img_path in tqdm(rgb_images, desc="Contour images"):
            output_name = "contour_" + os.path.basename(img_path)
            output_file = os.path.join(self.contour_images_path, output_name)
            bbox_images = cv2.imread(img_path, 0)
            args_list.append((bbox_images, output_file, self.cfg.kernel_median_contour, self.cfg.canny_low_thr,
                              self.cfg.canny_high_thr))

        with ThreadPoolExecutor(max_workers=self.cfg.max_worker) as executor:
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

        Args:
            args: A tuple containing the following elements:
                - cropped_image: A numpy array representing the cropped input image.
                - output_path: A string representing the path to save the output texture image.
                - kernel_size: A tuple of integers representing the size of the Gaussian blur kernel.

        Returns:
            None
        """

        cropped_image, output_path, kernel_size = args
        blured_image = cv2.GaussianBlur(cropped_image, kernel_size, 0)
        blured_image = cv2.GaussianBlur(blured_image, (15, 15), 0)
        sub_img = cv2.subtract(cropped_image, blured_image)
        cv2.imwrite(output_path, sub_img * 15)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- S A V E   T E X T U R E   I M G S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_texture_images(self) -> None:
        """
        Save texture images using Gaussian Blur filter.

        Returns:
             None
        """

        rgb_images = file_reader(self.rgb_images_path, "jpg")
        args_list = []

        for img_path in tqdm(rgb_images, desc="Texture images"):
            output_name = "texture_" + os.path.basename(img_path)
            output_file = os.path.join(self.texture_images_path, output_name)
            bbox_images = cv2.imread(img_path, 0)
            args_list.append((bbox_images, output_file,
                              (self.cfg.kernel_gaussian_texture, self.cfg.kernel_gaussian_texture)))

        with ThreadPoolExecutor(max_workers=self.cfg.max_worker) as executor:
            futures = [executor.submit(self.create_texture_images, args) for args in args_list]
            wait(futures)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- P R O C E S S   L B P   I M A G E S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def process_lbp_image(img_gray: np.ndarray, dst_image_path: str) -> None:
        """
        Process the LBP image for a given image file and save it to the destination directory.

        Args:
            img_gray: The BGR image as a numpy array.
            dst_image_path: The destination path to save the LBP image.

        Returns:
             None.
        """

        lbp_image = local_binary_pattern(image=img_gray, P=8, R=2, method="default")
        lbp_image = np.clip(lbp_image, 0, 255)
        cv2.imwrite(dst_image_path, lbp_image)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- S A V E   L B P   I M A G E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_lbp_images(self) -> None:
        """
        Saves LBP images in parallel.

        Returns:
             None.
        """

        rgb_images = file_reader(self.rgb_images_path, "jpg")

        with ThreadPoolExecutor(max_workers=self.cfg.max_worker) as executor:
            futures = []
            for img_path in tqdm(rgb_images, desc="LBP images"):
                output_name = "lbp_" + os.path.basename(img_path)
                output_file = os.path.join(self.lbp_images_path, output_name)
                bbox_images = cv2.imread(img_path, 0)
                future = executor.submit(self.process_lbp_image, bbox_images, str(output_file))
                futures.append(future)

            for future in futures:
                future.result()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ C R E A T E   L A B E L   D I R S -------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_label_dirs(self, rgb_path: str, contour_path: str, texture_path: str, lbp_path: str) -> None:
        """
        Create labeled directories for image files based on the given dataset.

        Args:
            rgb_path: str, path to the directory containing RGB images.
            contour_path: str, path to the directory containing contour images.
            texture_path: str, path to the directory containing texture images.
            lbp_path: str, path to the directory containing LBP images.

        Returns:
             None.
        """

        files_rgb = os.listdir(rgb_path)
        files_contour = os.listdir(contour_path)
        files_texture = os.listdir(texture_path)
        files_lbp = os.listdir(lbp_path)
        value = None

        for idx, (file_rgb, file_contour, file_texture, file_lbp) in \
                tqdm(enumerate(zip(files_rgb, files_contour, files_texture, files_lbp)), desc="Copying image files"):

            if self.cfg.dataset_type == 'ogyei':
                if "s_" in file_rgb:
                    match = re.search(r'^(.*?)_s_\d{3}\.jpg$', file_rgb)
                elif "u_" in file_rgb:
                    match = re.search(r'^(.*?)_u_\d{3}\.jpg$', file_rgb)
                else:
                    match = None

                if match:
                    value = match.group(1)
                else:
                    raise ValueError(f"Wrong file name: {file_rgb}")

            elif self.cfg.dataset_type == 'cure_two_sided':
                value = os.path.basename(file_rgb).split("_")[0]

            elif self.cfg.dataset_type == 'cure_one_sided':
                value = os.path.basename(file_rgb).split("_")[0] + "_" + os.path.basename(file_rgb).split("_")[1]
            else:
                raise ValueError("Wrong dataset type has given!")

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

        Returns:
             None
        """

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


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        proc_unet_images = CreateStreamImages()
        proc_unet_images.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
