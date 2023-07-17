import cv2
import logging
import numpy as np
import re
import os
import shutil

from concurrent.futures import ThreadPoolExecutor, wait
from glob import glob
from tqdm import tqdm
from typing import Tuple

from config.const import DATASET_PATH, IMAGES_PATH
from config.config import ConfigGeneral
from config.logger_setup import setup_logger
from utils.utils import measure_execution_time, numerical_sort


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++ C R E A T E   S T R E A M   I M A G E S +++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CreateStreamImages:
    def __init__(self, op: str = "train"):
        setup_logger()
        logging.info("Selected operation: %s" % op)
        self.cfg = ConfigGeneral().parse()
        self.path_to_images = self.path_selector(op)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- P A T H   S E L E C T O R -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def path_selector(op: str):
        """
        Selects the correct directory paths based on the given operation string.

        :param op: A string indicating the operation mode (train or test).
        :return: A dictionary containing directory paths for images, masks, and other related files.
        :raises ValueError: If the operation string is not "train" or "test".
        """

        if op.lower() == "train":
            path_to_images = {
                "images": DATASET_PATH.get_data_path("ogyi_v2_splitted_train_images"),
                "masks": DATASET_PATH.get_data_path("ogyi_v2_splitted_gt_train_masks"),
                "contour": IMAGES_PATH.get_data_path("ref_train_contour"),
                "lbp": IMAGES_PATH.get_data_path("ref_train_lbp"),
                "rgb": IMAGES_PATH.get_data_path("ref_train_rgb"),
                "texture": IMAGES_PATH.get_data_path("ref_train_texture"),
            }
        elif op.lower() == "valid":
            path_to_images = {
                "images": DATASET_PATH.get_data_path("ogyi_v2_splitted_valid_images"),
                "masks": DATASET_PATH.get_data_path("ogyi_v2_splitted_gt_valid_masks"),
                "contour": IMAGES_PATH.get_data_path("ref_valid_contour"),
                "lbp": IMAGES_PATH.get_data_path("ref_valid_lbp"),
                "rgb": IMAGES_PATH.get_data_path("ref_valid_rgb"),
                "texture": IMAGES_PATH.get_data_path("ref_valid_texture"),
            }
        elif op.lower() == "test":
            path_to_images = {
                "images": DATASET_PATH.get_data_path("ogyi_v2_splitted_test_images"),
                "masks": IMAGES_PATH.get_data_path("unet_out"),
                "contour": IMAGES_PATH.get_data_path("query_contour"),
                "lbp": IMAGES_PATH.get_data_path("query_lbp"),
                "rgb": IMAGES_PATH.get_data_path("query_rgb"),
                "texture": IMAGES_PATH.get_data_path("query_texture"),
            }
        else:
            raise ValueError("Wrong operation!")

        return path_to_images

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ C R E A T E   L A B E L   D I R S -------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_label_dirs(rgb_path: str, contour_path: str, texture_path: str, lbp_path: str) -> None:
        """
        Function create labels. Goes through a directory, yield the name of the medicine(s) from the file name, and
        create a corresponding directory, if that certain directory does not exist. Finally, it copies every image with
        the same label to the corresponding directory.

        :param rgb_path: string, path to the directory.
        :param texture_path:
        :param contour_path:
        :param lbp_path:
        :return: None
        """

        files_rgb = os.listdir(rgb_path)
        files_contour = os.listdir(contour_path)
        files_texture = os.listdir(texture_path)
        files_lbp = os.listdir(lbp_path)

        for idx, (file_rgb, file_contour, file_texture, file_lbp) in \
                tqdm(enumerate(zip(files_rgb, files_contour, files_texture, files_lbp)), desc="Copying image files"):
            if file_rgb.endswith(".png"):
                match = re.search(r'^id_\d{3}_([a-zA-Z0-9_]+)_\d{3}\.png$', file_rgb)
                if match:
                    value = match.group(1)
                    # value = os.path.basename(file_rgb).split("_")[0]
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
        output_file = os.path.join(self.path_to_images.get("rgb"), output_name)
        c_imgs = cv2.imread(color_path, 1)
        m_imgs = cv2.imread(mask_path, 0)
        self.draw_bounding_box(c_imgs, m_imgs, output_file)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------- S A V E   B O U N D I N G   B O X   I M G S ----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_rgb_images(self) -> None:
        """
        Reads in the images, draws the bounding box, and saves the images in parallel.
        :return: None
        """

        color_images = sorted(glob(self.path_to_images.get("images") + "/*.png"))
        mask_images = sorted(glob(self.path_to_images.get("masks") + "/*.png"))

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

        contour_images = sorted(glob(self.path_to_images.get("rgb") + "/*.png"), key=numerical_sort)
        args_list = []

        for img_path in tqdm(contour_images, desc="Contour images"):
            output_name = "contour_" + os.path.basename(img_path)
            output_file = (os.path.join(self.path_to_images.get("contour"), output_name))
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

        bbox_images = sorted(glob(self.path_to_images.get("rgb") + "/*.png"), key=numerical_sort)
        args_list = []

        for img_path in tqdm(bbox_images, desc="Texture images"):
            output_name = "texture_" + os.path.basename(img_path)
            output_file = (os.path.join(self.path_to_images.get("texture"), output_name))
            bbox_imgs = cv2.imread(img_path, 0)
            args_list.append((bbox_imgs, output_file,
                              (self.cfg.kernel_gaussian_texture, self.cfg.kernel_gaussian_texture)))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.create_texture_images, args) for args in args_list]
            wait(futures)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- G E T   P I X E L ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_pixel(img: np.ndarray, center: int, x: int, y: int) -> int:
        """
        Get the pixel value based on the given image and coordinates.

        :param img: The image as a numpy array.
        :param center: The center value for comparison.
        :param x: The x-coordinate of the pixel.
        :param y: The y-coordinate of the pixel.
        :return: The new pixel value.
        """

        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except IndexError:
            pass
        return new_value

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- P R O C E S S   L B P   I M A G E S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def process_lbp_image(self, img_gray: np.ndarray, dest_image_path: str) -> None:
        """
        Process the LBP image for a given image file and save it to the destination directory.

        :param img_gray: The BGR image as a numpy array.
        :param dest_image_path: The destination path to save the LBP image.
        :return: None.
        """

        height, width = img_gray.shape
        img_lbp = np.zeros((height, width), np.uint8)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center = img_gray[i, j]
                val_ar = [self.get_pixel(img_gray, center, i - 1, j - 1), self.get_pixel(img_gray, center, i - 1, j),
                          self.get_pixel(img_gray, center, i - 1, j + 1), self.get_pixel(img_gray, center, i, j + 1),
                          self.get_pixel(img_gray, center, i + 1, j + 1), self.get_pixel(img_gray, center, i + 1, j),
                          self.get_pixel(img_gray, center, i + 1, j - 1), self.get_pixel(img_gray, center, i, j - 1)]
                power_val = [1, 2, 4, 8, 16, 32, 64, 128]
                val = 0
                for k in range(len(val_ar)):
                    val += val_ar[k] * power_val[k]
                img_lbp[i, j] = val

        cv2.imwrite(dest_image_path, img_lbp)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- S A V E   L B P   I M A G E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_lbp_images(self) -> None:
        """
        Saves LBP images in parallel.
        :return: None
        """

        bbox_images = sorted(glob(self.path_to_images.get("rgb") + "/*.png"), key=numerical_sort)

        with ThreadPoolExecutor() as executor:
            futures = []
            for img_path in tqdm(bbox_images, desc="LBP images"):
                output_name = "lbp_" + os.path.basename(img_path)
                output_file = os.path.join(self.path_to_images.get("lbp"), output_name)
                bbox_imgs = cv2.imread(img_path, 0)
                bbox_imgs = cv2.resize(bbox_imgs, (224, 224))
                future = executor.submit(self.process_lbp_image, bbox_imgs, output_file)
                futures.append(future)

            for future in futures:
                future.result()

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def main(self) -> None:
        """
        Executes the functions to create the images.

        :return: None
        """

        self.save_rgb_images()
        self.save_contour_images()
        self.save_texture_images()
        self.save_lbp_images()
        self.create_label_dirs(rgb_path=self.path_to_images.get("rgb"),
                               contour_path=self.path_to_images.get("contour"),
                               texture_path=self.path_to_images.get("texture"),
                               lbp_path=self.path_to_images.get("lbp"))


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        operation = "test"
        proc_unet_imgs = CreateStreamImages(op=operation)
        proc_unet_imgs.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
