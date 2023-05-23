"""
File: augment_data.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The augment_images function performs the image augmentation process (brightness, color distortion,
rotation, salt and pepper noise, Gaussian smoothing), and the do_augmentation function is called to initiate the
augmentation using predefined input and output directories.
"""

import cv2
import logging
import numpy as np
import os
import random

from glob import glob
from tqdm import tqdm
from typing import List, Tuple
from config.const import CONST
from config.logger_setup import setup_logger
from skimage.util import random_noise


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- C H A N G E   B R I G H T N E S S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def change_brightness(image: np.ndarray, exposure_factor: float) -> np.ndarray:
    """
    This function changes the brightness level of the image.
    :param image: Input image to modify.
    :param exposure_factor: Factor to adjust the image brightness. Values less than 1 decrease the brightness, while
    values greater than 1 increase it.
    :return: Modified image with adjusted brightness.
    """

    image = image.astype(np.float32) / 255.0
    adjusted_image = image * exposure_factor
    adjusted_image = np.clip(adjusted_image, 0, 1)
    adjusted_image = (adjusted_image * 255).astype(np.uint8)

    return adjusted_image


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- A D D   G A U S S I A N   N O I S E ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def add_gaussian_noise(image: np.ndarray, mean: int = 0, std_dev: int = 20) -> np.ndarray:
    """
    This function adds Gaussian noise to the image.
    :param image: Input image to add noise to.
    :param mean: Expected value.
    :param std_dev: Standard deviation.
    :return: Image with added Gaussian noise.
    """

    noisy_image = np.clip(image + np.random.normal(mean, std_dev, image.shape), 0, 255).astype(np.uint8)
    return noisy_image


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------ A D D   S A L T   &   P E P P E R   N O I S E -----------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def add_salt_and_pepper_noise(image: np.ndarray, salt_vs_pepper_ratio: float, amount: float) -> np.ndarray:
    """
    This function adds salt and pepper noise to the image.
    :param image: Input image to add noise to.
    :param salt_vs_pepper_ratio: Ratio of salt to pepper noise.
    :param amount: Amount of noise to be added.
    :return: Image with added salt and pepper noise.
    """

    noisy_image = random_noise(image, mode='s&p', salt_vs_pepper=salt_vs_pepper_ratio, amount=amount)
    noisy_image = (255 * noisy_image).astype(np.uint8)
    return noisy_image


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- R O T A T E   I M A G E ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def rotate_image(image: np.ndarray, angle: int = 180) -> np.ndarray:
    """
    This function rotates the image by the specified angle.
    :param image: Input image to rotate.
    :param angle: Angle of rotation in degrees. Default is 180 degrees.
    :return: Rotated image.
    """

    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- D I S T O R T   C O L O R ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def distort_color(image: np.ndarray, domain: Tuple[float, float]) -> np.ndarray:
    """
    This function distorts the color of the image by applying a random color shift within the specified domain.
    :param image: Input image to distort.
    :param domain: Color shift domain as a tuple (min_value, max_value) for each color channel.
    :return: Distorted image.
    """

    image = image.astype(np.float32) / 255.0

    color_shift = np.random.uniform(low=domain[0], high=domain[1], size=(1, 3))

    distorted_image = image * color_shift
    distorted_image = np.clip(distorted_image, 0, 1)
    distorted_image = (distorted_image * 255).astype(np.uint8)

    return distorted_image


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------- R O T A T E   S E G M E N T A T I O N   A N N O T A T I O N S ---------------------------
# ----------------------------------------------------------------------------------------------------------------------
def rotate_segmentation_annotations(annotations: List[str], image_width: int, image_height: int, angle: int) -> \
        List[List[float]]:
    """
    Rotates the segmentation annotations by the given angle.

    :param annotations: A list of strings representing the annotations.
    :param image_width: An integer indicating the width of the image.
    :param image_height: An integer indicating the height of the image.
    :param angle: An integer representing the angle of rotation.
    :return: A list of lists representing the rotated annotations.
    """

    rotated_annotations = []
    for annotation in annotations:
        annotation_values = annotation.split(' ')
        class_id = int(annotation_values[0])
        mask_coords = [float(coord) for coord in annotation_values[1:] if coord]
        mask_coords = np.array(mask_coords, dtype=float)
        mask_coords = mask_coords.reshape((-1, 2))
        mask_coords[:, 0] *= image_width
        mask_coords[:, 1] *= image_height

        # Rotate the coordinates
        rotation_matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), angle, 1)
        rotated_coords = np.matmul(np.hstack((mask_coords, np.ones((mask_coords.shape[0], 1)))), rotation_matrix.T)
        rotated_coords = rotated_coords[:, :2]

        # Convert the rotated coordinates back to YOLO format
        rotated_coords[:, 0] /= image_width
        rotated_coords[:, 1] /= image_height
        rotated_coords = rotated_coords.flatten().tolist()

        rotated_annotations.append([class_id] + rotated_coords)

    return rotated_annotations


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- S A V E   F I L E S ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def save_files(image_path: str, output_img_dir: str, output_txt_dir: str, annotations: List[str],
               image_to_save: np.ndarray, type_of_image: str) -> None:
    """
    Saves the image and annotations to the specified directories.

    :param image_path: The path of the original image file.
    :param output_img_dir: The directory where the output image will be saved.
    :param output_txt_dir: The directory where the output annotations will be saved.
    :param annotations: A list of annotation strings.
    :param image_to_save: The image to be saved.
    :param type_of_image: A string indicating the type of the image.
    :return: None
    """

    name = os.path.basename(image_path).split(".")[0]
    img_file_name = name + "_%s.png" % type_of_image
    txt_file_name = name + "_%s.txt" % type_of_image
    output_image_path = os.path.join(output_img_dir, img_file_name)
    cv2.imwrite(output_image_path, image_to_save)
    output_annotations_path = os.path.join(output_txt_dir, txt_file_name)
    with open(output_annotations_path, 'w') as f:
        f.writelines(annotations)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- S A V E   R O T A T E D   I M A G E S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def save_rotated_images(image: np.ndarray, image_path: str, output_img_dir: str, output_txt_dir: str,
                        annotations: List[str], rotated_image: np.ndarray) -> None:
    """
    Saves the rotated image and annotations to the specified directories.

    :param image: The original image.
    :param image_path: The path of the original image file.
    :param output_img_dir: The directory where the output image will be saved.
    :param output_txt_dir: The directory where the output annotations will be saved.
    :param annotations: A list of annotation strings.
    :param rotated_image: The rotated image to be saved.
    :return: None
    """

    image_width = image.shape[1]
    image_height = image.shape[0]
    name = os.path.basename(image_path).split(".")[0]
    img_file_name = name + "_rotated.png"
    txt_file_name = name + "_rotated.txt"
    output_image_path = os.path.join(output_img_dir, img_file_name)
    rotated_annotations = rotate_segmentation_annotations(annotations, image_width, image_height, 180)
    cv2.imwrite(output_image_path, rotated_image)
    output_annotations_path = os.path.join(output_txt_dir, txt_file_name)

    with open(output_annotations_path, 'w') as f:
        for annotation in rotated_annotations:
            f.write(' '.join([str(coord) for coord in annotation]) + '\n')


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- A U G M E N T   I M A G E S ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def augment_images(input_img_dir: str, input_txt_dir: str, output_img_dir: str, output_txt_dir: str) -> None:
    """
    Augments the images and saves the augmented versions to the specified directories.

    :param input_img_dir: The directory containing the input images.
    :param input_txt_dir: The directory containing the input annotations.
    :param output_img_dir: The directory where the augmented images will be saved.
    :param output_txt_dir: The directory where the augmented annotations will be saved.
    :return: None
    """

    salt_vs_pepper_ratio = 0.5
    noise_amount = 0.05
    distort_color_domain = (0.7, 1.2)
    brightness_factor = random.uniform(0.5, 1.5)
    kernel = (7, 7)

    image_paths = glob(os.path.join(input_img_dir, '*.png'))
    txt_paths = glob(os.path.join(input_txt_dir, "*.txt"))

    for idx, (image_path, annotations_path) in tqdm(enumerate(zip(image_paths, txt_paths)), total=len(image_paths),
                                                    desc="Image augmentation"):
        image = cv2.imread(image_path)

        with open(annotations_path, 'r') as f:
            annotations = f.readlines()

        rotated_image = rotate_image(image)
        save_rotated_images(image, image_path, output_img_dir, output_txt_dir, annotations, rotated_image)

        brightness_changed_images = change_brightness(image, brightness_factor)
        save_files(image_path, output_img_dir, output_txt_dir, annotations, brightness_changed_images, "brightness")

        smoothed_image = cv2.GaussianBlur(image, kernel, 0)
        save_files(image_path, output_img_dir, output_txt_dir, annotations, smoothed_image, "smoothed")

        noisy_image = add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, noise_amount)
        save_files(image_path, output_img_dir, output_txt_dir, annotations, noisy_image, "noisy")

        color_distorted_image = distort_color(image, distort_color_domain)
        save_files(image_path, output_img_dir, output_txt_dir, annotations, color_distorted_image, "color_distorted")


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    """
    Executes the functions to complete image augmentation.
    :return: None
    """

    setup_logger()
    augment_images(input_img_dir="C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyi/full_img_size/splitted/"
                                 "train/images",
                   input_txt_dir="C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyi/full_img_size/splitted/"
                                 "train/labels",
                   output_img_dir=CONST.dir_aug_img,
                   output_txt_dir=CONST.dir_aug_labels)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        logging.error(kie)
