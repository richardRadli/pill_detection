"""
File: augmentation_utils.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 05, 2023

Description: The program stores the functions for the augmentation process.
"""

import cv2
import gc
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil

from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
from typing import Tuple, List, Optional


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- R E N A M E   F I L E --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def rename_file(src_path: str, dst_path: str, op: str) -> str:
    """
    Rename the file by appending the operation name and a counter to the filename.

    Args:
        src_path (str): The path of the original file.
        dst_path (str): The path of the destination file.
        op (str): The operation name to be appended.

    Returns:
        str: The new file path with the renamed filename.
    """

    # Split the original file path into directory and filename
    directory = os.path.dirname(src_path)
    filename = os.path.basename(src_path)

    # Split the filename into name and extension
    name, extension = os.path.splitext(filename)

    # Construct the new file path with the desired filename
    new_filename = f"{name}_{op}"
    counter = 1
    final_file_name = os.path.join(dst_path, f"{new_filename}_{counter}{extension}")

    while os.path.isfile(final_file_name):
        counter += 1
        final_file_name = os.path.join(directory, f"{new_filename}_{counter}{extension}")

    return final_file_name


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- U N I Q U E   C O U N T   A P P ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def unique_count_app(img: np.ndarray) -> tuple:
    """
    Get the most common color in the image based on unique color counts.

    Args:
        img (np.ndarray): The input image.

    Returns:
        tuple: The most common color in the image as a tuple of (R, G, B) values.
    """

    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    colors, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    return tuple(colors[count.argmax()])


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- D I S T O R T   C O L O R ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def change_white_balance(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                         aug_annotation_path: str, aug_mask_path: str,
                         domain: Tuple[float, float] = (0.7, 1.2)) -> None:
    """
    Apply white balance distortion to an image and save the distorted image.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file
        mask_path (str): Path to the mask.
        aug_img_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation file.
        aug_mask_path (str): Path to the augmented mask image.
        domain (Tuple[float, float], optional): Range of scaling factors for white balance distortion.
            Defaults to (0.7, 1.2).
    """

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    filename = "distorted_colour"

    # Generate random scaling factors for each color channel
    scale_factors = np.random.uniform(low=domain[0], high=domain[1], size=(3,))

    # Apply the scaling factors to the image
    adjusted_image = image * scale_factors

    # Clip the pixel values to the valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255)
    adjusted_image = adjusted_image.astype(np.uint8)

    # Save new image
    new_image_file_name = rename_file(src_path=image_path, dst_path=aug_img_path, op=filename)
    cv2.imwrite(new_image_file_name, adjusted_image)

    # Save mask
    new_mask_file_name = rename_file(src_path=mask_path, dst_path=aug_mask_path, op=filename)
    cv2.imwrite(new_mask_file_name, mask)

    # Save annotation
    new_annotation_file_name = rename_file(src_path=annotation_path, dst_path=aug_annotation_path, op=filename)
    shutil.copy(annotation_path, new_annotation_file_name)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- G A U S S I A N   S M O O T H --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def gaussian_smooth(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                    aug_annotation_path: str, aug_mask_path: str, kernel: tuple) -> None:
    """
    Apply Gaussian smoothing to an image and save the smoothed image.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file.
        mask_path (str): Path to save the mask image.
        aug_img_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation file.
        aug_mask_path (str): Path to the augmented mask image.
        kernel (tuple): Kernel size for Gaussian smoothing in the form of (width, height).
    """

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    filename = "gaussian_%s" % str(kernel[0])

    smoothed_image = cv2.GaussianBlur(image, kernel, 0)

    new_image_file_name = rename_file(src_path=image_path, dst_path=aug_img_path, op=filename)
    cv2.imwrite(new_image_file_name, smoothed_image)

    new_mask_file_name = rename_file(src_path=mask_path, dst_path=aug_mask_path, op=filename)
    cv2.imwrite(new_mask_file_name, mask)

    new_annotation_file_name = rename_file(src_path=annotation_path, dst_path=aug_annotation_path, op=filename)
    shutil.copy(annotation_path, new_annotation_file_name)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- C H A N G E   B R I G H T N E S S ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def change_brightness(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                      aug_annotation_path: str, aug_mask_path: str, exposure_factor: float) -> None:
    """
    Adjust the brightness of an image and save the adjusted image.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file.
        mask_path (str): Path to save the mask image.
        aug_img_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation file.
        aug_mask_path (str): Path to the augmented mask image.
        exposure_factor (float): Factor to adjust the brightness.
                                 Values > 1 increase brightness, values < 1 decrease brightness.
    """

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    filename = "brightness"

    image = image.astype(np.float32) / 255.0
    adjusted_image = image * exposure_factor
    adjusted_image = np.clip(adjusted_image, 0, 1)
    adjusted_image = (adjusted_image * 255).astype(np.uint8)

    new_image_file_name = rename_file(src_path=image_path, dst_path=aug_img_path, op=filename)
    cv2.imwrite(new_image_file_name, adjusted_image)

    new_mask_file_name = rename_file(src_path=mask_path, dst_path=aug_mask_path, op=filename)
    cv2.imwrite(new_mask_file_name, mask)

    new_annotation_file_name = rename_file(src_path=annotation_path, dst_path=aug_annotation_path, op=filename)
    shutil.copy(annotation_path, new_annotation_file_name)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- R O T A T E   I M A G E -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def rotate_image(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                 aug_annotation_path: str, aug_mask_path: str, angle: int) -> None:
    """
    Rotate an image and save the rotated image.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file.
        mask_path (str): Path to save the rotated image.
        aug_img_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation file.
        aug_mask_path (str): Path to the augmented mask image.
        angle (int): Angle of rotation in degrees.
    """

    original_image = Image.open(image_path)
    mask = Image.open(mask_path)
    filename = f"rotated_{angle}"

    rotated_image = original_image.rotate(angle, expand=True)
    rotated_mask = mask.rotate(angle, expand=True)

    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    rotated_annotations = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.strip().split())

        # Convert angles to radians
        angle_rad = math.radians(angle)

        # Rotate the center coordinates
        x_center_rotated = (x_center * original_image.width - original_image.width / 2) * math.cos(angle_rad) - \
                           (y_center * original_image.height - original_image.height / 2) * math.sin(angle_rad) + \
                           rotated_image.width / 2

        y_center_rotated = (x_center * original_image.width - original_image.width / 2) * math.sin(angle_rad) + \
                           (y_center * original_image.height - original_image.height / 2) * math.cos(angle_rad) + \
                           rotated_image.height / 2

        width_rotated = width * original_image.width
        height_rotated = height * original_image.height

        rotated_box = (
            x_center_rotated,
            y_center_rotated,
            x_center_rotated + width_rotated,
            y_center_rotated + height_rotated
        )

        rotated_annotation = f"{int(class_id)} {rotated_box[0] / rotated_image.width:.6f} " \
                             f"{rotated_box[1] / rotated_image.height:.6f} " \
                             f"{(rotated_box[2] - rotated_box[0]) / rotated_image.width:.6f} " \
                             f"{(rotated_box[3] - rotated_box[1]) / rotated_image.height:.6f}"

        rotated_annotations.append(rotated_annotation)

    rotated_image_path = rename_file(src_path=image_path, dst_path=aug_img_path, op=filename)
    rotated_image.save(rotated_image_path)

    rotated_mask_path = rename_file(src_path=mask_path, dst_path=aug_mask_path, op=filename)
    rotated_mask.save(rotated_mask_path)

    new_annotation_file_name = rename_file(src_path=annotation_path, dst_path=aug_annotation_path, op=filename)
    with open(new_annotation_file_name, 'w') as f:
        f.write('\n'.join(rotated_annotations))


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- S H I F T   I M A G E ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def shift_image(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                aug_annotation_path: str, aug_mask_path: str, shift_x: int = 50, shift_y: int = 100) -> None:
    """
    Shift an image and save the shifted image.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file.
        mask_path (str): Path to save the mask image.
        aug_img_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation file.
        aug_mask_path (str): Path to the augmented mask image.
        shift_x (int, optional): Amount of horizontal shift. Defaults to 50.
        shift_y (int, optional): Amount of vertical shift. Defaults to 100.
    """

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    filename = f"shifting_{shift_x}_{shift_y}"

    # Shift image in x and y directions
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (cols, rows))
    shifted_mask = cv2.warpAffine(mask, M, (cols, rows))

    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    shifted_annotations = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.strip().split())
        shifted_annotation = (int(class_id), x_center + shift_x / cols, y_center + shift_y / rows, width, height)
        shifted_annotations.append(shifted_annotation)

    shifted_annotations = [f"{anno[0]} {anno[1]} {anno[2]} {anno[3]} {anno[4]}" for anno in shifted_annotations]

    # Save shifted image
    new_image_file_name = rename_file(src_path=image_path, dst_path=aug_img_path, op=filename)
    cv2.imwrite(new_image_file_name, shifted_image)

    new_mask_file_name = rename_file(src_path=mask_path, dst_path=aug_mask_path, op=filename)
    cv2.imwrite(new_mask_file_name, shifted_mask)

    new_annotation_file_name = rename_file(src_path=annotation_path, dst_path=aug_annotation_path, op=filename)
    with open(new_annotation_file_name, 'w') as f:
        f.write('\n'.join(shifted_annotations))


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Z O O M   I N   O B J E C T --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def zoom_in_object(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                   aug_annotation_path: str, aug_mask_path: str, crop_size: int) -> None:
    """
    Zoom in on an object in an image and save the zoomed image.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file.
        mask_path (str): Path to the mask image.
        aug_img_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation file.
        aug_mask_path (str): Path to the augmented mask image.
        crop_size (int): Size of the crop to zoom in on.
    """

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    if height > 801 and width > 801:
        filename = "zoomed"
        mask = cv2.imread(mask_path)

        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        cropped_image = image[start_y:end_y, start_x:end_x]
        zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

        cropped_mask = mask[start_y:end_y, start_x:end_x]
        zoomed_mask = cv2.resize(cropped_mask, (width, height), interpolation=cv2.INTER_LINEAR)

        with open(annotation_path, "r") as file:
            annotations = file.readlines()

        adjusted_annotations = []
        for annotation in annotations:
            class_id, x_center, y_center, obj_width, obj_height = map(float, annotation.strip().split())

            x_center = ((x_center * width) - start_x) / crop_size
            y_center = ((y_center * height) - start_y) / crop_size

            obj_width /= (crop_size / width)
            obj_height /= (crop_size / height)

            adjusted_annotations.append(f"{int(class_id)} {x_center} {y_center} {obj_width} {obj_height}")

        new_image_file_name = rename_file(src_path=image_path, dst_path=aug_img_path, op=filename)
        cv2.imwrite(new_image_file_name, zoomed_image)

        new_mask_file_name = rename_file(src_path=image_path, dst_path=aug_mask_path, op=filename)
        cv2.imwrite(new_mask_file_name, zoomed_mask)

        new_annotation_file_name = rename_file(src_path=annotation_path, dst_path=aug_annotation_path, op=filename)
        with open(new_annotation_file_name, 'w') as f:
            f.write('\n'.join(adjusted_annotations))


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Z O O M   I N   O B J E C T --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def copy_original_images(src_path: str, dst_path: str) -> None:
    """
    Copies files from one directory to another

    Args:
        src_path (str): Source path.
        dst_path (str): Destination path.
    """

    shutil.copy(src_path, dst_path)


# ------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- C H A N G E   B A C K G R O U N D ---------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def change_background_dtd(image_path: str, mask_path: str, annotation_path: str, aug_image_path: str,
                          aug_annotation_path: str, backgrounds_path: str) -> None:
    """
    Change the background of an image using images from a specified directory.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        annotation_path (str): Path to the annotation file.
        aug_image_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation files.
        backgrounds_path (str): Path to the directory containing background images.

    Returns:
        None
    """

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

    background_files = os.listdir(backgrounds_path)
    background_file = random.choice(background_files)
    background_image_path = os.path.join(backgrounds_path, background_file)
    background = cv2.imread(background_image_path)

    try:
        if background.size != 0:
            background = cv2.resize(background, (image.shape[1], image.shape[0]))

            foreground = cv2.bitwise_and(image, image, mask=mask)
            background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

            output_image = cv2.add(foreground, background)

            new_image_file_name = rename_file(src_path=image_path, dst_path=aug_image_path, op="changed_background")
            cv2.imwrite(new_image_file_name, output_image)

            new_annotation_file_name = rename_file(src_path=annotation_path, dst_path=aug_annotation_path,
                                                   op="changed_background")
            shutil.copy(annotation_path, new_annotation_file_name)
        else:
            logging.info(f"The background image {os.path.basename(background_image_path)} is empty.")
    except AttributeError:
        logging.info(f'Image {os.path.basename(background_image_path)} is wrong!')


# ------------------------------------------------------------------------------------------------------------------
# ----------------------------------- P L A C E   M E D I C I N E   O N   T R A Y ----------------------------------
# ------------------------------------------------------------------------------------------------------------------
def place_medicine_on_tray(pill_image_path: str, pill_mask_path: str, tray_image_path: str, save_path: str,
                           scaling_factor: float) -> None:
    """
    This function places the randomly selected medicine pill to the augmented trey image.

    Args:
        pill_image_path (str): Path to the randomly selected pill image.
        pill_mask_path (str): Path to the corresponding pill image mask.
        tray_image_path (str): Path to the tray image.
        save_path (str): Path to where the images will be saved.
        scaling_factor (float): Amount of scaling of the pill image.
    """

    pill_image = cv2.imread(pill_image_path)
    pill_mask = cv2.imread(pill_mask_path, cv2.IMREAD_GRAYSCALE)
    tray_image = cv2.imread(tray_image_path)

    unique_values = np.unique(pill_mask)
    if any((value != 0 and value != 255) for value in unique_values):
        _, binary_mask = cv2.threshold(pill_mask, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    pill_roi = pill_image[y:y + h, x:x + w]

    tray_height, tray_width = tray_image.shape[:2]
    resized_width = int(w * scaling_factor)
    resized_height = int(h * scaling_factor)
    resized_pill_roi = resize(pill_roi, (resized_height, resized_width), preserve_range=True).astype(np.uint8)

    x_offset = random.randint(0, tray_width - resized_width)
    y_offset = random.randint(0, tray_height - resized_height)
    pill_mask_roi = pill_mask[y:y + h, x:x + w]

    # Resize the pill mask to match the resized pill image
    resized_pill_mask_roi = resize(pill_mask_roi, (resized_height, resized_width), preserve_range=True).astype(
        np.uint8)

    # Convert pill_mask_roi to 3-channel mask
    pill_mask_roi_3ch = cv2.cvtColor(resized_pill_mask_roi, cv2.COLOR_GRAY2BGR)

    tray_image[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = np.where(
        pill_mask_roi_3ch,
        resized_pill_roi,
        tray_image[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width]
    )

    cv2.imwrite(save_path, tray_image)


# ------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- S U B T R A C T   I M A G E S ------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def abs_diff_images(empty_tray_image_path: str, aug_tray_img_w_pill_aug: str, save_results: bool, save_path: str,
                    save_plots_path) \
        -> None:
    """
    This function subtracts the empty tray and augmented trays with pills images, and generates absolute difference
    images. Plotting of three images are also available.

    Args:
        empty_tray_image_path (str): Path to the empty tray image.
        aug_tray_img_w_pill_aug (str): Path to the tray with pill image.
        save_results (bool): Either save the results (three images next to each other) or not.
        save_path (str): Path to the saving location.
        save_plots_path (str): Path to save the plots.
    """

    img1 = cv2.imread(empty_tray_image_path, 1)
    img2 = cv2.imread(aug_tray_img_w_pill_aug, 1)

    diff = cv2.absdiff(img1, img2)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    if save_results:
        # Create a figure and subplots
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Display img1
        axs[0].imshow(img1, cmap='gray')
        axs[0].set_title("Image 1")

        # Display img2
        axs[1].imshow(img2, cmap='gray')
        axs[1].set_title("Image 2")

        # Display diff
        axs[2].imshow(diff, cmap='gray')
        axs[2].set_title("Absolute Difference")

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.savefig(os.path.join(save_plots_path, os.path.basename(aug_tray_img_w_pill_aug)))

        plt.close()
        plt.close("all")
        plt.close()
        gc.collect()

    cv2.imwrite(save_path, diff)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- C R E A T E   D I R E C T O R I E S ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_directories(classes: List[str], images_dir: str, dataset_path: str,
                       masks_dir: Optional[str] = None, masks_path: Optional[str] = None) -> None:
    """
    Create directories for each class and copy images and masks to the corresponding directories.

    Args:
        classes (List[str]): List of class names.
        images_dir (str): Directory to save the images.
        dataset_path (str): Path to the dataset containing images.
        masks_dir (str, optional): Directory to save the masks. Defaults to None.
        masks_path (str, optional): Path to the dataset containing masks. Defaults to None.
    """

    for class_name in classes:
        class_path = os.path.join(images_dir, class_name)
        os.makedirs(class_path, exist_ok=True)

        images = [image for image in os.listdir(dataset_path) if image.startswith(f"{class_name}_")]
        for image in tqdm(images, total=len(images), desc="Copying images"):
            src_path = os.path.join(dataset_path, image)
            dest_path = os.path.join(class_path, image)
            shutil.copy(src_path, dest_path)

        if masks_dir and masks_path is not None:
            class_path_mask = os.path.join(masks_dir, class_name)
            os.makedirs(class_path_mask, exist_ok=True)

            # Copy masks to the corresponding class directory
            masks = [mask for mask in os.listdir(masks_path) if mask.startswith(f"{class_name}_")]
            for mask in tqdm(masks, total=len(masks), desc="Copying masks"):
                src_mask_path = os.path.join(masks_path, mask)
                dest_mask_path = os.path.join(class_path_mask, mask)
                shutil.copy(src_mask_path, dest_mask_path)
