"""
File: augmentation_utils.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 05, 2023

Description: The program stores the functions for the augmentation process.
"""

import cv2
import logging
import numpy as np
import os
import random
import shutil

from typing import Tuple


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
    filename = os.path.basename(src_path)

    # Split the filename into name and extension
    name, extension = os.path.splitext(filename)

    # Construct the new file path with the desired filename
    new_filename = f"{name}_{op}"
    counter = 1
    final_file_name = os.path.join(dst_path, f"{new_filename}_{counter}{extension}")

    while os.path.isfile(final_file_name):
        counter += 1
        final_file_name = os.path.join(dst_path, f"{new_filename}_{counter}{extension}")

    return final_file_name


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------S A V E   D A T A ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def save_data(
        image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path, image, mask, filename,
        txt_op, annotation=None
):
    new_image_file_name = rename_file(src_path=image_path, dst_path=aug_img_path, op=filename)
    cv2.imwrite(new_image_file_name, image)

    new_mask_file_name = rename_file(src_path=mask_path, dst_path=aug_mask_path, op=filename)
    cv2.imwrite(new_mask_file_name, mask)

    new_annotation_file_name = rename_file(src_path=annotation_path, dst_path=aug_annotation_path, op=filename)
    if txt_op == "copy":
        shutil.copy(annotation_path, new_annotation_file_name)
    elif txt_op == "overwrite":
        with open(new_annotation_file_name, 'w') as f:
            f.write('\n'.join(annotation))
    elif txt_op == "segmentation":
        with open(new_annotation_file_name, 'w') as f:
            f.write(annotation)
    else:
        raise ValueError(f"Wrong txt operation: {txt_op}")


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- D I S T O R T   C O L O R ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def change_white_balance(image_path: str, annotation_path: str, mask_path: str,
                         aug_img_path: str, aug_annotation_path: str, aug_mask_path: str,
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

    save_data(
        image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
        adjusted_image, mask, filename, txt_op="copy"
    )


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

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
              smoothed_image, mask, filename, txt_op="copy"
              )


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

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
              adjusted_image, mask, filename, txt_op="copy"
              )


def rotate_operation(image_path, mask_path, angle):
    # Read the original image and mask
    original_image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Get the center of the image
    center = (original_image.shape[1] // 2, original_image.shape[0] // 2)

    # Perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_image = cv2.warpAffine(original_image, rotation_matrix, (original_image.shape[1], original_image.shape[0]))
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))

    return original_image, rotated_image, rotated_mask, center


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
    filename = f"rotated_{angle}"

    original_image, rotated_image, rotated_mask, center = rotate_operation(image_path, mask_path, angle)

    # Read the annotations file
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    rotated_annotations = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.strip().split())

        # Convert angles to radians
        angle_rad = np.radians(angle)

        # Rotate the center coordinates
        x_center_rotated = (x_center * original_image.shape[1] - center[0]) * np.cos(angle_rad) - \
                           (y_center * original_image.shape[0] - center[1]) * np.sin(angle_rad) + center[0]

        y_center_rotated = (x_center * original_image.shape[1] - center[0]) * np.sin(angle_rad) + \
                           (y_center * original_image.shape[0] - center[1]) * np.cos(angle_rad) + center[1]

        width_rotated = width * original_image.shape[1]
        height_rotated = height * original_image.shape[0]

        rotated_box = (
            x_center_rotated,
            y_center_rotated,
            x_center_rotated + width_rotated,
            y_center_rotated + height_rotated
        )

        rotated_annotation = f"{int(class_id)} {rotated_box[0] / rotated_image.shape[1]:.6f} " \
                             f"{rotated_box[1] / rotated_image.shape[0]:.6f} " \
                             f"{(rotated_box[2] - rotated_box[0]) / rotated_image.shape[1]:.6f} " \
                             f"{(rotated_box[3] - rotated_box[1]) / rotated_image.shape[0]:.6f}"

        rotated_annotations.append(rotated_annotation)

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
              rotated_image, rotated_mask, filename, txt_op="overwrite", annotation=rotated_annotations)


def rotate_image_segmentation(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                              aug_annotation_path: str, aug_mask_path: str, angle: int) -> None:
    """
    Rotate an image and save the rotated image.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file.
        mask_path (str): Path to the mask image.
        aug_img_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation file.
        aug_mask_path (str): Path to the augmented mask image.
        angle (int): Angle of rotation in degrees.
    """
    filename = f"rotated_{angle}"
    original_image, rotated_image, rotated_mask, center = rotate_operation(image_path, mask_path, angle)

    # Read the annotations file
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    rotated_annotations = []
    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = parts[0]
        coordinates = list(map(float, parts[1:]))

        rotated_coordinates = []
        for i in range(0, len(coordinates), 2):
            x = coordinates[i] * original_image.shape[1]  # Scale x-coordinate
            y = coordinates[i + 1] * original_image.shape[0]  # Scale y-coordinate

            # Convert angles to radians
            angle_rad = np.radians(angle)

            # Rotate the coordinates around the center
            x_rotated = (x - center[0]) * np.cos(angle_rad) - (y - center[1]) * np.sin(angle_rad) + center[0]
            y_rotated = (x - center[0]) * np.sin(angle_rad) + (y - center[1]) * np.cos(angle_rad) + center[1]

            # Normalize rotated coordinates
            x_rotated /= rotated_image.shape[1]
            y_rotated /= rotated_image.shape[0]

            rotated_coordinates.extend([x_rotated, y_rotated])

        rotated_annotation = ' '.join([class_id] + list(map(str, rotated_coordinates)))
        rotated_annotations.append(rotated_annotation)

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
              rotated_image, rotated_mask, filename, txt_op="overwrite", annotation=rotated_annotations)


def shift_image_operation(image_path, mask_path, aug_img_path, aug_mask_path, shift_x, shift_y):
    """

    :param image_path:
    :param mask_path:
    :param aug_img_path:
    :param aug_mask_path:
    :param shift_x:
    :param shift_y:
    :return:
    """

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Shift image in x and y directions
    rows, cols, _ = image.shape
    mtx = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, mtx, (cols, rows))
    shifted_mask = cv2.warpAffine(mask, mtx, (cols, rows))

    # Write the shifted image and mask
    cv2.imwrite(aug_img_path, shifted_image)
    cv2.imwrite(aug_mask_path, shifted_mask)

    return shifted_image, shifted_mask, rows, cols


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

    shifted_image, shifted_mask, rows, cols = shift_image_operation(
        image_path,
        mask_path,
        aug_img_path,
        aug_mask_path,
        shift_x,
        shift_y
    )
    filename = f"shifting_{shift_x}_{shift_y}"

    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    shifted_annotations = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.strip().split())
        shifted_annotation = (int(class_id), x_center + shift_x / cols, y_center + shift_y / rows, width, height)
        shifted_annotations.append(shifted_annotation)

    shifted_annotations = [f"{anno[0]} {anno[1]} {anno[2]} {anno[3]} {anno[4]}" for anno in shifted_annotations]

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path, shifted_image,
              shifted_mask, filename, txt_op="overwrite", annotation=shifted_annotations)


def shift_image_segmentation(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                             aug_annotation_path: str, aug_mask_path: str, shift_x: int = 50,
                             shift_y: int = 100) -> None:
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

    shifted_image, shifted_mask, rows, cols = shift_image_operation(
        image_path,
        mask_path,
        aug_img_path,
        aug_mask_path,
        shift_x,
        shift_y
    )
    filename = f"shifting_{shift_x}_{shift_y}"

    # Modify annotations
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    shifted_annotation_str = None
    shifted_annotations_list = []
    for annotation in annotations:
        parts = annotation.strip().split()
        class_index = parts[0]
        coordinates = list(map(float, parts[1:]))
        shifted_annotations = []

        # Update coordinates based on the shift
        for i in range(0, len(coordinates), 2):
            x = coordinates[i] + shift_x / cols
            y = coordinates[i + 1] + shift_y / rows
            shifted_annotations.extend([x, y])

        # Add class ID to the shifted coordinates and concatenate into a single string
        shifted_annotation_str = ' '.join([class_index] + list(map(str, shifted_annotations)))
        shifted_annotations_list.append(shifted_annotation_str)

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path, shifted_image,
              shifted_mask, filename, txt_op="segmentation", annotation=shifted_annotation_str)


def zoom_operation(image_path, mask_path, crop_size):
    """

    :param image_path:
    :param mask_path:
    :param crop_size:
    :return:
    """

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    mask = cv2.imread(mask_path)

    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    cropped_image = image[start_y:end_y, start_x:end_x]
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

    cropped_mask = mask[start_y:end_y, start_x:end_x]
    zoomed_mask = cv2.resize(cropped_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed_image, zoomed_mask, width, height, start_x, start_y


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

    zoomed_image, zoomed_mask, width, height, start_x, start_y = zoom_operation(image_path, mask_path, crop_size)
    filename = "zoomed"

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

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
              zoomed_image, zoomed_mask, filename, txt_op="overwrite", annotation=adjusted_annotations)


def zoom_in_object_segmentation(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
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

    zoomed_image, zoomed_mask, width, height, start_x, start_y = zoom_operation(image_path, mask_path, crop_size)
    filename = "zoomed"

    with open(annotation_path, "r") as file:
        annotations = file.readlines()

    adjusted_annotations = []
    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = parts[0]
        coordinates = list(map(float, parts[1:]))

        adjusted_coordinates = []
        for i in range(0, len(coordinates), 2):
            x = ((coordinates[i] * width) - start_x) / crop_size
            y = ((coordinates[i + 1] * height) - start_y) / crop_size
            adjusted_coordinates.extend([x, y])

        adjusted_annotation = ' '.join([class_id] + list(map(str, adjusted_coordinates)))
        adjusted_annotations.append(adjusted_annotation)

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
              zoomed_image, zoomed_mask, filename, txt_op="overwrite", annotation=adjusted_annotations)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Z O O M   I N   O B J E C T --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def copy_original_images(image_path: str, mask_path: str, annotation_path: str, aug_image_path: str, aug_mask_path: str,
                         aug_annotation_path: str) -> None:
    """
    Copies files from one directory to another

    Args:
        image_path:
        mask_path:
        annotation_path:
        aug_image_path:
        aug_mask_path:
        aug_annotation_path:

    Returns:
        None
    """

    shutil.copy(image_path, aug_image_path)
    shutil.copy(mask_path, aug_mask_path)
    shutil.copy(annotation_path, aug_annotation_path)


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
