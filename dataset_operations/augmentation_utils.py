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
import threading
import shutil


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


def save_data(
        image_path: str,
        aug_img_path: str,
        mask_path: str,
        aug_mask_path: str,
        annotation_path: str,
        aug_annotation_path: str,
        image,
        mask,
        filename: str,
        txt_op: str,
        annotation=None
):
    """
    Saves augmented images, masks, and annotations to specified paths,
    with options to copy, overwrite, or handle segmentation annotations.

    Args:
        image_path (str): Path to the original image file.
        aug_img_path (str): Path to save the augmented image file.
        mask_path (str): Path to the original mask file.
        aug_mask_path (str): Path to save the augmented mask file.
        annotation_path (str): Path to the original annotation file.
        aug_annotation_path (str): Path to save the augmented annotation file.
        image: Augmented image data to save (compatible with `cv2.imwrite`).
        mask: Augmented mask data to save (compatible with `cv2.imwrite`).
        filename (str): Base filename for saving augmented files.
        txt_op (str): Operation to perform on the annotation file; options are
                      "copy", "overwrite", or "segmentation".
        annotation (Optional[List[str]]): New annotation content for "overwrite"
                                          or "segmentation" modes.

    Returns:
        None
    """

    with threading.Lock():
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


def change_white_balance(image_path: str, annotation_path: str, mask_path: str,
                         aug_img_path: str, aug_annotation_path: str, aug_mask_path: str,
                         domain: tuple) -> None:
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

    scale_factors = np.random.uniform(low=domain[0], high=domain[1], size=(3,))

    adjusted_image = image * scale_factors

    adjusted_image = np.clip(adjusted_image, 0, 255)
    adjusted_image = adjusted_image.astype(np.uint8)

    save_data(
        image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
        adjusted_image, mask, filename, txt_op="copy"
    )


def gaussian_smooth(image_path: str, annotation_path: str, mask_path: str, aug_img_path: str,
                    aug_annotation_path: str, aug_mask_path: str, kernel: int) -> None:
    """
    Apply Gaussian smoothing to an image and save the smoothed image.

    Args:
        image_path (str): Path to the input image.
        annotation_path (str): Path to the annotation file.
        mask_path (str): Path to save the mask image.
        aug_img_path (str): Path to the augmented image.
        aug_annotation_path (str): Path to the augmented annotation file.
        aug_mask_path (str): Path to the augmented mask image.
        kernel (int): Kernel size for Gaussian smoothing.
    """

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    filename = "gaussian_%s" % str(kernel)

    smoothed_image = cv2.GaussianBlur(image, (kernel, kernel), 0)

    save_data(image_path, aug_img_path, mask_path, aug_mask_path, annotation_path, aug_annotation_path,
              smoothed_image, mask, filename, txt_op="copy"
              )


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


def rotate_operation(image_path: str, mask_path: str, angle: float) -> tuple:
    """
    Rotates the specified image and mask by a given angle and returns the original
    and rotated images, rotated mask, and rotation center.

    Args:
        image_path (str): Path to the image file to rotate.
        mask_path (str): Path to the mask file to rotate.
        angle (float): Angle in degrees by which to rotate the image and mask.

    Returns:
        Tuple: A tuple containing:
            - original_image: The original unrotated image.
            - rotated_image: The rotated image.
            - rotated_mask: The rotated mask.
            - center: The center point of rotation as a tuple (x, y).
    """

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

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Shift image in x and y directions
    rows, cols, _ = image.shape
    mtx = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, mtx, (cols, rows))
    shifted_mask = cv2.warpAffine(mask, mtx, (cols, rows))

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


def change_background_dtd(image_path: str, mask_path: str, annotation_path: str, backgrounds_path: str) -> None:
    """
    Change the background of an image using images from a specified directory.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        annotation_path (str): Path to the annotation file.
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

            new_image_file_name = (
                rename_file(
                    src_path=image_path,
                    dst_path=os.path.dirname(image_path),
                    op="changed_background")
            )
            cv2.imwrite(new_image_file_name, output_image)

            new_annotation_file_name = (
                rename_file(
                    src_path=annotation_path,
                    dst_path=os.path.dirname(annotation_path),
                    op="changed_background"
                )
            )
            shutil.copy(annotation_path, new_annotation_file_name)

            # Save the original mask file in the new path
            new_mask_file_name = (
                rename_file(
                    src_path=mask_path,
                    dst_path=os.path.dirname(mask_path),
                    op="changed_background"
                )
            )
            shutil.copy(mask_path, new_mask_file_name)
        else:
            logging.info(f"The background image {os.path.basename(background_image_path)} is empty.")
    except AttributeError:
        logging.info(f'Image {os.path.basename(background_image_path)} is wrong!')