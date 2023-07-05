import cv2
import numpy as np
import os
import shutil

from tqdm import tqdm
from typing import Tuple


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- R E N A M E   F I L E --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def rename_file(image_path, op):
    """
    This function is used in the augmentation files.
    :param image_path:
    :param op:
    :return:
    """

    # Split the original file path into directory and filename
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)

    # Split the filename into name and extension
    name, extension = os.path.splitext(filename)

    # Construct the new file path with the desired filename
    new_filename = f"{name}_{op}"
    counter = 1
    final_file_name = os.path.join(directory, f"{new_filename}_{counter}{extension}")

    while os.path.isfile(final_file_name):
        counter += 1
        final_file_name = os.path.join(directory, f"{new_filename}_{counter}{extension}")

    return final_file_name


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- U N I Q U E   C O U N T   A P P ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def unique_count_app(img):
    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    colors, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    return tuple(colors[count.argmax()])


# ------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- G A U S S I A N   S M O O T H ------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def gaussian_smooth(image_path, aug_path, kernel, mask_path=None) -> None:
    image = cv2.imread(image_path)
    smoothed_image = cv2.GaussianBlur(image, kernel, 0)
    new_image_file_name = rename_file(aug_path, op="gaussian_%s" % str(kernel[0]))
    cv2.imwrite(new_image_file_name, smoothed_image)

    if mask_path is not None:
        mask = cv2.imread(mask_path)
        new_mask_file_name = rename_file(mask_path, op="gaussian_%s" % str(kernel[0]))
        cv2.imwrite(new_mask_file_name, mask)


# ------------------------------------------------------------------------------------------------------------------
# --------------------------------------- C H A N G E   B R I G H T N E S S ----------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def change_brightness(image_path, aug_path, exposure_factor: float, mask_path=None) -> None:
    image = cv2.imread(image_path)

    image = image.astype(np.float32) / 255.0
    adjusted_image = image * exposure_factor
    adjusted_image = np.clip(adjusted_image, 0, 1)
    adjusted_image = (adjusted_image * 255).astype(np.uint8)

    new_image_file_name = rename_file(aug_path, op="brightness")
    cv2.imwrite(new_image_file_name, adjusted_image)

    if mask_path is not None:
        mask = cv2.imread(mask_path)
        new_mask_file_name = rename_file(mask_path, op="brightness")
        cv2.imwrite(new_mask_file_name, mask)


# ------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- R O T A T E   I M A G E ---------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def rotate_image(image_path, aug_path, angle: int, mask_path=None) -> None:
    image = cv2.imread(image_path)

    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    clr = unique_count_app(image)
    clr = tuple(value.item() for value in clr)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=clr)
    new_image_file_name = rename_file(aug_path, op="rotated_%s" % str(angle))
    cv2.imwrite(new_image_file_name, rotated_image)

    if mask_path is not None:
        mask = cv2.imread(mask_path)
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (width, height))
        new_mask_file_name = rename_file(mask_path, op="rotated_%s" % str(angle))
        cv2.imwrite(new_mask_file_name, rotated_mask)


# ------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- S H I F T   I M A G E ----------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def shift_image(image_path, aug_path, shift_x: int = 50, shift_y: int = 100, mask_path=None):
    image = cv2.imread(image_path)

    height, width = image.shape[:2]

    mtx = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    clr = unique_count_app(image)
    clr = tuple(value.item() for value in clr)

    shifted_image = cv2.warpAffine(image, mtx, (width, height), borderValue=clr)
    new_image_file_name = rename_file(aug_path, op="shifted")
    cv2.imwrite(new_image_file_name, shifted_image)

    if mask_path is not None:
        mask = cv2.imread(mask_path)
        shifted_mask = cv2.warpAffine(mask, mtx, (width, height))
        new_mask_file_name = rename_file(mask_path, op="shifted")
        cv2.imwrite(new_mask_file_name, shifted_mask)


# ------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- D I S T O R T   C O L O R -------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def change_white_balance(image_path: str, aug_path, domain: Tuple[float, float] = (0.7, 1.2), mask_path=None) -> None:
    image = cv2.imread(image_path)

    # Generate random scaling factors for each color channel
    scale_factors = np.random.uniform(low=domain[0], high=domain[1], size=(3,))

    # Apply the scaling factors to the image
    adjusted_image = image * scale_factors

    # Clip the pixel values to the valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255)
    adjusted_image = adjusted_image.astype(np.uint8)

    new_image_file_name = rename_file(aug_path, op="distorted_colour")
    cv2.imwrite(new_image_file_name, adjusted_image)

    if mask_path is not None:
        mask = cv2.imread(mask_path)
        new_mask_file_name = rename_file(mask_path, op="distorted_colour")
        cv2.imwrite(new_mask_file_name, mask)


def copy_original_images(src_path, dst_path):
    shutil.copy(src_path, dst_path)


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Z O O M   I N   O B J E C T ------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def zoom_in_object(image_path, aug_path, crop_size, mask_path=None):
    image = cv2.imread(image_path)

    height, width = image.shape[:2]

    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    cropped_image = image[start_y:end_y, start_x:end_x]
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

    new_image_file_name = rename_file(aug_path, op="zoomed")
    cv2.imwrite(new_image_file_name, zoomed_image)

    if mask_path is not None:
        mask = cv2.imread(mask_path)
        cropped_mask = mask[start_y:end_y, start_x:end_x]
        zoomed_mask = cv2.resize(cropped_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        _, zoomed_mask = cv2.threshold(zoomed_mask, 128, 255, cv2.THRESH_BINARY)
        new_mask_file_name = rename_file(mask_path, op="zoomed")
        cv2.imwrite(new_mask_file_name, zoomed_mask)


# ------------------------------------------------------------------------------------------------------------------
# --------------------------------------- C R E A T E   D I R E C T O R I E S --------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def create_directories(classes, images_dir, dataset_path, masks_dir=None, masks_path=None):
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
