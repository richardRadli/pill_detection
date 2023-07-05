import cv2
import logging
import numpy as np
import os
import random
import shutil

from tqdm import tqdm
from typing import Tuple

from config.const import DATASET_PATH
from config.logger_setup import setup_logger
from utils.utils import rename_file, unique_count_app


class AugmentCUREDataset:
    def __init__(self):
        setup_logger()

        self.dataset_path = DATASET_PATH.get_data_path("cure_reference")
        self.mask_path = DATASET_PATH.get_data_path("cure_reference_mask")
        self.backgrounds = DATASET_PATH.get_data_path("dtd_images")

        self.training_images = DATASET_PATH.get_data_path("cure_train")
        self.training_masks = DATASET_PATH.get_data_path("cure_train_mask")
        self.test_images = DATASET_PATH.get_data_path("cure_test")
        self.test_masks = DATASET_PATH.get_data_path("cure_test_mask")

        # Set the number of training and test classes
        num_training_classes = 156

        # Get the list of all images
        image_files = os.listdir(self.dataset_path)
        image_files.sort()

        # Collect the unique classes from the image filenames
        all_classes = set()
        for image_file in image_files:
            class_name = image_file.split('_')[0]
            all_classes.add(class_name)
        all_classes = sorted(list(all_classes))

        # Split the classes into training and test sets
        self.training_classes = all_classes[:num_training_classes]
        self.test_classes = all_classes[num_training_classes:]

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- C R E A T E   D I R E C T O R I E S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_directories(classes, images_dir, masks_dir, dataset_path, masks_path):
        """

        :param classes:
        :param images_dir:
        :param masks_dir:
        :param dataset_path:
        :param masks_path:
        :return:
        """

        # Create directories for each class in the training set
        for class_name in classes:
            class_path = os.path.join(images_dir, class_name)
            os.makedirs(class_path, exist_ok=True)

            class_path_mask = os.path.join(masks_dir, class_name)
            os.makedirs(class_path_mask, exist_ok=True)

            # Copy images to the corresponding class directory
            images = [image for image in os.listdir(dataset_path) if image.startswith(f"{class_name}_")]
            for image in tqdm(images, total=len(images), desc="Copying images"):
                src_path = os.path.join(dataset_path, image)
                dest_path = os.path.join(class_path, image)
                shutil.copy(src_path, dest_path)

            # Copy masks to the corresponding class directory
            masks = [mask for mask in os.listdir(masks_path) if mask.startswith(f"{class_name}_")]
            for mask in tqdm(masks, total=len(masks), desc="Copying masks"):
                src_mask_path = os.path.join(masks_path, mask)
                dest_mask_path = os.path.join(class_path_mask, mask)
                shutil.copy(src_mask_path, dest_mask_path)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- D I S T O R T   C O L O R -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def distort_color(self, image_path: str, mask_path: str, domain: Tuple[float, float]) -> None:
        """

        :param image_path:
        :param mask_path:
        :param domain:
        :return:
        """

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # Generate random scaling factors for each color channel
        scale_factors = np.random.uniform(low=domain[0], high=domain[1], size=(3,))

        # Apply the scaling factors to the image
        adjusted_image = image * scale_factors

        # Clip the pixel values to the valid range [0, 255]
        adjusted_image = np.clip(adjusted_image, 0, 255)
        adjusted_image = adjusted_image.astype(np.uint8)

        new_image_file_name = rename_file(image_path, op="distorted_colour")
        new_mask_file_name = rename_file(mask_path, op="distorted_colour")

        cv2.imwrite(new_image_file_name, adjusted_image)
        cv2.imwrite(new_mask_file_name, mask)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- G A U S S I A N   S M O O T H ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def gaussian_smooth(self, image_path, mask_path, kernel) -> None:
        """

        :param image_path:
        :param mask_path:
        :param kernel:
        :return:
        """

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        smoothed_image = cv2.GaussianBlur(image, kernel, 0)

        new_image_file_name = rename_file(image_path, op="gaussian_%s" % str(kernel[0]))
        new_mask_file_name = rename_file(mask_path, op="gaussian_%s" % str(kernel[0]))

        cv2.imwrite(new_image_file_name, smoothed_image)
        cv2.imwrite(new_mask_file_name, mask)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- C H A N G E   B R I G H T N E S S ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def change_brightness(self, image_path, mask_path, exposure_factor: float) -> None:
        """

        :param image_path:
        :param mask_path:
        :param exposure_factor:
        :return:
        """

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        image = image.astype(np.float32) / 255.0
        adjusted_image = image * exposure_factor
        adjusted_image = np.clip(adjusted_image, 0, 1)
        adjusted_image = (adjusted_image * 255).astype(np.uint8)

        new_image_file_name = rename_file(image_path, op="brightness")
        new_mask_file_name = rename_file(mask_path, op="brightness")

        cv2.imwrite(new_image_file_name, adjusted_image)
        cv2.imwrite(new_mask_file_name, mask)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- R O T A T E   I M A G E ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def rotate_image(self, image_path, mask_path, angle: int) -> None:
        """

        :param image_path:
        :param mask_path:
        :param angle:
        :return:
        """

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        clr = unique_count_app(image)
        clr = tuple(value.item() for value in clr)

        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=clr)
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (width, height))

        new_image_file_name = rename_file(image_path, op="rotated_%s" % str(angle))
        new_mask_file_name = rename_file(mask_path, op="rotated_%s" % str(angle))

        cv2.imwrite(new_image_file_name, rotated_image)
        cv2.imwrite(new_mask_file_name, rotated_mask)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- S H I F T   I M A G E ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def shift_image(self, image_path, mask_path, shift_x: int = 50, shift_y: int = 100):
        """

        :param image_path:
        :param mask_path:
        :param shift_x:
        :param shift_y:
        :return:
        """

        # Read the image
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # Get the original image size
        height, width = image.shape[:2]

        # Create a transformation matrix for shifting
        mtx = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Apply the shift transformation
        clr = unique_count_app(image)
        clr = tuple(value.item() for value in clr)

        shifted_image = cv2.warpAffine(image, mtx, (width, height), borderValue=clr)
        shifted_mask = cv2.warpAffine(mask, mtx, (width, height))

        new_image_file_name = rename_file(image_path, op="shifted")
        new_mask_file_name = rename_file(mask_path, op="shifted")

        cv2.imwrite(new_image_file_name, shifted_image)
        cv2.imwrite(new_mask_file_name, shifted_mask)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- Z O O M   I N   O B J E C T ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def zoom_in_object(self, image_path, mask_path, crop_size):
        """

        :param image_path:
        :param mask_path:
        :param crop_size:
        :return:
        """

        # Read the image
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # Get the original image size
        height, width = image.shape[:2]

        # Calculate the coordinates for cropping the desired region
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        # Crop the region of interest
        cropped_image = image[start_y:end_y, start_x:end_x]
        croppe_mask = mask[start_y:end_y, start_x:end_x]

        # Resize the cropped image back to the original size
        zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
        zoomed_mask = cv2.resize(croppe_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        _, zoomed_mask = cv2.threshold(zoomed_mask, 128, 255, cv2.THRESH_BINARY)

        new_image_file_name = rename_file(image_path, op="zoomed")
        new_mask_file_name = rename_file(mask_path, op="zoomed")

        cv2.imwrite(new_image_file_name, zoomed_image)
        cv2.imwrite(new_mask_file_name, zoomed_mask)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- F L I P   I M A G E ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def flip_image(self, image_path, mask_path, flip_direction):
        # Read the image
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        # Flip the image based on the specified direction
        if flip_direction == 'horizontal':
            flipped_image = cv2.flip(image, 1)  # Flip horizontally (around the y-axis)
            flipped_mask = cv2.flip(mask, 1)
        elif flip_direction == 'vertical':
            flipped_image = cv2.flip(image, 0)  # Flip vertically (around the x-axis)
            flipped_mask = cv2.flip(mask, 0)
        else:
            raise ValueError("Invalid flip direction. Must be 'horizontal' or 'vertical'.")

        new_image_file_name = rename_file(image_path, op="flipped_%s" % flip_direction)
        new_mask_file_name = rename_file(mask_path, op="flipped_%s" % flip_direction)

        cv2.imwrite(new_image_file_name, flipped_image)
        cv2.imwrite(new_mask_file_name, flipped_mask)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C H A N G E   B A C K G R O U N D ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def change_background(self, image_path: str, mask_path: str, background_path: str) -> None:
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # Get a list of files in the backgrounds directory
        background_files = os.listdir(self.backgrounds)

        # Randomly select a background image file
        background_file = random.choice(background_files)

        # Build the path to the randomly selected background image
        background_image_path = os.path.join(self.backgrounds, background_file)

        background = cv2.imread(background_image_path)

        try:
            if background.size != 0:
                # Ensure mask and background have the same size
                background = cv2.resize(background, (image.shape[1], image.shape[0]))

                foreground = cv2.bitwise_and(image, image, mask=mask)
                background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

                output_image = cv2.add(foreground, background)

                new_image_file_name = rename_file(image_path, op="changed_background")
                new_mask_file_name = rename_file(mask_path, op="changed_background")

                cv2.imwrite(new_image_file_name, output_image)
                cv2.imwrite(new_mask_file_name, mask)
            else:
                logging.info(f"The background image {os.path.basename(background_image_path)} is empty.")
        except AttributeError:
            logging.info(f'Image {os.path.basename(background_image_path)} is wrong!')

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- M A I N -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self, split_dataset: bool, do_aug: bool, change_background: bool) -> None:
        """

        :param split_dataset:
        :param do_aug:
        :param change_background:
        :return:
        """

        if split_dataset:
            self.create_directories(self.training_classes, self.training_images, self.training_masks, self.dataset_path,
                                    self.mask_path)

            self.create_directories(self.test_classes, self.test_images, self.test_masks, self.dataset_path,
                                    self.mask_path)

        for class_name in self.training_classes:
            class_dir = os.path.join(self.training_images, class_name)
            class_dir_mask = os.path.join(self.training_masks, class_name)

            image_files = os.listdir(class_dir)
            mask_files = os.listdir(class_dir_mask)

            for idx, (image_file, mask_file) in tqdm(enumerate(zip(image_files, mask_files)), total=len(image_files)):
                full_image_path = os.path.join(class_dir, image_file)
                full_mask_path = os.path.join(class_dir_mask, mask_file)

                if do_aug:
                    self.distort_color(full_image_path, full_mask_path, (0.7, 1.2))
                    self.distort_color(full_image_path, full_mask_path, (0.7, 1.2))
                    self.gaussian_smooth(full_image_path, full_mask_path, (7, 7))
                    self.change_brightness(full_image_path, full_mask_path, exposure_factor=random.uniform(0.5, 1.5))
                    self.change_brightness(full_image_path, full_mask_path, exposure_factor=random.uniform(0.5, 1.5))
                    self.rotate_image(full_image_path, full_mask_path, angle=random.randint(35, 270))
                    self.rotate_image(full_image_path, full_mask_path, angle=random.randint(35, 270))
                    self.rotate_image(full_image_path, full_mask_path, angle=random.randint(35, 270))
                    self.shift_image(full_image_path, full_mask_path, 150, 200)
                    self.zoom_in_object(full_image_path, full_mask_path, 1500)
                    self.flip_image(full_image_path, full_mask_path, 'horizontal')
                    self.flip_image(full_image_path, full_mask_path, 'vertical')

                if change_background:
                    self.change_background(full_image_path, full_mask_path, self.backgrounds)

            break


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- __M A I N__ -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    aug = AugmentCUREDataset()
    aug.main(split_dataset=False, do_aug=True, change_background=False)
