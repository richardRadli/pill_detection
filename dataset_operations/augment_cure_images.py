"""
File: augment_cure_images.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jun 27, 2023

Description: The program  augments images for the CURE dataset. It can split the dataset, augment it and change
background on the images.
"""

import os
import random

from tqdm import tqdm

from augmentation_utils import change_brightness, gaussian_smooth, rotate_image, create_directories, shift_image, \
    zoom_in_object, change_white_balance, flip_image, change_background_dtd
from config.config import ConfigAugmentation
from config.const import DATASET_PATH
from utils.utils import setup_logger


class AugmentCUREDataset:
    def __init__(self):
        """
        Initialize the AugmentCUREDataset object.

        - Set up the logger.
        - Set the paths for dataset, masks, backgrounds, training images, training masks, test images, and test masks.
        - Set the number of training and test classes.
        - Get the list of all images and collect the unique classes.
        - Split the classes into training and test sets.
        """

        setup_logger()

        self.dataset_path = DATASET_PATH.get_data_path("cure_customer")
        self.mask_path = DATASET_PATH.get_data_path("cure_customer_mask")
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
    # ---------------------------------------------------- M A I N -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self, split_dataset: bool, do_aug: bool, change_background: bool) -> None:
        """
        Perform the main operations for augmenting the CURE dataset.

        :param split_dataset: Flag indicating whether to split the dataset into training and test sets.
        :param do_aug: Flag indicating whether to perform data augmentation.
        :param change_background: Flag indicating whether to change the background of the images.
        :return: None
        """

        aug_cfg = ConfigAugmentation().parse()

        if split_dataset:
            # Create directories for the train images
            create_directories(classes=self.training_classes,
                               images_dir=self.training_images,
                               dataset_path=self.dataset_path,
                               masks_dir=None,
                               masks_path=None)

            # Create directories for the test images
            create_directories(classes=self.test_classes,
                               images_dir=self.test_images,
                               dataset_path=self.dataset_path,
                               masks_dir=None,
                               masks_path=None)

        for class_name in self.training_classes:
            print(class_name)
            class_dir = os.path.join(self.training_images, class_name)
            class_dir_mask = os.path.join(self.training_masks, class_name)

            image_files = os.listdir(class_dir)
            mask_files = os.listdir(class_dir_mask)

            for idx, (image_file, mask_file) in tqdm(enumerate(zip(image_files, mask_files)), total=len(image_files)):
                full_image_path = os.path.join(class_dir, image_file)
                full_mask_path = os.path.join(class_dir_mask, mask_file)

                if do_aug:
                    change_white_balance(image_path=full_image_path,
                                         aug_path=full_image_path,
                                         domain=(aug_cfg.wb_low_thr, aug_cfg.wb_high_thr),
                                         mask_path=full_mask_path)

                    change_white_balance(image_path=full_image_path,
                                         aug_path=full_image_path,
                                         domain=(aug_cfg.wb_low_thr_2nd_aug, aug_cfg.wb_high_thr_2nd_aug),
                                         mask_path=full_mask_path)

                    gaussian_smooth(image_path=full_image_path,
                                    aug_path=full_image_path,
                                    kernel=(aug_cfg.kernel_size, aug_cfg.kernel_size),
                                    mask_path=full_mask_path)

                    change_brightness(image_path=full_image_path,
                                      aug_path=full_image_path,
                                      exposure_factor=random.uniform(aug_cfg.brightness_low_thr,
                                                                     aug_cfg.brightness_high_thr),
                                      mask_path=full_mask_path)

                    change_brightness(image_path=full_image_path,
                                      aug_path=full_image_path,
                                      exposure_factor=random.uniform(aug_cfg.brightness_low_thr_2nd_aug,
                                                                     aug_cfg.brightness_high_thr_2nd_aug),
                                      mask_path=full_mask_path)

                    rotate_image(image_path=full_image_path,
                                 aug_path=full_image_path,
                                 angle=random.randint(aug_cfg.rotate_low_thr, aug_cfg.rotate_high_thr),
                                 mask_path=full_mask_path)

                    rotate_image(image_path=full_image_path,
                                 aug_path=full_image_path,
                                 angle=random.randint(aug_cfg.rotate_low_thr, aug_cfg.rotate_high_thr),
                                 mask_path=full_mask_path)

                    rotate_image(image_path=full_image_path,
                                 aug_path=full_image_path,
                                 angle=random.randint(aug_cfg.rotate_low_thr, aug_cfg.rotate_high_thr),
                                 mask_path=full_mask_path)

                    shift_image(image_path=full_image_path,
                                aug_path=full_image_path,
                                shift_x=aug_cfg.shift_low_thr,
                                shift_y=aug_cfg.shift_high_thr,
                                mask_path=full_mask_path)

                    zoom_in_object(image_path=full_image_path,
                                   aug_path=full_image_path,
                                   crop_size=aug_cfg.crop_size,
                                   mask_path=full_mask_path)

                    flip_image(image_path=full_image_path,
                               aug_path=full_image_path,
                               flip_direction='horizontal',
                               mask_path=full_mask_path)

                    flip_image(image_path=full_image_path,
                               aug_path=full_image_path,
                               flip_direction='vertical',
                               mask_path=full_mask_path)

                if change_background:
                    change_background_dtd(image_path=full_image_path,
                                          mask_path=full_mask_path,
                                          backgrounds_path=self.backgrounds)


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- __M A I N__ -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    aug = AugmentCUREDataset()
    aug.main(split_dataset=True, do_aug=False, change_background=False)
