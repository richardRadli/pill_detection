"""
File: augment_cure_images.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jun 27, 2023
"""

import os
import random

from tqdm import tqdm

from augmentation_utils import (change_brightness, gaussian_smooth, rotate_image, shift_image, zoom_in_object,
                                change_white_balance, flip_image)


class AugmentOGYEIDataset:
    def __init__(self):
        self.dataset_path = \
            "C:/Users/ricsi/Desktop/ogyei_v2_splitted_20_classes"

        # Get the list of all images
        self.classes = os.listdir(self.dataset_path)
        self.classes.sort()

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- M A I N -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self) -> None:
        """
        Perform the main operations for augmenting the CURE dataset.

        :return: None
        """

        for class_name in self.classes:
            class_dir = os.path.join(self.dataset_path, class_name)

            image_files = os.listdir(class_dir)

            for _, image_file in tqdm(enumerate(image_files), total=len(image_files)):
                full_image_path = os.path.join(class_dir, image_file)

                change_white_balance(image_path=full_image_path, aug_path=full_image_path, domain=(0.7, 1.2))
                change_white_balance(image_path=full_image_path, aug_path=full_image_path, domain=(0.7, 1.2))
                gaussian_smooth(image_path=full_image_path, aug_path=full_image_path, kernel=(7, 7))
                change_brightness(image_path=full_image_path, aug_path=full_image_path,
                                  exposure_factor=random.uniform(0.5, 1.5))
                change_brightness(image_path=full_image_path, aug_path=full_image_path,
                                  exposure_factor=random.uniform(0.5, 1.5))
                rotate_image(image_path=full_image_path, aug_path=full_image_path, angle=random.randint(35, 270))
                rotate_image(image_path=full_image_path, aug_path=full_image_path, angle=random.randint(35, 270))
                rotate_image(image_path=full_image_path, aug_path=full_image_path, angle=random.randint(35, 270))
                shift_image(image_path=full_image_path, aug_path=full_image_path, shift_x=150, shift_y=200)
                zoom_in_object(image_path=full_image_path, aug_path=full_image_path, crop_size=1500)
                flip_image(image_path=full_image_path, aug_path=full_image_path, flip_direction='horizontal')
                flip_image(image_path=full_image_path, aug_path=full_image_path, flip_direction='vertical')


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- __M A I N__ -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    aug = AugmentOGYEIDataset()
    aug.main()
