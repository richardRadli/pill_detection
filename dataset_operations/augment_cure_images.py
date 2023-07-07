import os
import random

from tqdm import tqdm

from config.const import DATASET_PATH
from config.logger_setup import setup_logger
from augmentation_utils import change_brightness, gaussian_smooth, rotate_image, create_directories, shift_image, \
    zoom_in_object, change_white_balance, flip_image, change_background_dtd


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
            # Create directories for the train images
            create_directories(classes=self.training_classes, images_dir=self.training_images,
                               dataset_path=self.dataset_path, masks_dir=self.training_masks, masks_path=self.mask_path)

            # Create directories for the test images
            create_directories(classes=self.test_classes, images_dir=self.test_images,
                               dataset_path=self.dataset_path, masks_dir=self.test_masks, masks_path=self.mask_path)

        for class_name in self.training_classes:
            class_dir = os.path.join(self.training_images, class_name)
            class_dir_mask = os.path.join(self.training_masks, class_name)

            image_files = os.listdir(class_dir)
            mask_files = os.listdir(class_dir_mask)

            for idx, (image_file, mask_file) in tqdm(enumerate(zip(image_files, mask_files)), total=len(image_files)):
                full_image_path = os.path.join(class_dir, image_file)
                full_mask_path = os.path.join(class_dir_mask, mask_file)

                if do_aug:
                    change_white_balance(image_path=full_image_path, aug_path=full_image_path, domain=(0.7, 1.2),
                                         mask_path=full_mask_path)
                    change_white_balance(image_path=full_image_path, aug_path=full_image_path, domain=(0.7, 1.2),
                                         mask_path=full_mask_path)
                    gaussian_smooth(image_path=full_image_path, aug_path=full_image_path, kernel=(7, 7),
                                    mask_path=full_mask_path)
                    change_brightness(image_path=full_image_path, aug_path=full_image_path,
                                      exposure_factor=random.uniform(0.5, 1.5), mask_path=full_mask_path)
                    change_brightness(image_path=full_image_path, aug_path=full_image_path,
                                      exposure_factor=random.uniform(0.5, 1.5), mask_path=full_mask_path)
                    rotate_image(image_path=full_image_path, aug_path=full_image_path, angle=random.randint(35, 270),
                                 mask_path=full_mask_path)
                    rotate_image(image_path=full_image_path, aug_path=full_image_path, angle=random.randint(35, 270),
                                 mask_path=full_mask_path)
                    rotate_image(image_path=full_image_path, aug_path=full_image_path, angle=random.randint(35, 270),
                                 mask_path=full_mask_path)
                    shift_image(image_path=full_image_path, aug_path=full_image_path, shift_x=150, shift_y=200,
                                mask_path=full_mask_path)
                    zoom_in_object(image_path=full_image_path, aug_path=full_image_path, crop_size=1500,
                                   mask_path=full_mask_path)
                    flip_image(image_path=full_image_path, aug_path=full_mask_path, flip_direction='horizontal',
                               mask_path=full_mask_path)
                    flip_image(image_path=full_image_path, aug_path=full_mask_path, flip_direction='vertical',
                               mask_path=full_mask_path)

                if change_background:
                    change_background_dtd(image_path=full_image_path, mask_path=full_mask_path,
                                          backgrounds_path=self.backgrounds)


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- __M A I N__ -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    aug = AugmentCUREDataset()
    aug.main(split_dataset=False, do_aug=True, change_background=False)
