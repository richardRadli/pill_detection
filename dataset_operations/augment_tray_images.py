import cv2
import logging
import numpy as np
import os
import random
import re
import matplotlib.pyplot as plt

from skimage.transform import resize
from tqdm import tqdm

from config.config import ConfigAugmentation
from config.const import DATASET_PATH
from config.logger_setup import setup_logger
from augmentation_utils import rename_file, gaussian_smooth, change_brightness, rotate_image, shift_image, \
    change_white_balance, copy_original_images, create_directories


class AugmentTrayImages:
    def __init__(self):
        setup_logger()

        self.cfg = ConfigAugmentation().parse()

        self.root = "C:/Users/ricsi/Desktop/tray"

        self.tray_images = os.path.join(self.root, "2023-07-04")
        self.tray_images_aug = os.path.join(self.root, "tray_images_aug")
        self.tray_images_aug_w_med = os.path.join(self.root, "tray_images_aug_w_med")
        self.tray_images_aug_w_med_aug = os.path.join(self.root, "tray_images_aug_w_med_aug")
        self.diff_images = os.path.join(self.root, "diff_images")

        self.medicine_images = DATASET_PATH.get_data_path("cure_reference")
        self.medicine_masks = DATASET_PATH.get_data_path("cure_reference_mask")

        # Get the list of all images
        image_files = os.listdir(self.tray_images)
        image_files.sort()

        # Collect the unique classes from the image filenames
        all_classes = set()
        for image_file in image_files:
            class_name = image_file.split('_')[0]
            all_classes.add(class_name)
        self.all_classes = sorted(list(all_classes))

    @staticmethod
    def menu():
        phase1_title = "Augmenting empty tray images"
        phase2_title = "Placing pills on the augmented tray images"
        phase3_title = "Augmenting the images of the previous phase"
        phase4_title = "Creating absolute difference images"

        # Print the help menu
        logging.info("=== Help Menu ===")
        logging.info("Please select one of the following phases:")
        logging.info(f"1. {phase1_title}")
        logging.info(f"2. {phase2_title}")
        logging.info(f"3. {phase3_title}")
        logging.info(f"4. {phase4_title}")

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- F L I P   I M A G E ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def flip_image(image_path, aug_path, flip_direction):
        # Read the image
        image = cv2.imread(image_path)

        # Flip the image based on the specified direction
        if flip_direction == 'horizontal':
            flipped_image = cv2.flip(image, 1)  # Flip horizontally (around the y-axis)
        elif flip_direction == 'vertical':
            flipped_image = cv2.flip(image, 0)  # Flip vertically (around the x-axis)
        else:
            raise ValueError("Invalid flip direction. Must be 'horizontal' or 'vertical'.")

        new_image_file_name = rename_file(aug_path, op="flipped_%s" % flip_direction)
        cv2.imwrite(new_image_file_name, flipped_image)

    @staticmethod
    def place_medicine_on_tray(pill_image_path, pill_mask_path, tray_image_path, save_path, scaling_factor):
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

    @staticmethod
    def subtract_images(empty_tray_image_path, aug_tray_img_w_pill_aug, display_results, save_path=None):
        img1 = cv2.imread(empty_tray_image_path, 1)
        img2 = cv2.imread(aug_tray_img_w_pill_aug, 1)

        diff = cv2.absdiff(img1, img2)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        if display_results:
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
            plt.show()

        cv2.imwrite(save_path, diff)

    def main(self, create_class_dirs: bool, first_phase, second_phase, third_phase, fourth_phase) -> None:
        if create_class_dirs:
            create_directories(self.all_classes, self.tray_images, self.tray_images)

        for class_name in tqdm(self.all_classes, total=len(self.all_classes), desc="Processing classes"):
            # Collects the class folders
            class_dir = os.path.join(self.tray_images, class_name)
            class_dir_tray_images_aug = os.path.join(self.tray_images_aug, class_name)
            class_dir_tray_images_aug_w_med = os.path.join(self.tray_images_aug_w_med, class_name)
            class_dir_tray_images_aug_w_med_aug = os.path.join(self.tray_images_aug_w_med_aug, class_name)
            class_dir_diff_images = os.path.join(self.diff_images, class_name)

            # Create the directories if they don't exist
            os.makedirs(class_dir_tray_images_aug, exist_ok=True)
            os.makedirs(class_dir_tray_images_aug_w_med, exist_ok=True)
            os.makedirs(class_dir_tray_images_aug_w_med_aug, exist_ok=True)
            os.makedirs(class_dir_diff_images, exist_ok=True)

            # Collect file names
            image_files = os.listdir(class_dir)
            image_files_tray_images_aug = os.listdir(class_dir_tray_images_aug)
            image_files_tray_images_aug_w_med = os.listdir(class_dir_tray_images_aug_w_med)
            image_files_tray_images_aug_w_med_aug = os.listdir(class_dir_tray_images_aug_w_med_aug)

            if first_phase:
                # First phase, we are augmenting empty tray images
                for _, image_file in tqdm(enumerate(image_files), total=len(image_files),
                                          desc='Augmenting empty tray images'):
                    full_path_image = os.path.join(class_dir, image_file)
                    full_path_image_aug = os.path.join(class_dir_tray_images_aug, image_file)

                    copy_original_images(full_path_image, full_path_image_aug)
                    change_white_balance(full_path_image, full_path_image_aug,
                                         domain=(self.cfg.wb_low_thr, self.cfg.wb_high_thr))
                    gaussian_smooth(image_path=full_path_image, aug_path=full_path_image_aug,
                                    kernel=(self.cfg.kernel_size, self.cfg.kernel_size))
                    change_brightness(full_path_image, full_path_image_aug,
                                      exposure_factor=random.uniform(self.cfg.brightness_low_thr,
                                                                     self.cfg.brightness_high_thr))
                    rotate_image(full_path_image, full_path_image_aug,
                                 angle=random.randint(self.cfg.rotate_low_thr, self.cfg.rotate_high_thr))
                    shift_image(full_path_image, full_path_image_aug,
                                shift_x=self.cfg.shift_low_thr, shift_y=self.cfg.shift_high_thr)
                    self.flip_image(full_path_image, full_path_image_aug, flip_direction="horizontal")
                    self.flip_image(full_path_image, full_path_image_aug, flip_direction="vertical")

            if second_phase:
                # Second phase, we are placing pills on the augmented images
                for _, image_file in tqdm(enumerate(image_files_tray_images_aug),
                                          total=len(image_files_tray_images_aug),
                                          desc="Placing pills on the augmented tray images"):
                    full_path_tray_image_aug = os.path.join(class_dir_tray_images_aug, image_file)
                    full_path_tray_image_aug_w_med = os.path.join(class_dir_tray_images_aug_w_med, image_file)

                    pill_image_file = random.choice(os.listdir(self.medicine_images))
                    pill_image_path = os.path.join(self.medicine_images, pill_image_file)
                    pill_mask_path = os.path.join(self.medicine_masks, pill_image_file)
                    self.place_medicine_on_tray(
                        pill_image_path, pill_mask_path, full_path_tray_image_aug, full_path_tray_image_aug_w_med,
                        scaling_factor=self.cfg.scale_pill_img)

            if third_phase:
                for _, image_file in tqdm(enumerate(image_files_tray_images_aug_w_med),
                                          total=len(image_files_tray_images_aug_w_med),
                                          desc='Augmenting augmented tray images with pills on them'):
                    full_path_image_aug = os.path.join(class_dir_tray_images_aug_w_med, image_file)
                    full_path_tray_aug = os.path.join(class_dir_tray_images_aug_w_med_aug, image_file)

                    change_white_balance(full_path_image_aug, full_path_tray_aug, domain=(0.9, 1.0))
                    change_brightness(full_path_image_aug, full_path_tray_aug,
                                      exposure_factor=random.uniform(0.5, 1.2))

            if fourth_phase:
                for img_file_tray_img_aug in tqdm(image_files_tray_images_aug, total=len(image_files_tray_images_aug),
                                                  desc="Selecting empty tray images"):
                    if re.search(r"\d+\.png$", img_file_tray_img_aug) is None:
                        # Ignore file names without a number before .png extension
                        continue

                    for img_file_tray_img_aug_w_med_aug in tqdm(image_files_tray_images_aug_w_med_aug,
                                                                total=len(image_files_tray_images_aug_w_med_aug),
                                                                desc=
                                                                "Selecting augmented tray images with pills on them"):
                        if re.search(r"\d+\.png$", img_file_tray_img_aug_w_med_aug) is None:
                            # Ignore file names without a number before .png extension
                            continue

                        if img_file_tray_img_aug.split(".")[0] in img_file_tray_img_aug_w_med_aug.split(".")[0]:
                            full_path_diff_images = os.path.join(class_dir_diff_images, img_file_tray_img_aug_w_med_aug)
                            full_path_tray_image_aug = os.path.join(class_dir_tray_images_aug, img_file_tray_img_aug)
                            full_path_tray_image_aug_w_med_aug = \
                                os.path.join(class_dir_tray_images_aug_w_med_aug, img_file_tray_img_aug_w_med_aug)

                            self.subtract_images(empty_tray_image_path=full_path_tray_image_aug,
                                                 aug_tray_img_w_pill_aug=full_path_tray_image_aug_w_med_aug,
                                                 display_results=False,
                                                 save_path=full_path_diff_images)


if __name__ == "__main__":
    try:
        aug = AugmentTrayImages()
        aug.menu()
        aug.main(create_class_dirs=False,
                 first_phase=False,
                 second_phase=False,
                 third_phase=False,
                 fourth_phase=True)
    except KeyboardInterrupt as kie:
        logging.error("Keyboard interrupt")
