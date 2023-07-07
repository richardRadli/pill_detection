import logging
import os
import random
import re

from tqdm import tqdm

from config.config import ConfigAugmentation
from config.const import DATASET_PATH
from config.logger_setup import setup_logger
from augmentation_utils import gaussian_smooth, change_brightness, rotate_image, shift_image, abs_diff_images, \
    change_white_balance, copy_original_images, create_directories, flip_image, place_medicine_on_tray


class AugmentTrayImages:
    def __init__(self):
        setup_logger()

        self.cfg = ConfigAugmentation().parse()

        self.tray_images = DATASET_PATH.get_data_path("tray_original_images")
        self.tray_images_aug = DATASET_PATH.get_data_path("tray_images_aug")
        self.tray_images_aug_w_med = DATASET_PATH.get_data_path("tray_images_aug_w_med")
        self.tray_images_aug_w_med_aug = DATASET_PATH.get_data_path("tray_images_aug_w_med_aug")
        self.diff_images = DATASET_PATH.get_data_path("tray_diff_images")
        self.plots = DATASET_PATH.get_data_path("tray_plots")

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

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- M E N U --------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
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
    # ----------------------------------------------------- M A I N ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def main(self, create_class_dirs: bool, first_phase: bool, second_phase: bool, third_phase: bool,
             fourth_phase: bool) -> None:
        """
        Main function of the program. Executes the pipeline.

        Args:
            create_class_dirs (bool):
            first_phase (bool):
            second_phase (bool):
            third_phase (bool):
            fourth_phase (bool):
        """

        if create_class_dirs:
            create_directories(classes=self.all_classes, images_dir=self.tray_images, dataset_path=self.tray_images)

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

            # First phase: augmenting empty tray images
            if first_phase:
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
                    flip_image(image_path=full_path_image, aug_path=full_path_image_aug, flip_direction="horizontal")
                    flip_image(image_path=full_path_image, aug_path=full_path_image_aug, flip_direction="vertical")

            # Second phase: placing pills on the augmented images
            if second_phase:
                for _, image_file in tqdm(enumerate(image_files_tray_images_aug),
                                          total=len(image_files_tray_images_aug),
                                          desc="Placing pills on the augmented tray images"):
                    full_path_tray_image_aug = os.path.join(class_dir_tray_images_aug, image_file)
                    full_path_tray_image_aug_w_med = os.path.join(class_dir_tray_images_aug_w_med, image_file)

                    pill_image_file = random.choice(os.listdir(self.medicine_images))
                    pill_image_path = os.path.join(self.medicine_images, pill_image_file)
                    pill_mask_path = os.path.join(self.medicine_masks, pill_image_file)
                    place_medicine_on_tray(pill_image_path, pill_mask_path, full_path_tray_image_aug,
                                           full_path_tray_image_aug_w_med, scaling_factor=self.cfg.scale_pill_img)

            # Third phase: augmentation of the previously generated images
            if third_phase:
                for _, image_file in tqdm(enumerate(image_files_tray_images_aug_w_med),
                                          total=len(image_files_tray_images_aug_w_med),
                                          desc='Augmenting augmented tray images with pills on them'):
                    full_path_image_aug = os.path.join(class_dir_tray_images_aug_w_med, image_file)
                    full_path_tray_aug = os.path.join(class_dir_tray_images_aug_w_med_aug, image_file)

                    change_white_balance(full_path_image_aug, full_path_tray_aug, domain=(self.cfg.wb_low_thr_2nd_aug,
                                                                                          self.cfg.wb_high_thr_2nd_aug))
                    change_brightness(full_path_image_aug, full_path_tray_aug,
                                      exposure_factor=random.uniform(self.cfg.brightness_low_thr_2nd_aug,
                                                                     self.cfg.brightness_high_thr_2nd_aug))

            # Forth phase: Creating difference images
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

                            abs_diff_images(empty_tray_image_path=full_path_tray_image_aug,
                                            aug_tray_img_w_pill_aug=full_path_tray_image_aug_w_med_aug,
                                            save_results=True,
                                            save_path=full_path_diff_images,
                                            save_plots_path=self.plots)


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
