import colorama
import concurrent.futures
import numpy as np
import random

from PIL import Image
from tqdm import tqdm
from typing import Tuple, List

from augmentation_utils import (change_brightness, gaussian_smooth, rotate_image_segmentation, shift_image_segmentation,
                                zoom_in_object_segmentation, change_white_balance, change_background_dtd)
from config.json_config import json_config_selector
from config.dataset_paths_selector import dataset_images_path_selector
from utils.utils import file_reader, load_config_json


class AugmentCUREDataset:
    def __init__(self):
        colorama.init()

        # Load configuration for augmentation settings
        self.cfg_aug = (
            load_config_json(
                json_schema_filename=json_config_selector("augmentation").get("schema"),
                json_filename=json_config_selector("augmentation").get("config")
            )
        )

        # Load dataset paths
        dataset_name = self.cfg_aug.get("dataset_name")
        self.training_images_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("reference").get("reference_images")
        )
        self.train_masks_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("reference").get("reference_mask_images")
        )
        self.training_annotations_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("reference").get("reference_labels")
        )
        self.train_aug_img_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("images")
        )
        self.train_aug_annotation_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("yolo_labels")
        )
        self.train_aug_mask_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("mask_images")
        )
        self.background_images = (
            dataset_images_path_selector(dataset_name="dtd").get("dataset_path")
        )

    @staticmethod
    def path_select(training_images, training_annotations, train_masks) -> Tuple[List[str], List[str], List[str]]:
        """
        Selects and returns paths for images, annotations, and masks.

        Args:
            training_images (str): Path to the directory containing image files.
            training_annotations (str): Path to the directory containing annotation files.
            train_masks (str): Path to the directory containing mask files.

        Returns:
            Tuple[List[str], List[str], List[str]]: Lists of file paths for images, annotations, and masks.
        """

        image_files = file_reader(training_images, "jpg")
        annotation_files = file_reader(training_annotations, "txt")
        mask_files = file_reader(train_masks, "jpg")

        return image_files, annotation_files, mask_files

    def process_image(self, image_path: str, anno_path: str, mask_path: str, aug_image_path: str, aug_anno_path: str,
                      aug_mask_path: str) -> None:
        """
        Perform augmentations like white balance, Gaussian smoothing, brightness, rotation, shift, and zoom
        Each function call should have unique parameters or randomized values per round for variation

        Args:
             image_path: Path to the image.
             anno_path: Path to the annotations.
             mask_path: Path to the mask images.
             aug_image_path: Path to save augmented images.
             aug_anno_path: Path to save augmented annotations.
             aug_mask_path: Path to save augmented mask images.
        """

        change_white_balance(
            image_path=image_path,
            annotation_path=anno_path,
            mask_path=mask_path,
            aug_img_path=aug_image_path,
            aug_annotation_path=aug_anno_path,
            aug_mask_path=aug_mask_path,
            domain=(self.cfg_aug.get("wb_low_thr"), self.cfg_aug.get("wb_high_thr"))
        )
        gaussian_smooth(
            image_path=image_path,
            annotation_path=anno_path,
            mask_path=mask_path,
            aug_img_path=aug_image_path,
            aug_annotation_path=aug_anno_path,
            aug_mask_path=aug_mask_path,
            kernel=random.choice([3, 5, 7])
        )
        change_brightness(
            image_path=image_path,
            annotation_path=anno_path,
            mask_path=mask_path,
            aug_img_path=aug_image_path,
            aug_annotation_path=aug_anno_path,
            aug_mask_path=aug_mask_path,
            exposure_factor=random.uniform(
                self.cfg_aug.get("brightness_low_thr"), self.cfg_aug.get("brightness_high_thr")
            )
        )
        rotate_image_segmentation(
            image_path=image_path,
            annotation_path=anno_path,
            mask_path=mask_path,
            aug_img_path=aug_image_path,
            aug_annotation_path=aug_anno_path,
            aug_mask_path=aug_mask_path,
            angle=random.randint(self.cfg_aug.get("rotate_1"), self.cfg_aug.get("rotate_2"))
        ),
        shift_image_segmentation(
            image_path=image_path,
            annotation_path=anno_path,
            mask_path=mask_path,
            aug_img_path=aug_image_path,
            aug_annotation_path=aug_anno_path,
            aug_mask_path=aug_mask_path,
            shift_x=random.randint(self.cfg_aug.get("shift_x_1"), self.cfg_aug.get("shift_x_2")),
            shift_y=random.randint(self.cfg_aug.get("shift_y_1"), self.cfg_aug.get("shift_y_2")),
        )
        zoom_in_object_segmentation(
            image_path=image_path,
            annotation_path=anno_path,
            mask_path=mask_path,
            aug_img_path=aug_image_path,
            aug_annotation_path=aug_anno_path,
            aug_mask_path=aug_mask_path,
            crop_size=random.randint(self.cfg_aug.get("zoom_1"), self.cfg_aug.get("zoom_2"))
        )

    def change_bg(self, image_path: str, anno_path: str, mask_path: str) -> None:
        """
        Changes the background of the given image based on the specified mask.

        Args:
            image_path (str): Path to the input image file.
            anno_path (str): Path to the annotation file associated with the image.
            mask_path (str): Path to the mask file to be used for background change.

        Returns:
            None
        """

        change_background_dtd(
            image_path=image_path,
            mask_path=mask_path,
            annotation_path=anno_path,
            backgrounds_path=self.background_images
        )

    def aug(self) -> None:
        """
        Augments images by applying transformations to reach a target number of augmented images.
        Loads image, annotation, and mask files, calculates the required number of augmentation cycles,
        and applies augmentations using concurrent processing.

        Returns:
            None
        """
        image_files, annotation_files, mask_files = (
            self.path_select(self.training_images_path, self.training_annotations_path, self.train_masks_path)
        )

        assert len(image_files) == len(annotation_files) == len(mask_files)

        original_images_count = len(image_files)
        target_augmentations = self.cfg_aug.get("number_of_aug_images")
        augmentations_per_image = 6
        num_aug_images_one_cycle = original_images_count * augmentations_per_image

        total_augmentation_cycles = int(np.round(target_augmentations / num_aug_images_one_cycle))
        final_image_count = original_images_count * augmentations_per_image * total_augmentation_cycles

        print(f"Total number of images in the end: {final_image_count * 2}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.cfg_aug.get("max_workers")) as executor:
            futures = []

            for image_path, annotation_path, mask_path in zip(image_files, annotation_files, mask_files):
                for i in range(total_augmentation_cycles):
                    futures.append(
                        executor.submit(
                            self.process_image,
                            image_path=image_path,
                            anno_path=annotation_path,
                            mask_path=mask_path,
                            aug_image_path=self.train_aug_img_path,
                            aug_anno_path=self.train_aug_annotation_path,
                            aug_mask_path=self.train_aug_mask_path
                        )
                    )

            # Wait for all tasks to complete
            for future in tqdm(futures, desc="Augmenting"):
                future.result()

    def change_background(self) -> None:
        """
        Changes the backgrounds of all augmented images using masks.

        Loads augmented image, annotation, and mask files, and applies
        background change using concurrent processing.

        Returns:
            None
        """

        aug_image_files, aug_annotation_files, aug_mask_files = (
            self.path_select(
                self.train_aug_img_path,
                self.train_aug_annotation_path,
                self.train_aug_mask_path
            )
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.cfg_aug.get("max_workers")) as executor:
            futures = []
            for image_path, annotation_path, mask_path in zip(aug_image_files, aug_annotation_files, aug_mask_files):
                if not self.is_valid_image(image_path):
                    print(f"Skipping corrupt image: {image_path}")
                    continue

                future = executor.submit(
                    self.change_bg, image_path, annotation_path, mask_path
                )
                futures.append(future)

            for _ in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures),
                          desc=colorama.Fore.CYAN + "Changing backgrounds"):
                pass

    @staticmethod
    def is_valid_image(image_path: str) -> bool:
        """
        Check if the image is a valid, non-corrupt JPEG.

        Args:
            image_path: Path of the image.

        """

        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError) as e:
            return False

    def main(self) -> None:
        """
        Executes the main processing pipeline: image augmentation followed by background change.

        This function first calls the `aug` method to augment images, then
        calls `change_background` to alter the backgrounds of the augmented images.

        Returns:
            None
        """

        self.aug()
        self.change_background()


if __name__ == "__main__":
    try:
        aug_cure = AugmentCUREDataset()
        aug_cure.main()
    except KeyboardInterrupt:
        print(f"Caught keyboard interrupt. Exiting...")
