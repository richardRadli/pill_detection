import colorama
import concurrent.futures
import gc
import random

from tqdm import tqdm

from augmentation_utils import (change_brightness, gaussian_smooth, rotate_image_segmentation, shift_image_segmentation,
                                zoom_in_object_segmentation, change_white_balance, change_background_dtd)
from config.json_config import json_config_selector
from config.dataset_paths_selector import dataset_images_path_selector
from utils.utils import file_reader, load_config_json


class AugmentCUREDataset:
    def __init__(self, operation):
        colorama.init()
        self.cfg_aug = (
            load_config_json(
                json_schema_filename=json_config_selector("augmentation").get("schema"),
                json_filename=json_config_selector("augmentation").get("config")
            )
        )
        self.operation = operation
        dataset_name = self.cfg_aug.get("dataset_name")
        
        # Train data paths
        self.training_images = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("images")
        )
        self.training_annotations = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("segmentation_labels")
        )
        self.train_masks = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("mask_images")
        )
        self.train_aug_img_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("aug_images")
        )
        self.train_aug_annotation_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("aug_yolo_labels")
        )
        self.train_aug_mask_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("train").get("aug_mask_images")
        )

        # Valid data paths
        self.valid_images = (
            dataset_images_path_selector(dataset_name=dataset_name).get("valid").get("images")
        )
        self.valid_annotations = (
            dataset_images_path_selector(dataset_name=dataset_name).get("valid").get("segmentation_labels")
        )
        self.valid_masks = (
            dataset_images_path_selector(dataset_name=dataset_name).get("valid").get("mask_images")
        )
        self.valid_aug_img_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("valid").get("aug_images")
        )
        self.valid_aug_annotation_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("valid").get("aug_yolo_labels")
        )
        self.valid_aug_mask_path = (
            dataset_images_path_selector(dataset_name=dataset_name).get("valid").get("aug_mask_images")
        )

        self.background_images = (
            dataset_images_path_selector(dataset_name="dtd").get("dataset_path"))

    def path_select(self, operation):
        """
        
        Args
             operation:
        
        Returns:
        
        """

        image = self.training_images if operation == "train" else self.valid_images
        anno = self.training_annotations if operation == "train" else self.valid_annotations
        mask = self.train_masks if operation == "train" else self.valid_masks

        aug_image = self.train_aug_img_path if operation == "train" else self.valid_aug_img_path
        aug_anno = self.train_aug_annotation_path if operation == "train" else self.valid_aug_annotation_path
        aug_mask = self.train_aug_mask_path if operation == "train" else self.valid_aug_mask_path

        image_files = file_reader(image, "jpg")
        annotation_files = file_reader(anno, "txt")
        mask_files = file_reader(mask, "jpg")

        return image_files, annotation_files, mask_files, aug_image, aug_anno, aug_mask

    def process_image(self, image_path: str, anno_path: str, mask_path: str, aug_image_path: str, aug_anno_path: str,
                      aug_mask_path: str) -> None:
        """
        This function executes the various image augmentation operations.

        Args:
             image_path: Path to the images.
             anno_path: Path to the annotations.
             mask_path: Path to the mask images.
             aug_image_path: Path to the augmented images.
             aug_anno_path: Path to the augmented annotations.
             aug_mask_path: Path to the augmented mask images.

        Returns:
             None
        """

        change_white_balance(image_path=image_path,
                             annotation_path=anno_path,
                             mask_path=mask_path,
                             aug_img_path=aug_image_path,
                             aug_annotation_path=aug_anno_path,
                             aug_mask_path=aug_mask_path,
                             domain=(self.cfg_aug.get("wb_low_thr"), self.cfg_aug.get("wb_high_thr")))
        gaussian_smooth(image_path=image_path,
                        annotation_path=anno_path,
                        mask_path=mask_path,
                        aug_img_path=aug_image_path,
                        aug_annotation_path=aug_anno_path,
                        aug_mask_path=aug_mask_path,
                        kernel=(self.cfg_aug.get("kernel_size"), self.cfg_aug.get("kernel_size")))
        change_brightness(image_path=image_path,
                          annotation_path=anno_path,
                          mask_path=mask_path,
                          aug_img_path=aug_image_path,
                          aug_annotation_path=aug_anno_path,
                          aug_mask_path=aug_mask_path,
                          exposure_factor=random.uniform(
                              self.cfg_aug.get("brightness_low_thr"), self.cfg_aug.get("brightness_high_thr")
                          ))
        rotate_image_segmentation(image_path=image_path,
                                  annotation_path=anno_path,
                                  mask_path=mask_path,
                                  aug_img_path=aug_image_path,
                                  aug_annotation_path=aug_anno_path,
                                  aug_mask_path=aug_mask_path,
                                  angle=self.cfg_aug.get("rotate_1"))
        rotate_image_segmentation(image_path=image_path,
                                  annotation_path=anno_path,
                                  mask_path=mask_path,
                                  aug_img_path=aug_image_path,
                                  aug_annotation_path=aug_anno_path,
                                  aug_mask_path=aug_mask_path,
                                  angle=self.cfg_aug.get("rotate_2"))
        shift_image_segmentation(image_path=image_path,
                                 annotation_path=anno_path,
                                 mask_path=mask_path,
                                 aug_img_path=aug_image_path,
                                 aug_annotation_path=aug_anno_path,
                                 aug_mask_path=aug_mask_path,
                                 shift_x=self.cfg_aug.get("shift_x"),
                                 shift_y=self.cfg_aug.get("shift_y")
                                 )
        zoom_in_object_segmentation(image_path=image_path,
                                    annotation_path=anno_path,
                                    mask_path=mask_path,
                                    aug_img_path=aug_image_path,
                                    aug_annotation_path=aug_anno_path,
                                    aug_mask_path=aug_mask_path,
                                    crop_size=self.cfg_aug.get("zoom"))

    def change_bg(self, image_path, anno_path, mask_path, aug_img_path, aug_anno_path):
        """
        Args:
             image_path:
             anno_path:
             mask_path:
             aug_img_path:
             aug_anno_path:
             
        Returns:

        """

        change_background_dtd(image_path=image_path,
                              mask_path=mask_path,
                              annotation_path=anno_path,
                              aug_image_path=aug_img_path,
                              aug_annotation_path=aug_anno_path,
                              backgrounds_path=self.background_images)

    def main(self):
        image_files, annotation_files, mask_files, aug_image, aug_anno, aug_mask = self.path_select(self.operation)

        assert len(image_files) == len(annotation_files) == len(mask_files)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg_aug.get("max_workers")) as executor:
            futures = []
            for image_path, annotation_path, mask_path in zip(image_files, annotation_files, mask_files):
                future = \
                    executor.submit(
                        self.process_image, image_path, annotation_path, mask_path, aug_image, aug_anno, aug_mask
                    )
                futures.append(future)

            for _ in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures),
                          desc=colorama.Fore.BLUE + "Augmenting images"):
                pass

        assert len(image_files) == len(annotation_files) == len(mask_files)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg_aug.get("max_workers")) as executor:
            futures = []
            for image_path, annotation_path, mask_path in zip(image_files, annotation_files, mask_files):
                future = executor.submit(self.change_bg, image_path, annotation_path, mask_path, aug_image, aug_anno)
                futures.append(future)
                gc.collect()

            for _ in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures),
                          desc=colorama.Fore.CYAN + "Changing backgrounds"):
                pass


if __name__ == "__main__":
    aug_cure = AugmentCUREDataset(operation="train")
    aug_cure.main()