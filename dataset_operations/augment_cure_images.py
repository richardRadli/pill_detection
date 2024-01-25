import os
import random

from glob import glob
from tqdm import tqdm

from augmentation_utils import change_brightness, gaussian_smooth, rotate_image, create_directories, shift_image, \
    zoom_in_object, change_white_balance, change_background_dtd
from config.config_selector import dataset_images_path_selector
from utils.utils import setup_logger


class AugmentCUREDataset:
    def __init__(self):
        self.training_images = dataset_images_path_selector().get("cure").get("train_images")
        self.training_annotations = dataset_images_path_selector().get("cure").get("train_yolo_labels")
        self.train_masks = dataset_images_path_selector().get("cure").get("train_mask_images")

        self.valid_images = dataset_images_path_selector().get("cure").get("valid_images")
        self.valid_annotations = dataset_images_path_selector().get("cure").get("valid_yolo_labels")
        self.valid_masks = dataset_images_path_selector().get("cure").get("valid_mask_images")

        self.background_images = dataset_images_path_selector().get("dtd").get("dataset_path")

    def main(self):
        image_files = sorted(glob(self.training_images + "/*.jpg"))
        annotation_files = sorted(glob(self.training_annotations + "/*.txt"))
        mask_files = sorted(glob(self.train_masks + "/*.jpg"))

        for idx, (image_path, anno_path) in enumerate(zip(image_files, annotation_files)):
            # change_white_balance(image_path=image_path,
            #                      annotation_path=anno_path,
            #                      aug_path=image_path,
            #                      domain=(0.7, 1.2))
            # change_white_balance(image_path=image_path,
            #                      annotation_path=anno_path,
            #                      aug_path=image_path,
            #                      domain=(0.7, 1.2))
            # gaussian_smooth(image_path=image_path,
            #                 annotation_path=anno_path,
            #                 aug_path=image_path,
            #                 kernel=(7, 7))
            # change_brightness(image_path=image_path,
            #                   annotation_path=anno_path,
            #                   aug_path=image_path,
            #                   exposure_factor=random.uniform(0.5, 1.5))
            # change_brightness(image_path=image_path,
            #                   annotation_path=anno_path,
            #                   aug_path=image_path,
            #                   exposure_factor=random.uniform(0.5, 1.5))
            # rotate_image(image_path=image_path,
            #              annotation_path=anno_path,
            #              aug_path=image_path,
            #              angle=180)
            # rotate_image(image_path=image_path,
            #              annotation_path=anno_path,
            #              aug_path=image_path,
            #              angle=-180)
            # shift_image(image_path=image_path,
            #             annotation_path=anno_path,
            #             aug_path=image_path,
            #             )
            zoom_in_object(image_path=image_path,
                           annotation_path=anno_path,
                           aug_path=image_path,
                           crop_size=1500)
            break

        # for idx, (image_path, mask_path, anno_path) in enumerate(zip(image_files, mask_files, annotation_files)):
        #     change_background_dtd(image_path=image_path,
        #                           mask_path=mask_path,
        #                           aug_path=image_path,
        #                           annotation_path=anno_path,
        #                           backgrounds_path=self.background_images)
        #     break


if __name__ == "__main__":
    aug_cure = AugmentCUREDataset()
    aug_cure.main()
