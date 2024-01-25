import concurrent.futures
import random

from glob import glob
from tqdm import tqdm

from augmentation_utils import (change_brightness, gaussian_smooth, rotate_image, shift_image, zoom_in_object,
                                change_white_balance, change_background_dtd)
from config.config_selector import dataset_images_path_selector


class AugmentCUREDataset:
    def __init__(self):
        self.training_images = dataset_images_path_selector().get("cure").get("train_images")
        self.training_annotations = dataset_images_path_selector().get("cure").get("train_yolo_labels")
        self.train_masks = dataset_images_path_selector().get("cure").get("train_mask_images")

        self.valid_images = dataset_images_path_selector().get("cure").get("valid_images")
        self.valid_annotations = dataset_images_path_selector().get("cure").get("valid_yolo_labels")
        self.valid_masks = dataset_images_path_selector().get("cure").get("valid_mask_images")

        self.background_images = dataset_images_path_selector().get("dtd").get("dataset_path")

    @staticmethod
    def process_image(image_path, anno_path, mask_path):
        change_white_balance(image_path=image_path,
                             annotation_path=anno_path,
                             mask_path=mask_path,
                             domain=(0.7, 1.2))
        gaussian_smooth(image_path=image_path,
                        annotation_path=anno_path,
                        mask_path=mask_path,
                        kernel=(7, 7))
        change_brightness(image_path=image_path,
                          annotation_path=anno_path,
                          mask_path=mask_path,
                          exposure_factor=random.uniform(0.5, 1.5))
        rotate_image(image_path=image_path,
                     annotation_path=anno_path,
                     mask_path=mask_path,
                     angle=180)
        rotate_image(image_path=image_path,
                     annotation_path=anno_path,
                     mask_path=mask_path,
                     angle=-180)
        shift_image(image_path=image_path,
                    annotation_path=anno_path,
                    mask_path=mask_path,
                    )
        zoom_in_object(image_path=image_path,
                       annotation_path=anno_path,
                       mask_path=mask_path,
                       crop_size=1500)

    def change_bg(self, image_path, anno_path, mask_path):
        change_background_dtd(image_path=image_path,
                              mask_path=mask_path,
                              aug_path=image_path,
                              annotation_path=anno_path,
                              backgrounds_path=self.background_images)

    def main(self, operation: str):
        image = self.training_images if operation == "train" else self.valid_images
        anno = self.training_annotations if operation == "train" else self.valid_annotations
        mask = self.train_masks if operation == "train" else self.valid_masks

        image_files = sorted(glob(image + "/*.jpg"))
        annotation_files = sorted(glob(anno + "/*.txt"))
        mask_files = sorted(glob(mask + "/*.jpg"))

        assert len(image_files) == len(annotation_files) == len(mask_files)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        #     futures = []
        #     for image_path, annotation_path, mask_path in zip(image_files, annotation_files, mask_files):
        #         future = executor.submit(self.process_image, image_path, annotation_path, mask_path)
        #         futures.append(future)
        #
        #     for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        #         pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for image_path, annotation_path, mask_path in zip(image_files, annotation_files, mask_files):
                future = executor.submit(self.change_bg, image_path, annotation_path, mask_path)
                futures.append(future)

            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass


if __name__ == "__main__":
    aug_cure = AugmentCUREDataset()
    aug_cure.main(operation="train")
