import numpy as np
import os
import random
import shutil

from tqdm import tqdm

from config.dataset_paths_selector import dataset_images_path_selector


def copy_images():
    np.random.seed(1234)

    test_images_path = dataset_images_path_selector("cure").get("customer").get("customer_images")
    images_list = os.listdir(test_images_path)

    test_masks_path = dataset_images_path_selector("cure").get("customer").get("customer_mask_images")
    masks_list = os.listdir(test_masks_path)

    assert len(images_list) == len(masks_list)

    test_size = int(0.2 * len(images_list))

    selected_images = random.choices(images_list, k=test_size)
    dst_folder_images = dataset_images_path_selector("cure").get("test").get("test_images")
    dst_folder_masks = dataset_images_path_selector("cure").get("test").get("test_masks")

    src_files_images = [os.path.join(test_images_path, images_list_i) for images_list_i in selected_images]
    src_files_masks = [os.path.join(test_masks_path, masks_list_i) for masks_list_i in selected_images]

    assert len(src_files_images) == len(src_files_masks)

    for src_image, src_mask in tqdm(zip(src_files_images, src_files_masks),
                                    total=len(src_files_images),
                                    desc="Copying images"):
        shutil.copy(src_image, dst_folder_images)
        shutil.copy(src_mask, dst_folder_masks)


if __name__ == '__main__':
    copy_images()
