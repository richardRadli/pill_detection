import cv2
import logging
import numpy as np
import os

from glob import glob
from typing import List, Tuple
from tqdm import tqdm
from config.config import ConfigAugment
from config.const import CONST
from config.logger_setup import setup_logger

cfg = ConfigAugment().parse()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- R E A D   I M A G E   T O   L I S T ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def read_image_to_list() -> Tuple[np.ndarray, List[str]]:
    """
    Reads all images in a directory and returns them as a numpy array.

    :return: A tuple containing the numpy array of images and a list of their corresponding file names.
    """

    images = sorted(glob(CONST.dir_train_images + "*.png"))
    file_names = []
    images_list = []

    for idx, img_path in tqdm(enumerate(images), desc="Reading images", total=len(images)):
        file_names.append(os.path.basename(img_path))
        train_img = cv2.imread(img_path, 1)
        images_list.append(train_img)
    return np.array(images_list), file_names


def augment_data(training_images, file_names):
    augmented_image = []
    f_names = []

    for num in tqdm(range(0, training_images.shape[0])):
        for i in range(0, cfg.augmentation_factor):
            if cfg.use_original:
                augmented_image.append(training_images[num])
                f_names.append(file_names[num])
            if cfg.use_gaussian_noise:
                augmented_image.append(cv2.GaussianBlur(training_images[num], (cfg.kernel_size, cfg.kernel_size), 0))
                f_names.append(file_names[num])
            if cfg.use_horizontal_flip:
                augmented_image.append(cv2.flip(training_images[num], 1))
                f_names.append(file_names[num])
            if cfg.use_vertical_flip:
                augmented_image.append(cv2.flip(training_images[num], 0))
                f_names.append(file_names[num])
        break

    return np.array(augmented_image), f_names


def do_augmentation():
    """

    :return:
    """
    setup_logger()

    train_imgs, file_names = read_image_to_list()
    aug_imgs, f_names = augment_data(train_imgs, file_names)

    logging.info(f'\nNumber of images: {aug_imgs.shape[0]}')

    for idx, (value_img, f_name) in tqdm(enumerate(zip(aug_imgs, f_names))):
        cv2.imshow(os.path.join(CONST.dir_aug_img, str(idx) + f_name), value_img)
        cv2.waitKey()


if __name__ == "__main__":
    try:
        do_augmentation()
    except KeyboardInterrupt as kie:
        logging.error(kie)
