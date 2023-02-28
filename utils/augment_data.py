import cv2
import numpy as np
import os
import tensorflow_addons as tfa

from tqdm import tqdm

from config import ConfigAugment
from const import CONST
from utils.utils import read_image_and_mask

cfg = ConfigAugment().parse()


def augment_data(training_images, training_masks):
    """

    param training_images:
    param training_masks:
    :return:
    """

    augmented_image = []
    augmented_masks = []

    for num in tqdm(range(0, training_images.shape[0])):
        for i in range(0, cfg.augmentation_factor):
            if cfg.use_original:
                augmented_image.append(training_images[num])
                augmented_masks.append(training_masks[num])

            if cfg.use_random_rotation:
                augmented_image.append(
                    tfa.image.rotate(training_images[num], cfg.rotation_angle))
                augmented_masks.append(
                    tfa.image.rotate(training_masks[num], cfg.rotation_angle))

    return np.array(augmented_image), np.array(augmented_masks)


def do_augmentation():
    train_imgs, train_masks = read_image_and_mask()
    aug_imgs, aug_masks = augment_data(train_imgs, train_masks)

    for idx, (value_img, value_mask) in enumerate(zip(aug_imgs, aug_masks)):
        cv2.imwrite(os.path.join(CONST.dir_aug_img, str(idx) + ".png"), value_img)
        cv2.imwrite(os.path.join(CONST.dir_aug_mask, str(idx) + ".png"), value_mask)


if __name__ == "__main__":
    do_augmentation()
