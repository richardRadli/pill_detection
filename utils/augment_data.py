import cv2
import numpy as np
import os

from tqdm import tqdm
from config import ConfigAugment
from const import CONST
from utils import read_image_to_list

cfg = ConfigAugment().parse()


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

    train_imgs, file_names = read_image_to_list()
    aug_imgs, f_names = augment_data(train_imgs, file_names)

    print(f'\nNumber of images: {aug_imgs.shape[0]}')

    for idx, (value_img, f_name) in tqdm(enumerate(zip(aug_imgs, f_names))):
        cv2.imshow(os.path.join(CONST.dir_aug_img, str(idx) + f_name), value_img)
        cv2.waitKey()


if __name__ == "__main__":
    try:
        do_augmentation()
    except KeyboardInterrupt as kie:
        print(kie)
