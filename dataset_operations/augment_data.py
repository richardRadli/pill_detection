import cv2
import logging
import numpy as np
import os
import random

from glob import glob
from tqdm import tqdm
from config.const import CONST
from config.logger_setup import setup_logger
from skimage.util import random_noise


def change_brightness(image, exposure_factor):
    image = image.astype(np.float32) / 255.0
    adjusted_image = image * exposure_factor
    adjusted_image = np.clip(adjusted_image, 0, 1)
    adjusted_image = (adjusted_image * 255).astype(np.uint8)

    return adjusted_image


def add_gaussian_noise(image):
    mean = 0
    std_dev = 20
    noisy_image = np.clip(image + np.random.normal(mean, std_dev, image.shape), 0, 255).astype(np.uint8)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, amount):
    noisy_image = random_noise(image, mode='s&p', salt_vs_pepper=salt_vs_pepper_ratio, amount=amount)
    noisy_image = (255 * noisy_image).astype(np.uint8)
    return noisy_image


def rotate_image(image, angle: int = 180):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def rotate_segmentation_annotations(annotations, image_width, image_height, angle):
    rotated_annotations = []
    for annotation in annotations:
        annotation_values = annotation.split(' ')
        class_id = int(annotation_values[0])
        mask_coords = [float(coord) for coord in annotation_values[1:] if coord]
        mask_coords = np.array(mask_coords, dtype=float)
        mask_coords = mask_coords.reshape((-1, 2))
        mask_coords[:, 0] *= image_width
        mask_coords[:, 1] *= image_height

        # Rotate the coordinates
        rotation_matrix = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), angle, 1)
        rotated_coords = np.matmul(np.hstack((mask_coords, np.ones((mask_coords.shape[0], 1)))), rotation_matrix.T)
        rotated_coords = rotated_coords[:, :2]

        # Convert the rotated coordinates back to YOLO format
        rotated_coords[:, 0] /= image_width
        rotated_coords[:, 1] /= image_height
        rotated_coords = rotated_coords.flatten().tolist()

        rotated_annotations.append([class_id] + rotated_coords)

    return rotated_annotations


def augment_images(input_img_dir, input_txt_dir, output_img_dir, output_txt_dir):
    salt_vs_pepper_ratio = 0.5
    noise_amount = 0.05

    image_paths = glob(os.path.join(input_img_dir, '*.png'))
    txt_paths = glob(os.path.join(input_txt_dir, "*.txt"))

    for idx, (image_path, annotations_path) in tqdm(enumerate(zip(image_paths, txt_paths)), total=len(image_paths),
                                                    desc="Image augmentation"):
        image = cv2.imread(image_path)

        with open(annotations_path, 'r') as f:
            annotations = f.readlines()

        rotated_image = rotate_image(image)
        image_width = image.shape[1]
        image_height = image.shape[0]
        output_image_path = os.path.join(output_img_dir, f"rotated_{os.path.basename(image_path)}")
        rotated_annotations = rotate_segmentation_annotations(annotations, image_width, image_height, 180)
        cv2.imwrite(output_image_path, rotated_image)
        output_annotations_path = os.path.join(output_txt_dir, f"rotated_{os.path.basename(annotations_path)}")

        with open(output_annotations_path, 'w') as f:
            for annotation in rotated_annotations:
                f.write(' '.join([str(coord) for coord in annotation]) + '\n')

        brightness_factor = random.uniform(0.5, 1.5)
        brightness_changed_image = change_brightness(image, brightness_factor)
        output_image_path = os.path.join(output_img_dir, f"brightness_{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, brightness_changed_image)
        output_annotations_path = os.path.join(output_txt_dir, f"brightness_{os.path.basename(annotations_path)}")
        with open(output_annotations_path, 'w') as f:
            f.writelines(annotations)

        smoothed_image = cv2.GaussianBlur(image, (7, 7), 0)
        output_image_path = os.path.join(output_img_dir, f"smoothed_{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, smoothed_image)
        output_annotations_path = os.path.join(output_txt_dir, f"smoothed_{os.path.basename(annotations_path)}")
        with open(output_annotations_path, 'w') as f:
            f.writelines(annotations)

        noisy_image = add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, noise_amount)
        output_image_path = os.path.join(output_img_dir, f"noisy_{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, noisy_image)
        output_annotations_path = os.path.join(output_txt_dir, f"noisy_{os.path.basename(annotations_path)}")
        with open(output_annotations_path, 'w') as f:
            f.writelines(annotations)


def do_augmentation():
    setup_logger()
    augment_images(input_img_dir="C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyi/full_img_size/splitted/"
                                 "train/images",
                   input_txt_dir="C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyi/full_img_size/splitted/"
                                 "train/labels",
                   output_img_dir=CONST.dir_aug_img,
                   output_txt_dir=CONST.dir_aug_labels)


if __name__ == "__main__":
    try:
        do_augmentation()
    except KeyboardInterrupt as kie:
        logging.error(kie)
