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
    """

    :param image:
    :param exposure_factor:
    :return:
    """

    image = image.astype(np.float32) / 255.0
    adjusted_image = image * exposure_factor
    adjusted_image = np.clip(adjusted_image, 0, 1)
    adjusted_image = (adjusted_image * 255).astype(np.uint8)

    return adjusted_image


def add_gaussian_noise(image):
    """

    :param image:
    :return:
    """

    mean = 0
    std_dev = 20
    noisy_image = np.clip(image + np.random.normal(mean, std_dev, image.shape), 0, 255).astype(np.uint8)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, amount):
    """

    :param image:
    :param salt_vs_pepper_ratio:
    :param amount:
    :return:
    """

    noisy_image = random_noise(image, mode='s&p', salt_vs_pepper=salt_vs_pepper_ratio, amount=amount)
    noisy_image = (255 * noisy_image).astype(np.uint8)
    return noisy_image


def rotate_image(image, angle: int = 180):
    """

    :param image:
    :param angle:
    :return:
    """

    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def scale_image(image: np.ndarray, scale_percent: int) -> np.ndarray:
    """
    Scale an image by a given factor and save it to the output path.

    :param image: The path to the input image file.
    :param scale_percent: The factor by which to scale the image.
    """

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    scaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return scaled_image


def rotate_segmentation_annotations(annotations, image_width, image_height, angle):
    """

    :param annotations:
    :param image_width:
    :param image_height:
    :param angle:
    :return:
    """

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


def save_files(image_path, output_img_dir, output_txt_dir, annotations, image_to_save, type_of_image: str) -> None:
    """

    :param image_path:
    :param output_img_dir:
    :param output_txt_dir:
    :param annotations:
    :param image_to_save:
    :param type_of_image:
    :return: None
    """

    name = os.path.basename(image_path).split(".")[0]
    img_file_name = name + "_%s.png" % type_of_image
    txt_file_name = name + "_%s.txt" % type_of_image
    output_image_path = os.path.join(output_img_dir, img_file_name)
    cv2.imwrite(output_image_path, image_to_save)
    output_annotations_path = os.path.join(output_txt_dir, txt_file_name)
    with open(output_annotations_path, 'w') as f:
        f.writelines(annotations)


def save_rotated_images(image, image_path, output_img_dir, output_txt_dir, annotations, rotated_image):
    """

    :param image:
    :param image_path:
    :param output_img_dir:
    :param output_txt_dir:
    :param annotations:
    :param rotated_image:
    :return:
    """

    image_width = image.shape[1]
    image_height = image.shape[0]
    name = os.path.basename(image_path).split(".")[0]
    img_file_name = name + "_rotated.png"
    txt_file_name = name + "_rotated.txt"
    output_image_path = os.path.join(output_img_dir, img_file_name)
    rotated_annotations = rotate_segmentation_annotations(annotations, image_width, image_height, 180)
    cv2.imwrite(output_image_path, rotated_image)
    output_annotations_path = os.path.join(output_txt_dir, txt_file_name)

    with open(output_annotations_path, 'w') as f:
        for annotation in rotated_annotations:
            f.write(' '.join([str(coord) for coord in annotation]) + '\n')


def augment_images(input_img_dir, input_txt_dir, output_img_dir, output_txt_dir):
    """

    :param input_img_dir:
    :param input_txt_dir:
    :param output_img_dir:
    :param output_txt_dir:
    :return:
    """

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
        save_rotated_images(image, image_path, output_img_dir, output_txt_dir, annotations, rotated_image)

        brightness_factor = random.uniform(0.5, 1.5)
        save_files(image_path, output_img_dir, output_txt_dir, annotations, brightness_factor, "brightness")

        smoothed_image = cv2.GaussianBlur(image, (7, 7), 0)
        save_files(image_path, output_img_dir, output_txt_dir, annotations, smoothed_image, "smoothed")

        noisy_image = add_salt_and_pepper_noise(image, salt_vs_pepper_ratio, noise_amount)
        save_files(image_path, output_img_dir, output_txt_dir, annotations, noisy_image, "noisy")

        scaled_image = scale_image(image, scale_percent=50)
        save_files(image_path, output_img_dir, output_txt_dir, annotations, scaled_image, "scaled")


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
