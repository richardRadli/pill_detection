import cv2
import concurrent.futures
import numpy as np
import os
import shutil

from typing import List, Tuple
from glob import glob
from tqdm import tqdm


def convert_yolo_format_to_pixels(image: np.ndarray, annotation: list) -> list:
    """
    Converts YOLO format annotation to pixel coordinates based on the provided image.

    :param image: The input image.
    :param annotation: YOLO format annotation.
    :return: List of pixel coordinates.
    """

    cropped_height, cropped_width, _ = image.shape

    annotation_points = []
    if len(annotation) % 2 == 0:
        for i in range(0, len(annotation), 2):
            x = int(annotation[i] * cropped_width)
            y = int(annotation[i + 1] * cropped_height)
            annotation_points.append((x, y))

    return annotation_points


def convert_coordinates_to_original(cropped_coordinates: Tuple[int, int], cropped_width: int, cropped_height: int,
                                    original_width: int, original_height: int) -> Tuple[int, int]:
    """
    Converts cropped coordinates to original image coordinates based on the original and cropped dimensions.

    :param cropped_coordinates: The cropped coordinates (x, y).
    :param cropped_width: The width of the cropped image.
    :param cropped_height: The height of the cropped image.
    :param original_width: The width of the original image.
    :param original_height: The height of the original image.
    :return: The corresponding original coordinates (x, y).
    """

    cropped_x, cropped_y = cropped_coordinates

    # Calculate the center position of the cropped image
    cropped_center_x = original_width / 2
    cropped_center_y = original_height / 2

    # Calculate the offset from the center due to cropping
    offset_x = cropped_center_x - cropped_width / 2
    offset_y = cropped_center_y - cropped_height / 2

    # Calculate the corresponding positions on the original image
    original_x = int(cropped_x + offset_x)
    original_y = int(cropped_y + offset_y)

    return original_x, original_y


def convert_pixels_to_yolo_format(image_width: int, image_height: int, coordinates: List[Tuple[int, int]],
                                  class_id: int) -> List[float]:
    """
    Converts pixel coordinates to YOLO format based on the image dimensions.

    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :param coordinates: The pixel coordinates [(x, y)].
    :param class_id: The class ID.
    :return: The coordinates in YOLO format [class_id, x_normalized, y_normalized, ...].
    """

    yolo_coordinates = [class_id]
    for x, y in coordinates:
        x_normalized = x / image_width
        y_normalized = y / image_height
        yolo_coordinates.extend([x_normalized, y_normalized])
    return yolo_coordinates


def read_image_to_list(dir_train_images: str) -> List[str]:
    """
    Reads image files from a directory and its subdirectories.

    :param dir_train_images: The directory path containing the images.
    :return: A list of image file paths.
    """

    subdirs = sorted(glob(os.path.join(dir_train_images, "*", "")))
    file_names = []

    for subdir in subdirs:
        images = sorted(glob(os.path.join(subdir, "*.png")))
        for idx, img_path in tqdm(enumerate(images), total=len(images), desc="Collecting image file names from subdir "
                                                                             f"{subdir}"):
            file_names.append(img_path)

    return file_names


def read_yolo_annotations_to_list(yolo_dir: str) -> List[str]:
    """
    Reads image files from a directory.

    :param yolo_dir: The directory path containing the images.
    :return: A list of image file paths.
    """

    txt_files = sorted(glob(os.path.join(yolo_dir, "*.txt")))
    file_names = []

    for _, txt_file in tqdm(enumerate(txt_files), total=len(txt_files), desc="Collecting txt file names"):
        file_names.append(txt_file)

    return file_names


def save_text_to_file(data_list: List[float], file_path: str) -> None:
    """
    Saves a list of data to a text file.

    :param data_list: The list of data.
    :param file_path: The file path to save the data.
    :return: None
    """

    with open(file_path, "w") as file:
        for item in data_list:
            file.write(str(item) + "\n")


def create_subdirectories_and_copy_files(source_dir: str, target_dir: str) -> None:
    """
    Creates subdirectories in the target directory and copies corresponding files from the source directory.

    :param source_dir: The source directory containing the files.
    :param target_dir: The target directory to create subdirectories and copy files.
    :return: None
    """

    # Get the list of subdirectories in the source directory
    subdirs = [name for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, name))]

    # Create subdirectories in the target directory
    for subdir in subdirs:
        subdir_path = os.path.join(target_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        # Get the list of files in the source subdirectory
        source_subdir = os.path.join(source_dir, subdir)
        files = os.listdir(source_subdir)

        # Copy the corresponding .txt files to the target subdirectory
        for file in files:
            if file.endswith(".png"):
                filename = os.path.splitext(file)[0] + ".txt"
                source_file = os.path.join(target_dir, filename)
                target_file = os.path.join(subdir_path, filename)
                try:
                    shutil.move(source_file, target_file)
                except FileNotFoundError as fnfe:
                    print(fnfe)


def process_image(main_dir: str, ori: str, cropped: str, yolo_annotation: str) -> None:
    """
    Process an image by reading the cropped and original images, converting annotations, and saving the result.

    :param main_dir: The main directory.
    :param ori: The path to the original image.
    :param cropped: The path to the cropped image.
    :param yolo_annotation: The path to the YOLO annotation file.
    :return: None
    """

    cropped_img = cv2.imread(cropped)
    original_img = cv2.imread(ori)
    with open(yolo_annotation, "r") as file:
        annotation_text = file.readline().strip()

    annotation_list = list(map(float, annotation_text.split()))
    class_id = int(annotation_list[0])
    annotation_list = annotation_list[1:]
    original_pixel_coordinates = []

    annotation_points = convert_yolo_format_to_pixels(image=cropped_img, annotation=annotation_list)
    for c in annotation_points:
        original_x, original_y = convert_coordinates_to_original(c, cropped_img.shape[1], cropped_img.shape[0],
                                                                 original_img.shape[1], original_img.shape[0])
        original_pixel_coordinates.append((original_x, original_y))

    yolo_coordinates = convert_pixels_to_yolo_format(original_img.shape[1], original_img.shape[0],
                                                     original_pixel_coordinates, class_id)

    save_text_to_file(data_list=yolo_coordinates,
                      file_path=os.path.join(main_dir, "train_labels_undistorted", os.path.basename(yolo_annotation)))


def main():
    # Directories
    main_dir = "D:/project/IVM"
    original_images_dir_name = "captured_OGYEI_pill_photos_undistorted"
    original_images_labels_dir_name = "train_labels_undistorted"
    cropped_images_dir_name = "captured_OGYEI_pill_photos_undistorted_cropped"
    cropped_images_labels_dir_name = "train_labels_cropped"

    # Read in data
    original_imgs_file_names = read_image_to_list(os.path.join(main_dir, original_images_dir_name))
    cropped_imgs_file_names = read_image_to_list(os.path.join(main_dir, cropped_images_dir_name))
    yolo_annotations = read_yolo_annotations_to_list(os.path.join(main_dir, cropped_images_labels_dir_name))

    # Threaded execution of the processing of images
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, main_dir, ori, cropped, yolo_annotation)
                   for ori, cropped, yolo_annotation in zip(original_imgs_file_names,
                                                            cropped_imgs_file_names,
                                                            yolo_annotations)]

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    # Creating directories and moving files there
    labels_dir = os.path.join(main_dir, original_images_labels_dir_name)
    images_dir = os.path.join(main_dir, original_images_dir_name)
    create_subdirectories_and_copy_files(images_dir, labels_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        print(kie)
