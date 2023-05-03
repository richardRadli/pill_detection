import cv2
import numpy as np
import os

from glob import glob
from tqdm import tqdm
from typing import List

from const import CONST
from convert_yolo import convert_yolo_format_to_pixels, read_yolo_annotations_to_list


def read_image_to_list(dir_train_images: str) -> List[str]:
    """
    Reads image files from a directory and its subdirectories.

    :param dir_train_images: The directory path containing the images.
    :return: A list of image file paths.
    """

    img_files = sorted(glob(os.path.join(dir_train_images, "*.png")))
    file_names = []

    for _, img_file in tqdm(enumerate(img_files), total=len(img_files), desc="Collecting image file names"):
        file_names.append(img_file)

    return file_names


def main():
    main_dir = os.path.join(CONST.PROJECT_ROOT, "datasets/ogyi/full_img_size/splitted/test")
    original_imgs_file_names = read_image_to_list(main_dir + "/images")
    yolo_annotations = read_yolo_annotations_to_list(main_dir + "/labels")

    for _, (img, txt) in tqdm(enumerate(zip(original_imgs_file_names, yolo_annotations)),
                              total=len(original_imgs_file_names)):
        print("\n", f'Image name: {os.path.basename(img)}\ntxt name: {os.path.basename(txt)}')
        image = cv2.imread(img)

        with open(txt, "r") as file:
            annotation_text = file.readline().strip()

        annotation_list = list(map(float, annotation_text.split()))
        annotation_list = annotation_list[1:]
        annotation_points = convert_yolo_format_to_pixels(image=image, annotation=annotation_list)

        annotation_points = np.array(annotation_points, dtype=np.int32)
        annotation_points = annotation_points.reshape((-1, 1, 2))  # Reshape the array
        cv2.polylines(image, [annotation_points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("", cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2)))
        cv2.waitKey(100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        print(kie)
