"""
File: crop_yolo_detect_images.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 15, 2023

Description: Crop images based on bounding box annotations and save the cropped images.
"""

import cv2
import os

from tqdm import tqdm

from config.data_paths import DATASET_PATH
from utils.utils import file_reader


def crop_bounding_boxes(image_path: str, annotation_path: str, out_path: str):
    """
    Crop images based on bounding box annotations and save the cropped images.
    
    Args:
        image_path: Path to the directory containing the images.
        annotation_path: Path to the directory containing the annotation files.
        out_path: Path to the directory where the cropped images will be saved.

    Returns:
         None
    """

    images = file_reader(image_path, ".png")

    for img in tqdm(images, total=len(images)):
        img_name = os.path.splitext(os.path.basename(img))[0]
        ann = os.path.join(annotation_path, img_name + ".txt")

        if os.path.isfile(ann):
            image = cv2.imread(img)
            height, width, _ = image.shape

            with open(ann, 'r') as f:
                lines = f.readlines()

            # Parse annotation lines and extract relevant information
            annotations = [line.strip().split() for line in lines]
            annotations = [
                [int(annotation[0]), float(annotation[1]), float(annotation[2]), float(annotation[3]),
                 float(annotation[4])] for annotation in annotations]

            for annotation in annotations:
                _, x_center, y_center, box_width, box_height = annotation

                # Convert relative coordinates to absolute coordinates
                x_min = int((x_center - box_width / 2) * width)
                y_min = int((y_center - box_height / 2) * height)
                x_max = int((x_center + box_width / 2) * width)
                y_max = int((y_center + box_height / 2) * height)

                # Calculate the longer side of the bounding box
                box_size = max(x_max - x_min, y_max - y_min)

                # Calculate the center coordinates of the square bounding box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # Calculate the new coordinates of the square bounding box
                x_min = center_x - box_size // 2
                y_min = center_y - box_size // 2
                x_max = center_x + box_size // 2
                y_max = center_y + box_size // 2

                # Calculate the crop coordinates
                left = x_min - 20
                upper = y_min - 20
                right = x_max + 20
                lower = y_max + 20

                # Ensure the crop coordinates are within image boundaries
                left = max(left, 0)
                upper = max(upper, 0)
                right = min(right, width)
                lower = min(lower, height)

                # Crop the image using NumPy array indexing if the crop coordinates are valid
                if left < right and upper < lower:
                    cropped_image_array = image[upper:lower, left:right]

                    # Save the cropped image
                    cv2.imwrite(os.path.join(out_path, img_name + ".png"), cropped_image_array)
                else:
                    print(f"Invalid crop coordinates for image: {os.path.basename(img)}")
        else:
            print(f"No annotation file found for image: {os.path.basename(img)}")


if __name__ == "__main__":
    img_path = DATASET_PATH.get_data_path("cure_customer")
    ann_path = "C:/Users/ricsi/Documents/yolov7/runs/detect/yolov7_cure_100k_binary_labels_cure_customer2/labels"
    out_path = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/cure/Customer_bbox"
    crop_bounding_boxes(img_path, ann_path, out_path)
