"""
File: convert_bbox_to_yolo.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 04, 2023

Description: This program converts bbox annotations to yolo format.
"""


import cv2
import numpy as np
import os

from glob import glob
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ G E T   B O U N D I N G   B O X -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_bounding_box(mask: np.ndarray):
    """
    Get the bounding box coordinates of the largest contour in a binary mask.

    Args:
        mask: Binary mask image.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) representing the bounding box coordinates.
        Returns None if no contours are found.
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        return x, y, x + w, y + h
    return None


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- M A S K   T O   Y O L O   F O R M A T -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def masks_to_yolo_format(masks_dir: str, output_dir: str):
    """
    Convert mask images to YOLO format annotations.

    Args:
        masks_dir (str): Directory containing mask images.
        output_dir (str): Directory to save the YOLO annotation files.

    Returns:
        None
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of mask files
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]

    for mask_file in tqdm(mask_files, total=len(mask_files), desc="Converting to bounding box annotations"):
        # Load the mask image
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Get the bounding box coordinates
        bbox = get_bounding_box(mask)

        if bbox is not None:
            # Convert the coordinates to YOLO format
            height, width = mask.shape[:2]
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / (2 * width)
            y_center = (y_min + y_max) / (2 * height)
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            # Save the YOLO annotation file
            annotation_file = mask_file.replace('.png', '.txt')
            annotation_path = os.path.join(output_dir, annotation_file)
            with open(annotation_path, 'w') as f:
                f.write(f'0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}')

        else:
            print(f'No bounding box found for mask: {mask_file}')

    print('Conversion completed.')


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- D I S P L A Y   A N N O T A T I O N S -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def display_annotations(image_path: str, annotation_path: str):
    """
    Display annotated images with bounding box rectangles.

    Args:
        image_path: Path to the directory containing the images.
        annotation_path: Path to the directory containing the annotation files.
    Returns:
        None
    """

    images = sorted(glob(image_path + "*.png"))
    annotationz = sorted(glob(annotation_path + "*.txt"))

    for idx, (img, ann) in enumerate(zip(images, annotationz)):
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
            class_index, x_center, y_center, box_width, box_height = annotation

            # Convert relative coordinates to absolute coordinates
            x_min = int((x_center - box_width / 2) * width)
            y_min = int((y_center - box_height / 2) * height)
            x_max = int((x_center + box_width / 2) * width)
            y_max = int((y_center + box_height / 2) * height)

            # Draw the bounding box rectangle on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)

        cv2.imshow("Annotated Image + %s" % str(idx), cv2.resize(image, (image.shape[1]//3, image.shape[0]//3)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- M A I N --------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main():
    """
    The main function, executes the functions.
    :return:
    """

    # Example usage
    masks_directory = "C:/Users/ricsi/Desktop/train/masks/"
    output_directory = "C:/Users/ricsi/Desktop/train/labels/"  # Replace with your desired output directory
    masks_to_yolo_format(masks_directory, output_directory)
    display_annotations(masks_directory, output_directory)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- __M A I N__ ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
