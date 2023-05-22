import cv2
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tqdm import tqdm
from typing import Tuple

from config.const import CONST
from convert_yolo import convert_yolo_format_to_pixels
from utils.utils import measure_execution_time


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ C H A N G E   B A C K G R O U N D -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def change_background(image_path: str, annotations_path: str, background_color: Tuple[int, int, int]) -> None:
    """
    The function reads in the path of an image and a corresponding annotation file. Converts the YOLO annotation values
     to pixel coordinates, after that it applies bitwise and operation in order to subtract the fore and background.
     Finally, it adds together the images. Results should be an image with the pill(s) on it, and a homogeneous
     background.

    :param image_path: Path to where the images are located.
    :param annotations_path: Path to where the annotations are located.
    :param background_color: Color of the background.
    :return: None
    """

    image = cv2.imread(image_path)

    with open(annotations_path, 'r') as file:
        annotation_text = file.readlines()

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    background = np.zeros(image.shape, dtype=np.uint8)
    background[:] = background_color
    foreground = None

    for anno_text in annotation_text:
        annotation_list = list(map(float, anno_text.split()))
        annotation_list = annotation_list[1:]
        annotation_points = convert_yolo_format_to_pixels(image=image, annotation=annotation_list)

        annotation_points = np.array(annotation_points, dtype=np.int32)
        annotation_points = annotation_points.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [annotation_points], color=(255, 255, 255))

        foreground = cv2.bitwise_and(image, image, mask=mask)
        background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

    output_image = cv2.add(foreground, background)

    output_file_name = os.path.join(CONST.dir_wo_background, os.path.basename(image_path))
    cv2.imwrite(output_file_name, output_image)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
@measure_execution_time
def main() -> None:
    """
    Main function of the file. Collects the files and in a parallelized way, it executes the change_background()
    function. As to demonstrate technical capability, the execution time is measured with the measure_execution_time()
    decorator function.

    :return: None
    """

    images = sorted(glob(CONST.dir_ogyi_multi_splitted_train_images + "/*.png"))
    annotations = sorted(glob(CONST.dir_ogyi_multi_splitted_train_labels + "/*.txt"))

    background_color = (100, 100, 100)

    with ThreadPoolExecutor() as executor:
        futures = []
        for img, txt in tqdm(zip(images, annotations), total=len(images), desc="Processing images"):
            future = executor.submit(change_background, img, txt, background_color)
            futures.append(future)

        for future in tqdm(futures, total=len(futures), desc='Saving images'):
            future.result()


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
