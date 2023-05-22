import cv2
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tqdm import tqdm

from config.const import CONST
from convert_yolo import convert_yolo_format_to_pixels
from utils.utils import measure_execution_time


def change_background(image_path, annotations_path, background_color):
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


@measure_execution_time
def main():
    image_path = CONST.dir_ogyi_multi_splitted_train_images
    annotations_path = CONST.dir_ogyi_multi_splitted_train_labels

    images = sorted(glob(image_path + "/*.png"))
    annotations = sorted(glob(annotations_path + "/*.txt"))

    background_color = (100, 100, 100)

    with ThreadPoolExecutor() as executor:
        futures = []
        for img, txt in tqdm(zip(images, annotations), total=len(images), desc="Processing images"):
            future = executor.submit(change_background, img, txt, background_color)
            futures.append(future)

        # Wait for all tasks to complete
        for future in tqdm(futures, total=len(futures), desc='Saving images'):
            future.result()


if __name__ == "__main__":
    main()
