import cv2
import numpy as np
import os

from glob import glob

from config.const import CONST
from convert_yolo import convert_yolo_format_to_pixels


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


def main():
    image_path = os.path.join(CONST.DATASET_ROOT, 'ogyi_multi//splitted/train/images')
    annotations_path = os.path.join(CONST.DATASET_ROOT, 'ogyi_multi/splitted/train/labels')

    images = sorted(glob(image_path + "/*.png"))
    annotations = sorted(glob(annotations_path + "/*.txt"))

    background_color = (120, 120, 120)

    for idx, (img, txt) in enumerate(zip(images, annotations)):
        change_background(img, txt, background_color)
        break


if __name__ == "__main__":
    main()
