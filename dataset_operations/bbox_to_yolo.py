import cv2
import os

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import file_reader


def path_select(cfg, operation):
    train_images = dataset_images_path_selector(cfg.dataset_name).get("train").get("images")
    train_bboxes = dataset_images_path_selector(cfg.dataset_name).get("train").get("bbox_pixel_labels")
    train_yolo_annotations = dataset_images_path_selector(cfg.dataset_name).get("train").get("yolo_labels")

    valid_images = dataset_images_path_selector(cfg.dataset_name).get("valid").get("images")
    valid_bboxes = dataset_images_path_selector(cfg.dataset_name).get("valid").get("bbox_pixel_labels")
    valid_yolo_annotations = dataset_images_path_selector(cfg.dataset_name).get("valid").get("yolo_labels")

    test_images = dataset_images_path_selector(cfg.dataset_name).get("test").get("images")
    test_bboxes = dataset_images_path_selector(cfg.dataset_name).get("test").get("bbox_pixel_labels")
    test_yolo_annotations = dataset_images_path_selector(cfg.dataset_name).get("test").get("yolo_labels")

    images = train_images if operation == "train" else (valid_images if operation == "valid" else test_images)
    bboxes = train_bboxes if operation == "train" else (valid_bboxes if operation == "valid" else test_bboxes)
    yolo_annotations = train_yolo_annotations if operation == "train" else (
        valid_yolo_annotations if operation == "valid" else test_yolo_annotations
    )

    return images, bboxes, yolo_annotations


def read_annotations_from_file(file_path):
    with open(file_path, 'r') as file:
        annotations = [list(map(int, line.strip().split())) for line in file]
    return annotations


def convert_bbox_to_yolo(bbox, image_width, image_height):
    class_id, x_min, y_min, x_max, y_max = bbox

    width = x_max - x_min
    height = y_max - y_min

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return class_id, x_center, y_center, width, height


def convert_annotations_to_yolo_file(input_file, output_file, image_width, image_height):
    annotations = read_annotations_from_file(input_file)

    with open(output_file, 'w') as file:
        for bbox_annotation in annotations:
            yolo_annotation = convert_bbox_to_yolo(bbox_annotation, image_width, image_height)
            file.write(' '.join(map(str, yolo_annotation)) + '\n')


def main(operation: str = "train"):
    cfg = ConfigAugmentation().parse()

    images, bboxes, yolo_annotations = path_select(cfg, operation)

    images_path = file_reader(images, "jpg")
    annotation_path = file_reader(bboxes, "txt")

    for idx, (img, annotation) in enumerate(zip(images_path, annotation_path)):
        src_img = cv2.imread(img)
        image_width = src_img.shape[1]
        image_height = src_img.shape[0]
        output_file_path = os.path.join(yolo_annotations, os.path.basename(annotation))
        convert_annotations_to_yolo_file(annotation, output_file_path, image_width, image_height)


if __name__ == '__main__':
    main()