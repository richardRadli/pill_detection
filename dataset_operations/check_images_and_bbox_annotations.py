import cv2

from pathlib import Path
from tqdm import tqdm

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector


def read_yolo_annotations(annotation_file: str) -> list:
    """
    Reads YOLO annotation and processes its content.
    :param annotation_file: Path to the annotation file.
    :return: List of annotation.
    """

    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x, y, w, h = map(float, data[1:])
        annotations.append((class_id, x, y, w, h))

    return annotations


def plot_bbox_on_image(img_path: str, annotations: list) -> None:
    """
    Plots bounding boxes on an image given the corresponding annotation.
    :param img_path: Path to the image to plot.
    :param annotations: List of annotation.
    :return: None
    """

    image = cv2.imread(img_path)
    height, width, _ = image.shape

    for annotation in annotations:
        class_id, x, y, w, h = annotation
        x_min = int((x - w / 2) * width)
        y_min = int((y - h / 2) * height)
        x_max = int((x + w / 2) * width)
        y_max = int((y + h / 2) * height)

        color = (0, 255, 0)
        thickness = 3
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    cv2.imshow('Image with Bounding Boxes', cv2.resize(image, (image.shape[1]//3, image.shape[0]//3)))
    cv2.waitKey(50)


def main(operation: str = "train") -> None:
    """
    Executes the plotting of the bounding boxes.
    :param operation: Which set to process.
    :return: None
    """

    cfg_aug = ConfigAugmentation().parse()

    train_aug_img_path = (
        dataset_images_path_selector(dataset_name=cfg_aug.dataset_name).get("train_aug_images")
    )
    train_aug_annotation_path = (
        dataset_images_path_selector(dataset_name=cfg_aug.dataset_name).get("train_aug_yolo_labels")
    )

    valid_aug_img_path = (
        dataset_images_path_selector(dataset_name=cfg_aug.dataset_name).get("valid_aug_images")
    )
    valid_aug_annotation_path = (
        dataset_images_path_selector(dataset_name=cfg_aug.dataset_name).get("valid_aug_yolo_labels")
    )

    aug_image = train_aug_img_path if operation == "train" else valid_aug_img_path
    aug_anno = train_aug_annotation_path if operation == "train" else valid_aug_annotation_path

    image_file = [str(file) for file in Path(aug_image).glob("*.jpg")]
    yolo_annotation_file = [str(file) for file in Path(aug_anno).glob("*.txt")]

    for idx, (image_path, annotation_path) in tqdm(enumerate(zip(image_file, yolo_annotation_file)),
                                                   total=len(image_file),
                                                   desc="Checking annotations"):
        yolo_annotations = read_yolo_annotations(annotation_path)
        plot_bbox_on_image(image_path, yolo_annotations)


if __name__ == "__main__":
    main()
