import cv2
import os

from glob import glob


def convert_bbox_to_yolo(bbox, image_width, image_height):
    class_id, xmin, ymin, xmax, ymax = bbox

    width = xmax - xmin
    height = ymax - ymin

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return class_id, x_center, y_center, width, height


def read_annotations_from_file(file_path):
    with open(file_path, 'r') as file:
        annotations = [list(map(int, line.strip().split())) for line in file]
    return annotations


def convert_annotations_to_yolo_file(input_file, output_file, image_width, image_height):
    annotations = read_annotations_from_file(input_file)

    with open(output_file, 'w') as file:
        for bbox_annotation in annotations:
            yolo_annotation = convert_bbox_to_yolo(bbox_annotation, image_width, image_height)
            file.write(' '.join(map(str, yolo_annotation)) + '\n')


def main():
    images_path = sorted(glob("D:/storage/IVM/datasets/cure/test_dir/images/*.jpg"))
    annotation_path = sorted(glob("D:/storage/IVM/datasets/cure/test_dir/bbox_labels/*.txt"))
    out_path = "D:/storage/IVM/datasets/cure/test_dir/yolo_labels/"
    os.makedirs(out_path, exist_ok=True)

    for idx, (img, annotation) in enumerate(zip(images_path, annotation_path)):
        src_img = cv2.imread(img)
        image_width = src_img.shape[1]
        image_height = src_img.shape[0]
        output_file_path = os.path.join(out_path, os.path.basename(annotation))
        convert_annotations_to_yolo_file(annotation, output_file_path, image_width, image_height)


if __name__ == '__main__':
    main()
