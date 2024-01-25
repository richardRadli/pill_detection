import cv2

from glob import glob


def read_yolo_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x, y, w, h = map(float, data[1:])
        annotations.append((class_id, x, y, w, h))

    return annotations


def plot_bbox_on_image(image_path, annotations):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    for annotation in annotations:
        class_id, x, y, w, h = annotation
        x_min = int((x - w / 2) * width)
        y_min = int((y - h / 2) * height)
        x_max = int((x + w / 2) * width)
        y_max = int((y + h / 2) * height)

        color = (0, 255, 0)  # Green color for bounding boxes
        thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    cv2.imshow('Image with Bounding Boxes', cv2.resize(image, (image.shape[1]//3, image.shape[0]//3)))
    cv2.waitKey(100)


if __name__ == "__main__":
    yolo_annotation_file = sorted(glob('D:/storage/IVM/datasets/cure/valid_dir/yolo_labels/*.txt'))
    image_file = sorted(glob('D:/storage/IVM/datasets/cure/valid_dir/images/*.jpg'))

    for idx, (image_path, annotation_path) in enumerate(zip(image_file, yolo_annotation_file)):
        print(annotation_path)
        yolo_annotations = read_yolo_annotations(annotation_path)
        plot_bbox_on_image(image_path, yolo_annotations)
