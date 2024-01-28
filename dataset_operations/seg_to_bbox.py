import os
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm


def convert_yolo_to_bbox(yolo_annotation, img_width, img_height):
    yolo_id = int(yolo_annotation[0])
    points = list(map(float, yolo_annotation.split()[1:]))
    x_coordinates = points[::2]
    y_coordinates = points[1::2]

    x_min = int(min(x_coordinates) * img_width)
    y_min = int(min(y_coordinates) * img_height)
    x_max = int(max(x_coordinates) * img_width)
    y_max = int(max(y_coordinates) * img_height)

    return yolo_id, x_min, y_min, x_max, y_max


def plot_bbox(image_path, bboxes):
    image = cv2.imread(image_path)
    for bbox in bboxes:
        _, x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def convert_and_plot_annotations(img_folder, annotation_folder, dst_path):
    for img_filename in tqdm(os.listdir(img_folder)):
        if img_filename.endswith(".jpg"):
            img_path = os.path.join(img_folder, img_filename)
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(annotation_folder, label_filename)

            with open(label_path, 'r') as label_file:
                annotations = label_file.readlines()

            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape

            bboxes = []
            for annotation in annotations:
                bbox = convert_yolo_to_bbox(annotation, img_width, img_height)
                bboxes.append(bbox)

            plot_bbox(img_path, bboxes)

            save_filename = os.path.join(dst_path, label_filename)
            with open(save_filename, 'w') as save_file:
                for bbox in bboxes:
                    save_file.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")


if __name__ == "__main__":
    image_folder = "D:/storage/IVM/datasets/cure/customer_annotated/train/images"
    label_folder = "D:/storage/IVM/datasets/cure/customer_annotated/train/labels"
    save_folder = "D:/storage/IVM/datasets/cure/customer_annotated/train/bbox_labels"
    os.makedirs(save_folder, exist_ok=True)
    convert_and_plot_annotations(image_folder, label_folder, save_folder)
