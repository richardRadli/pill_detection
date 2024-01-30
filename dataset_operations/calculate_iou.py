from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import file_reader


def average_of_list(list_to_average):
    """

    :param list_to_average:
    :return:
    """

    return sum(list_to_average) / len(list_to_average)


def calculate_iou(box1, box2):
    """

    :param box1:
    :param box2:
    :return:
    """

    box1 = [box1[1] - box1[3] / 2, box1[2] - box1[4] / 2, box1[1] + box1[3] / 2, box1[2] + box1[4] / 2]
    box2 = [box2[1] - box2[3] / 2, box2[2] - box2[4] / 2, box2[1] + box2[3] / 2, box2[2] + box2[4] / 2]

    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou


def read_annotations_from_file(file_path: str):
    """

    :param file_path:
    :return:
    """

    with open(file_path, 'r') as file:
        annotations = [list(map(float, line.strip().split())) for line in file]
    return annotations


def calculate_average_iou(gt_annotations, predicted_annotations):
    """

    :param gt_annotations:
    :param predicted_annotations:
    :return:
    """

    total_iou = 0
    num_annotations = len(gt_annotations)

    for i in range(num_annotations):
        iou = calculate_iou(gt_annotations[i], predicted_annotations[i])
        total_iou += iou

    return total_iou / num_annotations


def main():
    cfg = ConfigAugmentation().parse()
    gt_path = dataset_images_path_selector(dataset_name=cfg.dataset_name).get("test").get("yolo_labels")
    gt_files = file_reader(gt_path, "txt")
    prediction_path = "C:/Users/ricsi/Documents/yolov7/runs/detect/cure_binary_classification_64_16_203/labels"
    prediction_files = file_reader(prediction_path, "txt")
    avg_iou = []

    for idx, (gt_file, prediction_file) in enumerate(zip(gt_files, prediction_files)):
        gt_annotations = read_annotations_from_file(gt_file)
        predicted_annotations = read_annotations_from_file(prediction_file)

        iou = calculate_average_iou(gt_annotations, predicted_annotations)
        avg_iou.append(iou)

    avg_list = average_of_list(avg_iou)
    print(avg_list)


if __name__ == "__main__":
    main()
