from glob import glob


def average_of_list(list_to_average):
    return sum(list_to_average) / len(list_to_average)


def calculate_iou(box1, box2):
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


def read_annotations_from_file(file_path):
    with open(file_path, 'r') as file:
        annotations = [list(map(float, line.strip().split())) for line in file]
    return annotations


def calculate_average_iou(gt_annotations, predicted_annotations):
    total_iou = 0
    num_annotations = len(gt_annotations)

    for i in range(num_annotations):
        iou = calculate_iou(gt_annotations[i], predicted_annotations[i])
        total_iou += iou

    return total_iou / num_annotations


def main():
    gt_path = sorted(glob("D:/storage/IVM/datasets/cure/test_dir/yolo_labels/*.txt"))
    pred_path = sorted(glob("C:/Users/ricsi/Documents/yolov7/runs/detect/cure-test/labels/*.txt"))
    avg_iou = []

    for idx, (gt_file, pred_file) in enumerate(zip(gt_path, pred_path)):
        gt_annotations = read_annotations_from_file(gt_file)
        predicted_annotations = read_annotations_from_file(pred_file)

        iou = calculate_average_iou(gt_annotations, predicted_annotations)
        avg_iou.append(iou)

    avg_list = average_of_list(avg_iou)
    print(avg_list)


if __name__ == "__main__":
    main()
