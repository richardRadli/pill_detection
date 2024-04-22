import cv2
import numpy as np
import os

from tqdm import tqdm

from utils.utils import file_reader


def calculate_tp_fp_fn(image1, image2):
    tp = np.sum(np.logical_and(image1, image2))
    fp = np.sum(np.logical_and(image2, np.logical_not(image1)))
    fn = np.sum(np.logical_and(image1, np.logical_not(image2)))

    return tp, fp, fn


def average_of_list(list_to_average):
    """
    Calculate the average of a list of numbers.

    Args:
        list_to_average (list): List of numbers.

    Returns:
        float: The average value.
    """
    return sum(list_to_average) / len(list_to_average)


def calculate_iou_custom(tp, fp, fn):
    iou = tp / (tp + fp + fn)
    iou = min(max(iou, 0.0), 1.0)
    return iou


def main():
    gt_path = "D:/storage/pill_detection/datasets/cure_two_sided/Customer/test_dir/masks"
    gt_files = file_reader(gt_path, "jpg")
    prediction_path = "C:/Users/ricsi/Documents/yolov8/runs/segment/predict/masks"
    prediction_files = file_reader(prediction_path, "jpg")
    avg_iou = []

    for gt_file in tqdm(gt_files, total=len(gt_files)):
        match_found = False
        gt_file_name = os.path.basename(gt_file)
        for prediction_file in prediction_files:
            prediction_file_name = os.path.basename(prediction_file)
            if gt_file_name == prediction_file_name:
                match_found = True
                gt_annotations = cv2.imread(gt_file, 0)
                predicted_annotations = cv2.imread(prediction_file, 0)
                tp, fp, fn = calculate_tp_fp_fn(gt_annotations, predicted_annotations)
                iou_custom = calculate_iou_custom(tp, fp, fn)
                avg_iou.append(iou_custom)
                break
        if not match_found:
            continue

    avg_iou_value = average_of_list(avg_iou)
    print("Average IoU:", avg_iou_value)


if __name__ == "__main__":
    main()
