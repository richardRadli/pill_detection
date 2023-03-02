import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from glob import glob
from sklearn.metrics import confusion_matrix, classification_report

from const import CONST
from utils.utils import plot_diagrams, numerical_sort


# ----------------------------------------------------------------------------------------#
# --------------------------C A L C U L A T E   F P R   T P R-----------------------------#
# ----------------------------------------------------------------------------------------#
def calculate_metrics():
    images_true = sorted(glob(CONST.dir_test_mask + '/*.png'), key=numerical_sort)
    images_pred = sorted(glob(CONST.dir_unet_output + '/*.png'), key=numerical_sort)

    # False Positive Rate
    fpr_list = []
    # True Positive Rate - Recall
    tpr_list = []
    # Positive Predictive Value - Precision
    ppv_list = []
    # IoU
    iou_list = []

    for _, (true_img_idx, pred_img_idx) in enumerate(zip(images_true, images_pred)):
        print(true_img_idx.split("\\")[2])
        print(pred_img_idx.split("\\")[2])

        y_true = cv2.imread(true_img_idx, 0)
        y_true_norm = y_true.ravel()

        y_pred = cv2.imread(pred_img_idx, 0)
        y_pred_norm = y_pred.ravel()

        cm = confusion_matrix(y_true_norm, y_pred_norm)

        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        # FPR
        fpr = fp / (fp + tn)
        fpr_list.append(fpr)

        # TPR
        tpr = 0
        if tp != 0 or fn != 0:
            tpr = tp / (tp + fn)
        # print("TPR, y: ",y)
        tpr_list.append(tpr)

        # PPV
        ppv = 0
        if tp != 0 or fp != 0:
            ppv = tp / (tp + fp)
        ppv_list.append(ppv)

        # IoU
        iou = tp / (tp + fp + fn)
        iou_list.append(iou)

        print(classification_report(y_true_norm, y_pred_norm))

    fpr_sorted = sorted(fpr_list)
    tpr_sorted = sorted(tpr_list)
    iou_sorted = sorted(iou_list)
    ppv_sorted = sorted(ppv_list)
    ppv_sorted = np.nan_to_num(ppv_sorted)
    ppv_sorted = np.where(ppv_sorted == 0, 1, ppv_sorted)

    return np.mean(fpr_sorted), np.mean(tpr_sorted), np.mean(ppv_sorted), np.mean(iou_sorted)


def plot_results():
    images_input = sorted(glob(CONST.dir_test_images + "/*.png"), key=numerical_sort)
    images_true = sorted(glob(CONST.dir_test_mask + '/*.png'), key=numerical_sort)
    images_pred = sorted(glob(CONST.dir_unet_output + '/*.png'), key=numerical_sort)

    for idx, (input_img_val, true_img_val, pred_img_val) in enumerate(zip(images_input, images_true, images_pred)):
        save_path = input_img_val
        file_name = save_path.split("\\")[2]
        output_path = (os.path.join(CONST.dir_unet_output_2, file_name))

        in_img = plt.imread(input_img_val)
        gt_img = plt.imread(true_img_val)
        pred_img = plt.imread(pred_img_val)
        plot_diagrams(in_img, gt_img, pred_img, output_path)


if __name__ == "__main__":
    # plot_results()
    fpr_avg, tpr_avg, ppv_avg, iou_avg = calculate_metrics()
    print(f"Average FPR: {fpr_avg},\nAverage Recall: {tpr_avg},\nAverage Precision: {ppv_avg}, "
          f"\nAverage IoU: {iou_avg}")
