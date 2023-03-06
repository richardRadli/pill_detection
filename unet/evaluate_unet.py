import concurrent.futures
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from glob import glob
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from const import CONST
from utils.utils import plot_diagrams, numerical_sort


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- C A L C U L A T E   M E T R I C S    T H R E A D ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def calculate_metrics_thread() -> (float, float, float, float):
    """
    Calculates the  fpr, tpr, ppv, iou values in a multi-threading way.
    :return: mean values of fpr, tpr, ppv, iou.
    """

    images_true = sorted(glob(CONST.dir_test_mask + "/*.png"))
    images_pred = sorted(glob(CONST.dir_unet_output + "/*.png"))

    fpr_list = []
    tpr_list = []
    ppv_list = []
    io_u_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _, (true_img_idx, pred_img_idx) in tqdm(enumerate(zip(images_true, images_pred)), total=len(images_true)):
            future = executor.submit(calculate_metrics, true_img_idx, pred_img_idx)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            fpr, tpr, ppv, iou = future.result()
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            ppv_list.append(ppv)
            io_u_list.append(iou)

    fpr_sorted = sorted(fpr_list)
    tpr_sorted = sorted(tpr_list)
    iou_sorted = sorted(io_u_list)
    ppv_sorted = sorted(ppv_list)
    ppv_sorted = np.nan_to_num(ppv_sorted)
    ppv_sorted = np.where(ppv_sorted == 0, 1, ppv_sorted)

    return np.mean(fpr_sorted), np.mean(tpr_sorted), np.mean(ppv_sorted), np.mean(iou_sorted)


# ---------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------- C A L C U L A T E   M E T R I C S -----------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
def calculate_metrics(true_img_path: str, pred_img_path: str) -> (float, float, float, float):
    """
    This function calculates the fpr, tpr, ppv, iou values in the current thread.
    :param true_img_path: Path to the actual ground truth image
    :param pred_img_path: Path to the actual predicted image
    :return:  fpr, tpr, ppv, iou values
    """

    y_true = cv2.imread(true_img_path, 0)
    y_true_norm = y_true.ravel()
    y_pred = cv2.imread(pred_img_path, 0)
    y_pred_norm = y_pred.ravel()

    cm = confusion_matrix(y_true_norm, y_pred_norm)

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    # FPR
    fpr = fp / (fp + tn)

    # TPR
    tpr = 0
    if tp != 0 or fn != 0:
        tpr = tp / (tp + fn)

    # PPV
    ppv = 0
    if tp != 0 or fp != 0:
        ppv = tp / (tp + fp)

    # IoU
    iou = tp / (tp + fp + fn)

    return fpr, tpr, ppv, iou


def plot_results() -> None:
    """
    Plots the ground truth image, mask and the predicted mask on a figure.

    :return: None
    """

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
    fpr_res, tpr_res, ppvc_res, iou_res = calculate_metrics_thread()
    print(f" Fall out: {fpr_res: .4f}\n Recall: {tpr_res: .4f}\n Precision: {ppvc_res: .4f}\n IoU: {iou_res: .4f}\n")
    plot_results()
