import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report

from const import CONST
from utils.utils import plot_diagrams, numerical_sort


# ---------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------- C A L C U L A T E   M E T R I C S -----------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
def calculate_metrics() -> None:
    """

    :return: None
    """

    images_true = sorted(os.listdir(CONST.dir_test_mask))
    images_pred = sorted(os.listdir(CONST.dir_unet_output))

    with ThreadPoolExecutor() as executor:
        y_true_list = list(executor.map(lambda img: cv2.imread(os.path.join(CONST.dir_test_mask, img), 0).ravel(),
                                        images_true))
        y_pred_list = list(executor.map(lambda img: cv2.imread(os.path.join(CONST.dir_unet_output, img), 0).ravel(),
                                        images_pred))
        cm_list = list(executor.map(lambda y_true_norm, y_pred_norm: confusion_matrix(y_true_norm, y_pred_norm),
                                    y_true_list, y_pred_list))

    tn_list = [cm[0][0] for cm in cm_list]
    fp_list = [cm[0][1] for cm in cm_list]
    fn_list = [cm[1][0] for cm in cm_list]
    tp_list = [cm[1][1] for cm in cm_list]

    fpr_list = [fp / (fp + tn) for fp, tn in zip(fp_list, tn_list)]
    tpr_list = [tp / (tp + fn) for tp, fn in zip(tp_list, fn_list)]
    ppv_list = [tp / (tp + fp) if (tp + fp) != 0 else 0 for tp, fp in zip(tp_list, fp_list)]
    io_u_list = [tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0 for tp, fp, fn in zip(tp_list, fp_list, fn_list)]

    print("FPR:", sum(fpr_list) / len(fpr_list))
    print("TPR:", sum(tpr_list) / len(tpr_list))
    print("PPV:", sum(ppv_list) / len(ppv_list))
    print("IoU:", sum(io_u_list) / len(io_u_list))

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    print(classification_report(y_true, y_pred))


def plot_results():
    """

    :return:
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
    calculate_metrics()
    plot_results()
