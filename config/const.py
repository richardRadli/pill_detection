"""
File: const.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program stores the const values of different variables. There is a main class, named _Const(), and 3
other classes are inherited from that (Images, Data, Dataset).
"""

import logging
import os

from utils.utils import setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    # Setup logger
    setup_logger()

    # Select user and according paths
    user = os.getlogin()
    root_mapping = {
        "ricsi": {
            "PROJECT_ROOT":
                "C:/Users/ricsi/Documents/project/storage/IVM",
            "DATASET_ROOT":
                "C:/Users/ricsi/Documents/project/storage/IVM/datasets"
        },
        "keplab": {
            "PROJECT_ROOT":
                "/home/keplab/Documents/users/radli_richard/storage/IVM",
            "DATASET_ROOT":
                "/home/keplab/Documents/users/radli_richard/datasets/IVM"
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        PROJECT_ROOT = root_info["PROJECT_ROOT"]
        DATASET_ROOT = root_info["DATASET_ROOT"]
    else:
        raise ValueError("Wrong user!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def create_directories(cls, dirs, root_type) -> None:
        """
        Class method that creates the missing directories.
        :param dirs: These are the directories that the function checks.
        :param root_type: Either PROJECT or DATASET.
        :return: None
        """

        for _, path in dirs.items():
            if root_type == "PROJECT":
                dir_path = os.path.join(cls.PROJECT_ROOT, path)
            elif root_type == "DATASET":
                dir_path = os.path.join(cls.DATASET_ROOT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++ I M A G E S +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Images(_Const):
    dirs_images = {
        # --------------------------------------------------- O G Y E I ------------------------------------------------
        "stream_images_ogyei":
            "images/ogyei/stream_images",
        "stream_images_ogyei_test":
            "images/ogyei/test",

        # ---------------------------------------------- R E F   T R A I N ---------------------------------------------
        "train_contour_stream_ogyei":
            "images/ogyei/stream_images/contour/train",
        "train_lbp_stream_ogyei":
            "images/ogyei/stream_images/lbp/train",
        "train_rgb_stream_ogyei":
            "images/ogyei/stream_images/rgb/train",
        "train_texture_stream_ogyei":
            "images/ogyei/stream_images/texture/train",

        # ---------------------------------------------- R E F   V A L I D ---------------------------------------------
        "valid_contour_stream_ogyei":
            "images/ogyei/stream_images/contour/valid",
        "valid_lbp_stream_ogyei":
            "images/ogyei/stream_images/lbp/valid",
        "valid_rgb_stream_ogyei":
            "images/ogyei/stream_images/rgb/valid",
        "valid_texture_stream_ogyei":
            "images/ogyei/stream_images/texture/valid",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "test_query_ogyei":
            "images/ogyei/test/query",
        "test_contour_stream_query_ogyei":
            "images/ogyei/test/query/contour",
        "test_lbp_stream_query_ogyei":
            "images/ogyei/test/query/lbp",
        "test_rgb_stream_query_ogyei":
            "images/ogyei/test/query/rgb",
        "test_texture_stream_query_ogyei":
            "images/ogyei/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "test_ref_ogyei":
            "images/ogyei/test/ref",
        "test_contour_stream_ref_ogyei":
            "images/ogyei/test/ref/contour",
        "test_lbp_stream_ref_ogyei":
            "images/ogyei/test/ref/lbp",
        "test_rgb_stream_ref_ogyei":
            "images/ogyei/test/ref/rgb",
        "test_texture_stream_ref_ogyei":
            "images/ogyei/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_v2_ogyei":
            "images/ogyei/plotting/stream_net/plotting_efficient_net_v2",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "images/ogyei/plotting/fusion_net/plotting_fusion_network_efficient_net_v2_multi_head_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_v2_ogyei":
            "images/ogyei/conf_mtx/stream_net/conf_mtx_efficient_net_v2",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "images/ogyei/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_v2_multi_head_attention",

        # ------------------------------------- I M A G E   A U G M E N T A T I O N ------------------------------------
        "images_aug":
            "images/aug/images_aug",
        "wo_background":
            "images/aug/wo_background",

        # ------------------------------------------- C A L I B R A T I O N --------------------------------------------
        "calibration_images":
            "images/camera/calibration_images",
        "undistorted_images":
            "images/camera/undistorted_images",
        "pill_images":
            "images/camera/pill_images"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_images, "PROJECT")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_images.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Data(_Const):
    dirs_data = {
        # ------------------------------------------------- O T H E R --------------------------------------------------
        # Other
        "camera_matrix":
            "data/camera/camera_matrix",

        "camera_settings":
            "data/camera/camera_settings",

        "pill_names":
            "data/camera/pill_names",

        # k-fold directory names
        "k_folds":
            "data/other/k_folds",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I +++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNet V2 - StreamNetwork
        "weights_efficient_net_v2_contour_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_contour",
        "weights_efficient_net_v2_lbp_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_lbp",
        "weights_efficient_net_v2_rgb_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_rgb",
        "weights_efficient_net_v2_texture_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------

        "weights_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "data/ogyei/weights/fusion_net/efficient_net_v2_multi_head_attention",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNet V2
        "logs_efficient_net_v2_contour_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_contour",
        "logs_efficient_net_v2_lbp_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_lbp",
        "logs_efficient_net_v2_rgb_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_rgb",
        "logs_efficient_net_v2_texture_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "data/ogyei/logs/fusion_net/efficient_net_v2_multi_head",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_v2_ogyei":
            "data/ogyei/predictions/stream_net/predictions_efficient_net_v2",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        "predictions_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "data/ogyei/predictions/fusion_net/predictions_fusion_network_efficient_net_v2_multi_head_attention",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_v2_ogyei":
            "data/ogyei/ref_vec/stream_net/reference_vectors_efficient_net_v2",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "data/ogyei/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_v2_multi_head_attention",
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data, "PROJECT")

    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_data.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Datasets(_Const):
    dirs_dataset = {
        # D T D
        "dtd_images":
            "dtd_images",

        # O G Y E I   V 2
        # Single
        # Unsplitted
        "ogyei_v2_single_unsplitted_images":
            "ogyei_v2/single/unsplitted/images",
        "ogyei_v2_single_unsplitted_labels":
            "ogyei_v2/single/unsplitted/labels",
        "ogyei_v2_single_unsplitted_gt_masks":
            "ogyei_v2/single/unsplitted/gt_masks",
        "ogyei_v2_single_unsplitted_pred_masks":
            "ogyei_v2/single/unsplitted/pred_masks",

        # Train images, labels and masks
        "ogyei_v2_single_splitted_train_images":
            "ogyei_v2/single/splitted/train/images",
        "ogyei_v2_single_splitted_train_labels":
            "ogyei_v2/single/splitted/train/labels",
        "ogyei_v2_single_splitted_gt_train_masks":
            "ogyei_v2/single/splitted/train/gt_train_masks",
        "ogyei_v2_single_splitted_pred_train_masks":
            "ogyei_v2/single/splitted/train/pred_train_masks",

        # Validation images, labels and masks
        "ogyei_v2_single_splitted_valid_images":
            "ogyei_v2/single/splitted/valid/images",
        "ogyei_v2_single_splitted_valid_labels":
            "ogyei_v2/single/splitted/valid/labels",
        "ogyei_v2_single_splitted_gt_valid_masks":
            "ogyei_v2/single/splitted/valid/gt_valid_masks",
        "ogyei_v2_single_splitted_pred_valid_masks":
            "ogyei_v2/single/splitted/valid/pred_valid_masks",

        # Test images, labels and masks
        "ogyei_v2_single_splitted_test_images":
            "ogyei_v2/single/splitted/test/images",
        "ogyei_v2_single_splitted_test_labels":
            "ogyei_v2/single/splitted/test/labels",
        "ogyei_v2_single_splitted_gt_test_masks":
            "ogyei_v2/single/splitted/test/gt_test_masks",
        "ogyei_v2_single_splitted_pred_test_masks":
            "ogyei_v2/single/splitted/test/pred_test_masks",

        # T R A Y
        "tray_original_images":
            "tray/original_images",
        "tray_diff_images":
            "tray/diff_images",
        "tray_plots":
            "tray/plots",
        "tray_images_aug":
            "tray/tray_images_aug",
        "tray_images_aug_w_med":
            "tray/tray_images_aug_w_med",
        "tray_images_aug_w_med_aug":
            "tray/tray_images_aug_w_med_aug",
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset, "DATASET")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT, self.dirs_dataset.get(key, ""))


class NLPData(_Const):
    nlp_data = {
        "pill_names":
            "nlp/csv/pill_names",
        "full_sentence_csv":
            "nlp/csv/full_sentence_csv",
        "vector_distances":
            "nlp/csv/distances",

        "nlp_vector":
            "nlp/npy/nlp_vector",

        "word_vector_vis":
            "nlp/plot/word_vector",
        "elbow":
            "nlp/plot/elbow",
        "silhouette":
            "nlp/plot/silhouette",

        "patient_information_leaflet_doc":
            "nlp/documents/patient_information_leaflet_doc",
        "patient_information_leaflet_docx":
            "nlp/documents/patient_information_leaflet_docx",
        "extracted_features_files":
            "nlp/documents/extracted_features_files"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.nlp_data, "PROJECT")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.nlp_data.get(key, ""))


CONST: _Const = _Const()
IMAGES_PATH: Images = Images()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()
NLP_DATA_PATH: NLPData = NLPData()
