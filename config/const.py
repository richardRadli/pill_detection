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
        # -------------------------------------------------- U N E T ---------------------------------------------------
        "unet_out":
            "images/unet/unet_out",
        "unet_compare":
            "images/unet/unet_compare",

        # --------------------------------------------------- O G Y E I ------------------------------------------------
        # ---------------------------------------------- R E F   T R A I N ---------------------------------------------
        "train_contour_stream_ogyei":
            "images/stream_images/ogyei/contour/train",
        "train_lbp_stream_ogyei":
            "images/stream_images/ogyei/lbp/train",
        "train_rgb_stream_ogyei":
            "images/stream_images/ogyei/rgb/train",
        "train_texture_stream_ogyei":
            "images/stream_images/ogyei/texture/train",

        # ---------------------------------------------- R E F   V A L I D ---------------------------------------------
        "valid_contour_stream_ogyei":
            "images/stream_images/ogyei/contour/valid",
        "valid_lbp_stream_ogyei":
            "images/stream_images/ogyei/lbp/valid",
        "valid_rgb_stream_ogyei":
            "images/stream_images/ogyei/rgb/valid",
        "valid_texture_stream_ogyei":
            "images/stream_images/ogyei/texture/valid",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "test_contour_stream_ref_ogyei":
            "images/test/ogyei/ref/contour",
        "test_contour_stream_query_ogyei":
            "images/test/ogyei/query/contour",
        "test_lbp_stream_ref_ogyei":
            "images/test/ogyei/ref/lbp",
        "test_lbp_stream_query_ogyei":
            "images/test/ogyei/query/lbp",
        "test_rgb_stream_ref_ogyei":
            "images/test/ogyei/ref/rgb",
        "test_rgb_stream_query_ogyei":
            "images/test/ogyei/query/rgb",
        "test_texture_stream_ref_ogyei":
            "images/test/ogyei/ref/texture",
        "test_texture_stream_query_ogyei":
            "images/test/ogyei/query/texture",

        # ---------------------------------------------------- C U R E -------------------------------------------------
        # ---------------------------------------------- R E F   T R A I N ---------------------------------------------
        "train_contour_stream_cure":
            "images/stream_images/cure/contour/train",
        "train_lbp_stream_cure":
            "images/stream_images/cure/lbp/train",
        "train_rgb_stream_cure":
            "images/stream_images/cure/rgb/train",
        "train_texture_stream_cure":
            "images/stream_images/cure/texture/train",

        # ---------------------------------------------- R E F   V A L I D ---------------------------------------------
        "valid_contour_stream_cure":
            "images/stream_images/cure/contour/valid",
        "valid_lbp_stream_cure":
            "images/stream_images/cure/lbp/valid",
        "valid_rgb_stream_cure":
            "images/stream_images/cure/rgb/valid",
        "valid_texture_stream_cure":
            "images/stream_images/cure/texture/valid",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "test_contour_stream_ref_cure":
            "images/test/cure/ref/contour",
        "test_contour_stream_query_cure":
            "images/test/cure/query/contour",
        "test_lbp_stream_ref_cure":
            "images/test/cure/ref/lbp",
        "test_lbp_stream_query_cure":
            "images/test/cure/query/lbp",
        "test_rgb_stream_ref_cure":
            "images/test/cure/ref/rgb",
        "test_rgb_stream_query_cure":
            "images/test/cure/query/rgb",
        "test_texture_stream_ref_cure":
            "images/test/cure/ref/texture",
        "test_texture_stream_query_cure":
            "images/test/cure/query/texture",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        # CNN
        "contour_hardest_cnn_network":
            "images/hardest_samples/cnn_network/contour_hardest",
        "lbp_hardest_cnn_network":
            "images/hardest_samples/cnn_network/lbp_hardest",
        "rgb_hardest_cnn_network":
            "images/hardest_samples/cnn_network/rgb_hardest",
        "texture_hardest_cnn_network":
            "images/hardest_samples/cnn_network/texture_hardest",

        # EfficientNet b0
        "contour_hardest_efficient_net":
            "images/hardest_samples/efficient_net/contour_hardest",
        "lbp_hardest_efficient_net":
            "images/hardest_samples/efficient_net/lbp_hardest",
        "rgb_hardest_efficient_net":
            "images/hardest_samples/efficient_net/rgb_hardest",
        "texture_hardest_efficient_net":
            "images/hardest_samples/efficient_net/texture_hardest",

        # EfficientNetV2 small
        "contour_hardest_efficient_net_v2":
            "images/hardest_samples/efficient_net_v2/contour_hardest",
        "lbp_hardest_efficient_net_v2":
            "images/hardest_samples/efficient_net_v2/lbp_hardest",
        "rgb_hardest_efficient_net_v2":
            "images/hardest_samples/efficient_net_v2/rgb_hardest",
        "texture_hardest_efficient_net_v2":
            "images/hardest_samples/efficient_net_v2/texture_hardest",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_cnn_network":
            "images/plotting/stream_net/plotting_cnn_network",
        "plotting_efficient_net":
            "images/plotting/stream_net/plotting_efficient_net",
        "plotting_efficient_net_v2":
            "images/plotting/stream_net/plotting_efficient_net_v2",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_cnn":
            "images/plotting/fusion_net/plotting_fusion_network_cnn",
        "plotting_fusion_network_efficient_net_self_attention":
            "images/plotting/fusion_net/plotting_fusion_network_efficient_net_self_attention",
        "plotting_fusion_network_efficient_net_v2_self_attention":
            "images/plotting/fusion_net/plotting_fusion_network_efficient_net_v2_self_attention",
        "plotting_fusion_network_efficient_net_v2_multi_head_attention":
            "images/plotting/fusion_net/plotting_fusion_network_efficient_net_v2_multi_head_attention",
        "plotting_fusion_network_efficient_net_v2_mha_fmha":
            "images/plotting/fusion_net/plotting_fusion_network_efficient_net_v2_mha_fmha",

        # ------------------------------------- I M A G E   A U G M E N T A T I O N ------------------------------------
        "images_aug":
            "images/aug/images_aug",
        "wo_background":
            "images/aug/wo_background"
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
        # ----------------------------------------------- W E I G H T S ------------------------------------------------
        # UNet
        "weights_unet":
            "data/weights/weights_unet",

        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # CNN - StreamNetwork
        "weights_cnn_network_contour":
            "data/weights/stream_net/cnn/weights_cnn_network_contour",
        "weights_cnn_network_lbp":
            "data/weights/stream_net/cnn/weights_cnn_network_lbp",
        "weights_cnn_network_rgb":
            "data/weights/stream_net/cnn/weights_cnn_network_rgb",
        "weights_cnn_network_texture":
            "data/weights/stream_net/cnn/weights_cnn_network_texture",

        # EfficientNet b0 - StreamNetwork
        "weights_efficient_net_contour":
            "data/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp":
            "data/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb":
            "data/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture":
            "data/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # EfficientNet V2 - StreamNetwork
        "weights_efficient_net_v2_contour":
            "data/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_contour",
        "weights_efficient_net_v2_lbp":
            "data/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_lbp",
        "weights_efficient_net_v2_rgb":
            "data/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_rgb",
        "weights_efficient_net_v2_texture":
            "data/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_cnn":
            "data/weights/fusion_net/cnn",
        "weights_fusion_network_efficient_net_self_attention":
            "data/weights/fusion_net/efficient_net_self_attention",
        "weights_fusion_network_efficient_net_v2_self_attention":
            "data/weights/fusion_net/efficient_net_v2_self_attention",
        "weights_fusion_network_efficient_net_v2_multi_head_attention":
            "data/weights/fusion_net/efficient_net_v2_multi_head_attention",
        "weights_fusion_network_efficient_net_v2_mha_fmha":
            "data/weights/fusion_net/efficient_net_v2_mha_fmha",

        # -------------------------------------------------- L O G S ---------------------------------------------------
        # UNet
        "logs_unet":
            "data/logs/logs_unet",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # CNN
        "logs_cnn_contour":
            "data/logs/stream_net/cnn/logs_cnn_contour",
        "logs_cnn_lbp":
            "data/logs/stream_net/cnn/logs_cnn_lbp",
        "logs_cnn_rgb":
            "data/logs/stream_net/cnn/logs_cnn_rgb",
        "logs_cnn_texture":
            "data/logs/stream_net/cnn/logs_cnn_texture",

        # EfficientNet b0
        "logs_efficient_net_contour":
            "data/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp":
            "data/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb":
            "data/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture":
            "data/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # EfficientNet V2
        "logs_efficient_net_v2_contour":
            "data/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_contour",
        "logs_efficient_net_v2_lbp":
            "data/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_lbp",
        "logs_efficient_net_v2_rgb":
            "data/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_rgb",
        "logs_efficient_net_v2_texture":
            "data/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_cnn":
            "data/logs/fusion_net/cnn",
        "logs_fusion_network_efficient_net_self_attention":
            "data/logs/fusion_net/efficient_net_self_attention",
        "logs_fusion_network_efficient_net_v2_self_attention":
            "data/logs/fusion_net/efficient_net_v2_self_attention",
        "logs_fusion_network_efficient_net_v2_multi_head_attention":
            "data/logs/fusion_net/efficient_net_v2_multi_head",
        "logs_fusion_network_efficient_net_v2_mha_fmha":
            "data/logs/fusion_net/efficient_net_v2_mha_fmha",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_cnn_network":
            "data/predictions/stream_net/predictions_cnn_network",
        "predictions_efficient_net":
            "data/predictions/stream_net/predictions_efficient_net",
        "predictions_efficient_net_v2":
            "data/predictions/stream_net/predictions_efficient_net_v2",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_cnn":
            "data/predictions/fusion_net/predictions_fusion_network_cnn",
        "predictions_fusion_network_efficient_net_self_attention_net":
            "data/predictions/fusion_net/predictions_fusion_network_efficient_net_self_attention_net",
        "predictions_fusion_network_efficient_net_v2_self_attention_net":
            "data/predictions/fusion_net/predictions_fusion_network_efficient_net_v2_self_attention",
        "predictions_fusion_network_efficient_net_v2_multi_head_attention":
            "data/predictions/fusion_net/predictions_fusion_network_efficient_net_v2_multi_head_attention",
        "predictions_fusion_network_efficient_net_v2_mha_fmha":
            "data/predictions/fusion_net/predictions_fusion_network_efficient_net_v2_mha_fmha",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_cnn_network":
            "data/ref_vec/stream_net/reference_vectors_cnn_network",
        "reference_vectors_efficient_net":
            "data/ref_vec/stream_net/reference_vectors_efficient_net",
        "reference_vectors_efficient_net_v2":
            "data/ref_vec/stream_net/reference_vectors_efficient_net_v2",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_cnn":
            "data/ref_vec/fusion_net/reference_vectors_fusion_net_cnn",
        "reference_vectors_fusion_net_efficient_net_self_attention":
            "data/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_self_attention",
        "reference_vectors_fusion_net_efficient_net_v2_self_attention":
            "data/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_v2_self_attention",
        "reference_vectors_fusion_net_efficient_net_v2_multi_head_attention":
            "data/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_v2_multi_head_attention",
        "reference_vectors_fusion_net_efficient_net_v2_mha_fmha":
            "data/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_v2_mha_fmha",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        # Hardest samples
        "negative_cnn_network":
            "data/hardest_samples/cnn_network/negative",
        "positive_cnn_network":
            "data/hardest_samples/cnn_network/positive",

        "negative_efficient_net":
            "data/hardest_samples/efficient_net/negative",
        "positive_efficient_net":
            "data/hardest_samples/efficient_net/positive",

        "negative_efficient_net_v2":
            "data/hardest_samples/efficient_net_v2/negative",
        "positive_efficient_net_v2":
            "data/hardest_samples/efficient_net_v2/positive",

        # ------------------------------------------------- O T H E R --------------------------------------------------
        # Other
        "cam_data":
            "data/other/cam_data"
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
        # C U R E
        "cure_customer":
            "cure/Customer",
        "cure_customer_mask":
            "cure/Customer_mask",
        "cure_reference":
            "cure/Reference",
        "cure_reference_mask":
            "cure/Reference_mask",
        "cure_train":
            "cure/train",
        "cure_train_mask":
            "cure/train_mask",
        "cure_test":
            "cure/test",
        "cure_test_mask":
            "cure/test_mask",

        # D T D
        "dtd_images":
            "dtd_images",

        # O G Y E I   V 1
        # Single
        # Unsplitted
        "ogyei_v1_single_unsplitted_images":
            "ogyei_v1/single/unsplitted/images",
        "ogyei_v1_single_unsplitted_labels":
            "ogyei_v1/single/unsplitted/labels",
        "ogyei_v1_single_unsplitted_gt_masks":
            "ogyei_v1/single/unsplitted/gt_masks",
        "ogyei_v1_single_unsplitted_pred_masks":
            "ogyei_v1/single/unsplitted/pred_masks",

        # Train images, labels and masks
        "ogyei_v1_single_splitted_train_images":
            "ogyei_v1/single/splitted/train/images",
        "ogyei_v1_single_splitted_train_labels":
            "ogyei_v1/single/splitted/train/labels",
        "ogyei_v1_single_splitted_gt_train_masks":
            "ogyei_v1/single/splitted/train/gt_train_masks",
        "ogyei_v1_single_splitted_pred_train_masks":
            "ogyei_v1/single/splitted/train/pred_train_masks",

        # Validation images, labels and masks
        "ogyei_v1_single_splitted_valid_images":
            "ogyei_v1/single/splitted/valid/images",
        "ogyei_v1_single_splitted_valid_labels":
            "ogyei_v1/single/splitted/valid/labels",
        "ogyei_v1_single_splitted_gt_valid_masks":
            "ogyei_v1/single/splitted/valid/gt_valid_masks",
        "ogyei_v1_single_splitted_pred_valid_masks":
            "ogyei_v1/single/splitted/valid/pred_valid_masks",

        # Test images, labels and masks
        "ogyei_v1_single_splitted_test_images":
            "ogyei_v1/single/splitted/test/images",
        "ogyei_v1_single_splitted_test_labels":
            "ogyei_v1/single/splitted/test/labels",
        "ogyei_v1_single_splitted_gt_test_masks":
            "ogyei_v1/single/splitted/test/gt_test_masks",
        "ogyei_v1_single_splitted_pred_test_masks":
            "ogyei_v1/single/splitted/test/pred_test_masks",

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
            "tray/tray_images_aug_w_med_aug"
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
