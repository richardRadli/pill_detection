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
                "D:/storage/pill_detection",
            "DATASET_ROOT":
                "D:/storage/pill_detection/datasets"
        },
        "keplab": {
            "PROJECT_ROOT":
                "/home/keplab/Documents/users/radli_richard/storage/pill_detection",
            "DATASET_ROOT":
                "/home/keplab/Documents/users/radli_richard/datasets/pill_detection"
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
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ C U R E ++++++++++++++++++++++++++++++++++++++++++++++++++
        "stream_images_cure_anchor":
            "images/cure/stream_images/anchor",
        "stream_images_cure_pos_neg":
            "images/cure/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "train_contour_stream_cure_anchor":
            "images/cure/stream_images/anchor/contour",
        "train_lbp_stream_cure_anchor":
            "images/cure/stream_images/anchor/lbp",
        "train_rgb_stream_cure_anchor":
            "images/cure/stream_images/anchor/rgb",
        "train_texture_stream_cure_anchor":
            "images/cure/stream_images/anchor/texture",

        # ----------------------------------------------- P O S   N E G ------------------------------------------------
        "contour_stream_cure_pos_neg":
            "images/cure/stream_images/pos_neg/contour",
        "lbp_stream_cure_pos_neg":
            "images/cure/stream_images/pos_neg/lbp",
        "rgb_stream_cure_pos_neg":
            "images/cure/stream_images/pos_neg/rgb",
        "texture_stream_cure_pos_neg":
            "images/cure/stream_images/pos_neg/texture",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "test_query_cure":
            "images/cure/test/query",
        "test_contour_stream_query_cure":
            "images/cure/test/query/contour",
        "test_lbp_stream_query_cure":
            "images/cure/test/query/lbp",
        "test_rgb_stream_query_cure":
            "images/cure/test/query/rgb",
        "test_texture_stream_query_cure":
            "images/cure/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "test_ref_cure":
            "images/cure/test/ref",
        "test_contour_stream_ref_cure":
            "images/cure/test/ref/contour",
        "test_lbp_stream_ref_cure":
            "images/cure/test/ref/lbp",
        "test_rgb_stream_ref_cure":
            "images/cure/test/ref/rgb",
        "test_texture_stream_ref_cure":
            "images/cure/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_cure":
            "images/cure/plotting/stream_net/plotting_efficient_net",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_self_attention_cure":
            "images/cure/plotting/fusion_net/plotting_fusion_network_efficient_net_self_attention_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_cure":
            "images/cure/conf_mtx/stream_net/conf_mtx_efficient_net",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_self_attention_attention_cure":
            "images/cure/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_self_attention_attention",

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I ++++++++++++++++++++++++++++++++++++++++++++++++
        "stream_images_ogyei_anchor":
            "images/ogyei/stream_images/anchor",
        "stream_images_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "train_contour_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/contour",
        "train_lbp_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/lbp",
        "train_rgb_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/rgb",
        "train_texture_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/texture",

        # ----------------------------------------------- P O S   N E G ------------------------------------------------
        "contour_stream_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg/contour",
        "lbp_stream_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg/lbp",
        "rgb_stream_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg/rgb",
        "texture_stream_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg/texture",

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
        "plotting_efficient_net_ogyei":
            "images/ogyei/plotting/stream_net/plotting_efficient_net",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_self_attention_attention_ogyei":
            "images/ogyei/plotting/fusion_net/plotting_fusion_network_efficient_net_self_attention_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_ogyei":
            "images/ogyei/conf_mtx/stream_net/conf_mtx_efficient_net",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_self_attention_attention_ogyei":
            "images/ogyei/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_self_attention_attention",

        # ------------------------------------- I M A G E   A U G M E N T A T I O N ------------------------------------
        "images_aug":
            "images/aug/images_aug",
        "wo_background":
            "images/aug/wo_background",

        # -------------------------------------------------- U N E T ---------------------------------------------------
        "unet_out":
            "images/unet/unet_out",
        "unet_compare":
            "images/unet/unet_compare",

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

        "cure_k_fold":
            "data/cure/k_fold",
        "ogyei_k_fold":
            "data/ogyei/k_fold",

        # --------------------------------------------------- U N E T --------------------------------------------------
        # ----------------------------------------------- W E I G H T S ------------------------------------------------
        # UNet
        "weights_unet":
            "data/unet/weights",

        # -------------------------------------------------- L O G S ---------------------------------------------------
        # UNet
        "logs_unet":
            "data/unet/logs",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I +++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # CNN - StreamNetwork
        "weights_cnn_network_contour_ogyei":
            "data/ogyei/weights/stream_net/cnn/weights_cnn_network_contour",
        "weights_cnn_network_lbp_ogyei":
            "data/ogyei/weights/stream_net/cnn/weights_cnn_network_lbp",
        "weights_cnn_network_rgb_ogyei":
            "data/ogyei/weights/stream_net/cnn/weights_cnn_network_rgb",
        "weights_cnn_network_texture_ogyei":
            "data/ogyei/weights/stream_net/cnn/weights_cnn_network_texture",

        # EfficientNet b0 - StreamNetwork
        "weights_efficient_net_contour_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_cnn_ogyei":
            "data/ogyei/weights/fusion_net/cnn",
        "weights_fusion_network_efficient_net_self_attention_ogyei":
            "data/ogyei/weights/fusion_net/efficient_net_self_attention",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # CNN
        "logs_cnn_contour_ogyei":
            "data/ogyei/logs/stream_net/cnn/logs_cnn_contour",
        "logs_cnn_lbp_ogyei":
            "data/ogyei/logs/stream_net/cnn/logs_cnn_lbp",
        "logs_cnn_rgb_ogyei":
            "data/ogyei/logs/stream_net/cnn/logs_cnn_rgb",
        "logs_cnn_texture_ogyei":
            "data/ogyei/logs/stream_net/cnn/logs_cnn_texture",

        # EfficientNet b0
        "logs_efficient_net_contour_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_cnn_ogyei":
            "data/ogyei/logs/fusion_net/cnn",
        "logs_fusion_network_efficient_net_self_attention_ogyei":
            "data/ogyei/logs/fusion_net/efficient_net_self_attention",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_cnn_network_ogyei":
            "data/ogyei/predictions/stream_net/predictions_cnn_network",
        "predictions_efficient_net_ogyei":
            "data/ogyei/predictions/stream_net/predictions_efficient_net",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_cnn_ogyei":
            "data/ogyei/predictions/fusion_net/predictions_fusion_network_cnn",
        "predictions_fusion_network_efficient_net_self_attention_ogyei":
            "data/ogyei/predictions/fusion_net/predictions_fusion_network_efficient_net_self_attention_net",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_cnn_network_ogyei":
            "data/ogyei/ref_vec/stream_net/reference_vectors_cnn_network",
        "reference_vectors_efficient_net_ogyei":
            "data/ogyei/ref_vec/stream_net/reference_vectors_efficient_net",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_cnn_ogyei":
            "data/ogyei/ref_vec/fusion_net/reference_vectors_fusion_net_cnn",
        "reference_vectors_fusion_network_efficient_net_self_attention_ogyei":
            "data/ogyei/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_self_attention",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_cnn_contour_ogyei":
            "data/ogyei/hardest_samples/cnn/contour",
        "hardest_samples_cnn_lbp_ogyei":
            "data/ogyei/hardest_samples/cnn/lbp",
        "hardest_samples_cnn_rgb_ogyei":
            "data/ogyei/hardest_samples/cnn/rgb",
        "hardest_samples_cnn_texture_ogyei":
            "data/ogyei/hardest_samples/cnn/texture",

        "hardest_samples_efficient_net_contour_ogyei":
            "data/ogyei/hardest_samples/efficient_net/contour",
        "hardest_samples_efficient_net_lbp_ogyei":
            "data/ogyei/hardest_samples/efficient_net/lbp",
        "hardest_samples_efficient_net_rgb_ogyei":
            "data/ogyei/hardest_samples/efficient_net/rgb",
        "hardest_samples_efficient_net_texture_ogyei":
            "data/ogyei/hardest_samples/efficient_net/texture",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ C U R E ++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # CNN - StreamNetwork
        "weights_cnn_network_contour_cure":
            "data/cure/weights/stream_net/cnn/weights_cnn_network_contour",
        "weights_cnn_network_lbp_cure":
            "data/cure/weights/stream_net/cnn/weights_cnn_network_lbp",
        "weights_cnn_network_rgb_cure":
            "data/cure/weights/stream_net/cnn/weights_cnn_network_rgb",
        "weights_cnn_network_texture_cure":
            "data/cure/weights/stream_net/cnn/weights_cnn_network_texture",

        # EfficientNet b0 - StreamNetwork
        "weights_efficient_net_contour_cure":
            "data/cure/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp_cure":
            "data/cure/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb_cure":
            "data/cure/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture_cure":
            "data/cure/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_cnn_cure":
            "data/cure/weights/fusion_net/cnn",
        "weights_fusion_network_efficient_net_self_attention_cure":
            "data/cure/weights/fusion_net/efficient_net_self_attention",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # CNN
        "logs_cnn_contour_cure":
            "data/cure/logs/stream_net/cnn/logs_cnn_contour",
        "logs_cnn_lbp_cure":
            "data/cure/logs/stream_net/cnn/logs_cnn_lbp",
        "logs_cnn_rgb_cure":
            "data/cure/logs/stream_net/cnn/logs_cnn_rgb",
        "logs_cnn_texture_cure":
            "data/cure/logs/stream_net/cnn/logs_cnn_texture",

        # EfficientNet b0
        "logs_efficient_net_contour_cure":
            "data/cure/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp_cure":
            "data/cure/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb_cure":
            "data/cure/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture_cure":
            "data/cure/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_cnn_cure":
            "data/cure/logs/fusion_net/cnn",
        "logs_fusion_network_efficient_net_self_attention_cure":
            "data/cure/logs/fusion_net/efficient_net_self_attention",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_cnn_network_cure":
            "data/cure/predictions/stream_net/predictions_cnn_network",
        "predictions_efficient_net_cure":
            "data/cure/predictions/stream_net/predictions_efficient_net",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_cnn_cure":
            "data/cure/predictions/fusion_net/predictions_fusion_network_cnn",
        "predictions_fusion_network_efficient_net_self_attention_cure":
            "data/cure/predictions/fusion_net/predictions_fusion_network_efficient_net_self_attention_net",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_cnn_network_cure":
            "data/cure/ref_vec/stream_net/reference_vectors_cnn_network",
        "reference_vectors_efficient_net_cure":
            "data/cure/ref_vec/stream_net/reference_vectors_efficient_net",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_cnn_cure":
            "data/cure/ref_vec/fusion_net/reference_vectors_fusion_net_cnn",
        "reference_vectors_fusion_net_efficient_net_self_attention_cure":
            "data/cure/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_self_attention",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_cnn_contour_cure":
            "data/cure/hardest_samples/cnn/contour",
        "hardest_samples_cnn_lbp_cure":
            "data/cure/hardest_samples/cnn/lbp",
        "hardest_samples_cnn_rgb_cure":
            "data/cure/hardest_samples/cnn/rgb",
        "hardest_samples_cnn_texture_cure":
            "data/cure/hardest_samples/cnn/texture",

        "hardest_samples_efficient_net_contour_cure":
            "data/cure/hardest_samples/efficient_net/contour",
        "hardest_samples_efficient_net_lbp_cure":
            "data/cure/hardest_samples/efficient_net/lbp",
        "hardest_samples_efficient_net_rgb_cure":
            "data/cure/hardest_samples/efficient_net/rgb",
        "hardest_samples_efficient_net_texture_cure":
            "data/cure/hardest_samples/efficient_net/texture",
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
        # --------------------------------------------------- D T D ----------------------------------------------------
        "dtd_images":
            "dtd_images",

        # -------------------------------------------------- C U R E ---------------------------------------------------
        # CUSTOMER
        "cure_customer_images":
            "cure/Customer/images",
        "cure_customer_mask_images":
            "cure/Customer/mask_images",
        "cure_customer_segmentation_labels":
            "cure/Customer/segmentation_labels",

        # REFERENCE
        "cure_reference_images":
            "cure/Reference/images",
        "cure_reference_mask_images":
            "cure/Reference/mask_images",
        "cure_reference_segmentation_labels":
            "cure/Reference/segmentation_labels",
        "cure_reference_yolo_labels":
            "cure/Reference/yolo_labels",

        # CUSTOMER SPLITTED
        "cure_train_images":
            "cure/Customer_splitted/train_dir/images",
        "cure_train_mask_images":
            "cure/Customer_splitted/train_dir/mask_images",
        "cure_train_segmentation_labels":
            "cure/Customer_splitted/train_dir/segmentation_labels",

        "cure_valid_images":
            "cure/Customer_splitted/valid_dir/images",
        "cure_valid_mask_images":
            "cure/Customer_splitted/valid_dir/mask_images",
        "cure_valid_segmentation_labels":
            "cure/Customer_splitted/valid_dir/segmentation_labels",

        "cure_test_images":
            "cure/Customer_splitted/test_dir/images",
        "cure_test_mask_images":
            "cure/Customer_splitted/test_dir/mask_images",
        "cure_test_segmentation_labels":
            "cure/Customer_splitted/test_dir/segmentation_labels",

        # CUSTOMER SPLITTED AUGMENTED
        "cure_train_aug_images":
            "cure/Customer_splitted_aug/train_dir/images",
        "cure_train_aug_yolo_labels":
            "cure/Customer_splitted_aug/train_dir/yolo_labels",
        "cure_train_aug_mask_images":
            "cure/Customer_splitted_aug/train_dir/mask_images",

        "cure_valid_aug_images":
            "cure/Customer_splitted_aug/valid_dir/images",
        "cure_valid_aug_yolo_labels":
            "cure/Customer_splitted_aug/valid_dir/yolo_labels",
        "cure_valid_aug_mask_images":
            "cure/Customer_splitted_aug/valid_dir/mask_images",

        "cure_test_aug_images":
            "cure/Customer_splitted_aug/test_dir/images",
        "cure_test_aug_yolo_labels":
            "cure/Customer_splitted_aug/test_dir/yolo_labels",

        # STREAM - Customer
        "stream_images_cure_customer":
            "cure/Customer/stream_images",
        "stream_images_cure_customer_contour":
            "cure/Customer/stream_images/contour",
        "stream_images_cure_customer_lbp":
            "cure/Customer/stream_images/lbp",
        "stream_images_cure_customer_rgb":
            "cure/Customer/stream_images/rgb",
        "stream_images_cure_customer_texture":
            "cure/Customer/stream_images/texture",

        # STREAM - Reference
        "stream_images_cure_reference":
            "cure/Reference/stream_images",
        "stream_images_cure_reference_contour":
            "cure/Reference/stream_images/contour",
        "stream_images_cure_reference_lbp":
            "cure/Reference/stream_images/lbp",
        "stream_images_cure_reference_rgb":
            "cure/Reference/stream_images/rgb",
        "stream_images_cure_reference_texture":
            "cure/Reference/stream_images/texture",

        # ------------------------------------------------- O G Y E I --------------------------------------------------
        # CUSTOMER
        "ogyei_customer_images":
            "ogyei/Customer/images",
        "ogyei_customer_segmentation_labels":
            "ogyei/Customer/segmentation_labels",
        "ogyei_customer_mask_images":
            "ogyei/Customer/mask_images",

        # REFERENCE
        "ogyei_reference_images":
            "ogyei/Reference/images",
        "ogyei_reference_segmentation_labels":
            "ogyei/Reference/segmentation_labels",
        "ogyei_reference_mask_images":
            "ogyei/Reference/mask_images",

        # CUSTOMER SPLITTED
        "ogyei_train_images":
            "ogyei/Customer_splitted/train_dir/images",
        "ogyei_train_mask_images":
            "ogyei/Customer_splitted/train_dir/mask_images",
        "ogyei_train_segmentation_labels":
            "ogyei/Customer_splitted/train_dir/segmentation_labels",

        "ogyei_valid_images":
            "ogyei/Customer_splitted/valid_dir/images",
        "ogyei_valid_mask_images":
            "ogyei/Customer_splitted/valid_dir/mask_images",
        "ogyei_valid_segmentation_labels":
            "ogyei/Customer_splitted/valid_dir/segmentation_labels",

        "ogyei_test_images":
            "ogyei/Customer_splitted/test_dir/images",
        "ogyei_test_mask_images":
            "ogyei/Customer_splitted/test_dir/mask_images",
        "ogyei_test_segmentation_labels":
            "ogyei/Customer_splitted/test_dir/segmentation_labels",

        # STREAM - Customer
        "stream_images_ogyei_customer":
            "ogyei/Customer/stream_images",
        "stream_images_ogyei_customer_contour":
            "ogyei/Customer/stream_images/contour",
        "stream_images_ogyei_customer_lbp":
            "ogyei/Customer/stream_images/lbp",
        "stream_images_ogyei_customer_rgb":
            "ogyei/Customer/stream_images/rgb",
        "stream_images_ogyei_customer_texture":
            "ogyei/Customer/stream_images/texture",

        # STREAM - Reference
        "stream_images_ogyei_reference":
            "ogyei/Reference/stream_images",
        "stream_images_ogyei_reference_contour":
            "ogyei/Reference/stream_images/contour",
        "stream_images_ogyei_reference_lbp":
            "ogyei/Reference/stream_images/lbp",
        "stream_images_ogyei_reference_rgb":
            "ogyei/Reference/stream_images/rgb",
        "stream_images_ogyei_reference_texture":
            "ogyei/Reference/stream_images/texture"
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


CONST: _Const = _Const()
IMAGES_PATH: Images = Images()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()
