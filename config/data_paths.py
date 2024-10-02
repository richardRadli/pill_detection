"""
File: data_paths.py
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
            "STORAGE_ROOT ":
                "D:/storage/pill_detection/KDIR2023",
            "DATASET_ROOT":
                "D:/storage/pill_detection/KDIR2023/datasets",
            "PROJECT_ROOT":
                "C:/Users/ricsi/Documents/project/IVM",
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        STORAGE_ROOT = root_info["STORAGE_ROOT "]
        DATASET_ROOT = root_info["DATASET_ROOT"]
        PROJECT_ROOT = root_info["PROJECT_ROOT"]
    else:
        raise ValueError("Wrong user!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def create_directories(cls, dirs, root_type) -> None:
        """
        Class method that creates the missing directories.
        
        Args:
            dirs: These are the directories that the function checks.
            root_type: Either STORAGE or DATASET.

        Returns:
             None
        """

        for _, path in dirs.items():
            if root_type == "STORAGE":
                dir_path = os.path.join(cls.STORAGE_ROOT, path)
            elif root_type == "PROJECT":
                dir_path = os.path.join(cls.PROJECT_ROOT, path)
            elif root_type == "DATASET":
                dir_path = os.path.join(cls.DATASET_ROOT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


class ConfigFilePaths(_Const):
    dirs_config_paths = {
        "config_augmentation":
            "config/json_files/augmentation_config.json",
        "config_schema_augmentation":
            "config/json_files/augmentation_config_schema.json",

        "config_fusion_net":
            "config/json_files/fusion_net_config.json",
        "config_schema_fusion_net":
            "config/json_files/fusion_net_config_schema.json",

        "config_stream_images":
            "config/json_files/stream_images_config.json",
        "config_schema_stream_images":
            "config/json_files/stream_images_config_schema.json",

        "config_streamnet":
            "config/json_files/streamnet_config.json",
        "config_schema_streamnet":
            "config/json_files/streamnet_config_schema.json",

        "config_unet":
            "config/json_files/unet_config.json",
        "config_schema_unet":
            "config/json_files/unet_config_schema.json"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_config_paths.get(key, ""))


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
        "contour_stream_cure_anchor":
            "images/cure/stream_images/anchor/contour",
        "lbp_stream_cure_anchor":
            "images/cure/stream_images/anchor/lbp",
        "rgb_stream_cure_anchor":
            "images/cure/stream_images/anchor/rgb",
        "texture_stream_cure_anchor":
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
        "query_cure":
            "images/cure/test/query",
        "contour_stream_query_cure":
            "images/cure/test/query/contour",
        "lbp_stream_query_cure":
            "images/cure/test/query/lbp",
        "rgb_stream_query_cure":
            "images/cure/test/query/rgb",
        "texture_stream_query_cure":
            "images/cure/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "ref_cure":
            "images/cure/test/ref",
        "contour_stream_ref_cure":
            "images/cure/test/ref/contour",
        "lbp_stream_ref_cure":
            "images/cure/test/ref/lbp",
        "rgb_stream_ref_cure":
            "images/cure/test/ref/rgb",
        "texture_stream_ref_cure":
            "images/cure/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_v2_cure":
            "images/cure/plotting/stream_net/efficient_net_v2",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_v2_self_attention_cure":
            "images/cure/plotting/fusion_net/fusion_network_efficient_net_v2_self_attention",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_cure":
            "images/cure/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention",
        "plotting_fusion_network_efficient_net_v2_MHAFMHA_cure":
            "images/cure/plotting/fusion_net/fusion_network_efficient_net_v2_MHAFMHA",

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I ++++++++++++++++++++++++++++++++++++++++++++++++
        "stream_images_ogyei_anchor":
            "images/ogyei/stream_images/anchor",
        "stream_images_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "contour_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/contour",
        "lbp_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/lbp",
        "rgb_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/rgb",
        "texture_stream_ogyei_anchor":
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
        "query_ogyei":
            "images/ogyei/test/query",
        "contour_stream_query_ogyei":
            "images/ogyei/test/query/contour",
        "lbp_stream_query_ogyei":
            "images/ogyei/test/query/lbp",
        "rgb_stream_query_ogyei":
            "images/ogyei/test/query/rgb",
        "texture_stream_query_ogyei":
            "images/ogyei/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "ref_ogyei":
            "images/ogyei/test/ref",
        "contour_stream_ref_ogyei":
            "images/ogyei/test/ref/contour",
        "lbp_stream_ref_ogyei":
            "images/ogyei/test/ref/lbp",
        "rgb_stream_ref_ogyei":
            "images/ogyei/test/ref/rgb",
        "texture_stream_ref_ogyei":
            "images/ogyei/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_v2_ogyei":
            "images/ogyei/plotting/stream_net/efficient_net_v2",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_v2_self_attention_ogyei":
            "images/ogyei/plotting/fusion_net/fusion_network_efficient_net_v2_self_attention",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei":
            "images/ogyei/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention",
        "plotting_fusion_network_efficient_net_v2_MHAFMHA_ogyei":
            "images/ogyei/plotting/fusion_net/fusion_network_efficient_net_v2_MHAFMHA",
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_images, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_images.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Data(_Const):
    dirs_data = {
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNetV2 - StreamNetwork
        "weights_efficient_net_v2_contour_cure":
            "data/cure/weights/stream_net/efficient_net_v2/contour",
        "weights_efficient_net_v2_lbp_cure":
            "data/cure/weights/stream_net/efficient_net_v2/lbp",
        "weights_efficient_net_v2_rgb_cure":
            "data/cure/weights/stream_net/efficient_net_v2/rgb",
        "weights_efficient_net_v2_texture_cure":
            "data/cure/weights/stream_net/efficient_net_v2/texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_v2_self_attention_cure":
            "data/cure/weights/fusion_net/efficient_net_v2_self_attention",
        "weights_fusion_network_efficient_net_v2_multihead_attention_cure":
            "data/cure/weights/fusion_net/efficient_net_v2_multihead_attention",
        "weights_fusion_network_efficient_net_v2_MHAFMHA_cure":
            "data/cure/weights/fusion_net/efficient_net_v2_MHAFMHA",


        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNetV2
        "logs_efficient_net_v2_contour_cure":
            "data/cure/logs/stream_net/efficient_net_v2/contour",
        "logs_efficient_net_v2_lbp_cure":
            "data/cure/logs/stream_net/efficient_net_v2/lbp",
        "logs_efficient_net_v2_rgb_cure":
            "data/cure/logs/stream_net/efficient_net_v2/rgb",
        "logs_efficient_net_v2_texture_cure":
            "data/cure/logs/stream_net/efficient_net_v2/texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_v2_self_attention_cure":
            "data/cure/logs/fusion_net/efficient_net_v2_self_attention",
        "logs_fusion_network_efficient_net_v2_multihead_attention_cure":
            "data/cure/logs/fusion_net/efficient_net_v2_multihead_attention",
        "logs_fusion_network_efficient_net_v2_MHAFMHA_cure":
            "data/cure/logs/fusion_net/efficient_net_v2_MHAFMHA",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_v2_cure":
            "data/cure/predictions/stream_net/efficient_net_v2",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_v2_self_attention_cure":
            "data/cure/predictions/fusion_net/efficient_net_v2_self_attention",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_cure":
            "data/cure/predictions/fusion_net/efficient_net_v2_multihead_attention",
        "predictions_fusion_network_efficient_net_v2_MHAFMHA_cure":
            "data/cure/predictions/fusion_net/efficient_net_v2_MHAFMHA",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_v2_cure":
            "data/cure/ref_vec/stream_net/efficient_net_v2",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "ref_vec_fusion_network_efficient_net_v2_self_attention_cure":
            "data/cure/ref_vec/fusion_net/efficient_net_v2_self_attention",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure":
            "data/cure/ref_vec/fusion_net/efficient_net_v2_multihead_attention",
        "ref_vec_fusion_network_efficient_net_v2_MHAFMHA_cure":
            "data/cure/ref_vec/fusion_net/efficient_net_v2_MHAFMHA",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_v2_contour_cure":
            "data/cure/hardest_samples/efficient_net_v2/contour",
        "hardest_samples_efficient_net_v2_lbp_cure":
            "data/cure/hardest_samples/efficient_net_v2/lbp",
        "hardest_samples_efficient_net_v2_rgb_cure":
            "data/cure/hardest_samples/efficient_net_v2/rgb",
        "hardest_samples_efficient_net_v2_texture_cure":
            "data/cure/hardest_samples/efficient_net_v2/texture",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I +++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNetV2 - StreamNetwork
        "weights_efficient_net_v2_contour_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/contour",
        "weights_efficient_net_v2_lbp_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/lbp",
        "weights_efficient_net_v2_rgb_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/rgb",
        "weights_efficient_net_v2_texture_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_v2_self_attention_ogyei":
            "data/ogyei/weights/fusion_net/efficient_net_v2_self_attention",
        "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei":
            "data/ogyei/weights/fusion_net/efficient_net_v2_multihead_attention",
        "weights_fusion_network_efficient_net_v2_MHAFMHA_ogyei":
            "data/ogyei/weights/fusion_net/efficient_net_v2_MHAFMHA",


        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNetV2
        "logs_efficient_net_v2_contour_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/contour",
        "logs_efficient_net_v2_lbp_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/lbp",
        "logs_efficient_net_v2_rgb_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/rgb",
        "logs_efficient_net_v2_texture_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_v2_self_attention_ogyei":
            "data/ogyei/logs/fusion_net/efficient_net_v2_self_attention",
        "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei":
            "data/ogyei/logs/fusion_net/efficient_net_v2_multihead_attention",
        "logs_fusion_network_efficient_net_v2_MHAFMHA_ogyei":
            "data/ogyei/logs/fusion_net/efficient_net_v2_MHAFMHA",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_v2_ogyei":
            "data/ogyei/predictions/stream_net/efficient_net_v2",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_v2_self_attention_ogyei":
            "data/ogyei/predictions/fusion_net/efficient_net_v2_self_attention",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei":
            "data/ogyei/predictions/fusion_net/efficient_net_v2_multihead_attention",
        "predictions_fusion_network_efficient_net_v2_MHAFMHA_ogyei":
            "data/ogyei/predictions/fusion_net/efficient_net_v2_MHAFMHA",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_v2_ogyei":
            "data/ogyei/ref_vec/stream_net/efficient_net_v2",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "ref_vec_fusion_network_efficient_net_v2_self_attention_ogyei":
            "data/ogyei/ref_vec/fusion_net/efficient_net_v2_self_attention",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei":
            "data/ogyei/ref_vec/fusion_net/efficient_net_v2_multihead_attention",
        "ref_vec_fusion_network_efficient_net_v2_MHAFMHA_ogyei":
            "data/ogyei/ref_vec/fusion_net/efficient_net_v2_MHAFMHA",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_v2_contour_ogyei":
            "data/ogyei/hardest_samples/efficient_net_v2/contour",
        "hardest_samples_efficient_net_v2_lbp_ogyei":
            "data/ogyei/hardest_samples/efficient_net_v2/lbp",
        "hardest_samples_efficient_net_v2_rgb_ogyei":
            "data/ogyei/hardest_samples/efficient_net_v2/rgb",
        "hardest_samples_efficient_net_v2_texture_ogyei":
            "data/ogyei/hardest_samples/efficient_net_v2/texture"
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data, "STORAGE")

    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_data.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Datasets(_Const):
    dirs_dataset = {
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
            "ogyei_v2/Reference/stream_images/texture",

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

        # ------------------------------------------------- O G Y E I --------------------------------------------------
        # CUSTOMER
        "ogyei_customer_images":
            "ogyei_v2/Customer/images",
        "ogyei_customer_segmentation_labels":
            "ogyei_v2/Customer/segmentation_labels",
        "ogyei_customer_mask_images":
            "ogyei_v2/Customer/mask_images",

        # REFERENCE
        "ogyei_reference_images":
            "ogyei_v2/Reference/images",
        "ogyei_reference_segmentation_labels":
            "ogyei_v2/Reference/segmentation_labels",
        "ogyei_reference_mask_images":
            "ogyei_v2/Reference/mask_images",

        # STREAM - Customer
        "stream_images_ogyei_customer":
            "ogyei_v2/Customer/stream_images",
        "stream_images_ogyei_customer_contour":
            "ogyei_v2/Customer/stream_images/contour",
        "stream_images_ogyei_customer_lbp":
            "ogyei_v2/Customer/stream_images/lbp",
        "stream_images_ogyei_customer_rgb":
            "ogyei_v2/Customer/stream_images/rgb",
        "stream_images_ogyei_customer_texture":
            "ogyei_v2/Customer/stream_images/texture",

        # STREAM - Reference
        "stream_images_ogyei_reference":
            "ogyei_v2/Reference/stream_images",
        "stream_images_ogyei_reference_contour":
            "ogyei_v2/Reference/stream_images/contour",
        "stream_images_ogyei_reference_lbp":
            "ogyei_v2/Reference/stream_images/lbp",
        "stream_images_ogyei_reference_rgb":
            "ogyei_v2/Reference/stream_images/rgb",
        "stream_images_ogyei_reference_texture":
            "ogyei_v2/Reference/stream_images/texture",

        # UNSPLITTED
        "ogyei_images":
            "ogyei_v2/unsplitted/images",
        "ogyei_mask_images":
            "ogyei_v2/unsplitted/gt_masks",
        "ogyei_segmentation_labels":
            "ogyei_v2/unsplitted/labels",

        # SPLITTED
        "ogyei_train_images":
            "ogyei_v2/splitted/train/images",
        "ogyei_train_mask_images":
            "ogyei_v2/splitted/train/gt_train_masks",
        "ogyei_train_segmentation_labels":
            "ogyei_v2/splitted/train/labels",

        "ogyei_valid_images":
            "ogyei_v2/splitted/valid/images",
        "ogyei_valid_mask_images":
            "ogyei_v2/splitted/valid/gt_valid_masks",
        "ogyei_valid_segmentation_labels":
            "ogyei_v2/splitted/valid/labels",

        "ogyei_test_images":
            "ogyei_v2/splitted/test/images",
        "ogyei_test_mask_images":
            "ogyei_v2/splitted/test/gt_test_masks",
        "ogyei_test_segmentation_labels":
            "ogyei_v2/splitted/test/labels",

        "dtd":
            "dtd_images"
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
JSON_FILES_PATHS: ConfigFilePaths = ConfigFilePaths()
IMAGES_PATH: Images = Images()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()
