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
                "D:/storage/pill_detection/VISAPP2024",
            "DATASET_ROOT":
                "D:/storage/pill_detection/VISAPP2024/datasets",
            "PROJECT_ROOT":
                "C:/Users/ricsi/Documents/project/IVM",
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        STORAGE_ROOT = root_info["STORAGE_ROOT"]
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
            "config/json_files/streamnet_config_schema.json"
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
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I ++++++++++++++++++++++++++++++++++++++++++++++++
        "stream_images_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor",
        "stream_images_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "contour_stream_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor/contour",
        "lbp_stream_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor/lbp",
        "rgb_stream_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor/rgb",
        "texture_stream_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor/texture",

        # ----------------------------------------------- P O S   N E G ------------------------------------------------
        "contour_stream_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg/contour",
        "lbp_stream_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg/lbp",
        "rgb_stream_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg/rgb",
        "texture_stream_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg/texture",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "query_ogyei_v2":
            "images/ogyei_v2/test/query",
        "contour_stream_query_ogyei_v2":
            "images/ogyei_v2/test/query/contour",
        "lbp_stream_query_ogyei_v2":
            "images/ogyei_v2/test/query/lbp",
        "rgb_stream_query_ogyei_v2":
            "images/ogyei_v2/test/query/rgb",
        "texture_stream_query_ogyei_v2":
            "images/ogyei_v2/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "ref_ogyei_v2":
            "images/ogyei_v2/test/ref",
        "contour_stream_ref_ogyei_v2":
            "images/ogyei_v2/test/ref/contour",
        "lbp_stream_ref_ogyei_v2":
            "images/ogyei_v2/test/ref/lbp",
        "rgb_stream_ref_ogyei_v2":
            "images/ogyei_v2/test/ref/rgb",
        "texture_stream_ref_ogyei_v2":
            "images/ogyei_v2/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_v2_ogyei_v2_hmtl":
            "images/ogyei_v2/plotting/stream_net/efficient_net_v2/hmtl",
        "plotting_efficient_net_v2_ogyei_v2_dmtl":
            "images/ogyei_v2/plotting/stream_net/efficient_net_v2/dmtl",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "images/ogyei_v2/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/hmtl",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl":
            "images/ogyei_v2/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/dmtl"
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
        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I +++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNetV2 - StreamNetwork
        "weights_efficient_net_v2_contour_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/contour/hmtl",
        "weights_efficient_net_v2_lbp_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/lbp/hmtl",
        "weights_efficient_net_v2_rgb_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/rgb/hmtl",
        "weights_efficient_net_v2_texture_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/texture/hmtl",

        "weights_efficient_net_v2_contour_ogyei_v2_dmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/contour/dmtl",
        "weights_efficient_net_v2_lbp_ogyei_v2_dmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/lbp/dmtl",
        "weights_efficient_net_v2_rgb_ogyei_v2_dmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/rgb/dmtl",
        "weights_efficient_net_v2_texture_ogyei_v2_dmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/texture/dmtl",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl":
            "data/ogyei_v2/weights/fusion_net/efficient_net_v2_multihead_attention/dmtl",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNetV2
        "logs_efficient_net_v2_contour_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/contour/hmtl",
        "logs_efficient_net_v2_lbp_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/lbp/hmtl",
        "logs_efficient_net_v2_rgb_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/rgb/hmtl",
        "logs_efficient_net_v2_texture_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/texture/hmtl",

        "logs_efficient_net_v2_contour_ogyei_v2_dmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/contour/dmtl",
        "logs_efficient_net_v2_lbp_ogyei_v2_dmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/lbp/dmtl",
        "logs_efficient_net_v2_rgb_ogyei_v2_dmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/rgb/dmtl",
        "logs_efficient_net_v2_texture_ogyei_v2_dmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/texture/dmtl",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl":
            "data/ogyei_v2/logs/fusion_net/efficient_net_v2_multihead_attention/dmtl",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_v2_ogyei_v2_hmtl":
            "data/ogyei_v2/predictions/stream_net/efficient_net_v2/hmtl",
        "predictions_efficient_net_v2_ogyei_v2_dmtl":
            "data/ogyei_v2/predictions/stream_net/efficient_net_v2/dmtl",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "data/ogyei_v2/predictions/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl":
            "data/ogyei_v2/predictions/fusion_net/efficient_net_v2_multihead_attention/dmtl",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_v2_ogyei_v2_hmtl":
            "data/ogyei_v2/ref_vec/stream_net/efficient_net_v2/hmtl",
        "reference_vectors_efficient_net_v2_ogyei_v2_dmtl":
            "data/ogyei_v2/ref_vec/stream_net/efficient_net_v2/dmtl",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "data/ogyei_v2/ref_vec/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl":
            "data/ogyei_v2/ref_vec/fusion_net/efficient_net_v2_multihead_attention/dmtl",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_v2_contour_ogyei_v2_hmtl":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/contour/hmtl",
        "hardest_samples_efficient_net_v2_lbp_ogyei_v2_hmtl":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/lbp/hmtl",
        "hardest_samples_efficient_net_v2_rgb_ogyei_v2_hmtl":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/rgb/hmtl",
        "hardest_samples_efficient_net_v2_texture_ogyei_v2_hmtl":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/texture/hmtl",

        "hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl":
        "data/ogyei_v2/hardest_samples/efficient_net_v2/contour/dmtl",
        "hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl":
        "data/ogyei_v2/hardest_samples/efficient_net_v2/lbp/dmtl",
        "hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl":
        "data/ogyei_v2/hardest_samples/efficient_net_v2/rgb/dmtl",
        "hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl":
        "data/ogyei_v2/hardest_samples/efficient_net_v2/texture/dmtl",

        "ogyei_v2_k_fold":
            "data/ogyei_v2/k_fold"
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data, "STORAGE")

    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_data.get(key, ""))


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
        self.create_directories(self.nlp_data, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.nlp_data.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Datasets(_Const):
    dirs_dataset = {
        # ------------------------------------------------- O G Y E I --------------------------------------------------
        # CUSTOMER
        "ogyei_v2_customer_images":
            "ogyei_v2/Customer/images",
        "ogyei_v2_customer_segmentation_labels":
            "ogyei_v2/Customer/segmentation_labels",
        "ogyei_v2_customer_mask_images":
            "ogyei_v2/Customer/mask_images",

        # REFERENCE
        "ogyei_v2_reference_images":
            "ogyei_v2/Reference/images",
        "ogyei_v2_reference_segmentation_labels":
            "ogyei_v2/Reference/segmentation_labels",
        "ogyei_v2_reference_mask_images":
            "ogyei_v2/Reference/mask_images",

        # STREAM - Customer
        "stream_images_ogyei_v2_customer":
            "ogyei_v2/Customer/stream_images",
        "stream_images_ogyei_v2_customer_contour":
            "ogyei_v2/Customer/stream_images/contour",
        "stream_images_ogyei_v2_customer_lbp":
            "ogyei_v2/Customer/stream_images/lbp",
        "stream_images_ogyei_v2_customer_rgb":
            "ogyei_v2/Customer/stream_images/rgb",
        "stream_images_ogyei_v2_customer_texture":
            "ogyei_v2/Customer/stream_images/texture",

        # STREAM - Reference
        "stream_images_ogyei_v2_reference":
            "ogyei_v2/Reference/stream_images",
        "stream_images_ogyei_v2_reference_contour":
            "ogyei_v2/Reference/stream_images/contour",
        "stream_images_ogyei_v2_reference_lbp":
            "ogyei_v2/Reference/stream_images/lbp",
        "stream_images_ogyei_v2_reference_rgb":
            "ogyei_v2/Reference/stream_images/rgb",
        "stream_images_ogyei_v2_reference_texture":
            "ogyei_v2/Reference/stream_images/texture",

        # UNSPLITTED
        "ogyei_v2_images":
            "ogyei_v2/unsplitted/images",
        "ogyei_v2_mask_images":
            "ogyei_v2/unsplitted/gt_masks",
        "ogyei_v2_segmentation_labels":
            "ogyei_v2/unsplitted/labels",

        # SPLITTED
        "ogyei_v2_train_images":
            "ogyei_v2/splitted/train/images",
        "ogyei_v2_train_mask_images":
            "ogyei_v2/splitted/train/gt_train_masks",
        "ogyei_v2_train_segmentation_labels":
            "ogyei_v2/splitted/train/labels",

        "ogyei_v2_valid_images":
            "ogyei_v2/splitted/valid/images",
        "ogyei_v2_valid_mask_images":
            "ogyei_v2/splitted/valid/gt_valid_masks",
        "ogyei_v2_valid_segmentation_labels":
            "ogyei_v2/splitted/valid/labels",

        "ogyei_v2_test_images":
            "ogyei_v2/splitted/test/images",
        "ogyei_v2_test_mask_images":
            "ogyei_v2/splitted/test/gt_test_masks",
        "ogyei_v2_test_segmentation_labels":
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
NLP_DATA_PATH: NLPData = NLPData()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()
