import logging
import os


from config.logger_setup import setup_logger


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
            "PROJECT_ROOT": "C:/Users/ricsi/Documents/project/storage/IVM",
            "DATASET_ROOT": "C:/Users/ricsi/Documents/project/storage/IVM/datasets"
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
        "unet_out": "images/unet/unet_out",
        "unet_compare": "images/unet/unet_compare",

        "ref_train_contour": "images/references/contour/train",
        "ref_train_lbp": "images/references/lbp/train",
        "ref_train_rgb": "images/references/rgb/train",
        "ref_train_texture": "images/references/texture/train",

        "ref_valid_contour": "images/references/contour/valid",
        "ref_valid_lbp": "images/references/lbp/valid",
        "ref_valid_rgb": "images/references/rgb/valid",
        "ref_valid_texture": "images/references/texture/valid",

        "query_contour": "images/query/contour",
        "query_lbp": "images/query/lbp",
        "query_rgb": "images/query/rgb",
        "query_texture": "images/query/texture",

        "contour_hardest": "images/hardest_samples/contour_hardest",
        "lbp_hardest": "images/hardest_samples/lbp_hardest",
        "rgb_hardest": "images/hardest_samples/rgb_hardest",
        "texture_hardest": "images/hardest_samples/texture_hardest",

        "plotting_cnn_network": "images/plotting/plotting_cnn_network",
        "plotting_efficient_net": "images/plotting/plotting_efficient_net",
        "plotting_fusion_network": "images/plotting/plotting_fusion_network",
        "plotting_efficient_net_self_attention": "images/plotting/plotting_efficient_net_self_attention",

        "images_aug": "images/aug/images_aug",
        "wo_background": "images/aug/wo_background"
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
        # Weights
        "weights_unet": "data/weights/weights_unet",

        "weights_fusion_net": "data/weights/weights_fusion_net",

        "weights_cnn_network_contour": "data/weights/weights_cnn_network_contour",
        "weights_cnn_network_lbp": "data/weights/weights_cnn_network_lbp",
        "weights_cnn_network_rgb": "data/weights/weights_cnn_network_rgb",
        "weights_cnn_network_texture": "data/weights/weights_cnn_network_texture",

        "weights_efficient_net_contour": "data/weights/weights_efficient_net_contour",
        "weights_efficient_net_lbp": "data/weights/weights_efficient_net_lbp",
        "weights_efficient_net_rgb": "data/weights/weights_efficient_net_rgb",
        "weights_efficient_net_texture": "data/weights/weights_efficient_net_texture",

        "weights_efficient_net_self_attention_contour": "data/weights/weights_efficient_self_attention_net_contour",
        "weights_efficient_net_self_attention_lbp": "data/weights/weights_efficient_self_attention_net_lbp",
        "weights_efficient_net_self_attention_rgb": "data/weights/weights_efficient_self_attention_net_rgb",
        "weights_efficient_net_self_attention_texture": "data/weights/weights_efficient_self_attention_net_texture",

        # Logs
        "logs_unet": "data/logs/logs_unet",

        "logs_fusion_net": "data/logs/logs_fusion_net",

        "logs_cnn_contour": "data/logs/logs_cnn_contour",
        "logs_cnn_lbp": "data/logs/logs_cnn_lbp",
        "logs_cnn_rgb": "data/logs/logs_cnn_rgb",
        "logs_cnn_texture": "data/logs/logs_cnn_texture",

        "logs_efficient_net_contour": "data/logs/logs_efficient_net_contour",
        "logs_efficient_net_lbp": "data/logs/logs_efficient_net_lbp",
        "logs_efficient_net_rgb": "data/logs/logs_efficient_net_rgb",
        "logs_efficient_net_texture": "data/logs/logs_efficient_net_texture",

        "logs_efficient_net_self_attention_contour": "data/logs/logs_efficient_net_self_attention_contour",
        "logs_efficient_net_self_attention_lbp": "data/logs/logs_efficient_net_self_attention_lbp",
        "logs_efficient_net_self_attention_rgb": "data/logs/logs_efficient_net_self_attention_rgb",
        "logs_efficient_net_self_attention_texture": "data/logs/logs_efficient_net_self_attention_texture",

        # Predictions
        "predictions_cnn_network": "data/predictions/predictions_cnn_network",
        "predictions_fusion_network": "data/predictions/predictions_fusion_network",
        "predictions_efficient_net": "data/predictions/predictions_efficient_net",
        "predictions_efficient_self_attention_net": "data/predictions/predictions_efficient_self_attention_net",

        # Reference vectors
        "reference_vectors_cnn_network": "data/ref_vec/reference_vectors_cnn_network",
        "reference_vectors_efficient_net": "data/ref_vec/reference_vectors_efficient_net",
        "reference_vectors_efficient_net_self_attention": "data/ref_vec/reference_vectors_efficient_net_self_attention",

        # Hardest samples
        "negative": "data/hardest_samples/negative",

        # Other
        "augmented_train_data_labels": "data/other/augmented_train_data_labels",
        "cam_data": "data/other/cam_data",
        "train_labels": "data/other/train_labels",
        "test_labels": "data/other/test_labels"
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data, "PROJECT")

    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_data.get(key, ""))


class Datasets(_Const):
    dirs_dataset = {
        # C U R E
        "cure_customer": "cure/Customer",
        "cure_customer_mask": "cure/Customer_mask",
        "cure_reference": "cure/Reference",
        "cure_reference_mask": "cure/Reference_mask",
        "cure_train": "cure/train",
        "cure_train_mask": "cure/train_mask",
        "cure_test": "cure/test",
        "cure_test_mask": "cure/test_mask",

        # D T D
        "dtd_images": "dtd_images",

        # O G Y I   V 2
        # Unsplitted
        "ogyi_v2_unsplitted_images": "ogyi_v2/unsplitted/images",
        "ogyi_v2_unsplitted_labels": "ogyi_v2/unsplitted/labels",
        "ogyi_v2_unsplitted_gt_masks": "ogyi_v2/unsplitted/gt_masks",
        "ogyi_v2_unsplitted_pred_masks": "ogyi_v2/unsplitted/pred_masks",

        # Train images, labels and masks
        "ogyi_v2_splitted_train_images": "ogyi_v2/splitted/train/images",
        "ogyi_v2_splitted_train_labels": "ogyi_v2/splitted/train/labels",
        "ogyi_v2_splitted_gt_train_masks": "ogyi_v2/splitted/train/gt_train_masks",
        "ogyi_v2_splitted_pred_train_masks": "ogyi_v2/splitted/train/pred_train_masks",
        # Validation images, labels and masks
        "ogyi_v2_splitted_valid_images": "ogyi_v2/splitted/valid/images",
        "ogyi_v2_splitted_valid_labels": "ogyi_v2/splitted/valid/labels",
        "ogyi_v2_splitted_gt_valid_masks": "ogyi_v2/splitted/valid/gt_valid_masks",
        "ogyi_v2_splitted_pred_valid_masks": "ogyi_v2/splitted/valid/pred_valid_masks",
        # Test images, labels and masks
        "ogyi_v2_splitted_test_images": "ogyi_v2/splitted/test/images",
        "ogyi_v2_splitted_test_labels": "ogyi_v2/splitted/test/labels",
        "ogyi_v2_splitted_gt_test_masks": "ogyi_v2/splitted/test/gt_test_masks",
        "ogyi_v2_splitted_pred_test_masks": "ogyi_v2/splitted/test/pred_test_masks",

        # O G Y I   V 3
        "ogyi_v3_unsplitted_images": "ogyi_v3/unsplitted/images",
        "ogyi_v3_unsplitted_labels": "ogyi_v3/unsplitted/labels",
        "ogyi_v3_splitted_train_images": "ogyi_v3/splitted/train/images",
        "ogyi_v3_splitted_train_labels": "ogyi_v3/splitted/train/labels",
        "ogyi_v3_splitted_test_images": "ogyi_v3/splitted/test/images",
        "ogyi_v3_splitted_test_labels": "ogyi_v3/splitted/test/labels",
        "ogyi_v3_splitted_valid_images": "ogyi_v3/splitted/valid/images",
        "ogyi_v3_splitted_valid_labels": "ogyi_v3/splitted/valid/labels",

        # T R A Y
        "tray_original_images": "tray/original_images",
        "tray_diff_images": "tray/diff_images",
        "tray_plots": "tray/plots",
        "tray_images_aug": "tray/tray_images_aug",
        "tray_images_aug_w_med": "tray/tray_images_aug_w_med",
        "tray_images_aug_w_med_aug": "tray/tray_images_aug_w_med_aug"
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
