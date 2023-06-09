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
        "keplab": {
            "PROJECT_ROOT": "",
            "DATASET_ROOT": ""
        },
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
    def create_directories(cls, dirs):
        """

        :param dirs:
        :return:
        """

        for _, path in dirs.items():
            dir_path = os.path.join(cls.PROJECT_ROOT, path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++ I M A G E S +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Images(_Const):
    dirs_images = {
        "train_masks": "images/masks/train_masks",
        "test_masks": "images/masks/test_masks",
        "valid_masks": "images/masks/validation",

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

        "plotting_stream_network": "images/plotting/plotting_stream_network",
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
        self.create_directories(self.dirs_images)

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

        "weights_stream_network_contour": "data/weights/weights_stream_network_contour",
        "weights_stream_network_lbp": "data/weights/weights_stream_network_lbp",
        "weights_stream_network_rgb": "data/weights/weights_stream_network_rgb",
        "weights_stream_network_texture": "data/weights/weights_stream_network_texture",

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

        "logs_stream_contour": "data/logs/logs_stream_contour",
        "logs_stream_lbp": "data/logs/logs_stream_lbp",
        "logs_stream_rgb": "data/logs/logs_stream_rgb",
        "logs_stream_texture": "data/logs/logs_stream_texture",

        "logs_efficient_net_contour": "data/logs/logs_efficient_net_contour",
        "logs_efficient_net_lbp": "data/logs/logs_efficient_net_lbp",
        "logs_efficient_net_rgb": "data/logs/logs_efficient_net_rgb",
        "logs_efficient_net_texture": "data/logs/logs_efficient_net_texture",

        "logs_efficient_net_self_attention_contour": "data/logs/logs_efficient_net_self_attention_contour",
        "logs_efficient_net_sefl_attention_lbp": "data/logs/logs_efficient_net_self_attention_lbp",
        "logs_efficient_net_self_attention_rgb": "data/logs/logs_efficient_net_self_attention_rgb",
        "logs_efficient_net_self_attention_texture": "data/logs/logs_efficient_net_self_attention_texture",

        # Predictions
        "predictions_stream_network": "data/predictions/predictions_stream_network",
        "predictions_fusion_network": "data/predictions/predictions_fusion_network",
        "predictions_efficient_net": "data/predictions/predictions_efficient_net",
        "predictions_efficient_self_attention_net": "data/predictions/predictions_efficient_self_attention_net",

        # Reference vectors
        "reference_vectors_stream_network": "data/ref_vec/reference_vectors_stream_network",
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
        self.create_directories(self.dirs_data)

    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_data.get(key, ""))


class Datasets(_Const):
    dirs_dataset = {
        "ogyi_v2_unsplitted_images": "datasets/ogyi_v2/unsplitted/images",
        "ogyi_v2_unsplitted_labels": "datasets/ogyi_v2/unsplitted/labels",

        "ogyi_v2_splitted_train_images": "datasets/ogyi_v2/splitted/train/images",
        "ogyi_v2_splitted_train_labels": "datasets/ogyi_v2/splitted/train/labels",
        "ogyi_v2_splitted_valid_images": "datasets/ogyi_v2/splitted/valid/images",
        "ogyi_v2_splitted_valid_labels": "datasets/ogyi_v2/splitted/valid/labels",
        "ogyi_v2_splitted_test_images": "datasets/ogyi_v2/splitted/test/images",
        "ogyi_v2_splitted_test_labels": "datasets/ogyi_v2/splitted/test/labels",

        "ogyi_v3_unsplitted_images": "datasets/ogyi_v3/unsplitted/images",
        "ogyi_v3_unsplitted_labels": "datasets/ogyi_v3/unsplitted/labels",

        "ogyi_v3_splitted_train_images": "datasets/ogyi_v3/splitted/train/images",
        "ogyi_v3_splitted_train_labels": "datasets/ogyi_v3/splitted/train/labels",
        "ogyi_v3_splitted_test_images": "datasets/ogyi_v3/splitted/test/images",
        "ogyi_v3_splitted_test_labels": "datasets/ogyi_v3/splitted/test/labels",
        "ogyi_v3_splitted_valid_images": "datasets/ogyi_v3/splitted/valid/images",
        "ogyi_v3_splitted_valid_labels": "datasets/ogyi_v3/splitted/valid/labels",

        "dtd_images": "datasets/dtd_images"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_dataset.get(key, ""))


CONST: _Const = _Const()
IMAGES_PATH: Images = Images()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()
