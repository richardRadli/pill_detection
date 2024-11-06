from config.data_paths import DATA_PATH, DATASET_PATH, IMAGES_PATH


def dataset_images_path_selector(dataset_name: str):
    """
    Selects the correct directory paths based on the given operation string.

    Returns:
        A dictionary containing directory paths for images, masks, and other related files.

    Raises ValueError:
        If the operation string is not "train" or "test".
    """

    path_to_images = {
        # -------------------------------------------------- O G Y E I -------------------------------------------------
        "ogyeiv2": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("ogyei_v2_customer_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_v2_customer_segmentation_labels"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("ogyei_v2_customer_mask_images")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("ogyei_v2_reference_images"),
                "reference_segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_v2_reference_segmentation_labels"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("ogyei_v2_reference_mask_images")
            },

            "unsplitted": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_v2_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_v2_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_v2_segmentation_labels")
            },

            "train": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_v2_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_v2_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_v2_train_segmentation_labels")
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_v2_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_v2_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_v2_valid_segmentation_labels")
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_v2_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_v2_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_v2_test_segmentation_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_reference_texture"),
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_ogyei_v2_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_ogyei_v2_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("ref_ogyei_v2"),
                'query':
                    IMAGES_PATH.get_data_path("query_ogyei_v2")
            },

            "other": {
                "k_fold":
                    DATA_PATH.get_data_path("ogyei_v2_k_fold")
            }
        },
        "dtd": {
            "dataset_path":
                DATASET_PATH.get_data_path("dtd")
        }
    }

    return path_to_images[dataset_name]
