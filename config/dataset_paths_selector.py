from config.data_paths import DATASET_PATH, IMAGES_PATH


def dataset_images_path_selector(dataset_name: str):
    """
    Selects the correct directory paths based on the given operation string.

    Returns:
        A dictionary containing directory paths for images, masks, and other related files.

    Raises ValueError:
        If the operation string is not "train" or "test".
    """

    path_to_images = {
        # --------------------------------------------------- C U R E --------------------------------------------------
        "cure": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("cure_customer_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_customer_segmentation_labels"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("cure_customer_mask_images")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("cure_reference_images"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("cure_reference_mask_images"),
                "reference_labels":
                    DATASET_PATH.get_data_path("cure_reference_yolo_labels"),
            },

            "train": {
                "images":
                    DATASET_PATH.get_data_path("cure_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_train_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_train_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_train_aug_yolo_labels"),
                "aug_mask_images":
                    DATASET_PATH.get_data_path("cure_train_aug_mask_images"),
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path("cure_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_valid_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_valid_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_valid_aug_yolo_labels"),
                "aug_mask_images":
                    DATASET_PATH.get_data_path("cure_valid_aug_mask_images"),
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path("cure_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_test_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_test_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_test_aug_yolo_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_reference_texture"),
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_cure_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_cure_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("test_ref_cure"),
                'query':
                    IMAGES_PATH.get_data_path("test_query_cure")
            }
        },
        # -------------------------------------------------- O G Y E I -------------------------------------------------
        "ogyei": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("ogyei_customer_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_customer_segmentation_labels"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("ogyei_customer_mask_images")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("ogyei_reference_images"),
                "reference_segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_reference_segmentation_labels"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("ogyei_reference_mask_images")
            },

            "unsplitted": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_segmentation_labels")
            },

            "train": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_train_segmentation_labels")
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_valid_segmentation_labels")
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_test_segmentation_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference_texture"),
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_ogyei_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_ogyei_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("ref_ogyei"),
                'query':
                    IMAGES_PATH.get_data_path("query_ogyei")
            }
        },
        "dtd": {
            "dataset_path":
                DATASET_PATH.get_data_path("dtd")
        }
    }

    return path_to_images[dataset_name]
