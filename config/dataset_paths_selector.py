from config.data_paths import DATA_PATH, DATASET_PATH, IMAGES_PATH


def dataset_images_path_selector(dataset_name: str):
    """
    Selects the correct directory paths based on the given operation string.

    Returns:
        A dictionary containing directory paths for images, masks, and other related files.
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
                    "stream_mask_images":
                        DATASET_PATH.get_data_path("stream_images_ogyei_v2_reference_masks")
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
            },

            "dynamic_margin": {
                "pill_desc_xlsx":
                    DATA_PATH.get_data_path("pill_desc_xlsx_ogyei_v2"),
                "Fourier_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_ogyei_v2"),
                "Fourier_euclidean_distance":
                    IMAGES_PATH.get_data_path("Fourier_euclidean_distance_ogyei_v2"),
                "Fourier_images_by_shape":
                    IMAGES_PATH.get_data_path("Fourier_collected_images_by_shape_ogyei_v2"),
                "colour_vectors":
                    DATA_PATH.get_data_path("colour_vectors_ogyei_v2"),
                "imprint_vectors":
                    DATA_PATH.get_data_path("imprint_vectors_ogyei_v2"),
                "score_vectors":
                    DATA_PATH.get_data_path("score_vectors_ogyei_v2"),
                "concatenated_vectors":
                    DATA_PATH.get_data_path("concatenated_vectors_ogyei_v2"),
                "combined_vectors_euc_dst":
                    IMAGES_PATH.get_data_path("combined_vectors_euc_dst_ogyei_v2"),
                "euc_mtx_xlsx":
                    DATA_PATH.get_data_path("euc_mtx_xlsx_ogyei_v2")
            }
        },

        "cure_one_sided": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_segmentation_labels"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_mask_images")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_images"),
                "reference_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_segmentation_labels"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_mask_images")
            },

            "unsplitted": {
                "images":
                    DATASET_PATH.get_data_path("cure_one_sided_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_segmentation_labels")
            },

            "train": {
                "images":
                    DATASET_PATH.get_data_path("cure_one_sided_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_train_segmentation_labels")
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_segmentation_labels")
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path("cure_one_sided_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_test_segmentation_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_texture"),
                    "stream_mask_images":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_masks")
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_cure_one_sided_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_cure_one_sided_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("ref_cure_one_sided"),
                'query':
                    IMAGES_PATH.get_data_path("query_cure_one_sided")
            },

            "other": {
                "k_fold":
                    DATA_PATH.get_data_path("cure_one_sided_k_fold")
            },

            "dynamic_margin": {
                "pill_desc_xlsx":
                    DATA_PATH.get_data_path("pill_desc_xlsx_cure_one_sided"),
                "Fourier_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_cure_one_sided"),
                "Fourier_euclidean_distance":
                    IMAGES_PATH.get_data_path("Fourier_euclidean_distance_cure_one_sided"),
                "Fourier_saved_mean_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_cure_one_sided"),
                "Fourier_images_by_shape":
                    IMAGES_PATH.get_data_path("Fourier_collected_images_by_shape_cure_one_sided"),
                "colour_vectors":
                    DATA_PATH.get_data_path("colour_vectors_cure_one_sided"),
                "imprint_vectors":
                    DATA_PATH.get_data_path("imprint_vectors_cure_one_sided"),
                "score_vectors":
                    DATA_PATH.get_data_path("score_vectors_cure_one_sided"),
                "concatenated_vectors":
                    DATA_PATH.get_data_path("concatenated_vectors_cure_one_sided"),
                "combined_vectors_euc_dst":
                    IMAGES_PATH.get_data_path("combined_vectors_euc_dst_cure_one_sided"),
                "euc_mtx_xlsx":
                    DATA_PATH.get_data_path("euc_mtx_xlsx_cure_one_sided")
            }
        },

        "cure_two_sided": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("cure_two_sided_customer_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_customer_segmentation_labels"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_customer_mask_images")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("cure_two_sided_reference_images"),
                "reference_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_reference_segmentation_labels"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_reference_mask_images")
            },

            "unsplitted": {
                "images":
                    DATASET_PATH.get_data_path("cure_two_sided_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_segmentation_labels")
            },

            "train": {
                "images":
                    DATASET_PATH.get_data_path("cure_two_sided_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_train_segmentation_labels")
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_segmentation_labels")
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path("cure_two_sided_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_test_segmentation_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_texture"),
                    "stream_mask_images":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_masks")
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_cure_two_sided_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_cure_two_sided_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("ref_cure_two_sided"),
                'query':
                    IMAGES_PATH.get_data_path("query_cure_two_sided")
            },

            "other": {
                "k_fold":
                    DATA_PATH.get_data_path("cure_two_sided_k_fold")
            },

            "dynamic_margin": {
                "pill_desc_xlsx":
                    DATA_PATH.get_data_path("pill_desc_xlsx_cure_two_sided"),
                "Fourier_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_cure_two_sided"),
                "Fourier_euclidean_distance":
                    IMAGES_PATH.get_data_path("Fourier_euclidean_distance_cure_two_sided"),
                "Fourier_saved_mean_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_cure_two_sided"),
                "Fourier_images_by_shape":
                    IMAGES_PATH.get_data_path("Fourier_collected_images_by_shape_cure_two_sided"),
                "colour_vectors":
                    DATA_PATH.get_data_path("colour_vectors_cure_two_sided"),
                "imprint_vectors":
                    DATA_PATH.get_data_path("imprint_vectors_cure_two_sided"),
                "score_vectors":
                    DATA_PATH.get_data_path("score_vectors_cure_two_sided"),
                "concatenated_vectors":
                    DATA_PATH.get_data_path("concatenated_vectors_cure_two_sided"),
                "combined_vectors_euc_dst":
                    IMAGES_PATH.get_data_path("combined_vectors_euc_dst_cure_two_sided"),
                "euc_mtx_xlsx":
                    DATA_PATH.get_data_path("euc_mtx_xlsx_cure_two_sided")
            }
        },

        "dtd": {
            "dataset_path":
                DATASET_PATH.get_data_path("dtd")
        }
    }

    return path_to_images[dataset_name]
