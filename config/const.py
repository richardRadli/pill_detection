import logging
import os

from config.logger_setup import setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    setup_logger()

    # Root of the project
    user = os.getlogin()
    root_mapping = {
        "keplab": {
            "PROJECT_ROOT": "E:/users/ricsi/IVM",
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
    # ---------------------------------------------- I M A G E   F I L E S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Directories for the images
    dirs_images = ["images/test_images",                                # 0
                   "images/test_masks",                                 # 1
                   "images/rgb",                                        # 2
                   "images/contour",                                    # 3
                   "images/texture",                                    # 4
                   "images/unet_out",                                   # 5
                   "images/unet_compare",                               # 6
                   "images/query/rgb",                                  # 7
                   "images/query/contour",                              # 8
                   "images/query/texture",                              # 9
                   "images/contour_hardest",                            # 10
                   "images/rgb_hardest",                                # 11
                   "images/texture_hardest",                            # 12
                   "images/prediction_stream_network",                  # 13
                   "images/prediction_efficient_net",                   # 14
                   "images/prediction_fusion_network",                  # 15
                   "images/prediction_efficient_net_self_attention",    # 16
                   "images/images_aug",                                 # 17
                   "images/wo_background"]                              # 18

    # Directories for the train images. These two directories must exist, it won't be created by the program.
    dir_train_images = os.path.join(PROJECT_ROOT, 'images/train_images/')
    dir_train_masks = os.path.join(PROJECT_ROOT, 'images/train_masks/')

    # Aux variables for the other directories. These will be created by the program.
    # Directories for augmentation
    dir_aug_img = os.path.join(PROJECT_ROOT, dirs_images[17])
    dir_wo_background = os.path.join(PROJECT_ROOT, dirs_images[18])

    # Directories for stream networks, stage 1
    dir_rgb = os.path.join(PROJECT_ROOT, dirs_images[2])
    dir_contour = os.path.join(PROJECT_ROOT, dirs_images[3])
    dir_texture = os.path.join(PROJECT_ROOT, dirs_images[4])

    # Directories for test images
    dir_test_images = os.path.join(PROJECT_ROOT, dirs_images[0])
    dir_test_mask = os.path.join(PROJECT_ROOT, dirs_images[1])

    # Directories for the UNET
    dir_unet_output = os.path.join(PROJECT_ROOT, dirs_images[5])
    dir_unet_compare = os.path.join(PROJECT_ROOT, dirs_images[6])

    # Directories for the query images
    dir_query_rgb = os.path.join(PROJECT_ROOT, dirs_images[7])
    dir_query_contour = os.path.join(PROJECT_ROOT, dirs_images[8])
    dir_query_texture = os.path.join(PROJECT_ROOT, dirs_images[9])

    # Directories for the hard samples, steam networks, stage 2
    dir_contour_hardest = os.path.join(PROJECT_ROOT, dirs_images[10])
    dir_rgb_hardest = os.path.join(PROJECT_ROOT, dirs_images[11])
    dir_texture_hardest = os.path.join(PROJECT_ROOT, dirs_images[12])

    # Directories for plotting the comparison of the query and the reference images
    dir_stream_network_pred = os.path.join(PROJECT_ROOT, dirs_images[13])
    dir_efficient_net_prediction = os.path.join(PROJECT_ROOT, dirs_images[14])
    dir_fusion_net_pred = os.path.join(PROJECT_ROOT, dirs_images[15])
    dir_efficient_net_self_attention_prediction = os.path.join(PROJECT_ROOT, dirs_images[16])

    # At this part, the program creates the directories.
    directories_images = []
    for d in dirs_images:
        directories_images.append(os.path.join(PROJECT_ROOT, d))

    for d in directories_images:
        if not os.path.exists(d):
            os.makedirs(d)
            logging.info(f"Directory {d} has been created")

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- D A T A   F I L E S ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Directories for the data
    dirs_data = ["data/weights_unet",                                   # 0
                 "data/logs_unet",                                      # 1
                 "data/weights_stream_network_contour",                 # 2
                 "data/weights_stream_network_rgb",                     # 3
                 "data/weights_stream_network_texture",                 # 4
                 "data/predictions_stream_network",                     # 5
                 "data/logs_stream_contour",                            # 6
                 "data/logs_stream_rgb",                                # 7
                 "data/logs_stream_texture",                            # 8
                 "data/hardest_samples/negative",                       # 9
                 "data/hardest_samples/positive",                       # 10
                 "data/weights_fusion_net",                             # 11
                 "data/logs_fusion_net",                                # 12
                 "data/cam_data",                                       # 13
                 "data/train_labels",                                   # 14
                 "data/predictions_fusion_network",                     # 15
                 "data/weights_efficient_net_rgb",                      # 16
                 "data/weights_efficient_net_texture",                  # 17
                 "data/weights_efficient_net_contour",                  # 18
                 "data/predictions_efficient_net",                      # 19
                 "data/logs_efficient_net_rgb",                         # 20
                 "data/logs_efficient_net_texture",                     # 21
                 "data/logs_efficient_net_contour",                     # 22
                 "data/weights_efficient_self_attention_net_rgb",       # 23
                 "data/weights_efficient_self_attention_net_contour",   # 24
                 "data/weights_efficient_self_attention_net_texture",   # 25
                 "data/predictions_efficient_self_attention_net",       # 26
                 "data/logs_efficient_net_self_attention_rgb",          # 27
                 "data/logs_efficient_net_self_attention_texture",      # 28
                 "data/logs_efficient_net_self_attention_contour",      # 29
                 "data/augmented_train_data_labels",                    # 30
                 "data/reference_vectors_stream_network",               # 31
                 "data/reference_vectors_efficient_net",                # 32
                 "data/reference_vectors_efficient_net_self_attention"  # 33
                 ]

    # Aux variables for the other directories. These will be created by the program.
    # Directories for UNET
    dir_unet_checkpoint = os.path.join(PROJECT_ROOT, dirs_data[0])
    dir_unet_logs = os.path.join(PROJECT_ROOT, dirs_data[1])

    # Directories for stream networks, for saving the weights
    dir_stream_contour_model_weights = os.path.join(PROJECT_ROOT, dirs_data[2])
    dir_stream_rgb_model_weights = os.path.join(PROJECT_ROOT, dirs_data[3])
    dir_stream_texture_model_weights = os.path.join(PROJECT_ROOT, dirs_data[4])

    # Directories for stream networks, for saving the logs
    dir_logs_stream_net_contour = os.path.join(PROJECT_ROOT, dirs_data[6])
    dir_logs_stream_net_rgb = os.path.join(PROJECT_ROOT, dirs_data[7])
    dir_logs_stream_net_texture = os.path.join(PROJECT_ROOT, dirs_data[8])

    # Directory for saving the predictions
    dir_stream_network_predictions = os.path.join(PROJECT_ROOT, dirs_data[5])

    # Directories for the hard samples (txt files)
    dir_hardest_neg_samples = os.path.join(PROJECT_ROOT, dirs_data[9])
    dir_hardest_pos_samples = os.path.join(PROJECT_ROOT, dirs_data[10])

    # Directory for the hard samples (pt files)
    dir_fusion_net_weights = os.path.join(PROJECT_ROOT, dirs_data[11])

    # Directory for logging the fusion network
    dir_fusion_net_logs = os.path.join(PROJECT_ROOT, dirs_data[12])

    # Directory for saving the predictions
    dir_fusion_network_predictions = os.path.join(PROJECT_ROOT, dirs_data[15])

    # Directories for EfficientNet, for saving the weights
    dir_efficient_net_rgb_model_weights = os.path.join(PROJECT_ROOT, dirs_data[16])
    dir_efficient_net_texture_model_weights = os.path.join(PROJECT_ROOT, dirs_data[17])
    dir_efficient_net_contour_model_weights = os.path.join(PROJECT_ROOT, dirs_data[18])

    # Directories for EfficientNet, for saving the logs
    dir_logs_efficient_net_contour = os.path.join(PROJECT_ROOT, dirs_data[22])
    dir_logs_efficient_net_rgb = os.path.join(PROJECT_ROOT, dirs_data[20])
    dir_logs_efficient_net_texture = os.path.join(PROJECT_ROOT, dirs_data[21])

    # Directory for saving the predictions, EfficientNet
    dir_efficient_net_predictions = os.path.join(PROJECT_ROOT, dirs_data[19])

    # Directories for EfficientNetSelfAttention, for saving the weights
    dir_efficient_net_self_attention_rgb_model_weights = os.path.join(PROJECT_ROOT, dirs_data[23])
    dir_efficient_net_self_attention_contour_model_weights = os.path.join(PROJECT_ROOT, dirs_data[24])
    dir_efficient_net_self_attention_texture_model_weights = os.path.join(PROJECT_ROOT, dirs_data[25])

    # Directories for EfficientNetSelfAttention, for saving the logs
    dir_logs_efficient_net_self_attention_rgb = os.path.join(PROJECT_ROOT, dirs_data[27])
    dir_logs_efficient_net_self_attention_texture = os.path.join(PROJECT_ROOT, dirs_data[28])
    dir_logs_efficient_net_self_attention_contour = os.path.join(PROJECT_ROOT, dirs_data[29])

    # Directory for saving the predictions, EfficientNetSelfAttention
    dir_efficient_net_self_attention_predictions = os.path.join(PROJECT_ROOT, dirs_data[26])

    # Other
    dir_cam_data = os.path.join(PROJECT_ROOT, dirs_data[13])
    dir_labels_data = os.path.join(PROJECT_ROOT, dirs_data[14])
    dir_aug_labels = os.path.join(PROJECT_ROOT, dirs_data[30])

    # Reference vectors
    dir_ref_vectors_stream_net = os.path.join(PROJECT_ROOT, dirs_data[31])
    dir_ref_vectors_efficient_net = os.path.join(PROJECT_ROOT, dirs_data[32])
    dir_ref_vectors_efficient_net_self_attention = os.path.join(PROJECT_ROOT, dirs_data[33])

    directories_data = []
    for d in dirs_data:
        directories_data.append(os.path.join(PROJECT_ROOT, d))

    for d in directories_data:
        if not os.path.exists(d):
            os.makedirs(d)
            logging.info(f"Directory {d} has been created")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- D A T A S E T S ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    dirs_dataset = [
        'ogyi/full_img_size/unsplitted/images',             # 0
        'ogyi/full_img_size/unsplitted/labels',             # 1
        'ogyi/full_img_size/splitted/train/images',         # 2
        'ogyi/full_img_size/splitted/train/labels',         # 3
        'ogyi/full_img_size/splitted/test/images',          # 4
        'ogyi/full_img_size/splitted/test/labels',          # 5
        'ogyi/full_img_size/splitted/valid/images',         # 6
        'ogyi/full_img_size/splitted/valid/labels',         # 7
        'ogyi_multi/unsplitted/images',                     # 8
        'ogyi_multi/unsplitted/labels',                     # 9
        'ogyi_multi/splitted/train/images',                 # 10
        'ogyi_multi/splitted/train/labels',                 # 11
        'ogyi_multi/splitted/test/images',                  # 12
        'ogyi_multi/splitted/test/labels',                  # 13
        'ogyi_multi/splitted/valid/images',                 # 14
        'ogyi_multi/splitted/valid/labels'                  # 15
    ]
    # OGYI Single - Full img size - unsplitted
    dir_ogyi_single_unsplitted_images = os.path.join(DATASET_ROOT, dirs_dataset[0])
    dir_ogyi_single_unsplitted_labels = os.path.join(DATASET_ROOT, dirs_dataset[1])

    # OGYI Single - Full img size - splitted
    dir_ogyi_single_splitted_train_images = os.path.join(DATASET_ROOT, dirs_dataset[2])
    dir_ogyi_single_splitted_train_labels = os.path.join(DATASET_ROOT, dirs_dataset[3])
    dir_ogyi_single_splitted_test_images = os.path.join(DATASET_ROOT, dirs_dataset[4])
    dir_ogyi_single_splitted_test_labels = os.path.join(DATASET_ROOT, dirs_dataset[5])
    dir_ogyi_single_splitted_valid_images = os.path.join(DATASET_ROOT, dirs_dataset[6])
    dir_ogyi_single_splitted_valid_labels = os.path.join(DATASET_ROOT, dirs_dataset[7])

    # OGYI Multi - unsplitted
    dir_ogyi_multi_unsplitted_images = os.path.join(DATASET_ROOT, dirs_dataset[8])
    dir_ogyi_multi_unsplitted_labels = os.path.join(DATASET_ROOT, dirs_dataset[9])
    #
    # # OGYI Multi - splitted
    dir_ogyi_multi_splitted_train_images = os.path.join(DATASET_ROOT, dirs_dataset[10])
    dir_ogyi_multi_splitted_train_labels = os.path.join(DATASET_ROOT, dirs_dataset[11])
    dir_ogyi_multi_splitted_test_images = os.path.join(DATASET_ROOT, dirs_dataset[12])
    dir_ogyi_multi_splitted_test_labels = os.path.join(DATASET_ROOT, dirs_dataset[13])
    dir_ogyi_multi_splitted_val_images = os.path.join(DATASET_ROOT, dirs_dataset[14])
    dir_ogyi_multi_splitted_val_labels = os.path.join(DATASET_ROOT, dirs_dataset[15])

    directories_data_set = []
    for d in dirs_dataset:
        directories_data_set.append(os.path.join(DATASET_ROOT, d))

    for d in directories_data_set:
        if not os.path.exists(d):
            os.makedirs(d)
            logging.info(f"Directory {d} has been created")

    def __setattr__(self, *_):
        raise TypeError


CONST: _Const = _Const()
