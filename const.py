import os


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    # Root of the project
    user = os.getlogin()
    if user == "keplab":
        PROJECT_ROOT = "E:/users/ricsi/IVM"
    elif user == "ricsi":
        PROJECT_ROOT = "C:/Users/ricsi/Documents/project/storage/IVM"
    else:
        raise ValueError("Wrong user!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- I M A G E   F I L E S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Directories for the images
    dirs_images = ["images/test_images",                # 0
                   "images/test_masks",                 # 1
                   "images/rgb",                        # 2
                   "images/contour",                    # 3
                   "images/texture",                    # 4
                   "images/unet_out",                   # 5
                   "images/unet_compare",               # 6
                   "images/query/rgb",                  # 7
                   "images/query/contour",              # 8
                   "images/query/texture",              # 9
                   "images/contour_hardest",            # 10
                   "images/rgb_hardest",                # 11
                   "images/texture_hardest",            # 12
                   "images/stream_network_prediction",  # 13
                   "images/efficient_net_prediction",   # 14
                   "images/fusion_network_prediction",  # 15
                   "images/images_aug",                 # 16
                   "images/masks_aug"]                 # 17

    # Directories for the train images. These two directories must exist, it won't be created by the program.
    dir_train_images = os.path.join(PROJECT_ROOT, 'images/train_images/')
    dir_train_masks = os.path.join(PROJECT_ROOT, 'images/train_masks/')

    # Aux variables for the other directories. These will be created by the program.
    # Directories for augmentation
    dir_aug_img = os.path.join(PROJECT_ROOT, dirs_images[16])
    dir_aug_mask = os.path.join(PROJECT_ROOT, dirs_images[17])

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

    # At this part, the program creates the directories.
    directories_images = []
    for d in dirs_images:
        directories_images.append(os.path.join(PROJECT_ROOT, d))

    for d in directories_images:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Directory {d} has been created")

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- D A T A   F I L E S ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Directories for the data
    dirs_data = ["data/weights_unet",                           # 0
                 "data/logs_unet",                              # 1
                 "data/weights_stream_network_contour",         # 2
                 "data/weights_stream_network_rgb",             # 3
                 "data/weights_stream_network_texture",         # 4
                 "data/predictions_stream_network",             # 5
                 "data/logs_stream_contour",                    # 6
                 "data/logs_stream_rgb",                        # 7
                 "data/logs_stream_texture",                    # 8
                 "data/hardest_samples/negative",               # 9
                 "data/hardest_samples/positive",               # 10
                 "data/weights_fusion_net",                     # 11
                 "data/logs_fusion_net",                        # 12
                 "data/cam_data",                               # 13
                 "data/train_labels",                           # 14
                 "data/predictions_fusion_network",             # 15
                 "data/weights_efficient_net_rgb",              # 16
                 "data/weights_efficient_net_texture",          # 17
                 "data/weights_efficient_net_contour",          # 18
                 "data/predictions_efficient_net",              # 19
                 "data/logs_efficient_net_rgb",                 # 20
                 "data/logs_efficient_net_texture",             # 21
                 "data/logs_efficient_net_contour"              # 22
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

    # Directory for saving the predictions
    dir_efficient_net_predictions = os.path.join(PROJECT_ROOT, dirs_data[19])

    # Other
    dir_cam_data = os.path.join(PROJECT_ROOT, dirs_data[13])
    dir_labels_data = os.path.join(PROJECT_ROOT, dirs_data[14])

    directories_data = []
    for d in dirs_data:
        directories_data.append(os.path.join(PROJECT_ROOT, d))

    for d in directories_data:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Directory {d} has been created")

    def __setattr__(self, *_):
        raise TypeError


CONST: _Const = _Const()
