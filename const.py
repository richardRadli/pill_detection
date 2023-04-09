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
    dirs_images = ["images/images_aug", "images/masks_aug", "images/bounding_box", "images/contour", "images/texture",
                   "images/test_images", "images/test_masks", "images/unet_out", "images/unet_out_2",
                   "images/query/rgb", "images/query/contour", "images/query/texture", "images/query_ref_prediction",
                   "images/contour_hardest", "images/rgb_hardest", "images/texture_hardest"]

    # Directories for the train images. These two directories must exist, it won't be created by the program.
    dir_train_images = os.path.join(PROJECT_ROOT, 'images/train_images/')
    dir_train_masks = os.path.join(PROJECT_ROOT, 'images/train_masks/')

    # Aux variables for the other directories. These will be created by the program.
    # Directories for augmentation
    dir_aug_img = os.path.join(PROJECT_ROOT, dirs_images[0])
    dir_aug_mask = os.path.join(PROJECT_ROOT, dirs_images[1])

    # Directories for stream networks, stage 1
    dir_bounding_box = os.path.join(PROJECT_ROOT, dirs_images[2])
    dir_contour = os.path.join(PROJECT_ROOT, dirs_images[3])
    dir_texture = os.path.join(PROJECT_ROOT, dirs_images[4])

    # Directories for test images
    dir_test_images = os.path.join(PROJECT_ROOT, dirs_images[5])
    dir_test_mask = os.path.join(PROJECT_ROOT, dirs_images[6])

    # Directories for the UNET
    dir_unet_output = os.path.join(PROJECT_ROOT, dirs_images[7])
    dir_unet_output_2 = os.path.join(PROJECT_ROOT, dirs_images[8])

    # Directories for the query images
    dir_query_rgb = os.path.join(PROJECT_ROOT, dirs_images[9])
    dir_query_contour = os.path.join(PROJECT_ROOT, dirs_images[10])
    dir_query_texture = os.path.join(PROJECT_ROOT, dirs_images[11])
    dir_query_ref_pred = os.path.join(PROJECT_ROOT, dirs_images[12])

    # Directories for the hard samples, steam networks, stage 2
    dir_contour_hardest = os.path.join(PROJECT_ROOT, dirs_images[13])
    dir_rgb_hardest = os.path.join(PROJECT_ROOT, dirs_images[14])
    dir_texture_hardest = os.path.join(PROJECT_ROOT, dirs_images[15])

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
    dirs_data = ["data/unet_checkpoints", "data/stream_contour_model_weights", "data/stream_rgb_model_weights",
                 "data/stream_texture_model_weights", "data/logs_unet", "data/logs_contour", "data/logs_rgb",
                 "data/logs_texture", "data/hardest_negative_samples", "data/hardest_positive_samples",
                 "data/hardest_anchor_samples", "data/hardest_samples_weights", "data/cam_data", "data/train_labels"]

    # Aux variables for the other directories. These will be created by the program.
    # Directories for UNET
    dir_unet_checkpoint = os.path.join(PROJECT_ROOT, dirs_data[0])
    dir_unet_logs = os.path.join(PROJECT_ROOT, dirs_data[4])

    # Directories for stream networks, for saving the weights
    dir_stream_contour_model_weights = os.path.join(PROJECT_ROOT, dirs_data[1])
    dir_stream_rgb_model_weights = os.path.join(PROJECT_ROOT, dirs_data[2])
    dir_stream_texture_model_weights = os.path.join(PROJECT_ROOT, dirs_data[3])

    # Directories for stream networks, for saving the logs
    dir_contour_logs = os.path.join(PROJECT_ROOT, dirs_data[5])
    dir_rgb_logs = os.path.join(PROJECT_ROOT, dirs_data[6])
    dir_texture_logs = os.path.join(PROJECT_ROOT, dirs_data[7])

    # Directories for the hard samples (txt files)
    dir_hardest_neg_samples = os.path.join(PROJECT_ROOT, dirs_data[8])
    dir_hardest_pos_samples = os.path.join(PROJECT_ROOT, dirs_data[9])
    dir_hardest_anc_samples = os.path.join(PROJECT_ROOT, dirs_data[10])

    # Directory for the hard samples (pt files)
    dir_hardest_samples_weights = os.path.join(PROJECT_ROOT, dirs_data[11])

    dir_cam_data = os.path.join(PROJECT_ROOT, dirs_data[12])
    dir_labels_data = os.path.join(PROJECT_ROOT, dirs_data[13])

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
