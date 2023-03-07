import os


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    # Root of the project
    user = os.getlogin()
    if user == "keplab":
        PROJECT_ROOT = "E:/users/ricsi/IVM"
    elif user == "rrb12":
        PROJECT_ROOT = "D:/project/IVM"
    else:
        raise ValueError("Wrong user!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- I M A G E   F I L E S ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Directories for the images
    dirs_images = ["images/images_aug", "images/masks_aug", "images/bounding_box", "images/contour", "images/texture",
                   "images/text_locator", "images/test_images", "images/test_masks", "images/unet_out",
                   "images/unet_out_2", "images/query/rgb", "images/query/contour", "images/query/texture"]

    # Directories for the train images. These two directories must exist, it won't be created by the program.
    dir_train_images = os.path.join(PROJECT_ROOT, 'images/train_images/')
    dir_train_masks = os.path.join(PROJECT_ROOT, 'images/train_masks/')

    # Aux variables for the other directories. These will be created by the program.
    dir_aug_img = os.path.join(PROJECT_ROOT, dirs_images[0])
    dir_aug_mask = os.path.join(PROJECT_ROOT, dirs_images[1])
    dir_bounding_box = os.path.join(PROJECT_ROOT, dirs_images[2])
    dir_contour = os.path.join(PROJECT_ROOT, dirs_images[3])
    dir_texture = os.path.join(PROJECT_ROOT, dirs_images[4])
    dir_text_loc = os.path.join(PROJECT_ROOT, dirs_images[5])
    dir_test_images = os.path.join(PROJECT_ROOT, dirs_images[6])
    dir_test_mask = os.path.join(PROJECT_ROOT, dirs_images[7])
    dir_unet_output = os.path.join(PROJECT_ROOT, dirs_images[8])
    dir_unet_output_2 = os.path.join(PROJECT_ROOT, dirs_images[9])
    dir_query_rgb = os.path.join(PROJECT_ROOT, dirs_images[10])
    dir_query_contour = os.path.join(PROJECT_ROOT, dirs_images[11])
    dir_query_texture = os.path.join(PROJECT_ROOT, dirs_images[12])

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
                 "data/stream_texture_model_weights", "data/logs_checkpoint", "data/logs_contour",
                 "data/logs_rgb", "data/logs_texture"]

    # Aux variables for the other directories. These will be created by the program.
    dir_unet_checkpoint = os.path.join(PROJECT_ROOT, dirs_data[0])
    dir_stream_contour_model_weights = os.path.join(PROJECT_ROOT, dirs_data[1])
    dir_stream_rgb_model_weights = os.path.join(PROJECT_ROOT, dirs_data[2])
    dir_stream_texture_model_weights = os.path.join(PROJECT_ROOT, dirs_data[3])
    dir_unet_logs = os.path.join(PROJECT_ROOT, dirs_data[4])
    dir_contour_logs = os.path.join(PROJECT_ROOT, dirs_data[5])
    dir_rgb_logs = os.path.join(PROJECT_ROOT, dirs_data[6])
    dir_texture_logs = os.path.join(PROJECT_ROOT, dirs_data[7])

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
