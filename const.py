import os


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    # Root of the project
    PROJECT_ROOT = "E:/users/ricsi/IVM"

    # Directories for the images
    dirs_images = ["images/images_aug", "images/masks_aug", "images/bounding_box", "images/contour", "images/texture",
                   "images/text_locator", "images/test_images", "images/test_masks", "images/unet_out",
                   "images/unet_out_2"]

    # Directories for the data
    dirs_data = ["data/checkpoints", "data/stream_contour_model_weights", "data/stream_rgb_model_weights",
                 "data/stream_texture_model_weights", "data/wandb_logs_checkpoint", "data/wandb_logs_contour",
                 "data/wandb_logs_rgb", "data/wandb_logs_texture"]

    # Directories for the train images. These two directories must exist, it won't be created by the program.
    dir_img = os.path.join(PROJECT_ROOT, 'images/train_images/')
    dir_mask = os.path.join(PROJECT_ROOT, 'images/train_masks/')

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

    # At this part, the program creates the directories.
    directories_images = [os.path.join("E:/users/ricsi/IVM", d) for d in dirs_images]

    for d in directories_images:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Directory {d} has been created")

    # Aux variables for the other directories. These will be created by the program.
    dir_checkpoint = os.path.join(PROJECT_ROOT, dirs_data[0])
    dir_stream_contour_model_weights = os.path.join(PROJECT_ROOT, dirs_data[1])
    dir_stream_rgb_model_weights = os.path.join(PROJECT_ROOT, dirs_data[2])
    dir_stream_texture_model_weights = os.path.join(PROJECT_ROOT, dirs_data[3])
    dir_wandb_checkpoint_logs = os.path.join(PROJECT_ROOT, dirs_data[4])
    dir_wandb_contour_logs = os.path.join(PROJECT_ROOT, dirs_data[5])
    dir_wandb_rgb_logs = os.path.join(PROJECT_ROOT, dirs_data[6])
    dir_wandb_texture_logs = os.path.join(PROJECT_ROOT, dirs_data[7])

    directories_data = [os.path.join("E:/users/ricsi/IVM", d) for d in dirs_data]

    for d in directories_data:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Directory {d} has been created")

    def __setattr__(self, *_):
        raise TypeError


CONST: _Const = _Const()
