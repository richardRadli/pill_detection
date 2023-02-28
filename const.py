import os


class _Const(object):
    PROJECT_ROOT = "E:/users/ricsi/IVM"

    dirs_images = ["images/images_aug", "images/masks_aug", "images/bounding_box", "images/contour", "images/texture",
                   "images/text_locator", "images/test_images", "images/test_masks", "images/unet_out",
                   "images/unet_out_2"]

    dirs_data = ["data/checkpoints", "data/stream_contour_model_weights", "data/stream_rgb_model_weights",
                 "data/stream_texture_model_weights"]

    dir_img = os.path.join(PROJECT_ROOT, 'images/train_images/')
    dir_mask = os.path.join(PROJECT_ROOT, 'images/train_masks/')

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

    directories_images = [os.path.join("E:/users/ricsi/IVM", d) for d in dirs_images]

    for d in directories_images:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Directory {d} has been created")

    dir_checkpoint = os.path.join(PROJECT_ROOT, dirs_data[0])
    dir_stream_contour_model_weights = os.path.join(PROJECT_ROOT, dirs_data[1])
    dir_stream_rgb_model_weights = os.path.join(PROJECT_ROOT, dirs_data[2])
    dir_stream_texture_model_weights = os.path.join(PROJECT_ROOT, dirs_data[3])

    directories_data = [os.path.join("E:/users/ricsi/IVM", d) for d in dirs_data]

    for d in directories_data:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Directory {d} has been created")

    def __setattr__(self, *_):
        raise TypeError


CONST: _Const = _Const()
