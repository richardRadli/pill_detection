"""
File: config.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program holds the configurations for different python files.
"""

import argparse


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++ C O N F I G   A U G M E N T A T I O N +++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigAugmentation:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--wb_low_thr', type=float, default=0.7)
        self.parser.add_argument('--wb_high_thr', type=float, default=1.2)
        self.parser.add_argument('--wb_low_thr_2nd_aug', type=float, default=0.9)
        self.parser.add_argument('--wb_high_thr_2nd_aug', type=float, default=1.0)
        self.parser.add_argument('--kernel_size', type=int, default=7)
        self.parser.add_argument('--brightness_low_thr', type=float, default=0.7)
        self.parser.add_argument('--brightness_high_thr', type=float, default=1.3)
        self.parser.add_argument('--brightness_low_thr_2nd_aug', type=float, default=0.9)
        self.parser.add_argument('--brightness_high_thr_2nd_aug', type=float, default=1.1)
        self.parser.add_argument('--rotate_low_thr', type=int, default=35)
        self.parser.add_argument('--rotate_high_thr', type=int, default=270)
        self.parser.add_argument('--shift_low_thr', type=int, default=150)
        self.parser.add_argument('--shift_high_thr', type=int, default=200)
        self.parser.add_argument('--scale_pill_img', type=float, default=0.3)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++ C O N F I G   S T R E A M   N E T W O R K +++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigStreamNetwork:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--dataset_type", type=str, default="ogyei", help="cure | ogyei")
        self.parser.add_argument("--type_of_net", type=str, default="EfficientNet",
                                 help="CNN | EfficientNet | EfficientNetV2")
        self.parser.add_argument("--type_of_stream", type=str, default="LBP", help="Contour | LBP | RGB | Texture")
        self.parser.add_argument("--dataset_operation", type=str, default="valid", help="train | valid | test")

        self.parser.add_argument("--type_of_loss_func", type=str, default="tl", help="tl | hmtl")
        self.parser.add_argument("--num_triplets", type=int, default=6000, help="Number of triplets to be generated")
        self.parser.add_argument("--margin", type=float, default=0.5)

        self.parser.add_argument("--epochs", type=int, default=30)
        self.parser.add_argument("--batch_size", type=int, default=32)

        self.parser.add_argument("--train_valid_ratio", type=float, default=0.8)

        self.parser.add_argument("--learning_rate_cnn_rgb", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_cnn_con", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_cnn_lbp", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_cnn_tex", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_en_rgb", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_con", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_lbp", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_tex", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_env2_rgb", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_env2_con", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_env2_lbp", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_env2_tex", type=float, default=3e-4)
        self.parser.add_argument("--weight_decay", type=float, default=1e-5)
        self.parser.add_argument('--step_size', type=int, default=10,
                                 help="Number of epochs after which to decay the learning rate")
        self.parser.add_argument('--gamma', type=float, default=0.1, help="Factor by which to decay the learning rate")

        self.parser.add_argument("--img_size_cnn", type=int, default=128)
        self.parser.add_argument("--img_size_en", type=int, default=224)

        self.parser.add_argument("--load_ref_vector", type=bool, default=True)

        self.parser.add_argument("--threshold_area", type=int, default=100)
        self.parser.add_argument("--kernel_median_contour", type=int, default=7)
        self.parser.add_argument("--canny_low_thr", type=int, default=5)
        self.parser.add_argument("--canny_high_thr", type=int, default=25)
        self.parser.add_argument("--kernel_gaussian_texture", type=int, default=7)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++ C O N F I G   F U S I O N   N E T W O R K +++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigFusionNetwork:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--type_of_net", type=str, default="EfficientNetSelfAttention",
                                 help="CNNFusionNet | EfficientNetSelfAttention | EfficientNetV2SelfAttention "
                                      "| EfficientNetV2MultiHeadAttention | EfficientNetV2MHAFMHA")
        self.parser.add_argument("--type_of_loss_func", type=str, default="tl", help="tl | hmtl")
        self.parser.add_argument("--margin", type=float, default=0.5)
        self.parser.add_argument("--train_split", type=float, default=0.8)
        self.parser.add_argument("--epochs", type=int, default=15)
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--learning_rate", type=float, default=3e-4)
        self.parser.add_argument("--weight_decay", type=float, default=1e-5)
        self.parser.add_argument('--step_size', type=int, default=5,
                                 help="Number of epochs after which to decay the learning rate")
        self.parser.add_argument('--gamma', type=float, default=0.1, help="Factor by which to decay the learning rate")

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
