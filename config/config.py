"""
File: config.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program holds the configurations for different python files.
"""

import argparse
import os


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++ C O N F I G   A U G M E N T A T I O N +++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CameraAndCalibrationConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--size_coeff', type=int, default=3,
                                 help="The shown image will be resized by the given coefficient.")
        self.parser.add_argument('--height', type=int, default=2048, help="Height of the image.")
        self.parser.add_argument('--width', type=int, default=2448, help="Width of the image.")
        self.parser.add_argument('--cam_id', type=int, default=0, help="Default camera device index")
        self.parser.add_argument('--chs_col', type=int, default=8, help="Number of columns in the chessboard")
        self.parser.add_argument('--chs_row', type=int, default=6, help="Number of rows in the chessboard")
        self.parser.add_argument('--square_size', type=int, default=25, help="Square size of the chessboard")
        self.parser.add_argument('--error_threshold', type=float, default=0.2,
                                 help="Error threshold for the calibration")

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++ C O N F I G   A U G M E N T A T I O N +++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigAugmentation:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--dataset_name", type=str, default="nih", choices=["cure", "nih", "ogyei"])
        self.parser.add_argument('--wb_low_thr', type=float, default=0.7)
        self.parser.add_argument('--wb_high_thr', type=float, default=1.2)
        self.parser.add_argument('--kernel_size', type=int, default=7)
        self.parser.add_argument('--brightness_low_thr', type=float, default=0.7)
        self.parser.add_argument('--brightness_high_thr', type=float, default=1.3)
        self.parser.add_argument('--rotate_1', type=int, default=180)
        self.parser.add_argument('--rotate_2', type=int, default=-180)
        self.parser.add_argument('--shift_x', type=int, default=50)
        self.parser.add_argument('--shift_y', type=int, default=100)
        self.parser.add_argument('--zoom', type=int, default=1500)
        self.parser.add_argument('--scale_pill_img', type=float, default=0.3)
        self.parser.add_argument('--max_workers', type=int, default=os.cpu_count()//2)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++ C O N F I G   S T R E A M   I M A G E S ++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigStreamImages:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--dataset_type", type=str, default="ogyei", choices=["cure | ogyei | nih"])
        self.parser.add_argument("--operation", type=str, default="customer", choices=["reference", "customer"])
        self.parser.add_argument("--threshold_area", type=int, default=100)
        self.parser.add_argument("--kernel_median_contour", type=int, default=7)
        self.parser.add_argument("--canny_low_thr", type=int, default=10)
        self.parser.add_argument("--canny_high_thr", type=int, default=30)
        self.parser.add_argument("--kernel_gaussian_texture", type=int, default=7)
        self.parser.add_argument("--max_worker", type=int, default=os.cpu_count()//2)

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

        self.parser.add_argument("--dataset_type", type=str, default="ogyei", choices=["cure | ogyei | nih"])
        self.parser.add_argument("--type_of_net", type=str, default="EfficientNet", choices=["EfficientNet"])
        self.parser.add_argument("--type_of_stream", type=str, default="Texture",
                                 choices=["Contour | LBP | RGB | Texture"])

        self.parser.add_argument("--type_of_loss_func", type=str, default="hmtl", help="hmtl | dmtl")
        self.parser.add_argument("--mining_type", type=str, default="semihard", choices=["semihard", "hard", "easy"])
        self.parser.add_argument("--upper_norm_limit", type=float, default=4.0)
        self.parser.add_argument("--margin", type=float, default=0.5)

        self.parser.add_argument("--epochs", type=int, default=20)
        self.parser.add_argument("--batch_size", type=int, default=32)

        self.parser.add_argument("--train_valid_ratio", type=float, default=0.8)

        self.parser.add_argument("--learning_rate_en_con", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_lbp", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_rgb", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_tex", type=float, default=1e-4)
        self.parser.add_argument('--step_size', type=int, default=5,
                                 help="Number of epochs after which to decay the learning rate")
        self.parser.add_argument('--gamma', type=float, default=1/3, help="Factor by which to decay the learning rate")

        self.parser.add_argument("--img_size_en", type=int, default=224)

        self.parser.add_argument("--load_ref_vector", type=bool, default=False)

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
        self.parser.add_argument("--dataset_type", type=str, default="ogyei", choices=["cure", "nih", "ogyei"])
        self.parser.add_argument("--type_of_net", type=str, default="EfficientNetMultiHeadAttention")
        self.parser.add_argument("--type_of_loss_func", type=str, default="tl", help="tl | dmtl")
        self.parser.add_argument("--upper_norm_limit", type=float, default=4.0)
        self.parser.add_argument("--margin", type=float, default=0.5)
        self.parser.add_argument("--train_valid_ratio", type=float, default=0.8)
        self.parser.add_argument("--epochs", type=int, default=7)
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--learning_rate", type=float, default=1e-4)
        self.parser.add_argument("--weight_decay", type=float, default=1e-3)
        self.parser.add_argument('--step_size', type=int, default=2,
                                 help="Number of epochs after which to decay the learning rate")
        self.parser.add_argument('--gamma', type=float, default=1/3, help="Factor by which to decay the learning rate")

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt

