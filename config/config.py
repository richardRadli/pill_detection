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
# ++++++++++++++++++++++++++++++++++++++++++++ C O N F I G   T R A I N I N G +++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigTrainingUnet:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        self.parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
        self.parser.add_argument('--weight_decay', '-wd', type=float, default=1e-8)
        self.parser.add_argument('--momentum', type=float, default=0.999)
        self.parser.add_argument('--gradient_clipping', type=float, default=1.0)
        self.parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
        self.parser.add_argument('--img_scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
        self.parser.add_argument('--val', '-v', type=float, default=0.1,
                                 help='Percent of the data that is used as validation (0-100)')
        self.parser.add_argument('--amp', type=bool, default=True, help='Use mixed precision')
        self.parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
        self.parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
        self.parser.add_argument('--channels', '-ch', type=int, default=3, help="Number of channels")
        self.parser.add_argument('--save_checkpoint', '-sc', type=bool, default=True)
        self.parser.add_argument('--load_mask_values', type=bool, default=True)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ C O N F I G   T E S T I N G ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigTestingUnet:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        # Testing
        self.parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                                 help='Specify the file in which the model is stored')
        self.parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images',
                                 required=True)
        self.parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
        self.parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
        self.parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                                 help='Minimum probability value to consider a mask pixel white')
        self.parser.add_argument('--scale', '-s', type=float, default=0.5,
                                 help='Scale factor for the input images')
        self.parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
        self.parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
        self.parser.add_argument('--channels', type=int, default=3)

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

        self.parser.add_argument("--dataset_type", type=str, default="ogyei", choices=["cure | ogyei"])
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

        self.parser.add_argument("--dataset_type", type=str, default="ogyei", choices=["cure | ogyei"])
        self.parser.add_argument("--type_of_net", type=str, default="CNN", choices=["CNN | EfficientNet"])
        self.parser.add_argument("--type_of_stream", type=str, default="Texture",
                                 choices=["Contour | LBP | RGB | Texture"])

        self.parser.add_argument("--type_of_loss_func", type=str, default="hmtl", help="hmtl")
        self.parser.add_argument("--mining_type", type=str, default="semihard", choices=["semihard", "hard", "easy"])
        self.parser.add_argument("--upper_norm_limit", type=float, default=4.0)
        self.parser.add_argument("--margin", type=float, default=0.5)

        self.parser.add_argument("--epochs", type=int, default=20)
        self.parser.add_argument("--batch_size", type=int, default=32)

        self.parser.add_argument("--train_valid_ratio", type=float, default=0.8)

        self.parser.add_argument("--learning_rate_cnn_con", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_cnn_lbp", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_cnn_rgb", type=float, default=3e-4)
        self.parser.add_argument("--learning_rate_cnn_tex", type=float, default=3e-4)

        self.parser.add_argument("--learning_rate_en_con", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_lbp", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_rgb", type=float, default=1e-4)
        self.parser.add_argument("--learning_rate_en_tex", type=float, default=1e-4)
        self.parser.add_argument("--weight_decay", type=float, default=0.1)
        self.parser.add_argument('--step_size', type=int, default=5,
                                 help="Number of epochs after which to decay the learning rate")
        self.parser.add_argument('--gamma', type=float, default=1/3, help="Factor by which to decay the learning rate")

        self.parser.add_argument("--img_size_cnn", type=int, default=128)
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
        self.parser.add_argument("--dataset_type", type=str, default="ogyei", choices=["cure", "ogyei"])
        self.parser.add_argument("--type_of_net", type=str, default="EfficientNetSelfAttention",
                                 choices=["EfficientNetSelfAttention", "CNNFusionNet"])
        self.parser.add_argument("--type_of_loss_func", type=str, default="tl", help="tl")
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
