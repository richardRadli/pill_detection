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
# +++++++++++++++++++++++++++++++++++++++ C O N F I G   S T R E A M   I M A G E S ++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigStreamImages:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--dataset_type", type=str, default="cure_two_sided",
                                 choices=["cure_one_sided", "cure_two_sided", "ogyei"])
        self.parser.add_argument("--operation", type=str, default="reference", choices=["reference", "customer"])
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

        self.parser.add_argument("--dataset_type", type=str, default="cure_two_sided",
                                 choices=["cure_one_sided", "cure_two_sided", "ogyei"])
        self.parser.add_argument("--type_of_net", type=str, default="EfficientNet", choices=["EfficientNet"])
        self.parser.add_argument("--type_of_stream", type=str, default="Texture",
                                 choices=["Contour | LBP | RGB | Texture"])
        self.parser.add_argument("--type_of_loss_func", type=str, default="dmtl", help="hmtl | dmtl")
        self.parser.add_argument("--dmtl_type", type=str, default="feature", choices=["feature", "nlp"])
        self.parser.add_argument("--mining_type", type=str, default="easy", choices=["semihard", "hard", "easy"])
        self.parser.add_argument("--upper_norm_limit", type=float, default=3.0)
        self.parser.add_argument("--margin", type=float, default=0.2)
        self.parser.add_argument("--epochs", type=int, default=25)
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
        self.parser.add_argument("--reference_set", type=str, default="partial", choices=["full", "partial"])
        self.parser.add_argument("--fold", type=str, default="fold1",
                                 choices=["fold1", "fold2", "fold3", "fold4", "fold5"])

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
        self.parser.add_argument("--dataset_type", type=str, default="cure_two_sided",
                                 choices=["cure_one_sided", "cure_two_sided", "ogyei"])
        self.parser.add_argument("--type_of_net", type=str, default="EfficientNetMultiHeadAttention")
        self.parser.add_argument("--type_of_loss_func", type=str, default="hmtl", help="hmtl | dmtl")
        self.parser.add_argument("--dmtl_type", type=str, default="feature", choices=["feature", "nlp"])
        self.parser.add_argument("--upper_norm_limit", type=float, default=3.0)
        self.parser.add_argument("--margin", type=float, default=0.2)
        self.parser.add_argument("--reference_set", type=str, default="full", choices=["full", "partial"])
        self.parser.add_argument("--train_valid_ratio", type=float, default=0.8)
        self.parser.add_argument("--epochs", type=int, default=7)
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--learning_rate", type=float, default=1e-4)
        self.parser.add_argument("--weight_decay", type=float, default=1e-5)
        self.parser.add_argument('--step_size', type=int, default=2,
                                 help="Number of epochs after which to decay the learning rate")
        self.parser.add_argument('--gamma', type=float, default=1/3, help="Factor by which to decay the learning rate")
        self.parser.add_argument('--fold', type=str, default="fold1")

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++ C O N F I G   S T R E A M   I M A G E S ++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigWordEmbedding:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--dataset_type", type=str, default="cure_two_sided",
                                 choices=["cure_one_sided", "cure_two_sided", "ogyei"])
        self.parser.add_argument("-train_valid_ratio", type=float, default=0.8)
        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument("--learning_rate", type=float, default=5e-4)
        self.parser.add_argument("--weight_decay", type=float, default=1e-3)
        self.parser.add_argument("--epochs", type=int, default=10)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
