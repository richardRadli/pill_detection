"""
File: dataloader_fusion_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description:
The FusionDataset class is a custom dataset loader class that is used for loading and processing image datasets for
fusion networks.
"""

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from config.config import ConfigStreamNetwork
from fusion_network_phase.mine_hard_samples import get_hardest_samples


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++ F U S I O N D A T A S E T +++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class FusionDataset(Dataset):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- __ I N I T __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, image_size: int) -> None:
        """
        This is the __init__ function of the dataset loader class.

        :return: None
        """

        self.cfg = ConfigStreamNetwork().parse()
        self.hard_samples = get_hardest_samples()

        # Transforms for each dataset
        self.contour_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.lbp_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.rgb_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.texture_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- __ G E T I T E M __ ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index: int):
        """
        This is the __getitem__ method of a dataset loader class. Given an index, it loads the corresponding images from
        three different datasets (RGB, texture, contour, LBP), and applies some data augmentation (transforms) to each
        image. It then returns a tuple of twelve images.

        :param index: An integer representing the index of the sample to retrieve from the dataset.
        :return: A tuple of 12 elements, where each element corresponds to an image with a specific transformation.
        """

        sample = self.hard_samples[index]

        contour_anchor_img = self.contour_transform(Image.open(sample[0]))
        contour_positive_img = self.contour_transform(Image.open(sample[1]))
        contour_negative_img = self.contour_transform(Image.open(sample[2]))
        lbp_anchor_img = self.lbp_transform(Image.open(sample[3]))
        lbp_positive_img = self.lbp_transform(Image.open(sample[4]))
        lbp_negative_img = self.lbp_transform(Image.open(sample[5]))
        rgb_anchor_img = self.rgb_transform(Image.open(sample[6]))
        rgb_positive_img = self.rgb_transform(Image.open(sample[7]))
        rgb_negative_img = self.rgb_transform(Image.open(sample[8]))
        texture_anchor_img = self.texture_transform(Image.open(sample[9]))
        texture_positive_img = self.texture_transform(Image.open(sample[10]))
        texture_negative_img = self.texture_transform(Image.open(sample[11]))

        rgb_positive_img_path = sample[7]
        rgb_negative_img_path = sample[8]

        return (contour_anchor_img, lbp_anchor_img, rgb_anchor_img, texture_anchor_img,
                contour_positive_img, lbp_positive_img, rgb_positive_img, texture_positive_img,
                contour_negative_img, lbp_negative_img, rgb_negative_img, texture_negative_img,
                rgb_positive_img_path, rgb_negative_img_path)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __ L E N __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        """
        This is the __len__ method of a dataset loader class.

        :return:
        """

        return len(self.hard_samples)
