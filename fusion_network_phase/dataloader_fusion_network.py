"""
File: dataloader_fusion_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Feb 21, 2024

Description:
The FusionDataset class is a custom dataset loader class that is used for loading and processing image datasets for
fusion networks.
"""

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from fusion_network_phase.mine_hard_samples import get_hardest_samples


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++ F U S I O N D A T A S E T +++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class FusionDataset(Dataset):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- __ I N I T __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, image_size: int = 224) -> None:
        """
        This is the __init__ function of the dataset loader class.

        Args:
            image_size (int): Size of the image.

        Returns:
             None
        """

        # Transforms for each dataset
        self.grayscale_transform = transforms.Compose([
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

        self.hard_samples = get_hardest_samples()

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- __ G E T I T E M __ ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index: int):
        """
        This is the __getitem__ method of a dataset loader class. Given an index, it loads the corresponding images from
        three different datasets (RGB, texture, contour, LBP), and applies some data augmentation (transforms) to each
        image. It then returns a tuple of twelve images.

        Args:
            index (int): An integer representing the index of the sample to retrieve from the dataset.

        Returns:
            A tuple of 12 elements, where each element corresponds to an image with a specific transformation.
        """

        hard_samples = self.hard_samples[index]

        contour_anchor = self.grayscale_transform(Image.open(hard_samples[0]))
        contour_positive = self.grayscale_transform(Image.open(hard_samples[1]))
        contour_negative = self.grayscale_transform(Image.open(hard_samples[2]))
        lbp_anchor = self.grayscale_transform(Image.open(hard_samples[3]))
        lbp_positive = self.grayscale_transform(Image.open(hard_samples[4]))
        lbp_negative = self.grayscale_transform(Image.open(hard_samples[5]))
        rgb_anchor = self.rgb_transform(Image.open(hard_samples[6]))
        rgb_positive = self.rgb_transform(Image.open(hard_samples[7]))
        rgb_negative = self.rgb_transform(Image.open(hard_samples[8]))
        texture_anchor = self.grayscale_transform(Image.open(hard_samples[9]))
        texture_positive = self.grayscale_transform(Image.open(hard_samples[10]))
        texture_negative = self.grayscale_transform(Image.open(hard_samples[11]))

        return (contour_anchor, contour_positive, contour_negative,
                lbp_anchor, lbp_positive, lbp_negative,
                rgb_anchor, rgb_positive, rgb_negative,
                texture_anchor, texture_positive, texture_negative, hard_samples[7], hard_samples[8])

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __ L E N __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        """
        This is the __len__ method of a dataset loader class.

        Returns:
            int: the size of the dataset.
        """

        return len(self.hard_samples)