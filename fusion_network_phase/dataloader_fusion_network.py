"""
File: dataloader_fusion_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description:
The FusionDataset class is a custom dataset loader class that is used for loading and processing image datasets for
fusion networks.
"""

import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Tuple

from config.config import ConfigStreamNetwork
from config.config_selector import sub_stream_network_configs


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

        self.label_to_indices = None

        # Transforms for each dataset
        self.contour_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.RandomRotation(degrees=6),
            transforms.RandomRotation(degrees=354),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.lbp_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.RandomRotation(degrees=6),
            transforms.RandomRotation(degrees=354),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.rgb_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.RandomRotation(degrees=6),
            transforms.RandomRotation(degrees=354),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.texture_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.RandomRotation(degrees=6),
            transforms.RandomRotation(degrees=354),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        # Load datasets
        selected_subnetwork_config = sub_stream_network_configs(self.cfg)
        network_cfg_contour = selected_subnetwork_config.get("Contour")
        network_cfg_lpb = selected_subnetwork_config.get("LBP")
        network_cfg_rgb = selected_subnetwork_config.get("RGB")
        network_cfg_texture = selected_subnetwork_config.get("Texture")

        self.contour_dataset_anchor, _, _ = self.load_dataset([
            network_cfg_contour.get("train").get(self.cfg.dataset_type).get("anchor")
        ])
        self.contour_dataset_pos_neg, _, _ = self.load_dataset([
            network_cfg_contour.get("train").get(self.cfg.dataset_type).get("pos_neg")
        ])
        self.lbp_dataset_anchor, _, _ = self.load_dataset([
            network_cfg_lpb.get("train").get(self.cfg.dataset_type).get("anchor")
        ])
        self.lbp_dataset_pos_neg, _, _ = self.load_dataset([
            network_cfg_lpb.get("train").get(self.cfg.dataset_type).get("pos_neg")
        ])
        self.rgb_dataset_anchor, self.rgb_labels_set_anchor, self.rgb_label_to_indices_anchor = self.load_dataset([
            network_cfg_rgb.get("train").get(self.cfg.dataset_type).get("anchor")
        ])
        self.rgb_dataset_pos_neg, _, self.rgb_label_to_indices_pos_neg = self.load_dataset([
            network_cfg_rgb.get("train").get(self.cfg.dataset_type).get("pos_neg")
        ])
        self.texture_dataset_anchor, _, _ = self.load_dataset([
            network_cfg_texture.get("train").get(self.cfg.dataset_type).get("anchor")
        ])
        self.texture_dataset_pos_neg, _, _ = self.load_dataset([
            network_cfg_texture.get("train").get(self.cfg.dataset_type).get("pos_neg")
        ])

        self.triplets = self.generate_triplets(5000)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- __ G E T I T E M __ ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor, str, str]:
        """
        This is the __getitem__ method of a dataset loader class. Given an index, it loads the corresponding images from
        three different datasets (RGB, texture, contour, LBP), and applies some data augmentation (transforms) to each
        image. It then returns a tuple of twelve images.

        :param index: An integer representing the index of the sample to retrieve from the dataset.
        :return: A tuple of 12 elements, where each element corresponds to an image with a specific transformation.
        """

        anchor_index, positive_index, negative_index = self.triplets[index]

        # Load corresponding images from all datasets
        contour_anchor_img_path, _ = self.contour_dataset_anchor[anchor_index]
        lbp_anchor_img_path, _ = self.lbp_dataset_anchor[anchor_index]
        rgb_anchor_img_path, anchor_label = self.rgb_dataset_anchor[anchor_index]
        texture_anchor_img_path, _ = self.texture_dataset_anchor[anchor_index]

        # Load positive sample from the same class as anchor
        contour_positive_img_path, _ = self.contour_dataset_pos_neg[positive_index]
        lbp_positive_img_path, _ = self.lbp_dataset_pos_neg[positive_index]
        rgb_positive_img_path, _ = self.rgb_dataset_pos_neg[positive_index]
        texture_positive_img_path, _ = self.texture_dataset_pos_neg[positive_index]

        # Load negative sample from a different class
        contour_negative_img_path, _ = self.contour_dataset_pos_neg[negative_index]
        lbp_negative_img_path, _ = self.lbp_dataset_pos_neg[negative_index]
        rgb_negative_img_path, _ = self.rgb_dataset_pos_neg[negative_index]
        texture_negative_img_path, _ = self.texture_dataset_pos_neg[negative_index]

        # Load images and apply transforms
        contour_anchor_img = Image.open(contour_anchor_img_path)
        contour_positive_img = Image.open(contour_positive_img_path)
        contour_negative_img = Image.open(contour_negative_img_path)

        lbp_anchor_img = Image.open(lbp_anchor_img_path)
        lbp_positive_img = Image.open(lbp_positive_img_path)
        lbp_negative_img = Image.open(lbp_negative_img_path)

        rgb_anchor_img = Image.open(rgb_anchor_img_path)
        rgb_positive_img = Image.open(rgb_positive_img_path)
        rgb_negative_img = Image.open(rgb_negative_img_path)

        texture_anchor_img = Image.open(texture_anchor_img_path)
        texture_positive_img = Image.open(texture_positive_img_path)
        texture_negative_img = Image.open(texture_negative_img_path)

        contour_anchor_img = self.contour_transform(contour_anchor_img)
        contour_positive_img = self.contour_transform(contour_positive_img)
        contour_negative_img = self.contour_transform(contour_negative_img)

        lbp_anchor_img = self.lbp_transform(lbp_anchor_img)
        lbp_positive_img = self.lbp_transform(lbp_positive_img)
        lbp_negative_img = self.lbp_transform(lbp_negative_img)

        rgb_anchor_img = self.rgb_transform(rgb_anchor_img)
        rgb_positive_img = self.rgb_transform(rgb_positive_img)
        rgb_negative_img = self.rgb_transform(rgb_negative_img)

        texture_anchor_img = self.texture_transform(texture_anchor_img)
        texture_positive_img = self.texture_transform(texture_positive_img)
        texture_negative_img = self.texture_transform(texture_negative_img)

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

        return len(self.triplets)

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- L O A D   T H E   D A T A S E T -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def load_dataset(dataset_dirs):
        """
        Load the dataset from the specified directory paths.

        :return:
        """

        dataset = []
        labels = set()

        # Iterate over the dataset directories
        for dataset_path in dataset_dirs:
            # Iterate over the label directories in each dataset directory
            for label_name in os.listdir(dataset_path):
                label_path = os.path.join(dataset_path, label_name)

                # Skip non-directory files
                if not os.path.isdir(label_path):
                    continue

                label = label_name
                labels.add(label)

                # Iterate over the image files in the label directory
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    dataset.append((image_path, label))

        labels_all = np.array([x[1] for x in dataset])
        labels_set = set(labels_all)
        label_to_indices = \
            {label: np.where(labels_all == label)[0] for label in labels_set}

        return dataset, labels_set, label_to_indices

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- G E N E R A T E   T R I P L E T S ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def generate_triplets(self, num_triplets: int = 1000):
        """
        Generate triplets of indices for anchor, positive, and negative images.

        :param num_triplets: The number of triplets to generate.
        :return: A list of triplets, where each triplet contains the indices of anchor, positive, and negative images.
        """

        triplets = []

        for _ in range(num_triplets):
            # Select a random anchor class
            anchor_class = np.random.choice(list(self.rgb_labels_set_anchor))

            # Select a random anchor image from the anchor class
            anchor_index = np.random.choice(self.rgb_label_to_indices_anchor[anchor_class])

            # Select a random positive image from the same class as the anchor
            positive_index = np.random.choice(self.rgb_label_to_indices_pos_neg[anchor_class])

            # Select a random negative class different from the anchor class
            negative_class = np.random.choice(list(self.rgb_labels_set_anchor - {anchor_class}))

            # Select a random negative image from the negative class
            negative_index = np.random.choice(self.rgb_label_to_indices_pos_neg[negative_class])

            triplets.append((anchor_index, positive_index, negative_index))

        return triplets
