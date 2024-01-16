"""
File: dataloader_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program creates the different images (contour, lbp, rgb, texture) for the substreams.
"""

import numpy as np
import os
import random

from itertools import combinations
from PIL import Image
from typing import List, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from config.config import ConfigStreamNetwork


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ S T R E A M   D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class StreamDataset(Dataset):
    def __init__(self, dataset_dirs: List[str], type_of_stream: str, image_size: int, num_triplets: int = -1) -> None:
        """
        Initialize the StreamDataset class.

        :param dataset_dirs: A list of directory paths containing the dataset.
        :param type_of_stream: The type of stream (RGB, Contour, Texture, LBP).
        :param image_size: The size of the images.
        :param num_triplets: The number of triplets to generate (-1 means all possible triplets).
        """
        self.cfg = ConfigStreamNetwork().parse()

        self.labels_set = None
        self.label_to_indices = None
        self.labels = None
        self.dataset_paths = dataset_dirs
        self.type_of_stream = type_of_stream

        if self.type_of_stream == "RGB":
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif self.type_of_stream in ["Contour", "Texture", "LBP"]:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Wrong kind of network")

        self.dataset = self.load_dataset()
        self.triplets = self.generate_triplets(num_triplets)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   D A T A S E T ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_dataset(self) -> List[Tuple[str, str]]:
        """
        Load the dataset from the specified directory paths.

        :return: A list of tuples containing the image paths and their corresponding labels.
        """

        dataset = []
        labels = set()

        # Iterate over the dataset directories
        for dataset_path in self.dataset_paths:
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

        self.labels = np.array([x[1] for x in dataset])
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

        return dataset

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- G E N E R A T E   T R I P L E T S ----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def generate_triplets(self, num_triplets: int = 1000) -> List[Tuple[int, int, int]]:
        """
        Generate triplets of indices for anchor, positive, and negative images.

        :param num_triplets: The number of triplets to generate.
        :return: A list of triplets, where each triplet contains the indices of anchor, positive, and negative images.
        """

        triplets = []
        for label in self.labels_set:
            label_indices = self.label_to_indices[label]
            if len(label_indices) < 2:
                continue
            anchor_positive_pairs = list(combinations(label_indices, 2))
            for anchor_index, positive_index in anchor_positive_pairs:
                negative_label = np.random.choice(list(self.labels_set - {label}))
                negative_index = np.random.choice(self.label_to_indices[negative_label])
                triplets.append((anchor_index, positive_index, negative_index))

        # Shuffle the triplets and select a fixed number
        random.shuffle(triplets)
        triplets = triplets[:num_triplets]

        return triplets

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- __ G E T I T E M __ ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image, Image.Image, str, str]:
        """
        Retrieve a triplet of images (anchor, positive, negative) and their paths for a given index.

        :param index: The index of the anchor image.
        :return: A tuple of three images (anchor, positive, negative) and their paths.
        """

        anchor_index, positive_index, negative_index = self.triplets[index]

        anchor_img_path, _ = self.dataset[anchor_index]
        positive_img_path, _ = self.dataset[positive_index]
        negative_img_path, _ = self.dataset[negative_index]

        anchor_img = Image.open(anchor_img_path)
        positive_img = Image.open(positive_img_path)
        negative_img = Image.open(negative_img_path)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, negative_img_path, positive_img_path

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __ L E N __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        """
        This is the __len__ method of a dataset loader class.

        :return: The length of the dataset
        """

        return len(self.triplets)
