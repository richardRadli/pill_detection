"""
File: dataloader_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program creates the different images (contour, lbp, rgb, texture) for the substreams.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image
from typing import List, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from config.config import ConfigStreamNetwork


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ S T R E A M   D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class StreamDataset(Dataset):
    def __init__(self, dataset_dirs_anchor: List[str], dataset_dirs_pos_neg, type_of_stream: str, image_size: int,
                 num_triplets: int = -1) -> None:
        """
        Initialize the StreamDataset class.

        :param dataset_dirs_anchor: A list of directory paths containing the anchor dataset.
        :param dataset_dirs_pos_neg: A list of directory paths containing the positive and negative dataset
        :param type_of_stream: The type of stream (RGB, Contour, Texture, LBP).
        :param image_size: The size of the images.
        :param num_triplets: The number of triplets to generate (-1 means all possible triplets).
        """

        self.cfg = ConfigStreamNetwork().parse()

        self.dataset_dirs_anchor = dataset_dirs_anchor
        self.dataset_dirs_pos_neg = dataset_dirs_pos_neg
        self.type_of_stream = type_of_stream

        if self.type_of_stream == "RGB":
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
                transforms.RandomRotation(degrees=6),
                transforms.RandomRotation(degrees=12),
                transforms.RandomRotation(degrees=354),
                transforms.RandomRotation(degrees=348),
                transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.0),
                transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.5),
                transforms.ColorJitter(brightness=0.5, contrast=0.0, saturation=0.0, hue=0.0),
                transforms.ColorJitter(brightness=0.0, contrast=0.5, saturation=0.0, hue=0.0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif self.type_of_stream in ["Contour", "Texture", "LBP"]:
            zoom_factor = np.random.uniform(0.8, 1.2)
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
                transforms.RandomRotation(degrees=np.random.randint(6, 354)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(zoom_factor, zoom_factor)),
                transforms.ColorJitter(brightness=0.5, contrast=0.0, saturation=0.0, hue=0.0),
                transforms.ColorJitter(brightness=0.0, contrast=0.5, saturation=0.0, hue=0.0),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Wrong kind of network")

        self.dataset_anchor, self.labels_set_anchor, self.label_to_indices_anchor = (
            self.load_dataset(self.dataset_dirs_anchor)
        )
        self.dataset_pos_neg, _, self.label_to_indices_pos_neg = self.load_dataset(self.dataset_dirs_pos_neg)
        self.triplets = self.generate_triplets(num_triplets)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   D A T A S E T ---------------------------------------------
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
    def generate_triplets(self, num_triplets: int) -> List[Tuple[int, int, int]]:
        """
        Generate triplets of indices for anchor, positive, and negative images.

        :param num_triplets: The number of triplets to generate.
        :return: A list of triplets, where each triplet contains the indices of anchor, positive, and negative images.
        """

        triplets = []

        for anchor_class in self.labels_set_anchor:
            anchor_index = np.random.choice(self.label_to_indices_anchor[anchor_class])
            positive_index = np.random.choice(self.label_to_indices_pos_neg[anchor_class])

            negative_class = np.random.choice(list(self.labels_set_anchor - {anchor_class}))
            negative_index = np.random.choice(self.label_to_indices_pos_neg[negative_class])

            triplets.append((anchor_index, positive_index, negative_index))

        for _ in range(num_triplets - len(self.labels_set_anchor)):
            anchor_class = np.random.choice(list(self.labels_set_anchor))
            anchor_index = np.random.choice(self.label_to_indices_anchor[anchor_class])

            positive_index = np.random.choice(self.label_to_indices_pos_neg[anchor_class])

            negative_class = np.random.choice(list(self.labels_set_anchor - {anchor_class}))
            negative_index = np.random.choice(self.label_to_indices_pos_neg[negative_class])

            triplets.append((anchor_index, positive_index, negative_index))

        return triplets

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------- G E N E R A T E   A L L   T R I P L E T S ------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def generate_all_triplets(self) -> List[Tuple[int, int, int]]:
        """
        Generate all possible triplets of indices for anchor, positive, and negative images.

        :return: A list of all possible triplets, where each triplet contains the indices of anchor, positive, and
        negative images.
        """

        triplets = []

        for anchor_class in self.labels_set_anchor:
            for anchor_index in self.label_to_indices_anchor[anchor_class]:
                for positive_index in self.label_to_indices_pos_neg[anchor_class]:
                    for negative_class in (self.labels_set_anchor - {anchor_class}):
                        for negative_index in self.label_to_indices_pos_neg[negative_class]:
                            triplets.append((anchor_index, positive_index, negative_index))

        return triplets

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------- C O N V E R T   T E N S O R   T O   I M A G E  ---------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def convert_tensor_to_image(transformed_image):
        transformed_image_np = transformed_image.numpy()
        transformed_image_np = (transformed_image_np * 255.).astype(np.uint8)
        return transformed_image_np.T

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- P L O T   I M A G E S  ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def plot_images(self, a, p, n, index, path):
        """

        :param a:
        :param p:
        :param n:
        :param index:
        :param path:
        :return:
        """

        a = self.convert_tensor_to_image(a)
        p = self.convert_tensor_to_image(p)
        n = self.convert_tensor_to_image(n)

        plt.subplot(1, 3, 1)
        plt.imshow(a, cmap='gray')
        plt.title('a')

        plt.subplot(1, 3, 2)
        plt.imshow(p, cmap='gray')
        plt.title('p')

        plt.subplot(1, 3, 3)
        plt.imshow(n, cmap='gray')
        plt.title('n')

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'triplet_{index}.png'))

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- __ G E T I T E M __ ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image, Image.Image, str, str, str]:
        """
        Retrieve a triplet of images (anchor, positive, negative) and their paths for a given index.

        :param index: The index of the anchor image.
        :return: A tuple of three images (anchor, positive, negative) and their paths.
        """

        anchor_index, positive_index, negative_index = self.triplets[index]

        anchor_img_path, _ = self.dataset_anchor[anchor_index]
        positive_img_path, _ = self.dataset_pos_neg[positive_index]
        negative_img_path, _ = self.dataset_pos_neg[negative_index]

        anchor_img = Image.open(anchor_img_path)
        positive_img = Image.open(positive_img_path)
        negative_img = Image.open(negative_img_path)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_img_path, positive_img_path, negative_img_path

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __ L E N __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        """
        This is the __len__ method of a dataset loader class.

        :return: The length of the dataset
        """

        return len(self.triplets)
