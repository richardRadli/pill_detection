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

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from config.config import ConfigStreamNetwork


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++ F U S I O N D A T A S E T +++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class FusionDataset(Dataset):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- __ I N I T __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, dataset_dirs_anchor_con, dataset_dirs_pos_neg_con, dataset_dirs_anchor_lbp,
                 dataset_dirs_pos_neg_lbp, dataset_dirs_anchor_rgb, dataset_dirs_pos_neg_rgb, dataset_dirs_anchor_tex,
                 dataset_dirs_pos_neg_tex, image_size: int = 224) -> None:
        """
        This is the __init__ function of the dataset loader class.

        :return: None
        """

        self.cfg = ConfigStreamNetwork().parse()

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

        self.query_images_con, self.query_labels_con, self.reference_images_con, self.reference_labels_con = (
            self.get_stream_data(dataset_dirs_anchor_con, dataset_dirs_pos_neg_con)
        )
        self.query_images_lbp, self.query_labels_lbp, self.reference_images_lbp, self.reference_labels_lbp = (
            self.get_stream_data(dataset_dirs_anchor_lbp, dataset_dirs_pos_neg_lbp)
        )
        self.query_images_rgb, self.query_labels_rgb, self.reference_images_rgb, self.reference_labels_rgb = (
            self.get_stream_data(dataset_dirs_anchor_rgb, dataset_dirs_pos_neg_rgb)
        )
        self.query_images_tex, self.query_labels_tex, self.reference_images_tex, self.reference_labels_tex = (
            self.get_stream_data(dataset_dirs_anchor_tex, dataset_dirs_pos_neg_tex)
        )

    def get_stream_data(self, dataset_dirs_anchor, dataset_dirs_pos_neg):
        query_images, query_labels = self.load_dataset(dataset_dirs_anchor)
        reference_images, reference_labels = self.load_dataset(dataset_dirs_pos_neg)
        return query_images, query_labels, reference_images, reference_labels

    @staticmethod
    def load_dataset(dataset_dirs):
        dataset = []
        labels = []

        for dataset_path in dataset_dirs:
            for label_name in os.listdir(dataset_path):
                label_path = os.path.join(dataset_path, label_name)
                if not os.path.isdir(label_path):
                    continue
                label = label_name
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    dataset.append(image_path)
                    labels.append(label)

        return dataset, labels

    @staticmethod
    def collect_data(query_images, query_labels, reference_images, reference_labels, transform, index):
        query_image_path = query_images[index]
        query_image = transform(Image.open(query_image_path))
        query_label = query_labels[index]

        reference_label = query_label
        reference_indices = np.where(np.array(reference_labels) == reference_label)[0]

        if len(reference_indices) == 0:
            reference_image_path = reference_images[-1]
        else:
            reference_index = np.random.choice(reference_indices)
            reference_image_path = reference_images[reference_index]

        reference_image = transform(Image.open(reference_image_path))

        return query_image, query_label, query_image_path, reference_image, reference_label, reference_image_path

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

        (query_image_con, query_label_con, query_image_path_con, reference_image_con, reference_label_con,
         reference_image_path_con) = self.collect_data(self.query_images_con,
                                                       self.query_labels_con,
                                                       self.reference_images_con,
                                                       self.reference_labels_con,
                                                       self.grayscale_transform,
                                                       index)

        (query_image_lbp, query_label_lbp, query_image_path_lbp, reference_image_lbp, reference_label_lbp,
         reference_image_path_lbp) = self.collect_data(self.query_images_lbp,
                                                       self.query_labels_lbp,
                                                       self.reference_images_lbp,
                                                       self.reference_labels_lbp,
                                                       self.grayscale_transform,
                                                       index)

        (query_image_rgb, query_label_rgb, query_image_path_rgb, reference_image_rgb, reference_label_rgb,
         reference_image_path_rgb) = self.collect_data(self.query_images_rgb,
                                                       self.query_labels_rgb,
                                                       self.reference_images_rgb,
                                                       self.reference_labels_rgb,
                                                       self.rgb_transform,
                                                       index)

        (query_image_tex, query_label_tex, query_image_path_tex, reference_image_tex, reference_label_tex,
         reference_image_path_tex) = self.collect_data(self.query_images_tex,
                                                       self.query_labels_tex,
                                                       self.reference_images_tex,
                                                       self.reference_labels_tex,
                                                       self.grayscale_transform,
                                                       index)

        return (query_image_con,  reference_image_con,
                query_image_lbp,  reference_image_lbp,
                query_image_rgb, query_label_rgb, reference_image_rgb, reference_label_rgb,
                query_image_tex,  reference_image_tex)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __ L E N __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        """
        This is the __len__ method of a dataset loader class.

        :return:
        """

        return len(self.query_images_con)
