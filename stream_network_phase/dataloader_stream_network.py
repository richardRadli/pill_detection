import os
import numpy as np

from PIL import Image
from typing import List
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from config.config import ConfigStreamNetwork


class DataLoaderStreamNet(Dataset):
    def __init__(self, dataset_dirs_anchor: List[str], dataset_dirs_pos_neg: List[str]) -> None:
        """
        Initialize the StreamDataset class.

        :param dataset_dirs_anchor: A list of directory paths containing the anchor dataset.
        :param dataset_dirs_pos_neg: A list of directory paths containing the positive and negative dataset
        """

        self.cfg = ConfigStreamNetwork().parse()

        self.dataset_dirs_anchor = dataset_dirs_anchor
        self.dataset_dirs_pos_neg = dataset_dirs_pos_neg

        self.anchor_images, self.anchor_labels = self.load_dataset(self.dataset_dirs_anchor)
        self.pos_neg_images, self.pos_neg_labels = self.load_dataset(self.dataset_dirs_pos_neg)

        self.transform = self.get_transform()

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

    def get_transform(self):
        if self.cfg.type_of_stream == "RGB":
            return transforms.Compose([
                transforms.Resize(self.cfg.img_size_en),
                transforms.CenterCrop(self.cfg.img_size_en),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif self.cfg.type_of_stream in ["Contour", "Texture", "LBP"]:
            return transforms.Compose([
                transforms.Resize(self.cfg.img_size_en),
                transforms.CenterCrop(self.cfg.img_size_en),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Wrong kind of network")

    def __getitem__(self, index: int):
        anchor_image_path = self.anchor_images[index]
        anchor_image = self.transform(Image.open(anchor_image_path))
        anchor_label = self.anchor_labels[index]

        pos_neg_label = anchor_label
        pos_neg_indices = np.where(np.array(self.pos_neg_labels) == pos_neg_label)[0]

        if len(pos_neg_indices) == 0:
            pos_neg_image_path = self.pos_neg_images[-1]
        else:
            pos_neg_index = np.random.choice(pos_neg_indices)
            pos_neg_image_path = self.pos_neg_images[pos_neg_index]

        pos_neg_image = self.transform(Image.open(pos_neg_image_path))

        return anchor_image, anchor_label, pos_neg_image, pos_neg_label

    def __len__(self) -> int:
        return len(self.anchor_images)
