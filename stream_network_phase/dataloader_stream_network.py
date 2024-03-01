import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import LabelEncoder
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

        # Load datasets
        self.query_images, self.query_labels = self.load_dataset(dataset_dirs_anchor)
        self.reference_images, self.reference_labels = self.load_dataset(dataset_dirs_pos_neg)

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

        # Initialize label encoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        return dataset, encoded_labels

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

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, index: int):
        # Query (consumer) images
        query_image_path = self.query_images[index]
        query_image = self.transform(Image.open(query_image_path))
        query_label = self.query_labels[index]

        # Reference images
        reference_label = query_label
        reference_indices = np.where(np.array(self.reference_labels) == reference_label)[0]

        if len(reference_indices) == 0:
            reference_image_path = self.reference_images[-1]
        else:
            reference_index = np.random.choice(reference_indices)
            reference_image_path = self.reference_images[reference_index]

        reference_image = self.transform(Image.open(reference_image_path))

        return query_image, query_label, query_image_path, reference_image, reference_label, reference_image_path
