import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import LabelEncoder
from typing import Any, List, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from config.json_config import json_config_selector
from utils.utils import load_config_json


class DataLoaderStreamNet(Dataset):
    def __init__(self, dataset_dirs_anchor: List[str], dataset_dirs_pos_neg: List[str]) -> None:
        """
        Initialize the DataLoaderStreamNet class.
        
        Args:
            dataset_dirs_anchor: A list of directory paths containing the anchor dataset.
            dataset_dirs_pos_neg: A list of directory paths containing the positive and negative dataset

        Returns:
            None
        """

        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )

        network_type = self.cfg.get("type_of_net")
        network_cfg = self.cfg.get("networks").get(network_type)

        self.image_size = network_cfg.get("image_size")

        # Load datasets
        self.query_images, self.query_labels = (
            self.load_dataset(dataset_dirs_anchor)
        )

        self.reference_images, self.reference_labels = (
            self.load_dataset(dataset_dirs_pos_neg)
        )

        self.transform = self.get_transform()

    @staticmethod
    def load_dataset(dataset_dirs: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Load the dataset from the provided directories and encode labels.

        Args:
            dataset_dirs: A list of directory paths for the dataset.

        Returns:
            Tuple[List[str], np.ndarray]: A list of image file paths and an array of encoded labels.
        """

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

    def get_transform(self) -> transforms.Compose:
        """
        Get the appropriate transformation pipeline for the images based on the stream type.

        Returns:
            transforms.Compose: A transformation pipeline for the images.
        """

        if self.cfg.get("type_of_stream") == "RGB":
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif self.cfg.get("type_of_stream") in ["Contour", "Texture", "LBP"]:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Wrong kind of network")

    def __len__(self) -> int:
        """
        Get the total number of query images in the dataset.

        Returns:
            int: The total number of query images.
        """

        return len(self.reference_images)

    def __getitem__(self, index: int) -> Tuple[Any, np.ndarray, str, Any, np.ndarray, str]:
        """
        Get a query image and a reference image with their corresponding labels.

        Args:
            index (int): The index of the query image.

        Returns:
            Tuple[Any, np.ndarray, str, Any, np.ndarray, str]:
                A tuple containing the query image, query label, query image path, reference image, reference label,
                and reference image path.
        """

        # Reference image path
        reference_image_path = self.reference_images[index]
        reference_image = self.transform(Image.open(reference_image_path))
        reference_label = self.reference_labels[index]

        # Query images
        query_label = reference_label
        query_indices = np.where(np.array(self.query_labels) == query_label)[0]

        if len(query_indices) == 0:
            query_image_path = self.query_images[-1]
        else:
            query_index = np.random.choice(query_indices)
            query_image_path = self.query_images[query_index]

        query_image = self.transform(Image.open(query_image_path))

        return query_image, query_label, query_image_path, reference_image, reference_label, reference_image_path
