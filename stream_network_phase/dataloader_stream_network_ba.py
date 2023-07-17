import numpy as np
import os

from itertools import combinations
from PIL import Image
from typing import List, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ S T R E A M   D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class StreamDataset(Dataset):
    def __init__(self, dataset_dir: str, type_of_stream: str, image_size: int) -> None:
        self.labels_set = None
        self.label_to_indices = None
        self.labels = None
        self.dataset_path = dataset_dir
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
        self.triplets = self.generate_triplets()

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- L O A D   D A T A S E T ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_dataset(self) -> List[Tuple[str, str]]:
        """
        Load the dataset.

        :return: A list of tuples, where each tuple contains the path to an image and its corresponding label.
        """

        dataset = []
        labels = set()
        for label_name in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label_name)
            if not os.path.isdir(label_path):
                continue
            label = label_name
            labels.add(label)
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
    def generate_triplets(self) -> List[Tuple[int, int, int]]:
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
        return triplets

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- __ G E T I T E M __ ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index: int):
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

        return anchor_img, positive_img, negative_img, negative_img_path

    def __len__(self) -> int:
        return len(self.triplets)
