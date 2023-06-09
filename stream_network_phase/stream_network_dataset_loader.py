import numpy as np
import os

from PIL import Image
from typing import List, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ S T R E A M   D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class StreamDataset(Dataset):
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, dataset_dir: str, type_of_stream: str) -> None:
        self.labels_set = None
        self.label_to_indices = None
        self.labels = None
        self.dataset_path = dataset_dir
        self.type_of_stream = type_of_stream

        if self.type_of_stream == "RGB":
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif self.type_of_stream in ["Contour", "Texture", "LBP"]:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Wrong kind of network")

        self.dataset = self.load_dataset()

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- L O A D   T H E   D A T A S E T -----------------------------------------
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
    # ----------------------------------------------- __ G E T I T E M __ ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image, Image.Image, str, str]:
        """
        Retrieve a triplet of images (anchor, positive, negative) and their paths for a given index.

        :param index: The index of the anchor image.
        :return: A tuple of three images (anchor, positive, negative) and their paths.
        """

        anchor_img_path, anchor_label = self.dataset[index]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])

        positive_img_path, positive_label = self.dataset[positive_index]
        negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
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

        return len(self.dataset)
