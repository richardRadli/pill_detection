import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++ S T R E A M   D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class StreamDataset(Dataset):
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, dataset_dir: str, type_of_network: str) -> None:
        self.labels_set = None
        self.label_to_indices = None
        self.labels = None
        self.dataset_path = dataset_dir
        self.type_of_network = type_of_network

        if self.type_of_network == "RGB":
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif self.type_of_network in ["Contour", "Texture"]:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Wrong kind of network")

        self.dataset = self.load_dataset()

    def load_dataset(self):
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

    def __getitem__(self, index):
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

        return anchor_img, positive_img, negative_img, negative_img_path

    def __len__(self):
        return len(self.dataset)
