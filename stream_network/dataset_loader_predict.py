import os

from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class TestDataset(Dataset):
    def __init__(self, dataset_dir: str, type_of_network: str):
        self.dataset_dir = dataset_dir
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

        self.classes = os.listdir(self.dataset_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_dir = os.path.join(self.dataset_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.samples.append((file_path, class_name))

    def __getitem__(self, index: int) -> Tuple:
        img_path, label = self.samples[index]
        img = self._load_image(img_path)
        return img, self.class_to_idx[label]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> Image:
        with open(path, 'rb') as f:
            if self.type_of_network == "RGB":
                img = Image.open(f)
            elif self.type_of_network in ["Contour", "Texture"]:
                img = Image.open(f).convert("L")
            else:
                raise ValueError("Wrong type of network!")

            return self.transform(img)
