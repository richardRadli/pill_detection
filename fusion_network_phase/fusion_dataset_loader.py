import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class FusionNetDataset(Dataset):
    def __init__(self, dataset_path1, dataset_path2, dataset_path3):
        self.label_to_indices = None
        self.labels_set = None
        self.labels = None
        self.dataset_path1 = dataset_path1
        self.dataset_path2 = dataset_path2
        self.dataset_path3 = dataset_path3
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = self.load_dataset()

    def load_dataset(self):
        dataset = []
        labels = set()
        for label_name in os.listdir(self.dataset_path1):
            label_path1 = os.path.join(self.dataset_path1, label_name)
            if not os.path.isdir(label_path1):
                continue
            for label_name in os.listdir(self.dataset_path2):
                label_path2 = os.path.join(self.dataset_path2, label_name)
                if not os.path.isdir(label_path2):
                    continue
                for label_name in os.listdir(self.dataset_path3):
                    label_path3 = os.path.join(self.dataset_path3, label_name)
                    if not os.path.isdir(label_path3):
                        continue
                    label = label_name
                    labels.add(label)
                    for image_name1 in os.listdir(label_path1):
                        image_path1 = os.path.join(label_path1, image_name1)
                        for image_name2 in os.listdir(label_path2):
                            image_path2 = os.path.join(label_path2, image_name2)
                            for image_name3 in os.listdir(label_path3):
                                image_path3 = os.path.join(label_path3, image_name3)
                                dataset.append((image_path1, image_path2, image_path3, label))
        self.labels = np.array([x[3] for x in dataset])
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        return dataset

    def __getitem__(self, index):
        anchor_img_path, positive_img_path, negative_img_path, anchor_label = self.dataset[index]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        positive_img_path, _ = self.dataset[positive_index][:2]
        negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative_img_path, _, _ = self.dataset[negative_index][:3]
        anchor_img = Image.open(anchor_img_path)
        positive_img = Image.open(positive_img_path)
        negative_img = Image.open(negative_img_path)
        if self.transform:
            anchor_img = self.transform(anchor_img)
        positive_img = self.transform(positive_img)
        negative_img = self.transform(negative_img)
        return anchor_img, positive_img, negative_img, anchor_img_path, positive_img_path, negative_img_path

    def __len__(self):
        return len(self.dataset)
