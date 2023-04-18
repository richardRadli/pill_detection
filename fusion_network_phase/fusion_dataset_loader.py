import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class FusionDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        self.label_to_indices = None
        self.labels = None
        self.dataset_path = dataset_dir

        # Transforms for each dataset
        self.rgb_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.texture_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.contour_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # Load datasets
        self.rgb_dataset = self.load_dataset('rgb_hardest')
        self.labels_set = set(label for _, label in self.rgb_dataset)

        self.texture_dataset = self.load_dataset('texture_hardest')
        self.contour_dataset = self.load_dataset('contour_hardest')
        self.prepare_labels()

    def __getitem__(self, index):
        # Load corresponding images from all datasets
        rgb_anchor_img_path, anchor_label = self.rgb_dataset[index]
        texture_anchor_img_path, _ = self.texture_dataset[index]
        contour_anchor_img_path, _ = self.contour_dataset[index]

        # Load positive sample from the same class as anchor
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        rgb_positive_img_path, _ = self.rgb_dataset[positive_index]
        texture_positive_img_path, _ = self.texture_dataset[positive_index]
        contour_positive_img_path, _ = self.contour_dataset[positive_index]

        # Load negative sample from a different class
        negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        rgb_negative_img_path, _ = self.rgb_dataset[negative_index]
        texture_negative_img_path, _ = self.texture_dataset[negative_index]
        contour_negative_img_path, _ = self.contour_dataset[negative_index]

        # Load images and apply transforms
        rgb_anchor_img = Image.open(rgb_anchor_img_path)
        rgb_positive_img = Image.open(rgb_positive_img_path)
        rgb_negative_img = Image.open(rgb_negative_img_path)

        texture_anchor_img = Image.open(texture_anchor_img_path)
        texture_positive_img = Image.open(texture_positive_img_path)
        texture_negative_img = Image.open(texture_negative_img_path)

        contour_anchor_img = Image.open(contour_anchor_img_path)
        contour_positive_img = Image.open(contour_positive_img_path)
        contour_negative_img = Image.open(contour_negative_img_path)

        rgb_anchor_img = self.rgb_transform(rgb_anchor_img)
        rgb_positive_img = self.rgb_transform(rgb_positive_img)
        rgb_negative_img = self.rgb_transform(rgb_negative_img)

        texture_anchor_img = self.texture_transform(texture_anchor_img)
        texture_positive_img = self.texture_transform(texture_positive_img)
        texture_negative_img = self.texture_transform(texture_negative_img)

        contour_anchor_img = self.contour_transform(contour_anchor_img)
        contour_positive_img = self.contour_transform(contour_positive_img)
        contour_negative_img = self.contour_transform(contour_negative_img)

        return (rgb_anchor_img, texture_anchor_img, contour_anchor_img,
                rgb_positive_img, texture_positive_img, contour_positive_img,
                rgb_negative_img, texture_negative_img, contour_negative_img)

    def __len__(self):
        return len(self.rgb_dataset)

    def load_dataset(self, dataset_name: str):
        dataset = []
        labels = []
        dataset_path = os.path.join(self.dataset_path, dataset_name)
        for label_name in os.listdir(dataset_path):
            label_path = os.path.join(dataset_path, label_name)
            if not os.path.isdir(label_path):
                continue
            label = label_name
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                dataset.append((image_path, label))
                labels.append(label)
        self.labels = labels
        return dataset

    def prepare_labels(self):
        # Initialize dictionary that maps each label to corresponding indices in dataset
        self.label_to_indices = {label: [] for label in self.labels_set}
        for i in range(len(self.labels)):
            label = self.labels[i]
            self.label_to_indices[label].append(i)


# dt = FusionDataset('C:/Users/ricsi/Documents/project/storage/IVM/images')
