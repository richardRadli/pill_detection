import cv2
import numpy as np
import os
import torch

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class MaskRCNNDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert mask to binary image
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Find contours and generate bounding boxes and labels for each instance
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_instances = len(contours)
        boxes = []
        labels = []
        masks = []
        for i in range(num_instances):
            x, y, w, h = cv2.boundingRect(contours[i])
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # Class label for instances (change if necessary)
            instance_mask = (mask == 255).astype('uint8')  # Mask for current instance
            masks.append(instance_mask)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            masks = torch.stack([self.mask_transform(mask) for mask in masks])

        # Apply transformations
        # image = ToTensor()(image)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(num_instances, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        data = {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }

        return data
