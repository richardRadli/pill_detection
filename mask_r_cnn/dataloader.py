import json
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict


class CocoDataset(Dataset):
    def __init__(self, data_dir: str, ann_file: str, transforms: transforms.Compose):
        self.data_dir = data_dir
        self.transforms = transforms

        with open(ann_file, 'r') as f:
            ann_data = json.load(f)

        self.categories = {}
        for cat in ann_data['categories']:
            self.categories[cat['id']] = cat['name']

        self.images = {}
        for img in ann_data['images']:
            self.images[img['id']] = {
                'filename': img['file_name'],
                'width': img['width'],
                'height': img['height']
            }

        self.annotations = []
        for ann in ann_data['annotations']:
            self.annotations.append({
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'segmentation': ann['segmentation'],
                'iscrowd': ann['iscrowd']
            })

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_id = ann['image_id']
        img_info = self.images[img_id]
        img_path = os.path.join(self.data_dir, img_info['filename'])

        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        boxes = torch.as_tensor([ann['bbox']], dtype=torch.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = torch.as_tensor([ann['category_id']], dtype=torch.int64)

        image_id = torch.tensor([index])

        area = torch.as_tensor([ann['area']], dtype=torch.float32)

        iscrowd = torch.as_tensor([ann['iscrowd']], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annotations)


data_dir = "C:/Users/ricsi/Desktop/train"
ann_file = "C:/Users/ricsi/Desktop/train/annotations.json"


# Define transforms for data augmentation
transforms = transforms.Compose([
    transforms.ToTensor(),
    # Add additional transforms as needed
])

# Create dataset and dataloader
dataset = CocoDataset(data_dir, ann_file, transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

for img, target in dataloader:
    print(target)
    break