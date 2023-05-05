import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_info = self.annotations['images'][index]
        image_id = image_info['id']
        image_name = image_info['file_name']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        image_width = image.shape[2]
        image_height = image.shape[1]

        # Extract YOLO format segmentation annotations
        annotations = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
        bboxes = []
        labels = []
        for annotation in annotations:
            segmentations = annotation['segmentation']
            for segmentation in segmentations:
                # Convert segmentation to YOLO format [x_center, y_center, width, height]
                x, y, w, h = self._convert_segmentation_to_yolo(segmentation, image_width, image_height)
                bboxes.append([x, y, w, h])
                labels.append(annotation['category_id'])

        return image, bboxes, labels

    def _convert_segmentation_to_yolo(self, segmentation, image_width, image_height):
        # Convert segmentation to YOLO format [x_center, y_center, width, height]
        x_min = min(segmentation[::2])
        x_max = max(segmentation[::2])
        y_min = min(segmentation[1::2])
        y_max = max(segmentation[1::2])
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # Normalize values between 0 and 1
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        return x_center, y_center, width, height

from torchvision import transforms

# Define the image directory and annotation file paths
image_dir = 'C:/Users/ricsi/Desktop/train'
annotation_file = 'C:/Users/ricsi/Desktop/train/_annotations.coco.json'

# Define any desired image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create an instance of the SegmentationDataset
dataset = SegmentationDataset(image_dir, annotation_file, transform)

# Iterate over the dataset
for image, bboxes, labels in dataset:
    print(image)
