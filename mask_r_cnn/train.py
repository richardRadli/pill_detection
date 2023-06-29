import os
import cv2
import torch
import torchvision.models.segmentation

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class MaskRCNNDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)

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

        # Apply transformations
        image = ToTensor()(image)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(num_instances, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.uint8)

        data = {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }

        return data


# Example usage
image_dir = "C:/Users/ricsi/Desktop/ogyei_v2/train/images"
mask_dir = "C:/Users/ricsi/Desktop/ogyei_v2/train/masks"

dataset = MaskRCNNDataset(image_dir, mask_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = "cuda"

# load an instance segmentation model pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
model.to(device)  # move model to the right devic

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

for epoch in range(10):
    for batch in dataloader:
        images = list(img.to(device) for img in batch['image'])
        targets = []
        for i in range(len(images)):
            target = {}
            target['boxes'] = batch['boxes'][i].to(device)
            target['labels'] = batch['labels'][i].to(device)
            target['masks'] = batch['masks'][i].to(device)
            targets.append(target)

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        print(epoch, 'loss:', losses.item())