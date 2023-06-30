import logging
import os.path

import numpy as np
import torch
import torchvision.models.segmentation

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

from config.logger_setup import setup_logger
from config.config import ConfigTrainingMaskRCNN
from dataloader_mask_r_cnn import MaskRCNNDataset
from utils.utils import scale_down_image, use_gpu_if_available


class TrainMaskRCNN:
    def __init__(self):
        setup_logger()

        self.cfg = ConfigTrainingMaskRCNN().parse()

        self.device = use_gpu_if_available()

        # Example usage
        image_dir = "C:/Users/ricsi/Desktop/salt/images"
        mask_dir = "C:/Users/ricsi/Desktop/salt/masks"

        new_shape = scale_down_image(image_dir, scale_factor=self.cfg.img_scale)
        logging.info(f"The new shape after scaling with {self.cfg.img_scale} is {new_shape[1]} × {new_shape[0]}")

        # Define image transform
        image_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert the image to PIL Image
            transforms.Resize((new_shape[0], new_shape[1])),  # Resize the image
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

        # Define mask transform
        mask_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert the mask to PIL Image
            transforms.Resize((new_shape[0], new_shape[1])),  # Resize the mask
            transforms.ToTensor()  # Convert the mask to a tensor
        ])

        dataset = MaskRCNNDataset(image_dir, mask_dir, image_transform=None, mask_transform=None)
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size  # 20% for validation

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create data loaders for training and validation
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        # load an instance segmentation model pre-trained on COCO
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=self.cfg.num_of_classes)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.cfg.learning_rate)

        self.writer = SummaryWriter(log_dir="C:/Users/ricsi/Desktop/salt/logs")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- T R A I N ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train(self):
        train_losses = []
        valid_losses = []

        best_valid_loss = float('inf')
        best_model_path = None

        self.model.train()

        for epoch in tqdm(range(self.cfg.epochs), desc="Epochs"):
            for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc="Training loop"):
                images = list(img.to(self.device) for img in batch['image'])
                targets = []
                for i in range(len(images)):
                    target = {'boxes': batch['boxes'][i].to(self.device),
                              'labels': batch['labels'][i].to(self.device),
                              'masks': batch['masks'][i].to(self.device)}
                    targets.append(target)

                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
                train_losses.append(losses.item())

            for batch in tqdm(self.val_dataloader, total=len(self.val_dataloader), desc="Validation loop"):
                images = list(img.to(self.device) for img in batch['image'])
                targets = []
                for i in range(len(images)):
                    target = {'boxes': batch['boxes'][i].to(self.device),
                              'labels': batch['labels'][i].to(self.device),
                              'masks': batch['masks'][i].to(self.device)}
                    targets.append(target)

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                valid_losses.append(losses.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            logging.info(f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')
            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            train_losses.clear()
            valid_losses.clear()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join("C:/Users/ricsi/Desktop/salt/saves", str(epoch) + ".torch")
                torch.save(self.model.state_dict(), best_model_path)


if __name__ == "__main__":
    try:
        train_mask_r_cnn = TrainMaskRCNN()
        train_mask_r_cnn.train()
    except KeyboardInterrupt as kie:
        logging.error(kie)
