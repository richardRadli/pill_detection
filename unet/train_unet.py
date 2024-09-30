import colorama
import logging
import numpy as np
import os
import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from typing import List

from config.json_config import json_config_selector
from config.dataset_paths_selector import dataset_images_path_selector
from config.networks_paths_selector import unet_paths
from data_loader_unet import UnetDataLoader
from utils.utils import create_timestamp, load_config_json, use_gpu_if_available, setup_logger


class TrainUnet:
    def __init__(self):
        timestamp = create_timestamp()
        colorama.init()
        setup_logger()

        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("unet").get("schema"),
                json_filename=json_config_selector("unet").get("config")
            )
        )

        if self.cfg.get("seed"):
            torch.manual_seed(1234)

        self.device = (
            use_gpu_if_available()
        )

        self.model = (
            self.create_segmentation_model().to(self.device)
        )

        dataset_name = self.cfg.get("dataset_name")

        self.train_loader = (
            self.create_segmentation_dataset(
                images_dir=dataset_images_path_selector(dataset_name).get("train").get("images"),
                masks_dir=dataset_images_path_selector(dataset_name).get("train").get("mask_images"),
                batch_size=self.cfg.get("batch_size"),
                shuffle=True)
        )

        self.valid_loader = self.create_segmentation_dataset(
            images_dir=dataset_images_path_selector(dataset_name).get("valid").get("images"),
            masks_dir=dataset_images_path_selector(dataset_name).get("valid").get("mask_images"),
            batch_size=self.cfg.get("batch_size"),
            shuffle=True)

        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.get("learning_rate")
            )
        )

        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
        )

        unet_path = unet_paths(dataset_name)

        # Tensorboard
        tensorboard_log_dir = os.path.join(unet_path['logs_folder'], timestamp)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

        # Create save directory
        self.save_path = os.path.join(unet_path["weights_folder"], timestamp)
        os.makedirs(self.save_path, exist_ok=True)

    def create_segmentation_dataset(self, images_dir: str, masks_dir: str, batch_size: int, shuffle: bool) \
            -> DataLoader:
        """
        Creates a DataLoader for a segmentation task using a custom dataset.

        Args:
            images_dir (str): Directory containing input images.
            masks_dir (str): Directory containing corresponding ground truth masks.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            DataLoader: PyTorch DataLoader that provides batches of images and masks.
        """

        transform = transforms.Compose([
            transforms.Resize((self.cfg.get("resized_img_size"), self.cfg.get("resized_img_size"))),
            transforms.ToTensor(),
        ])

        dataset = UnetDataLoader(images_dir=images_dir,
                                 masks_dir=masks_dir,
                                 transform=transform)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def create_segmentation_model(self) -> smp.Unet:
        """
        Creates and returns a U-Net model for segmentation tasks using the segmentation_models_pytorch library.

        The configuration for the model is provided by the `self.cfg` dictionary, which includes:
        - encoder_name (str): Name of the encoder (backbone) for the U-Net model.
        - encoder_weights (str or None): Pre-trained weights for the encoder, or None for random initialization.
        - channels (int): Number of input channels (e.g., 3 for RGB images).
        - classes (int): Number of output segmentation classes.

        Returns:
            smp.Unet: A U-Net model instance ready for training or inference.
        """

        model = (
            smp.Unet(
                encoder_name=self.cfg.get("encoder_name"),
                encoder_weights=self.cfg.get("encoder_weights"),
                in_channels=self.cfg.get("channels"),
                classes=self.cfg.get("classes"))
        )

        return model

    def train_loop(self, batch_images: torch.Tensor, batch_masks: torch.Tensor, train_losses: List[float]) -> None:
        """
        Executes one training step in the loop: performs forward pass, calculates loss, backpropagates gradients, and
        updates model weights.

        Args:
            batch_images (torch.Tensor): A batch of input images (shape: [batch_size, channels, height, width]).
            batch_masks (torch.Tensor): Corresponding ground truth masks (shape: [batch_size, classes, height, width]).
            train_losses (List[float]): A list to track and store the training losses for each batch.

        Returns:
            None
        """

        batch_images, batch_masks = batch_images.to(self.device), batch_masks.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(batch_images)
        loss = self.criterion(outputs, batch_masks)
        loss.backward()
        self.optimizer.step()
        train_losses.append(loss.item())

    def valid_loop(self, batch_data: torch.Tensor, batch_masks: torch.Tensor, valid_losses: List[float]) -> None:
        """
        Executes one validation step: performs a forward pass, calculates the loss,
        and tracks validation losses.

        Args:
            batch_data (torch.Tensor): A batch of validation images (shape: [batch_size, channels, height, width]).
            batch_masks (torch.Tensor): Corresponding ground truth masks for validation
                                        (shape: [batch_size, classes, height, width]).
            valid_losses (List[float]): A list to track and store the validation losses for each batch.

        Returns:
            None
        """

        batch_data = batch_data.to(self.device)
        batch_masks = batch_masks.to(self.device)

        output = self.model(batch_data)

        t_loss = self.criterion(output, batch_masks)
        valid_losses.append(t_loss.item())

    def fit(self) -> None:
        """
        Method to execute the training and validation loop.

        Return:
            None
        """

        best_valid_loss = float("inf")
        best_model_path = None
        epoch_without_improvement = 0

        train_losses = []
        valid_losses = []

        for epoch in tqdm(range(self.cfg.get("epochs"))):

            self.model.train()
            for batch_data, batch_masks in tqdm(self.train_loader, total=len(self.train_loader)):
                self.train_loop(batch_data, batch_masks, train_losses)

            self.model.eval()
            for batch_data, batch_masks in tqdm(self.valid_loader, total=len(self.train_loader)):
                self.valid_loop(batch_data, batch_masks, valid_losses)

            train_loss = np.mean(train_losses)
            valid_loss = np.mean(valid_losses)

            logging.info(f'\ntrain_loss: {train_loss:.4f} ' + f'valid_loss: {valid_loss:.4f}')

            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            train_losses.clear()
            valid_losses.clear()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epoch_without_improvement = 0
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(self.save_path, f"best_model_epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f'New weights have been saved at epoch {epoch} with value of {best_valid_loss:.4f}')
            else:
                logging.warning(f"No new weights have been saved. Best valid loss was {best_valid_loss:.5f},\n "
                                f"current valid loss is {valid_loss:.5f}")
                epoch_without_improvement += 1
                if epoch_without_improvement >= self.cfg.get("patience"):
                    logging.warning(f"Early stopping counter: {epoch_without_improvement}")
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()
        self.writer.flush()


if __name__ == '__main__':
    train_unet = TrainUnet()
    train_unet.fit()
