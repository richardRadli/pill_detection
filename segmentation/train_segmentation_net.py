import colorama
import logging
import numpy as np
import os
import torch

from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from typing import List

from config.json_config import json_config_selector
from config.dataset_paths_selector import dataset_images_path_selector
from config.networks_paths_selector import segmentation_paths
from data_loader_segmentation_net import SegmentationDataLoader
from segmentation_network_models.segmentation_network_selector import SegmentationNetworkFactory
from utils.utils import create_timestamp, load_config_json, use_gpu_if_available, setup_logger


class TrainSegmentation:
    def __init__(self):
        timestamp = create_timestamp()
        colorama.init()
        setup_logger()

        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("segmentation_net").get("schema"),
                json_filename=json_config_selector("segmentation_net").get("config")
            )
        )

        if self.cfg.get("seed"):
            seed_number = 1234
            torch.manual_seed(seed_number)
            torch.cuda.manual_seed(seed_number)

        self.device = (
            use_gpu_if_available()
        )

        network_type = self.cfg["network_type"]
        dataset_name = self.cfg["dataset_name"]
        batch_size = self.cfg["batch_size"]
        learning_rate = self.cfg["learning_rate"]

        self.model = (
            SegmentationNetworkFactory().create_model(
                network_type=network_type,
                cfg=self.cfg
            )
        ).to(self.device)
        summary(self.model)

        train_dataset, valid_dataset = (
            self.create_segmentation_dataset(
                images_dir=dataset_images_path_selector(dataset_name).get("train").get("aug_images"),
                masks_dir=dataset_images_path_selector(dataset_name).get("train").get("aug_mask_images")
            )
        )
        logging.info(
            f"{dataset_name} train dataset size: {len(train_dataset)}, validation dataset size: {len(valid_dataset)}"
        )
        self.train_loader = (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
        )
        self.valid_loader = (
            DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        )

        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        )

        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
        )

        # LR scheduler
        self.scheduler = (
            StepLR(
                optimizer=self.optimizer,
                step_size=self.cfg.get("step_size"),
                gamma=self.cfg.get("gamma")
            )
        )

        seg_net_path = segmentation_paths(network_type, dataset_name)

        # Tensorboard
        tensorboard_log_dir = os.path.join(seg_net_path['logs_folder'], timestamp)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

        # Create save directory
        self.save_path = os.path.join(seg_net_path["weights_folder"], timestamp)
        os.makedirs(self.save_path, exist_ok=True)

    def create_segmentation_dataset(self, images_dir: str, masks_dir: str):
        """
        Creates a DataLoader for a segmentation task using a custom dataset.

        Args:
            images_dir (str): Directory containing input images.
            masks_dir (str): Directory containing corresponding ground truth masks.

        Returns:
            Dataset: PyTorch DataLoader that provides batches of images and masks.
        """

        transform = transforms.Compose([
            transforms.Resize((self.cfg.get("resized_img_size"), self.cfg.get("resized_img_size"))),
            transforms.ToTensor(),
        ])

        dataset = (
            SegmentationDataLoader(
                images_dir=images_dir,
                masks_dir=masks_dir,
                transform=transform
            )
        )

        train_size = int(self.cfg.get("train_ratio") * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        return train_set, val_set

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

        for epoch in tqdm(range(self.cfg.get("epochs")), desc=colorama.Fore.LIGHTYELLOW_EX + "Epochs"):

            self.model.train()
            for batch_data, batch_masks in tqdm(
                    self.train_loader, total=len(self.train_loader), desc=colorama.Fore.LIGHTRED_EX + "Training"
            ):
                self.train_loop(batch_data, batch_masks, train_losses)

            self.model.eval()
            for batch_data, batch_masks in tqdm(
                    self.valid_loader, total=len(self.valid_loader), desc=colorama.Fore.LIGHTBLUE_EX + "Validation"
            ):
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

            self.scheduler.step()

        self.writer.close()
        self.writer.flush()


if __name__ == '__main__':
    try:
        tm = TrainSegmentation()
        try:
            tm.fit()
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError!')
            torch.cuda.empty_cache()
    except KeyboardInterrupt as kbe:
        logging.error("Keyboard interrupt, program has been shut down!")
