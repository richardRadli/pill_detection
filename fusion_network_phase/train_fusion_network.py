"""
File: predict_fusion_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This program implements the training for fusion networks.
"""

import colorama
import logging
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from typing import Any, List

from config.json_config import json_config_selector
from config.networks_paths_selector import fusion_network_paths
from dataloader_fusion_network import FusionDataset
from fusion_network_models.fusion_network_selector import FusionNetworkFactory
from utils.utils import (create_timestamp, use_gpu_if_available, setup_logger, create_dataset,
                         measure_execution_time, load_config_json)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   M O D E L +++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainFusionNet:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.timestamp = create_timestamp()
        colorama.init()
        setup_logger()

        # Set up configuration
        self.cfg_fusion_net = (
            load_config_json(
                json_schema_filename=json_config_selector("fusion_net").get("schema"),
                json_filename=json_config_selector("fusion_net").get("config")
            )
        )
        self.cfg_stream_net = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )

        dataset_type = self.cfg_fusion_net.get("dataset_type")
        type_of_net = self.cfg_fusion_net.get("type_of_net")

        # Select the GPU if possible
        self.device = use_gpu_if_available()

        # Load datasets using FusionDataset
        dataset = FusionDataset()
        self.train_data_loader, self.valid_data_loader = (
            create_dataset(dataset=dataset,
                           train_valid_ratio=self.cfg_fusion_net.get("train_valid_ratio"),
                           batch_size=self.cfg_fusion_net.get("batch_size"))
        )

        # Set up model
        self.model = (
            self.setup_model()
        )

        # Set up loss function
        self.criterion = (
            torch.nn.TripletMarginLoss(
                margin=self.cfg_fusion_net.get("margin")
            )
        )

        # Specify optimizer
        self.optimizer = (
            torch.optim.Adam(
                params=list(self.model.fc1.parameters()) + list(self.model.fc2.parameters()),
                lr=self.cfg_fusion_net.get("learning_rate"),
                weight_decay=self.cfg_fusion_net.get("weight_decay")
            )
        )

        # LR scheduler
        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.cfg_fusion_net.get("step_size"),
                                gamma=self.cfg_fusion_net.get("gamma"))

        # Tensorboard
        tensorboard_log_dir = (
            self.create_save_dirs(
                fusion_network_paths(dataset_type=dataset_type, network_type=type_of_net).get("logs_folder")
            )
        )
        self.writer = (
            SummaryWriter(
                log_dir=tensorboard_log_dir
            )
        )

        # Create save path
        self.save_path = (
            self.create_save_dirs(
                fusion_network_paths(dataset_type=dataset_type, network_type=type_of_net).get("weights_folder")
            )
        )

        # Variables to save only the best epoch
        self.best_valid_loss = float('inf')
        self.best_model_path = None

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- S E T U P   M O D E L ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_model(self) -> Any:
        """
        Initialize and set up the fusion network model.

        Returns:
            The initialized fusion network model.
        """

        # Initialize the fusion network
        model = (
            FusionNetworkFactory.create_network(fusion_network_type=self.cfg_fusion_net.get("type_of_net"))
        )

        img_size = 128 if self.cfg_fusion_net.get("type_of_net") == "CNNFusionNet" else 224

        # Load model and upload it to the GPU
        model.to(self.device)

        summary(
            model=model,
            input_size=[
                    (1, img_size, img_size),
                    (1, img_size, img_size),
                    (3, img_size, img_size),
                    (1, img_size, img_size)
            ]
        )

        return model

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   S A V E   D I R S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_save_dirs(self, directory_path: str) -> str:
        """
        Create and return directory path for saving files.

        Args:
            directory_path (str): Main network configuration.

        Returns:
            str: Directory path created.
        """

        directory_to_create = (
            os.path.join(directory_path,
                         f"{self.timestamp}_{self.cfg_fusion_net.type_of_loss_func}_{self.cfg_fusion_net.fold}")
        )
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- T R A I N   L O O P ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train_loop(self, train_losses: List[float]) -> List[float]:
        """
        Train loop for the model.

        Args:
            train_losses (List[float]): List to track training losses.

        Returns:
            List[float]: Updated training losses.
        """

        for (contour_anchor, contour_positive, contour_negative,
             lbp_anchor, lbp_positive, lbp_negative,
             rgb_anchor, rgb_positive, rgb_negative,
             texture_anchor, texture_positive, texture_negative,
             anchor_img_path, negative_img_path) \
                in tqdm(self.train_data_loader,
                        total=len(self.train_data_loader),
                        desc=colorama.Fore.LIGHTCYAN_EX + "Training"):
            # Set the gradients to zero
            self.optimizer.zero_grad()

            # Upload data to the GPU
            contour_anchor = contour_anchor.to(self.device)
            contour_positive = contour_positive.to(self.device)
            contour_negative = contour_negative.to(self.device)
            lbp_anchor = lbp_anchor.to(self.device)
            lbp_positive = lbp_positive.to(self.device)
            lbp_negative = lbp_negative.to(self.device)
            rgb_anchor = rgb_anchor.to(self.device)
            rgb_positive = rgb_positive.to(self.device)
            rgb_negative = rgb_negative.to(self.device)
            texture_anchor = texture_anchor.to(self.device)
            texture_positive = texture_positive.to(self.device)
            texture_negative = texture_negative.to(self.device)

            anchor_embedding = self.model(contour_anchor, lbp_anchor, rgb_anchor, texture_anchor)
            positive_embedding = self.model(contour_positive, lbp_positive, rgb_positive, texture_positive)
            negative_embedding = self.model(contour_negative, lbp_negative, rgb_negative, texture_negative)

            # Compute triplet loss
            train_loss = self.criterion(anchor=anchor_embedding,
                                        positive=positive_embedding,
                                        negative=negative_embedding)

            # Backward pass
            train_loss.backward()
            self.optimizer.step()

            # Accumulate loss
            train_losses.append(train_loss.item())

        return train_losses

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- V A L I D   L O O P ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def valid_loop(self, valid_losses):
        """
        Validation loop for the model.

        Args:
            valid_losses (List[float]): List to track validation losses.

        Returns:
            List[float]: Updated validation losses.
        """

        # Validation loop
        with (torch.no_grad()):
            for (contour_anchor, contour_positive, contour_negative,
                 lbp_anchor, lbp_positive, lbp_negative,
                 rgb_anchor, rgb_positive, rgb_negative,
                 texture_anchor, texture_positive, texture_negative,
                 anchor_img_path, negative_img_path) in tqdm(
                self.valid_data_loader,
                total=len(self.valid_data_loader),
                desc=colorama.Fore.LIGHTWHITE_EX + "Validation"
            ):
                # Upload data to the GPU
                contour_anchor = contour_anchor.to(self.device)
                contour_positive = contour_positive.to(self.device)
                contour_negative = contour_negative.to(self.device)
                lbp_anchor = lbp_anchor.to(self.device)
                lbp_positive = lbp_positive.to(self.device)
                lbp_negative = lbp_negative.to(self.device)
                rgb_anchor = rgb_anchor.to(self.device)
                rgb_positive = rgb_positive.to(self.device)
                rgb_negative = rgb_negative.to(self.device)
                texture_anchor = texture_anchor.to(self.device)
                texture_positive = texture_positive.to(self.device)
                texture_negative = texture_negative.to(self.device)

                # Forward pass
                anchor_embedding = self.model(contour_anchor, lbp_anchor, rgb_anchor, texture_anchor)
                positive_embedding = self.model(contour_positive, lbp_positive, rgb_positive, texture_positive)
                negative_embedding = self.model(contour_negative, lbp_negative, rgb_negative, texture_negative)

                # Compute triplet loss
                valid_loss = self.criterion(anchor=anchor_embedding,
                                            positive=positive_embedding,
                                            negative=negative_embedding)

                # Accumulate loss
                valid_losses.append(valid_loss.item())

            return valid_losses

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- S A V E   M O D E L   W E I G H T S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_model_weights(self, epoch: int, valid_loss: float) -> None:
        """
        Saves the model and weights if the validation loss is improved.

        Args:
            epoch (int): Current epoch number.
            valid_loss (float): Validation loss.

        Returns:
            None
        """

        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            self.best_model_path = os.path.join(self.save_path, "epoch_" + str(epoch) + ".pt")
            torch.save(self.model.state_dict(), self.best_model_path)
            logging.info(f"New weights have been saved at epoch {epoch} with value of {valid_loss:.5f}")
        else:
            logging.warning(f"No new weights have been saved. Best valid loss was {self.best_valid_loss:.5f},\n "
                            f"current valid loss is {valid_loss:.5f}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ T R A I N I N G--------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def training(self) -> None:
        """
       Train the neural network.

       Returns:
           None
       """

        # To track the training loss as the model trains
        train_losses = []

        # To track the validation loss as the model trains
        valid_losses = []

        for epoch in tqdm(range(self.cfg_fusion_net.get("epochs")), desc=colorama.Fore.LIGHTGREEN_EX + "Epochs"):
            # Train loop
            train_losses = self.train_loop(train_losses)
            valid_losses = self.valid_loop(valid_losses)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            logging.info(f'train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}')
            train_losses.clear()
            valid_losses.clear()

            # Save model
            self.save_model_weights(epoch, valid_loss)

            # Decay the learning rate
            self.scheduler.step()

        # Close and flush SummaryWriter
        self.writer.close()
        self.writer.flush()


if __name__ == "__main__":
    try:
        tm = TrainFusionNet()
        try:
            tm.training()
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError!')
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        logging.error('Keyboard interrupt has been occurred!')
