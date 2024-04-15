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

from config.config import ConfigFusionNetwork, ConfigStreamNetwork
from config.config_selector import sub_stream_network_configs, fusion_network_config
from dataloader_fusion_network import FusionDataset
from fusion_network_models.fusion_network_selector import NetworkFactory
from utils.utils import (create_timestamp, create_dataset, print_network_config, use_gpu_if_available, setup_logger,
                         measure_execution_time)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   M O D E L +++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainFusionNet:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Create time stamp
        self.timestamp = create_timestamp()

        setup_logger()

        # Set up configuration
        self.cfg_fusion_net = ConfigFusionNetwork().parse()
        self.cfg_stream_net = ConfigStreamNetwork().parse()

        # Set up tqdm colours
        colorama.init()

        # Set up networks
        network_type = self.cfg_fusion_net.type_of_net
        main_network_config = fusion_network_config(network_type=network_type)
        subnetwork_config = sub_stream_network_configs(self.cfg_stream_net)
        network_cfg_contour = subnetwork_config.get("Contour")
        network_cfg_lbp = subnetwork_config.get("LBP")
        network_cfg_rgb = subnetwork_config.get("RGB")
        network_cfg_texture = subnetwork_config.get("Texture")

        # Print network configurations
        print_network_config(self.cfg_fusion_net)

        # Select the GPU if possible
        self.device = use_gpu_if_available()

        # Load datasets using FusionDataset
        dataset = FusionDataset()
        self.train_data_loader, self.valid_data_loader = (
            create_dataset(dataset=dataset,
                           train_valid_ratio=self.cfg_fusion_net.train_valid_ratio,
                           batch_size=self.cfg_fusion_net.batch_size)
        )

        # Set up model
        self.model = self.setup_model(network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture)

        # Specify loss function
        self.criterion = torch.nn.TripletMarginLoss(margin=self.cfg_fusion_net.margin)

        # Specify optimizer
        self.optimizer = torch.optim.Adam(
            params=list(self.model.fc1.parameters()) + list(self.model.fc2.parameters()),
            lr=self.cfg_fusion_net.learning_rate,
            weight_decay=self.cfg_fusion_net.weight_decay
        )

        # LR scheduler
        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.cfg_fusion_net.step_size,
                                gamma=self.cfg_fusion_net.gamma)

        # Tensorboard
        tensorboard_log_dir = self.create_save_dirs(main_network_config, "logs_folder")
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Create save path
        self.save_path = self.create_save_dirs(main_network_config, "weights_folder")

        # Variables to save only the best epoch
        self.best_valid_loss = float('inf')
        self.best_model_path = None

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- S E T U P   M O D E L ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_model(self, network_cfg_contour: dict, network_cfg_lbp: dict, network_cfg_rgb: dict,
                    network_cfg_texture: dict) -> Any:
        """
        Initialize and set up the fusion network model.

        Args:
            network_cfg_contour: Configuration for the contour stream network.
            network_cfg_lbp: Configuration for the LBP stream network.
            network_cfg_rgb: Configuration for the RGB stream network.
            network_cfg_texture: Configuration for the texture stream network.

        Returns:
            The initialized fusion network model.
        """

        # Initialize the fusion network
        model = NetworkFactory.create_network(fusion_network_type=self.cfg_fusion_net.type_of_net,
                                              type_of_net=self.cfg_stream_net.type_of_net,
                                              network_cfg_contour=network_cfg_contour,
                                              network_cfg_lbp=network_cfg_lbp,
                                              network_cfg_rgb=network_cfg_rgb,
                                              network_cfg_texture=network_cfg_texture)

        # Freeze the weights of the stream networks
        for param in model.contour_network.parameters():
            param.requires_grad = False
        for param in model.lbp_network.parameters():
            param.requires_grad = False
        for param in model.rgb_network.parameters():
            param.requires_grad = False
        for param in model.texture_network.parameters():
            param.requires_grad = False

        # Load model and upload it to the GPU
        model.to(self.device)

        # Display model
        summary(model=model,
                input_size=[(network_cfg_contour.get("channels")[0], network_cfg_contour.get("image_size"),
                             network_cfg_contour.get("image_size")),
                            (network_cfg_lbp.get("channels")[0], network_cfg_lbp.get("image_size"),
                             network_cfg_lbp.get("image_size")),
                            (network_cfg_rgb.get("channels")[0], network_cfg_rgb.get("image_size"),
                             network_cfg_rgb.get("image_size")),
                            (network_cfg_texture.get("channels")[0], network_cfg_texture.get("image_size"),
                             network_cfg_texture.get("image_size"))]
                )

        return model

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   S A V E   D I R S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_save_dirs(self, main_network_config: dict, directory_type: str) -> str:
        """
        Create and return directory path for saving files.

        Args:
            main_network_config (dict): Main network configuration.
            directory_type (str): Type of directory to create.

        Returns:
            str: Directory path created.
        """

        directory_path = main_network_config.get(directory_type).get(self.cfg_stream_net.dataset_type)
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
            if self.cfg_fusion_net.type_of_loss_func == "hmtl":
                train_loss = self.criterion(anchor=anchor_embedding,
                                            positive=positive_embedding,
                                            negative=negative_embedding)
            else:
                raise ValueError(f"Wrong loss function: {self.cfg_fusion_net}")

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
                if self.cfg_fusion_net.type_of_loss_func == "hmtl":
                    valid_loss = self.criterion(anchor=anchor_embedding,
                                                positive=positive_embedding,
                                                negative=negative_embedding)
                else:
                    raise ValueError(f"Wrong loss function: {self.cfg_fusion_net}")

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

        for epoch in tqdm(range(self.cfg_fusion_net.epochs), desc=colorama.Fore.LIGHTGREEN_EX + "Epochs"):
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
