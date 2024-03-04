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
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config.config import ConfigFusionNetwork, ConfigStreamNetwork
from config.config_selector import sub_stream_network_configs, fusion_network_config, nlp_configs
from dataloader_fusion_network import FusionDataset
from loss_functions.dynamic_margin_triplet_loss_fusion import DynamicMarginTripletLoss
from fusion_network_models.fusion_network_selector import NetworkFactory
from utils.utils import (create_timestamp, print_network_config, use_gpu_if_available, setup_logger,
                         get_embedded_text_matrix)


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
        self.train_data_loader, self.valid_data_loader = self.create_dataset()

        # Set up model
        self.model = self.setup_model(network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture)

        # Specify loss function
        path_to_excel_file = nlp_configs().get("vector_distances")
        if self.cfg_fusion_net.type_of_loss_func == "dmtl":
            df = get_embedded_text_matrix(path_to_excel_file)
            self.criterion = DynamicMarginTripletLoss(
                euc_dist_mtx=df,
                upper_norm_limit=self.cfg_fusion_net.upper_norm_limit,
                margin=self.cfg_fusion_net.margin
            )
        elif self.cfg_fusion_net.type_of_loss_func == "hmtl":
            self.criterion = torch.nn.TripletMarginLoss(margin=self.cfg_fusion_net.margin)
        else:
            raise ValueError(f"Wrong loss function: {self.cfg_fusion_net}")

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

        # Create save path for weights
        self.save_path = self.create_save_dirs(main_network_config, "weights_folder")

        # Variables to save only the best epoch
        self.best_valid_loss = float('inf')
        self.best_model_path = None

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- S E T U P   M O D E L ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_model(self, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture):
        """

        :param network_cfg_contour:
        :param network_cfg_lbp:
        :param network_cfg_rgb:
        :param network_cfg_texture:
        :return:
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
                input_size=[(1, network_cfg_contour.get("image_size"),
                             network_cfg_contour.get("image_size")),
                            (1, network_cfg_lbp.get("image_size"),
                             network_cfg_lbp.get("image_size")),
                            (3, network_cfg_rgb.get("image_size"),
                             network_cfg_rgb.get("image_size")),
                            (1, network_cfg_texture.get("image_size"),
                             network_cfg_texture.get("image_size"))]
                )

        return model

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   S A V E   D I R S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_save_dirs(self, main_network_config, directory_type):
        """

        :param main_network_config:
        :param directory_type:
        :return:
        """

        directory_path = main_network_config.get(directory_type).get(self.cfg_stream_net.dataset_type)
        directory_to_create = os.path.join(directory_path, f"{self.timestamp}_{self.cfg_fusion_net.type_of_loss_func}")
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- C R E A T E   D A T A S E T ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_dataset(self):
        """
        :return:
        """

        # Load dataset
        dataset = FusionDataset()

        # Calculate the number of samples for each set
        train_size = int(len(dataset) * self.cfg_fusion_net.train_valid_ratio)
        val_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])
        logging.info(f"Number of images in the train set: {len(train_dataset)}")
        logging.info(f"Number of images in the validation set: {len(valid_dataset)}")
        train_data_loader = DataLoader(train_dataset, batch_size=self.cfg_fusion_net.batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=self.cfg_fusion_net.batch_size, shuffle=True)

        return train_data_loader, valid_data_loader

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- T R A I N   L O O P ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train_loop(self, train_losses):
        """

        :param train_losses:
        :return:
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

            # Forward pass
            anchor_embedding = self.model(contour_anchor, lbp_anchor, rgb_anchor, texture_anchor)
            positive_embedding = self.model(contour_positive, lbp_positive, rgb_positive, texture_positive)
            negative_embedding = self.model(contour_negative, lbp_negative, rgb_negative, texture_negative)

            # Compute triplet loss
            if self.cfg_fusion_net.type_of_loss_func == "hmtl":
                train_loss = self.criterion(anchor=anchor_embedding,
                                            positive=positive_embedding,
                                            negative=negative_embedding)
            elif self.cfg_fusion_net.type_of_loss_func == "dmtl":
                train_loss = self.criterion(anchor_tensor=anchor_embedding,
                                            positive_tensor=positive_embedding,
                                            negative_tensor=negative_embedding,
                                            anchor_file_names=anchor_img_path, 
                                            negative_file_names=negative_img_path)
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

        :param valid_losses:
        :return:
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
                elif self.cfg_fusion_net.type_of_loss_func == "dmtl":
                    valid_loss = self.criterion(anchor_tensor=anchor_embedding,
                                                positive_tensor=positive_embedding,
                                                negative_tensor=negative_embedding,
                                                anchor_file_names=anchor_img_path,
                                                negative_file_names=negative_img_path)
                else:
                    raise ValueError(f"Wrong loss function: {self.cfg_fusion_net}")

                # Accumulate loss
                valid_losses.append(valid_loss.item())

            return valid_losses

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- S A V E   M O D E L   W E I G H T S --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_model_weights(self, epoch, valid_loss):
        """

        :param epoch:
        :param valid_loss:
        :return:
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
    def training(self):
        """

        :return:
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
