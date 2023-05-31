"""
File: train_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description:
"""

import json
import logging
import numpy as np
import os
import torch

from typing import Dict
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config.config import ConfigStreamNetwork
from config.const import CONST
from network_selector import NetworkFactory
from stream_network_dataset_loader import StreamDataset
from triplet_loss import TripletLossWithHardMining
from config.logger_setup import setup_logger
from utils.utils import create_timestamp, measure_execution_time, print_network_config, use_gpu_if_available


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   M O D E L +++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainModel:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Set up logger
        self.logger = setup_logger()

        # Set up configuration
        self.cfg = ConfigStreamNetwork().parse()

        # Print network configurations
        print_network_config(self.cfg)

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possibly
        self.device = use_gpu_if_available()

        # Setup network config
        if self.cfg.type_of_stream not in ["RGB", "Texture", "Contour", "LBP"]:
            raise ValueError("Wrong type was given!")
        network_config = self.subnetwork_configs()
        network_cfg = network_config.get(self.cfg.type_of_stream)

        # Load dataset
        self.dataset = StreamDataset(network_cfg.get('dataset_dir'), self.cfg.type_of_stream)
        train_size = int(self.cfg.train_rate * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        logging.info(f"Number of images in the train set: {train_size}")
        logging.info(f"Number of images in the validation set: {valid_size}")
        train_dataset, valid_dataset = random_split(self.dataset, [train_size, valid_size])
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # Load model and upload it to the GPU
        self.model = NetworkFactory.create_network(self.cfg.type_of_net, network_cfg)
        self.model.to(self.device)

        # Print model configuration
        summary(self.model, (network_cfg.get('channels')[0], self.cfg.img_size, self.cfg.img_size))

        # Specify loss function
        self.criterion = TripletLossWithHardMining(margin=self.cfg.margin)

        # Specify optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=network_cfg.get("learning_rate"),
                                          weight_decay=self.cfg.weight_decay)

        # Tensorboard
        tensorboard_log_dir = os.path.join(network_cfg.get('logs_dir'), self.timestamp)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Create save directory
        self.save_path = os.path.join(network_cfg.get('model_weights_dir'), self.timestamp)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------- S U B N E T W O R K   C O N F I G S  --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def subnetwork_configs(self) -> Dict:
        """
        Returns the dictionary containing the configuration details for the three different subnetworks
        (RGB, Texture, and Contour) used in the TripletLossModel.

        :return: A dictionary containing the configuration details for the three subnetworks.
        """

        network_config = {
            "RGB": {
                "channels": [3, 64, 96, 128, 256, 384, 512],
                "dataset_dir": CONST.dir_rgb,
                "model_weights_dir": {
                    "StreamNetwork": CONST.dir_stream_rgb_model_weights,
                    "EfficientNet": CONST.dir_efficient_net_rgb_model_weights,
                    "EfficientNetSelfAttention": CONST.dir_efficient_net_self_attention_rgb_model_weights
                }.get(self.cfg.type_of_net, CONST.dir_stream_contour_model_weights),
                "logs_dir": {
                    "StreamNetwork": CONST.dir_logs_stream_net_rgb,
                    "EfficientNet": CONST.dir_logs_efficient_net_rgb,
                    "EfficientNetSelfAttention": CONST.dir_logs_efficient_net_self_attention_rgb
                }.get(self.cfg.type_of_net, CONST.dir_logs_stream_net_contour),
                "learning_rate": {
                    "StreamNetwork": self.cfg.learning_rate_cnn_rgb,
                    "EfficientNet": self.cfg.learning_rate_en_rgb,
                    "EfficientNetSelfAttention": self.cfg.learning_rate_ensa_rgb
                }.get(self.cfg.type_of_net, self.cfg.learning_rate_cnn_rgb),
                "grayscale": False
            },

            "Texture": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "dataset_dir": CONST.dir_texture,
                "model_weights_dir": {
                    "StreamNetwork": CONST.dir_stream_texture_model_weights,
                    "EfficientNet": CONST.dir_efficient_net_texture_model_weights,
                    "EfficientNetSelfAttention": CONST.dir_efficient_net_self_attention_texture_model_weights
                }.get(self.cfg.type_of_net, CONST.dir_stream_contour_model_weights),
                "logs_dir": {
                    "StreamNetwork": CONST.dir_logs_stream_net_texture,
                    "EfficientNet": CONST.dir_logs_efficient_net_texture,
                    "EfficientNetSelfAttention": CONST.dir_logs_efficient_net_self_attention_texture
                }.get(self.cfg.type_of_net, CONST.dir_logs_stream_net_contour),
                "learning_rate": {
                    "StreamNetwork": self.cfg.learning_rate_cnn_con_tex,
                    "EfficientNet": self.cfg.learning_rate_en_con_tex,
                    "EfficientNetSelfAttention": self.cfg.learning_rate_ensa_con_tex
                }.get(self.cfg.type_of_net, self.cfg.learning_rate_cnn_con_tex),
                "grayscale": True
            },

            "Contour": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "dataset_dir": CONST.dir_contour,
                "model_weights_dir": {
                    "StreamNetwork": CONST.dir_stream_contour_model_weights,
                    "EfficientNet": CONST.dir_efficient_net_contour_model_weights,
                    "EfficientNetSelfAttention": CONST.dir_efficient_net_self_attention_contour_model_weights
                }.get(self.cfg.type_of_net, CONST.dir_stream_contour_model_weights),
                "logs_dir": {
                    "StreamNetwork": CONST.dir_logs_stream_net_contour,
                    "EfficientNet": CONST.dir_logs_efficient_net_contour,
                    "EfficientNetSelfAttention": CONST.dir_logs_efficient_net_self_attention_contour
                }.get(self.cfg.type_of_net, CONST.dir_logs_stream_net_contour),
                "learning_rate": {
                    "StreamNetwork": self.cfg.learning_rate_cnn_con_tex,
                    "EfficientNet": self.cfg.learning_rate_en_con_tex,
                    "EfficientNetSelfAttention": self.cfg.learning_rate_ensa_con_tex
                }.get(self.cfg.type_of_net, self.cfg.learning_rate_cnn_con_tex),
                "grayscale": True
            },

            "LBP": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "dataset_dir": CONST.dir_lbp,
                "model_weights_dir": {
                    "StreamNetwork": CONST.dir_stream_lbp_model_weights,
                    "EfficientNet": CONST.dir_efficient_net_lbp_model_weights,
                    "EfficientNetSelfAttention": CONST.dir_efficient_net_self_attention_lbp_model_weights
                }.get(self.cfg.type_of_net, CONST.dir_stream_lbp_model_weights),
                "logs_dir": {
                    "StreamNetwork": CONST.dir_logs_stream_net_lbp,
                    "EfficientNet": CONST.dir_logs_efficient_net_lbp,
                    "EfficientNetSelfAttention": CONST.dir_logs_efficient_net_self_attention_lbp
                }.get(self.cfg.type_of_net, CONST.dir_logs_stream_net_lbp),
                "learning_rate": {
                    "StreamNetwork": self.cfg.learning_rate_cnn_con_tex,
                    "EfficientNet": self.cfg.learning_rate_en_con_tex,
                    "EfficientNetSelfAttention": self.cfg.learning_rate_ensa_con_tex
                }.get(self.cfg.type_of_net, self.cfg.learning_rate_cnn_con_tex),
                "grayscale": True
            }
        }

        return network_config

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- G E T   H A R D  S A M P L E S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_hardest_samples(self, epoch: int, hard_neg_images: list, output_dir: str, op: str) -> None:
        """
        Saves the hardest negative images to a file in JSON format.
        :param epoch: The epoch number.
        :param hard_neg_images: A list of tensors of the hardest negative images.
        :param output_dir: The directory where the output file will be saved.
        :param op: A string representing the type of operation.
        :return: None
        """

        dict_name = os.path.join(output_dir, self.timestamp, self.cfg.type_of_stream)
        os.makedirs(dict_name, exist_ok=True)
        file_name = os.path.join(dict_name, self.timestamp + "_epoch_" + str(epoch) + "_%s.txt" % op)

        hardest_samples = []
        for i, hard_neg_tensor in enumerate(hard_neg_images):
            hardest_samples.append(hard_neg_tensor)

        with open(file_name, "w") as fp:
            json.dump(hardest_samples, fp)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------- R E C O R D   H A R D   S A M P L E S -------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def record_hard_samples(hard_samples: torch.Tensor, img_path: tuple, hard_images: list) -> list:
        """
        Records filenames of the hardest negative samples.

        :param hard_samples: A tensor containing the hardest negative samples.
        :param img_path: A tuple of image file paths.
        :param hard_images: A list to store the filenames of the hardest negative samples.
        :return: The updated list of hardest negative sample filenames.
        """

        if hard_samples is not None:
            for filename, sample in zip(img_path, hard_samples):
                if sample is not None:
                    hard_images.append(filename)

        return hard_images

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def fit(self) -> None:
        """
        This function is responsible for the training of the network.

        :return: None
        """

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to mine the hard negative samples
        hard_neg_images = []
        # to mine the hard positive samples
        hard_pos_images = []

        # Variables to save only the best weights and model
        best_valid_loss = float('inf')
        best_model_path = None

        for epoch in tqdm(range(self.cfg.epochs), desc="Epochs"):
            # Train loop
            for idx, (anchor, positive, negative, negative_img_path, positive_img_path) in \
                    tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader), desc="Train"):
                # Upload data to the GPU
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # Set the gradients to zero
                self.optimizer.zero_grad()

                # Forward pass
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)

                # Compute triplet loss
                loss, hard_neg, hard_pos = self.criterion(anchor_emb, positive_emb, negative_emb)

                # Backward pass, optimize and scheduler
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                train_losses.append(loss.item())

                # Collect hardest positive and negative samples
                hard_neg_images = self.record_hard_samples(hard_neg, negative_img_path, hard_neg_images)
                hard_pos_images = self.record_hard_samples(hard_pos, positive_img_path, hard_pos_images)

            # Validation loop
            with torch.no_grad():
                for idx, (anchor, positive, negative, _, _) in tqdm(enumerate(self.valid_data_loader),
                                                                    total=len(self.valid_data_loader),
                                                                    desc="Validation"):
                    # Upload data to GPU
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                    # Forward pass
                    anchor_emb = self.model(anchor)
                    positive_emb = self.model(positive)
                    negative_emb = self.model(negative)

                    # Compute triplet loss
                    val_loss, _, _ = self.criterion(anchor_emb, positive_emb, negative_emb)

                    # Accumulate loss
                    valid_losses.append(val_loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            # Log loss for epoch
            logging.info(f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')

            # Loop over the hard negative tensors
            self.get_hardest_samples(epoch, hard_neg_images, CONST.dir_hardest_neg_samples, "negative")
            self.get_hardest_samples(epoch, hard_pos_images, CONST.dir_hardest_pos_samples, "positive")

            # Clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()
            hard_neg_images.clear()
            hard_pos_images.clear()

            # Save the model and weights
            if valid_loss < best_valid_loss:
                # Save the best model
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(self.save_path, "epoch_" + str(epoch) + ".pt")
                torch.save(self.model.state_dict(), best_model_path)

        # Close and flush SummaryWriter
        self.writer.close()
        self.writer.flush()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        tm = TrainModel()
        tm.fit()
    except KeyboardInterrupt as kbe:
        logging.error("Keyboard interrupt, program has been shut down!")
