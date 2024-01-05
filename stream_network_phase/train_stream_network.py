"""
File: train_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This code implements the training for the stream network phase.
"""

import colorama
import json
import logging
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config.config import ConfigStreamNetwork
from config.network_configs import sub_stream_network_configs
from stream_network_models.stream_network_selector import NetworkFactory
from dataloader_stream_network import StreamDataset
from loss_functions.triplet_loss import TripletMarginLoss
from loss_functions.triplet_loss_hard_mining import TripletLossWithHardMining
from utils.utils import (create_timestamp, measure_execution_time, print_network_config, use_gpu_if_available,
                         setup_logger)


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

        # Set up tqdm colour
        colorama.init()

        # Print network configurations
        print_network_config(self.cfg)

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possible
        self.device = use_gpu_if_available()

        # Setup network config
        if self.cfg.type_of_stream not in ["RGB", "Texture", "Contour", "LBP"]:
            raise ValueError("Wrong type was given!")
        network_config = sub_stream_network_configs(self.cfg)
        network_cfg = network_config.get(self.cfg.type_of_stream)

        # Load dataset
        dataset = \
            StreamDataset(dataset_dirs=[network_cfg.get("train").get(self.cfg.dataset_type),
                                        network_cfg.get("valid").get(self.cfg.dataset_type)],
                          type_of_stream=self.cfg.type_of_stream,
                          image_size=network_cfg.get("image_size"),
                          num_triplets=self.cfg.num_triplets)

        # Calculate the number of samples for each set
        train_size = int(len(dataset) * self.cfg.train_valid_ratio)
        val_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])
        logging.info(f"Number of images in the train set: {len(train_dataset)}")
        logging.info(f"Number of images in the validation set: {len(valid_dataset)}")
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # Load model and upload it to the GPU
        self.model = NetworkFactory.create_network(self.cfg.type_of_net, network_cfg)
        self.model.to(self.device)

        # Print model configuration
        summary(self.model,
                (network_cfg.get('channels')[0], network_cfg.get("image_size"), network_cfg.get("image_size")))

        # Specify loss function
        if self.cfg.type_of_loss_func == "hmtl":
            self.criterion = TripletLossWithHardMining(margin=self.cfg.margin)
        elif self.cfg.type_of_loss_func == "tl":
            self.criterion = TripletMarginLoss(margin=self.cfg.margin)
        else:
            raise ValueError(f"Wrong type of loss function: {self.cfg.type_of_loss_func}")

        # Specify optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=network_cfg.get("learning_rate"),
                                          weight_decay=self.cfg.weight_decay)

        # LR scheduler
        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.cfg.step_size,
                                gamma=self.cfg.gamma)

        # Tensorboard
        tensorboard_log_dir = self.create_save_dirs(network_cfg.get('logs_dir'))
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Create save directory for model weights
        self.save_path = self.create_save_dirs(network_cfg.get('model_weights_dir'))

        # Hard samples paths
        self.hardest_negative_samples_path = self.create_save_dirs(network_cfg.get('hardest_negative_samples'))
        self.hardest_positive_samples_path = self.create_save_dirs(network_cfg.get('hardest_positive_samples'))

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   S A V E   D I R S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_save_dirs(self, network_cfg):
        """

        :param network_cfg:
        :return:
        """

        directory_path = network_cfg.get(self.cfg.type_of_net).get(self.cfg.dataset_type)
        directory_to_create = os.path.join(directory_path, f"{self.timestamp}_{self.cfg.type_of_loss_func}")
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- G E T   H A R D  S A M P L E S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_hardest_samples(self, epoch: int, hard_neg_images: list, dictionary_name: str, op: str) -> None:
        """
        Saves the hardest images to a file in JSON format.
        :param epoch: The epoch number.
        :param hard_neg_images: A list of tensors of the hardest negative images.
        :param dictionary_name: The directory where the output file will be saved.
        :param op: A string representing the type of operation.
        :return: None
        """

        file_name = os.path.join(dictionary_name, self.timestamp + "_epoch_" + str(epoch) + "_%s.txt" % op)

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
    def train(self) -> None:
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

        for epoch in tqdm(range(self.cfg.epochs), desc=colorama.Fore.GREEN + "Epochs"):
            # Train loop
            for idx, (anchor, positive, negative, negative_img_path, positive_img_path) in \
                    tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader),
                         desc=colorama.Fore.CYAN + "Train"):
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
                if self.cfg.type_of_loss_func == "hmtl":
                    loss, hard_neg, hard_pos = self.criterion(anchor_emb, positive_emb, negative_emb)
                    # Collect hardest positive and negative samples
                    hard_neg_images = self.record_hard_samples(hard_neg, negative_img_path, hard_neg_images)
                    hard_pos_images = self.record_hard_samples(hard_pos, positive_img_path, hard_pos_images)
                elif self.cfg.type_of_loss_func == "tl":
                    loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                else:
                    raise ValueError(f"Wrong type of loss function {self.cfg.type_of_loss_func}")

                # Backward pass, optimize and scheduler
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                train_losses.append(loss.item())

            # Validation loop
            with torch.no_grad():
                for idx, (anchor, positive, negative, negative_img_path, positive_img_path) \
                        in tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader),
                                desc=colorama.Fore.MAGENTA + "Validation"):
                    # Upload data to GPU
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                    # Forward pass
                    anchor_emb = self.model(anchor)
                    positive_emb = self.model(positive)
                    negative_emb = self.model(negative)

                    # Compute triplet loss
                    if self.cfg.type_of_loss_func == "hmtl":
                        val_loss, _, _ = self.criterion(anchor_emb, positive_emb, negative_emb)
                    elif self.cfg.type_of_loss_func == "tl":
                        val_loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                    else:
                        raise ValueError(f"Wrong type of loss function {self.cfg.type_of_loss_func}")

                    # Accumulate loss
                    valid_losses.append(val_loss.item())

            # Decay the learning rate
            self.scheduler.step()

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            # Log loss for epoch
            logging.info(f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')

            # Loop over the hard negative tensors
            self.get_hardest_samples(epoch, hard_neg_images, self.hardest_negative_samples_path, "negative")
            self.get_hardest_samples(epoch, hard_pos_images, self.hardest_positive_samples_path, "positive")

            # Clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()
            hard_neg_images.clear()
            hard_pos_images.clear()

            # Save the model and weights
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(self.save_path, "epoch_" + str(epoch) + ".pt")
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"New weights have been saved at epoch {epoch} with value of {valid_loss:.5f}")
            else:
                logging.warning(f"No new weights have been saved. Best valid loss was {best_valid_loss:.5f},\n "
                                f"current valid loss is {valid_loss:.5f}")

        # Close and flush SummaryWriter
        self.writer.close()
        self.writer.flush()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        tm = TrainModel()
        tm.train()
    except KeyboardInterrupt as kbe:
        logging.error("Keyboard interrupt, program has been shut down!")
