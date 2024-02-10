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
from config.config_selector import sub_stream_network_configs
from stream_network_models.stream_network_selector import NetworkFactory
from dataloader_stream_network import InitialStreamNetDataLoader
from loss_functions.triplet_loss_hard_mining import TripletLossWithHardMining
from utils.utils import (create_timestamp, measure_execution_time, print_network_config,
                         use_gpu_if_available, setup_logger)


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

        # Load model and upload it to the GPU
        self.model = NetworkFactory.create_network(self.cfg.type_of_net, network_cfg)
        self.model.to(self.device)

        # Print model configuration
        summary(
            self.model, (network_cfg.get('channels')[0], network_cfg.get("image_size"), network_cfg.get("image_size"))
        )

        # Specify loss function
        self.criterion = TripletLossWithHardMining(margin=self.cfg.margin)

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

        # Variables to save only the best weights and model
        self.best_valid_loss = float('inf')
        self.best_model_path = None

        # Hard samples paths
        if self.cfg.type_of_loss_func == "hmtl":
            self.hardest_samples_path = self.create_save_dirs(network_cfg.get('hardest_samples'))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ C R E A T E   D A T A S E T -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_dataset(self, network_cfg):
        """

        :param network_cfg:
        :return:
        """

        # Load dataset

        dataset = \
            InitialStreamNetDataLoader(
                dataset_dirs_anchor=[network_cfg.get("train").get(self.cfg.dataset_type).get("anchor")],
                dataset_dirs_pos_neg=[network_cfg.get("train").get(self.cfg.dataset_type).get("pos_neg")],
                type_of_stream=self.cfg.type_of_stream,
                image_size=network_cfg.get("image_size"),
                num_triplets=self.cfg.num_triplets,
                file_path=self.hardest_samples_path
            )

        # Calculate the number of samples for each set
        train_size = int(len(dataset) * self.cfg.train_valid_ratio)
        val_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])
        logging.info(f"Number of images in the train set: {len(train_dataset)}")
        logging.info(f"Number of images in the validation set: {len(valid_dataset)}")
        train_data_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        return dataset, train_data_loader, valid_data_loader

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
    def save_hardest_samples(self, dataset, epoch: int, hard_samples: list, dictionary_name: str) -> None:
        """
        Saves the hardest images to a file in JSON format.
        :param dataset:
        :param epoch: The epoch number.
        :param hard_samples: A list of tensors of the hardest images.
        :param dictionary_name: The directory where the output file will be saved.
        :return: None
        """

        file_name = self.timestamp + "_epoch_" + str(epoch) + ".txt"
        file_path = os.path.join(dictionary_name, file_name)

        hardest_samples = []
        for i, hard_tensor in enumerate(hard_samples):
            hardest_samples.append(hard_tensor)

        if self.cfg.num_triplets != len(hardest_samples):
            missing_num_triplets = self.cfg.num_triplets - len(hardest_samples)
            random_triplets = dataset.generate_triplets(missing_num_triplets)

        #TODO: merge random and hardest samples
        with open(file_path, "w") as fp:
            json.dump(hardest_samples, fp)

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------- R E C O R D   H A R D   S A M P L E S -------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def record_hard_samples(hard_samples: torch.Tensor, anc_img_path, pos_img_path, neg_img_path, hard_images: list) \
            -> list:
        """
        Records filenames of the hardest samples.

        :param hard_samples: A tensor containing the hardest samples.
        :param anc_img_path: A string of image file paths.
        :param pos_img_path: A string of image file paths.
        :param neg_img_path: A string of image file paths.
        :param hard_images: A list to store the filenames of the hardest samples.
        :return: The updated list of hardest sample filenames.
        """

        if hard_samples is not None:
            for anc_filename, pos_filename, neg_filename, sample in zip(
                    anc_img_path, pos_img_path, neg_img_path, hard_samples
            ):
                if sample is not None:
                    hard_images.append((anc_img_path, pos_img_path, neg_img_path))

        return hard_images

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- T R A I N   L O O P ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train_loop(self, train_data_loader, train_losses, hard_samples):
        for idx, (anchor, positive, negative, anchor_img_path, positive_img_path, negative_img_path) in \
                tqdm(enumerate(train_data_loader), total=len(train_data_loader),
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
            loss, hard_sample = self.criterion(anchor_emb, positive_emb, negative_emb)

            hard_samples = self.record_hard_samples(
                hard_sample, anchor_img_path, positive_img_path, negative_img_path, hard_samples
            )

            # Backward pass, optimize and scheduler
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            train_losses.append(loss.item())

        return train_losses, hard_samples

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- V A L I D   L O O P ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def valid_loop(self, valid_data_loader, valid_losses):
        """

        :param valid_data_loader:
        :param valid_losses:
        :return:
        """

        with torch.no_grad():
            for idx, (anchor, positive, negative, anchor_img_path, positive_img_path, negative_img_path) \
                    in tqdm(enumerate(valid_data_loader), total=len(valid_data_loader),
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
                val_loss, _ = self.criterion(anchor_emb, positive_emb, negative_emb)

                # Accumulate loss
                valid_losses.append(val_loss.item())

        return valid_losses

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- S A V E   M O D E L ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_model_weights(self, epoch, valid_loss):
        """
        Saves the model and weights

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
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def training(self) -> None:
        """
        This function is responsible for the training of the network.

        :return: None
        """

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to mine the hard negative samples
        hard_images = []

        network_config = sub_stream_network_configs(self.cfg)
        network_cfg = network_config.get(self.cfg.type_of_stream)

        dataset, train_data_loader, valid_data_loader = self.create_dataset(network_cfg)
        for epoch in tqdm(range(self.cfg.epochs), desc=colorama.Fore.GREEN + "Epochs"):
            # Train loop
            train_losses, hard_images = self.train_loop(train_data_loader, train_losses, hard_images)

            # Validation loop
            valid_losses = self.valid_loop(valid_data_loader, valid_losses)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            # Log loss for epoch
            logging.info(f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')

            # Loop over the hard negative tensors
            self.save_hardest_samples(dataset, epoch, hard_images, self.hardest_samples_path)
            hard_images.clear()

            # Clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()

            # Save model
            self.save_model_weights(epoch, valid_loss)

            self.scheduler.step()

        # Close and flush SummaryWriter
        self.writer.close()
        self.writer.flush()


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        tm = TrainModel()
        try:
            tm.training()
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError!')
            torch.cuda.empty_cache()
    except KeyboardInterrupt as kbe:
        logging.error("Keyboard interrupt, program has been shut down!")
