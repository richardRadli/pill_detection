"""
File: train_stream_network.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This code implements the training for the stream network phase.
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
from pytorch_metric_learning import losses, miners

from config.config import ConfigStreamNetwork
from config.config_selector import sub_stream_network_configs
from stream_network_models.stream_network_selector import NetworkFactory
from dataloader_stream_network import DataLoaderStreamNet
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
            model=self.model,
            input_size=(network_cfg.get('channels')[0], network_cfg.get("image_size"), network_cfg.get("image_size"))
        )

        self.criterion = losses.TripletMarginLoss(margin=self.cfg.margin)
        self.mining_func = miners.TripletMarginMiner(
            margin=self.cfg.margin, type_of_triplets=self.cfg.mining_type
        )

        # Specify optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=network_cfg.get("learning_rate"))

        # LR scheduler
        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.cfg.step_size,
                                gamma=self.cfg.gamma)

        # Tensorboard
        tensorboard_log_dir = self.create_save_dirs(network_cfg.get('logs_dir'))
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Create save directory for model weights
        self.save_path = self.create_save_dirs(network_cfg.get('model_weights_dir'))

        # Create save directory for hard samples
        self.hard_samples_path = self.create_save_dirs(network_cfg.get('hardest_samples'))

        # Variables to save only the best weights and model
        self.best_valid_loss = float('inf')
        self.best_model_path = None

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
            DataLoaderStreamNet(
                dataset_dirs_anchor=[network_cfg.get("train").get(self.cfg.dataset_type).get("anchor")],
                dataset_dirs_pos_neg=[network_cfg.get("train").get(self.cfg.dataset_type).get("pos_neg")]
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
    # ------------------------------------------ C O N V E R T   L A B E L S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def convert_labels(self, labels):
        """

        :param labels:
        :return:
        """

        labels = tuple(int(item) for item in labels)
        labels = torch.tensor(labels)
        return labels.to(self.device)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- T R A I N   L O O P ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train_loop(self, train_data_loader, train_losses, hard_samples):
        for idx, (consumer_images, consumer_labels, cp, reference_images, reference_labels, rp) in \
                tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=colorama.Fore.CYAN + "Train"):
            # Set the gradients to zero
            self.optimizer.zero_grad()

            # Upload data to the GPU
            consumer_images = consumer_images.to(self.device)
            reference_images = reference_images.to(self.device)

            consumer_labels = self.convert_labels(consumer_labels)
            reference_labels = self.convert_labels(reference_labels)

            # Forward pass
            consumer_embeddings = self.model(consumer_images)
            reference_embeddings = self.model(reference_images)

            # Get hard samples
            indices_tuple = self.mining_func(embeddings=consumer_embeddings,
                                             labels=consumer_labels,
                                             ref_emb=reference_embeddings,
                                             ref_labels=reference_labels)

            anchor_indices, positive_indices, negative_indices = indices_tuple

            anchor_filenames = [cp[i] for i in anchor_indices]
            positive_filenames = [rp[i] for i in positive_indices]
            negative_filenames = [rp[i] for i in negative_indices]

            hard_sample = [anchor_filenames, positive_filenames, negative_filenames]
            hard_samples.append(hard_sample)

            # Compute loss
            train_loss = self.criterion(embeddings=consumer_embeddings,
                                        labels=consumer_labels,
                                        ref_emb=reference_embeddings,
                                        ref_labels=reference_labels)

            # Backward pass, optimize
            train_loss.backward()
            self.optimizer.step()

            # Accumulate loss
            train_losses.append(train_loss.item())

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
            for idx, (consumer_images, consumer_labels, _, reference_images, reference_labels, _) \
                    in tqdm(enumerate(valid_data_loader), total=len(valid_data_loader),
                            desc=colorama.Fore.MAGENTA + "Validation"):
                # Upload data to GPU
                consumer_images = consumer_images.to(self.device)
                reference_images = reference_images.to(self.device)

                # Forward pass
                consumer_embeddings = self.model(consumer_images)
                reference_embeddings = self.model(reference_images)

                consumer_labels = self.convert_labels(consumer_labels)
                reference_labels = self.convert_labels(reference_labels)

                # Compute triplet loss
                val_loss = self.criterion(embeddings=consumer_embeddings,
                                          labels=consumer_labels,
                                          ref_emb=reference_embeddings,
                                          ref_labels=reference_labels)

                # Accumulate loss
                valid_losses.append(val_loss.item())

        return valid_losses

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- S A V E   H A R D   S A M P L E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_hard_samples(self, hard_samples, epoch):
        file_name = os.path.join(self.hard_samples_path, self.timestamp + "_epoch_" + str(epoch) + ".txt")
        with open(file_name, 'w') as file:
            file.write(str(hard_samples))

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
        # to track hard samples
        hard_samples = []

        network_config = sub_stream_network_configs(self.cfg)
        network_cfg = network_config.get(self.cfg.type_of_stream)

        dataset, train_data_loader, valid_data_loader = self.create_dataset(network_cfg)
        for epoch in tqdm(range(self.cfg.epochs), desc=colorama.Fore.GREEN + "Epochs"):
            # Train loop
            train_losses, hard_samples = self.train_loop(train_data_loader, train_losses, hard_samples)

            # Define the file path
            self.save_hard_samples(hard_samples, epoch)

            # Validation loop
            valid_losses = self.valid_loop(valid_data_loader, valid_losses)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            # Log loss for epoch
            logging.info(f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')

            # Clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()
            hard_samples.clear()

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
