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
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from typing import List, Tuple
from pytorch_metric_learning import miners

from config.json_config import json_config_selector
from config.networks_paths_selector import substream_paths
from config.networks_paths_selector import word_embedding_paths
from config.nlp_paths_selector import nlp_configs
from stream_network_models.stream_network_selector import StreamNetworkFactory
from loss_functions.dynamic_margin_triplet_loss_stream import DynamicMarginTripletLoss
from dataloader_stream_network import DataLoaderStreamNet
from utils.utils import (create_dataset, create_timestamp, get_embedded_text_matrix, measure_execution_time,
                         use_gpu_if_available, setup_logger, load_config_json)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   M O D E L +++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainModel:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Create time stamp
        self.timestamp = create_timestamp()

        # Set up logger
        self.logger = setup_logger()

        # Set up configuration
        self.cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )

        # Set up tqdm colour
        colorama.init()

        # Select the GPU if possible
        self.device = use_gpu_if_available()

        # Setup network config
        self.dataset_type = self.cfg.get("dataset_type")

        stream_type = self.cfg.get("type_of_stream")
        self.type_of_net = self.cfg.get("type_of_net")

        substream_network_cfg = self.cfg.get("streams").get(stream_type)
        backbone_network_cfg = self.cfg.get("networks").get(self.type_of_net)

        # Extract variables
        self.dataset_type = self.cfg.get("dataset_type")
        self.type_of_net = self.cfg.get("type_of_net")
        self.type_of_loss = self.cfg.get("type_of_loss_func")
        self.dmtl_type = self.cfg.get("dmtl_type")

        batch_size = self.cfg.get("batch_size")
        learning_rate = self.cfg.get("learning_rate")
        weight_decay = self.cfg.get("weight_decay")
        train_valid_ratio = self.cfg.get("train_valid_ratio")
        margin = self.cfg.get("margin")
        step_size = self.cfg.get("step_size")
        gamma = self.cfg.get("gamma")
        upper_norm_limit = self.cfg.get("upper_norm_limit")
        mining_type = self.cfg.get("mining_type")

        # Load model and upload it to the GPU
        self.model = StreamNetworkFactory.create_network(self.type_of_net, substream_network_cfg)
        self.model = self.model.to(self.device)

        # Print model configuration
        summary(
            model=self.model,
            input_size=(
                substream_network_cfg.get('channels')[0],
                backbone_network_cfg.get("image_size"),
                backbone_network_cfg.get("image_size")
            )
        )

        # Create dataset
        dataset_dirs_anchor = (
            substream_paths().get(stream_type).get(self.dataset_type).get(self.type_of_net).get("train").get("anchor")
        )
        dataset_dir_pos_neg = (
            substream_paths().get(stream_type).get(self.dataset_type).get(self.type_of_net).get("train").get("pos_neg")
        )
        dataset = \
            DataLoaderStreamNet(
                dataset_dirs_anchor=[dataset_dirs_anchor],
                dataset_dirs_pos_neg=[dataset_dir_pos_neg]
            )

        self.mapping = dataset.reference_encoding_map
        self.train_data_loader, self.valid_data_loader = (
            create_dataset(
                dataset=dataset,
                train_valid_ratio=train_valid_ratio,
                batch_size=batch_size
            )
        )

        # Set up loss and mining functions
        if self.type_of_loss == "dmtl":
            if self.dmtl_type == "textual":
                path_to_excel_file = (
                    nlp_configs().get("vector_distances")
                )
            else:
                path_to_excel_file = (
                    word_embedding_paths(self.dataset_type).get("emb_euc_mtx")
                )

            df = get_embedded_text_matrix(path_to_excel_file)
            self.criterion = (
                DynamicMarginTripletLoss(
                    euc_dist_mtx=df,
                    upper_norm_limit=upper_norm_limit,
                    margin=margin
                )
            )
        else:
            self.criterion = torch.nn.TripletMarginLoss(margin=margin)

        self.mining_func = (
            miners.TripletMarginMiner(
                margin=margin,
                type_of_triplets=mining_type
            )
        )

        # Specify optimizer
        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=backbone_network_cfg.get("learning_rate")
            )
        )

        # LR scheduler
        self.scheduler = (
            StepLR(
                optimizer=self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        )

        # Tensorboard
        tensorboard_log_dir = (
            self.create_save_dirs(
                network_cfg=substream_paths().get(stream_type),
                subdir="logs_dir",
                loss=self.type_of_loss,
                dmtl_type=self.dmtl_type
            )
        )
        self.writer = (
            SummaryWriter(
                log_dir=tensorboard_log_dir
            )
        )

        # Create save directory for hard samples
        self.hard_samples_path = (
            self.create_save_dirs(
                network_cfg=substream_paths().get(stream_type),
                subdir="hardest_samples",
                loss=self.type_of_loss,
                dmtl_type=self.dmtl_type
            )
        )

        # Create save path
        self.save_path = (
            self.create_save_dirs(
                network_cfg=substream_paths().get(stream_type),
                subdir="weights_folder",
                loss=self.type_of_loss,
                dmtl_type=self.dmtl_type
            )
        )

        # Variables to save only the best weights and model
        self.best_valid_loss = float('inf')
        self.best_model_path = None

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   S A V E   D I R S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def create_save_dirs(self, network_cfg, subdir, loss, dmtl_type) -> str:
        """
        Creates and returns a directory path based on the provided network configuration.

        Args:
            network_cfg: A dictionary containing network configuration information.
            subdir: The subdirectory to create the directory path for.
            loss: Loss type
            dmtl_type: Only set it when dmtl is used
        Returns:
            directory_to_create (str): The path of the created directory.
        """

        if loss == "hmtl":
            directory_path = network_cfg.get(subdir).get(loss)
        else:
            directory_path = network_cfg.get(subdir).get(loss).get(dmtl_type)

        directory_to_create = (
            os.path.join(directory_path, f"{self.timestamp}")
        )
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ C O N V E R T   L A B E L S -------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def convert_labels(self, labels: List[str]) -> torch.Tensor:
        """
        Convert labels to a tensor.

        Args:
            labels: A list of labels.

        Return:
             A tensor containing the converted labels.
        """

        labels = tuple(int(item) for item in labels)
        labels = torch.tensor(labels)
        return labels.to(self.device)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- T R A I N   L O O P ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train_loop(self, train_losses: List[float], hard_samples: List[list]) -> Tuple[List[float], List[List[str]]]:
        """
        Training loop for the model.

        Args:
            train_losses (list): List to store training losses.
            hard_samples (list): List to store hard samples.

        Returns:
            train_losses (list): Updated list of training losses.
            hard_samples (list): Updated list of hard samples.
        """

        for idx, (consumer_images, consumer_labels, cp, reference_images, reference_labels, rp) in \
                tqdm(enumerate(self.train_data_loader),
                     total=len(self.train_data_loader),
                     desc=colorama.Fore.CYAN + "Train"):
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
            indices_tuple = (
                self.mining_func(
                    embeddings=consumer_embeddings,
                    labels=consumer_labels,
                    ref_emb=reference_embeddings,
                    ref_labels=reference_labels
                )
            )

            anchor_indices, positive_indices, negative_indices = indices_tuple

            anchor_filenames = [cp[i] for i in anchor_indices]
            positive_filenames = [rp[i] for i in positive_indices]
            negative_filenames = [rp[i] for i in negative_indices]

            hard_sample = [anchor_filenames, positive_filenames, negative_filenames]
            hard_samples.append(hard_sample)

            # Compute loss
            train_loss = (
                self.criterion(
                    embeddings=consumer_embeddings,
                    labels=consumer_labels,
                    ref_emb=reference_embeddings,
                    ref_labels=reference_labels
                )
            )

            # Backward pass, optimize
            train_loss.backward()
            self.optimizer.step()

            # Accumulate loss
            train_losses.append(train_loss.item())

        return train_losses, hard_samples

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- V A L I D   L O O P ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def valid_loop(self, valid_losses: List[float]) -> List[float]:
        """
        Validation loop for the model.

        Args:
            valid_losses (list): List to store validation losses.

        Returns:
            valid_losses (list): Updated list of validation losses.
        """

        with (torch.no_grad()):
            for idx, (consumer_images, consumer_labels, _, reference_images, reference_labels, _) \
                    in tqdm(enumerate(self.valid_data_loader),
                            total=len(self.valid_data_loader),
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
                val_loss = (
                    self.criterion(
                        embeddings=consumer_embeddings,
                        labels=consumer_labels,
                        ref_emb=reference_embeddings,
                        ref_labels=reference_labels
                    )
                )

                # Accumulate loss
                valid_losses.append(val_loss.item())

        return valid_losses

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- S A V E   H A R D   S A M P L E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_hard_samples(self, hard_samples: List[List[str]], epoch: int) -> None:
        """
        Save hard samples to a file.

        Args:
            hard_samples (list): List of hard samples.
            epoch (int): Current epoch number.

        Returns:
            None
        """

        file_name = os.path.join(self.hard_samples_path, self.timestamp + "_epoch_" + str(epoch) + ".txt")
        with open(file_name, 'w') as file:
            file.write(str(hard_samples))

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- S A V E   M O D E L ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_model_weights(self, epoch: int, valid_loss: float) -> None:
        """
        Saves the model and weights if the validation loss is improved.

        Args:
            epoch (int): the current epoch
            valid_loss (float): the validation loss

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
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @measure_execution_time
    def training(self) -> None:
        """
        This function is responsible for the training of the network.

        Returns:
             None
        """

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track hard samples
        hard_samples = []

        for epoch in tqdm(range(self.cfg.epochs), desc=colorama.Fore.GREEN + "Epochs"):
            # Train loop
            train_losses, hard_samples = self.train_loop(train_losses, hard_samples)

            # Define the file path
            self.save_hard_samples(hard_samples, epoch)

            # Validation loop
            valid_losses = self.valid_loop(valid_losses)

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

        del train_loss, valid_loss, hard_samples


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
