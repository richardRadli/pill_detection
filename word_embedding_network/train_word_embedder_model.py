import logging
import os
import numpy as np
import torch

from pytorch_metric_learning import losses
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config.dataset_paths_selector import dataset_images_path_selector
from config.networks_paths_selector import word_embedding_paths
from config.json_config import json_config_selector
from utils.utils import (create_dataset, use_gpu_if_available, create_timestamp, setup_logger,
                         find_latest_file_in_directory, load_config_json)
from word_embedding_network.fully_connected_network import FullyConnectedNetwork
from word_embedding_network.dataloader_word_embedding_network import DataLoaderWordEmbeddingNetwork


class TrainWordEmbeddingNetwork:
    def __init__(self):
        self.timestamp = create_timestamp()
        self.logger = setup_logger()
        self.cfg = load_config_json(
            json_schema_filename=json_config_selector("word_embedding").get("schema"),
            json_filename=json_config_selector("word_embedding").get("config")
        )

        self.device = use_gpu_if_available()

        word_emb_model_confing = word_embedding_paths(self.cfg.get("dataset_type"))
        dataset_paths = dataset_images_path_selector(self.cfg.get("dataset_type"))

        json_file_path = (
            find_latest_file_in_directory(
                path=dataset_paths.get("dynamic_margin").get("concatenated_vectors"),
                extension="json"
            )
        )

        dataset = DataLoaderWordEmbeddingNetwork(json_file_path, self.cfg.get("neuron_split"))
        self.train_dataloader, self.valid_dataloader = (
            create_dataset(
                dataset=dataset,
                train_valid_ratio=self.cfg.get("train_valid_ratio"),
                batch_size=self.cfg.get("batch_size")
            )
        )

        self.criterion = losses.NTXentLoss(temperature=0.10)

        self.model = (
            FullyConnectedNetwork(
                neurons=self.cfg.get("neurons")
            )
        )

        self.model.to(self.device)
        summary(
            model=self.model,
            input_size=(
                self.cfg.get("batch_size"),
                self.cfg.get("neurons")[0]
            )
        )

        self.optimizer = (
            Adam(
                self.model.parameters(),
                lr=self.cfg.get("learning_rate")
            )
        )

        self.scheduler = (
            StepLR(
                optimizer=self.optimizer,
                step_size=1,
                gamma=0.01
            )
        )

        tensorboard_log_dir = self.create_save_dir(word_emb_model_confing.get("logs"))
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        self.save_path = self.create_save_dir(word_emb_model_confing.get("weights"))

        self.best_valid_loss = float('inf')
        self.best_model_path = None

    def create_save_dir(self, directory_path):
        directory_to_create = os.path.join(directory_path, f"{self.timestamp}")
        os.makedirs(directory_to_create, exist_ok=True)
        return directory_to_create

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

    def train_loop(self, train_losses: list):
        """
        Function that executes the train loop.

        Args:
            train_losses:

        Returns:
             train_losses:
        """

        for original, augmented in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc="Training"):
            self.optimizer.zero_grad()

            original = original.to(self.device)
            augmented = augmented.to(self.device)

            embeddings_original = self.model(original)
            embeddings_augmented = self.model(augmented)
            embeddings = torch.cat((embeddings_original, embeddings_augmented))
            indices = torch.arange(0, embeddings_original.size(0), device=self.device)
            labels = torch.cat((indices, indices))

            train_loss = self.criterion(embeddings, labels)

            train_loss.backward()

            self.optimizer.step()
            train_losses.append(train_loss.item())

        return train_losses

    def valid_loop(self, valid_losses):
        for original, augmented in tqdm(self.valid_dataloader, total=len(self.valid_dataloader), desc="Validation"):
            original = original.to(self.device)
            augmented = augmented.to(self.device)

            embeddings_original = self.model(original)
            embeddings_augmented = self.model(augmented)
            embeddings = torch.cat((embeddings_original, embeddings_augmented))
            indices = torch.arange(0, embeddings_original.size(0), device=self.device)
            labels = torch.cat((indices, indices))

            valid_loss = self.criterion(embeddings, labels)
            valid_losses.append(valid_loss.item())

        return valid_losses

    def fit(self):
        train_losses = []
        valid_losses = []

        for epoch in tqdm(range(self.cfg.get("epochs")), desc="Epochs"):
            self.train_loop(train_losses)
            self.valid_loop(valid_losses)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)
            logging.info(f'train_loss: {train_loss:.5f}, valid_loss: {valid_loss:.5f}')

            train_losses.clear()
            valid_losses.clear()

            self.save_model_weights(epoch, valid_loss)
            self.scheduler.step()

        self.writer.close()
        self.writer.flush()


if __name__ == '__main__':
    try:
        train_obj = TrainWordEmbeddingNetwork()
        train_obj.fit()
    except KeyboardInterrupt as kie:
        logging.error(f"\nInterrupted by user: {kie}")
