import json
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config import ConfigStreamNetwork
from const import CONST
from stream_dataset_loader import StreamDataset
from stream_network import StreamNetwork
from triplet_loss import TripletLossWithHardMining
from utils.utils import create_timestamp, print_network_config, use_gpu_if_available


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   M O D E L +++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainModel:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        # Set up configuration
        self.cfg = ConfigStreamNetwork().parse()

        # Print network configurations
        print_network_config(self.cfg)

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possibly
        self.device = use_gpu_if_available()

        # Setup network config
        if self.cfg.type_of_network not in ["RGB", "Texture", "Contour"]:
            raise ValueError("Wrong type was given!")

        network_config = {
            "RGB": {
                "channels": [3, 64, 96, 128, 256, 384, 512],
                "dataset_dir": CONST.dir_rgb,
                "model_weights_dir": CONST.dir_stream_rgb_model_weights,
                "logs_dir": CONST.dir_rgb_logs,
                "learning_rate": self.cfg.learning_rate_rgb
            },
            "Texture": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "dataset_dir": CONST.dir_texture,
                "model_weights_dir": CONST.dir_stream_texture_model_weights,
                "logs_dir": CONST.dir_texture_logs,
                "learning_rate": self.cfg.learning_rate_con_tex
            },
            "Contour": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "dataset_dir": CONST.dir_contour,
                "model_weights_dir": CONST.dir_stream_contour_model_weights,
                "logs_dir": CONST.dir_contour_logs,
                "learning_rate": self.cfg.learning_rate_con_tex
            }
        }

        # Set up network configuration
        network_cfg = network_config.get(self.cfg.type_of_network)
        if not network_cfg:
            raise ValueError("Wrong type was given!")

        # Load dataset
        self.dataset = StreamDataset(network_cfg.get('dataset_dir'), self.cfg.type_of_network)
        train_size = int(self.cfg.train_rate * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        print(f"\nSize of the train set: {train_size}\nSize of the validation set: {valid_size}\n")
        train_dataset, valid_dataset = random_split(self.dataset, [train_size, valid_size])
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # Load model and upload it to the GPU
        self.model = StreamNetwork(network_cfg.get('channels'))
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
    # ----------------------------------------- G E T   H A R D  S A M P L E S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_hardest_samples(self, epoch: int, hard_neg_images: list, output_dir: str, op: str) -> None:
        """
        :param epoch:
        :param hard_neg_images:
        :param output_dir:
        :param op:
        :return:
        """

        dict_name = os.path.join(output_dir, self.timestamp, self.cfg.type_of_network)
        os.makedirs(dict_name, exist_ok=True)
        file_name = os.path.join(dict_name, self.timestamp + "_epoch_" + str(epoch) + "_%s.txt" % op)

        hardest_samples = []
        for i, hard_neg_tensor in enumerate(hard_neg_images):
            hardest_samples.append(hard_neg_tensor)

        with open(file_name, "w") as fp:
            json.dump(hardest_samples, fp)

    @staticmethod
    def record_hard_samples(hard_samples: torch.Tensor, img_path: tuple, hard_images: list) -> list:
        """
        Record filenames of hardest negative samples

        :param hard_samples:
        :param img_path:
        :param hard_images:
        :return:
        """

        if hard_samples is not None:
            for filename, sample in zip(img_path, hard_samples):
                if sample is not None:
                    hard_images.append(filename)

        return hard_images

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
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

            # Print loss for epoch
            print(f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')

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
        print(f'{kbe}')
