import logging

import numpy as np
import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config.config import ConfigFusionNetwork
from config.const import DATA_PATH
from config.logger_setup import setup_logger
from models.fusion_network import FusionNet
from fusion_dataset_loader import FusionDataset
from utils.utils import create_timestamp, find_latest_file_in_latest_directory, print_network_config, \
    use_gpu_if_available


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   M O D E L +++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainFusionNet:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- __I N I T__ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        setup_logger()

        # Set up configuration
        self.cfg = ConfigFusionNetwork().parse()

        # Print network configurations
        print_network_config(self.cfg)

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possible
        self.device = use_gpu_if_available()

        # Load datasets using FusionDataset
        dataset = FusionDataset(image_size=self.cfg.img_size)
        train_size = int(self.cfg.train_split * len(dataset))
        valid_size = len(dataset) - train_size
        logging.info(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        # # Initialize the fusion network
        self.model = FusionNet()

        # Load the saved state dictionaries of the stream networks
        stream_con_state_dict = \
            (torch.load(find_latest_file_in_latest_directory(
                DATA_PATH.get_data_path("weights_cnn_network_contour"))))
        stream_rgb_state_dict = \
            (torch.load(find_latest_file_in_latest_directory(
                DATA_PATH.get_data_path("weights_cnn_network_rgb"))))
        stream_tex_state_dict = \
            (torch.load(find_latest_file_in_latest_directory(
                DATA_PATH.get_data_path("weights_cnn_network_texture"))))
        stream_lbp_state_dict = \
            (torch.load(find_latest_file_in_latest_directory(
                DATA_PATH.get_data_path("weights_cnn_network_lbp"))))

        # Update the state dictionaries of the fusion network's stream networks
        self.model.contour_network.load_state_dict(stream_con_state_dict)
        self.model.rgb_network.load_state_dict(stream_rgb_state_dict)
        self.model.texture_network.load_state_dict(stream_tex_state_dict)
        self.model.lbp_network.load_state_dict(stream_lbp_state_dict)

        # Freeze the weights of the stream networks
        for param in self.model.contour_network.parameters():
            param.requires_grad = False
        for param in self.model.rgb_network.parameters():
            param.requires_grad = False
        for param in self.model.texture_network.parameters():
            param.requires_grad = False
        for param in self.model.lbp_network.parameters():
            param.requires_grad = False

        # Load model and upload it to the GPU
        self.model.to(self.device)

        list_of_channels_con_tex_lbp = [1, 32, 48, 64, 128, 192, 256]
        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]
        summary(self.model, input_size=[(list_of_channels_con_tex_lbp[0], self.cfg.img_size, self.cfg.img_size),
                                        (list_of_channels_rgb[0], self.cfg.img_size, self.cfg.img_size),
                                        (list_of_channels_con_tex_lbp[0], self.cfg.img_size, self.cfg.img_size),
                                        (list_of_channels_con_tex_lbp[0], self.cfg.img_size, self.cfg.img_size)])

        # Specify loss function
        self.criterion = nn.TripletMarginLoss(margin=self.cfg.margin)

        # Specify optimizer
        self.optimizer = torch.optim.Adam(list(self.model.fc1.parameters()) + list(self.model.fc2.parameters()),
                                          lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        # LR scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.cfg.step_size, gamma=self.cfg.gamma)

        # Tensorboard
        tensorboard_log_dir = os.path.join(DATA_PATH.get_data_path("logs_fusion_net"), self.timestamp)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Create save path
        self.save_path = os.path.join(DATA_PATH.get_data_path("weights_fusion_net"), self.timestamp)
        os.makedirs(self.save_path, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train(self):
        # To track the training loss as the model trains
        train_losses = []

        # To track the validation loss as the model trains
        valid_losses = []

        # Variables to save only the best epoch
        best_valid_loss = float('inf')
        best_model_path = None

        for epoch in tqdm(range(self.cfg.epochs), desc="Epochs"):
            # Train loop
            for a_rgb, a_con, a_tex, a_lbp, p_rgb, p_con, p_tex, p_lbp, n_rgb, n_con, n_tex, n_lbp in \
                    tqdm(self.train_data_loader, total=len(self.train_data_loader), desc="Training"):
                # Uploading data to the GPU
                anchor_rgb = a_rgb.to(self.device)
                positive_rgb = p_rgb.to(self.device)
                negative_rgb = n_rgb.to(self.device)

                anchor_contour = a_con.to(self.device)
                positive_contour = p_con.to(self.device)
                negative_contour = n_con.to(self.device)

                anchor_texture = a_tex.to(self.device)
                positive_texture = p_tex.to(self.device)
                negative_texture = n_tex.to(self.device)

                anchor_lbp = a_lbp.to(self.device)
                positive_lbp = p_lbp.to(self.device)
                negative_lbp = n_lbp.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                anchor_out = self.model(anchor_contour, anchor_rgb, anchor_texture, anchor_lbp)
                positive_out = self.model(positive_contour, positive_rgb, positive_texture, positive_lbp)
                negative_out = self.model(negative_contour, negative_rgb, negative_texture, negative_lbp)

                # Compute triplet loss
                t_loss = self.criterion(anchor_out, positive_out, negative_out)

                # Backward pass
                t_loss.backward()
                self.optimizer.step()

                # Accumulate loss
                train_losses.append(t_loss.item())

            # Validation loop
            with torch.no_grad():
                for a_rgb, a_con, a_tex, a_lbp, p_rgb, p_con, p_tex, p_lbp, n_rgb, n_con, n_tex, n_lbp in \
                        tqdm(self.valid_data_loader, total=len(self.valid_data_loader), desc="Validation"):
                    # Uploading data to the GPU
                    anchor_rgb = a_rgb.to(self.device)
                    positive_rgb = p_rgb.to(self.device)
                    negative_rgb = n_rgb.to(self.device)

                    anchor_contour = a_con.to(self.device)
                    positive_contour = p_con.to(self.device)
                    negative_contour = n_con.to(self.device)

                    anchor_texture = a_tex.to(self.device)
                    positive_texture = p_tex.to(self.device)
                    negative_texture = n_tex.to(self.device)

                    anchor_lbp = a_lbp.to(self.device)
                    positive_lbp = p_lbp.to(self.device)
                    negative_lbp = n_lbp.to(self.device)

                    self.optimizer.zero_grad()

                    # Forward pass
                    anchor_out = self.model(anchor_contour, anchor_rgb, anchor_texture, anchor_lbp)
                    positive_out = self.model(positive_contour, positive_rgb, positive_texture, positive_lbp)
                    negative_out = self.model(negative_contour, negative_rgb, negative_texture, negative_lbp)

                    # Compute triplet loss
                    v_loss = self.criterion(anchor_out, positive_out, negative_out)

                    # Accumulate loss
                    valid_losses.append(v_loss.item())

            # Decay the learning rate
            self.scheduler.step()

            # Print loss for epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            logging.info(f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')

            # Record to tensorboard
            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            # Clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()

            # Save the best model and weights
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(self.save_path, "epoch_" + (str(epoch) + ".pt"))
                torch.save(self.model.state_dict(), best_model_path)

        # Close and flush SummaryWriter
        self.writer.close()
        self.writer.flush()


if __name__ == "__main__":
    try:
        tm = TrainFusionNet()
        tm.train()
    except KeyboardInterrupt as kbe:
        logging.error(f'{kbe}')
