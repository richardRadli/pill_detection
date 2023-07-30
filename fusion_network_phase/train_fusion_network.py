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

from config.config import ConfigFusionNetwork, ConfigStreamNetwork
from config.logger_setup import setup_logger
from config.network_configs import sub_stream_network_configs_training, fusion_network_config
from fusion_models.efficient_net_v2_s_multihead_attention import EfficientNetV2MultiHeadAttention
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
        self.cfg_fusion_net = ConfigFusionNetwork().parse()
        self.cfg_stream_net = ConfigStreamNetwork().parse()

        network_type = self.cfg_fusion_net.type_of_net
        main_network_config = fusion_network_config(network_type=network_type)
        subnetwork_config = sub_stream_network_configs_training(self.cfg_stream_net)
        network_cfg_contour = subnetwork_config.get("Contour")
        network_cfg_lbp = subnetwork_config.get("LBP")
        network_cfg_rgb = subnetwork_config.get("RGB")
        network_cfg_texture = subnetwork_config.get("Texture")

        # Print network configurations
        print_network_config(self.cfg_fusion_net)

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possible
        self.device = use_gpu_if_available()

        # Load datasets using FusionDataset
        dataset = FusionDataset(image_size=network_cfg_contour.get("image_size"))
        train_size = int(self.cfg_fusion_net.train_split * len(dataset))
        valid_size = len(dataset) - train_size
        logging.info(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        self.train_data_loader = DataLoader(train_dataset, batch_size=self.cfg_fusion_net.batch_size, shuffle=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.cfg_fusion_net.batch_size, shuffle=True)

        # Initialize the fusion network
        self.model = \
            EfficientNetV2MultiHeadAttention(type_of_net=self.cfg_stream_net.type_of_net,
                                      network_cfg_contour=network_cfg_contour,
                                      network_cfg_lbp=network_cfg_lbp,
                                      network_cfg_rgb=network_cfg_rgb,
                                      network_cfg_texture=network_cfg_texture)

        # Load the saved state dictionaries of the stream networks
        stream_con_state_dict = \
            (torch.load(find_latest_file_in_latest_directory(network_cfg_contour.get("model_weights_dir"))))
        stream_lbp_state_dict = \
            (torch.load(find_latest_file_in_latest_directory(network_cfg_lbp.get("model_weights_dir"))))
        stream_rgb_state_dict = \
            (torch.load(find_latest_file_in_latest_directory(network_cfg_rgb.get("model_weights_dir"))))
        stream_tex_state_dict = \
            (torch.load(find_latest_file_in_latest_directory(network_cfg_texture.get("model_weights_dir"))))

        # Update the state dictionaries of the fusion network's stream networks
        self.model.contour_network.load_state_dict(stream_con_state_dict)
        self.model.rgb_network.load_state_dict(stream_rgb_state_dict)
        self.model.texture_network.load_state_dict(stream_tex_state_dict)
        self.model.lbp_network.load_state_dict(stream_lbp_state_dict)

        # Freeze the weights of the stream networks
        # for param in self.model.contour_network.parameters():
        #     param.requires_grad = False
        # for param in self.model.rgb_network.parameters():
        #     param.requires_grad = False
        # for param in self.model.texture_network.parameters():
        #     param.requires_grad = False
        # for param in self.model.lbp_network.parameters():
        #     param.requires_grad = False

        # Load model and upload it to the GPU
        self.model.to(self.device)

        # Display model
        summary(self.model,
                input_size=[(network_cfg_contour.get("channels")[0], network_cfg_contour.get("image_size"),
                             network_cfg_contour.get("image_size")),
                            (network_cfg_lbp.get("channels")[0], network_cfg_lbp.get("image_size"),
                             network_cfg_lbp.get("image_size")),
                            (network_cfg_rgb.get("channels")[0], network_cfg_rgb.get("image_size"),
                             network_cfg_rgb.get("image_size")),
                            (network_cfg_texture.get("channels")[0], network_cfg_texture.get("image_size"),
                             network_cfg_texture.get("image_size"))])

        # Specify loss function
        self.criterion = nn.TripletMarginLoss(margin=self.cfg_fusion_net.margin)

        # Specify optimizer
        self.optimizer = torch.optim.Adam(params=list(self.model.fc1.parameters()) + list(self.model.fc2.parameters()),
                                          lr=self.cfg_fusion_net.learning_rate,
                                          weight_decay=self.cfg_fusion_net.weight_decay)

        # LR scheduler
        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.cfg_fusion_net.step_size,
                                gamma=self.cfg_fusion_net.gamma)

        # Tensorboard
        tensorboard_log_dir = os.path.join(main_network_config.get("logs_folder"), self.timestamp)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Create save path
        self.save_path = os.path.join(main_network_config.get("weights_folder"), self.timestamp)
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

        for epoch in tqdm(range(self.cfg_fusion_net.epochs), desc="Epochs"):
            # Train loop
            for a_con, a_lbp, a_rgb, a_tex, p_con, p_lbp, p_rgb, p_tex, n_con, n_lbp, n_rgb, n_tex in \
                    tqdm(self.train_data_loader, total=len(self.train_data_loader), desc="Training"):
                # Uploading data to the GPU
                anchor_contour = a_con.to(self.device)
                positive_contour = p_con.to(self.device)
                negative_contour = n_con.to(self.device)

                anchor_lbp = a_lbp.to(self.device)
                positive_lbp = p_lbp.to(self.device)
                negative_lbp = n_lbp.to(self.device)

                anchor_rgb = a_rgb.to(self.device)
                positive_rgb = p_rgb.to(self.device)
                negative_rgb = n_rgb.to(self.device)

                anchor_texture = a_tex.to(self.device)
                positive_texture = p_tex.to(self.device)
                negative_texture = n_tex.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                anchor_out = self.model(anchor_contour, anchor_lbp, anchor_rgb, anchor_texture)
                positive_out = self.model(positive_contour, positive_lbp, positive_rgb, positive_texture)
                negative_out = self.model(negative_contour, negative_lbp, negative_rgb, negative_texture)

                # Compute triplet loss
                t_loss = self.criterion(anchor_out, positive_out, negative_out)

                # Backward pass
                t_loss.backward()
                self.optimizer.step()

                # Accumulate loss
                train_losses.append(t_loss.item())

            # Validation loop
            with torch.no_grad():
                for a_con, a_lbp, a_rgb, a_tex, p_con, p_lbp, p_rgb, p_tex, n_con, n_lbp, n_rgb, n_tex in \
                        tqdm(self.valid_data_loader, total=len(self.valid_data_loader), desc="Validation"):
                    # Uploading data to the GPU
                    anchor_contour = a_con.to(self.device)
                    positive_contour = p_con.to(self.device)
                    negative_contour = n_con.to(self.device)

                    anchor_lbp = a_lbp.to(self.device)
                    positive_lbp = p_lbp.to(self.device)
                    negative_lbp = n_lbp.to(self.device)

                    anchor_rgb = a_rgb.to(self.device)
                    positive_rgb = p_rgb.to(self.device)
                    negative_rgb = n_rgb.to(self.device)

                    anchor_texture = a_tex.to(self.device)
                    positive_texture = p_tex.to(self.device)
                    negative_texture = n_tex.to(self.device)

                    self.optimizer.zero_grad()

                    # Forward pass
                    anchor_out = self.model(anchor_contour, anchor_lbp, anchor_rgb, anchor_texture)
                    positive_out = self.model(positive_contour, positive_lbp, positive_rgb, positive_texture)
                    negative_out = self.model(negative_contour, negative_lbp, negative_rgb, negative_texture)

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
    except KeyboardInterrupt:
        logging.error('Keyboard interrupt has been occurred!')
