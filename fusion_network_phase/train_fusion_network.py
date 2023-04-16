import numpy as np
import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

from config import ConfigFusionNetwork
from const import CONST
from fusion_network import FusionNet
from fusion_dataset_loader import FusionDataset
from utils.utils import create_timestamp, find_latest_file

cfg = ConfigFusionNetwork().parse()


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

        # Select the GPU if possibly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        list_of_channels_con_tex = [1, 32, 48, 64, 128, 192, 256]
        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]

        # Load dataset
        self.train_data_loader_rgb, self.valid_data_loader_rgb = self.dataset_load(CONST.dir_bounding_box, "RGB")
        self.train_data_loader_contour, self.valid_data_loader_contour = self.dataset_load(CONST.dir_contour, "Contour")
        self.train_data_loader_texture, self.valid_data_loader_texture = self.dataset_load(CONST.dir_texture, "Texture")

        # Initialize the fusion network
        self.model = FusionNet()

        # Load the saved state dictionaries of the stream networks
        stream_con_state_dict = (torch.load(find_latest_file(CONST.dir_stream_contour_model_weights)))
        stream_rgb_state_dict = (torch.load(find_latest_file(CONST.dir_stream_rgb_model_weights)))
        stream_tex_state_dict = (torch.load(find_latest_file(CONST.dir_stream_texture_model_weights)))

        # Update the state dictionaries of the fusion network's stream networks
        self.model.contour_network.load_state_dict(stream_con_state_dict)
        self.model.rgb_network.load_state_dict(stream_rgb_state_dict)
        self.model.texture_network.load_state_dict(stream_tex_state_dict)

        # Freeze the weights of the stream networks
        for param in self.model.contour_network.parameters():
            param.requires_grad = False
        for param in self.model.rgb_network.parameters():
            param.requires_grad = False
        for param in self.model.texture_network.parameters():
            param.requires_grad = False

        # Load model and upload it to the GPU
        self.model.to(self.device)
        summary(self.model, input_size=[(list_of_channels_con_tex[0], cfg.img_size, cfg.img_size),
                (list_of_channels_rgb[0], cfg.img_size, cfg.img_size),
                (list_of_channels_con_tex[0], cfg.img_size, cfg.img_size)])

        # Specify loss function
        self.criterion = nn.TripletMarginLoss(margin=cfg.margin)

        # Specify optimizer
        self.optimizer = torch.optim.Adam(list(self.model.fc1.parameters()) + list(self.model.fc2.parameters()),
                                          lr=cfg.learning_rate)

        # Specify scheduler
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=1 / 3)

        # Create save path
        self.save_path = os.path.join(CONST.dir_hardest_samples_weights, self.timestamp)
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def dataset_load(src_dir, operation):
        dataset = FusionDataset(src_dir, operation)
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        print(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)

        return train_data_loader, valid_data_loader

    def fit(self):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
            for loader_rgb, loader_contour, loader_texture in tqdm(zip(self.train_data_loader_rgb,
                                                                  self.train_data_loader_contour,
                                                                  self.train_data_loader_texture),
                                                                   total=len(self.train_data_loader_rgb),
                                                                   desc="Training"):
                for batch_rgb, batch_contour, batch_texture in zip(loader_rgb, loader_contour, loader_texture):
                    anchor_rgb, positive_rgb, negative_rgb, *_ = batch_rgb
                    anchor_contour, positive_contour, negative_contour, *_ = batch_contour
                    anchor_texture, positive_texture, negative_texture, *_ = batch_texture

                    anchor_rgb = anchor_rgb.to(self.device)
                    positive_rgb = positive_rgb.to(self.device)
                    negative_rgb = negative_rgb.to(self.device)

                    anchor_contour = anchor_contour.to(self.device)
                    positive_contour = positive_contour.to(self.device)
                    negative_contour = negative_contour.to(self.device)

                    anchor_texture = anchor_texture.to(self.device)
                    positive_texture = positive_texture.to(self.device)
                    negative_texture = negative_texture.to(self.device)

                    self.optimizer.zero_grad()

                    # Forward pass
                    anchor_out = self.model(anchor_contour, anchor_rgb, anchor_texture)
                    positive_out = self.model(positive_contour, positive_rgb, positive_texture)
                    negative_out = self.model(negative_contour, negative_rgb, negative_texture)

                    # Compute triplet loss
                    loss = self.criterion(anchor_out, positive_out, negative_out)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    train_losses.append(loss.item())

            # Validation
            with torch.no_grad():
                for loader_rgb, loader_contour, loader_texture in tqdm(zip(self.valid_data_loader_rgb,
                                                                           self.valid_data_loader_contour,
                                                                           self.valid_data_loader_texture),
                                                                       total=len(self.valid_data_loader_rgb),
                                                                       desc="Validation"):
                    for batch_rgb, batch_contour, batch_texture in zip(loader_rgb, loader_contour, loader_texture):
                        anchor_rgb, positive_rgb, negative_rgb, *_ = batch_rgb
                        anchor_contour, positive_contour, negative_contour, *_ = batch_contour
                        anchor_texture, positive_texture, negative_texture, *_ = batch_texture

                        anchor_rgb = anchor_rgb.to(self.device)
                        positive_rgb = positive_rgb.to(self.device)
                        negative_rgb = negative_rgb.to(self.device)

                        anchor_contour = anchor_contour.to(self.device)
                        positive_contour = positive_contour.to(self.device)
                        negative_contour = negative_contour.to(self.device)

                        anchor_texture = anchor_texture.to(self.device)
                        positive_texture = positive_texture.to(self.device)
                        negative_texture = negative_texture.to(self.device)

                        anchor_out = self.model(anchor_contour, anchor_rgb, anchor_texture)
                        positive_out = self.model(positive_contour, positive_rgb, positive_texture)
                        negative_out = self.model(negative_contour, negative_rgb, negative_texture)

                        val_loss = self.criterion(anchor_out, positive_out, negative_out)
                        valid_losses .append(val_loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print(f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')
            print('Learning rate: %e' % self.optimizer.param_groups[0]['lr'])

            # clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()

            if cfg.save and epoch % cfg.save_freq == 0:
                torch.save(self.model.state_dict(), self.save_path + "/" + "epoch_" + (str(epoch) + ".pt"))


if __name__ == "__main__":
    try:
        tm = TrainModel()
        tm.fit()
    except KeyboardInterrupt as kbe:
        print(kbe)
