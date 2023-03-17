import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

from config import ConfigStreamNetwork
from const import CONST
from fusion_network import FusionNet
from stream_network_phase.stream_dataset_loader import StreamDataset
from utils.utils import create_timestamp, find_latest_file

cfg = ConfigStreamNetwork().parse()


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
        dataset_rgb = StreamDataset(CONST.dir_bounding_box, "RGB")
        dataset_contour = StreamDataset(CONST.dir_contour, "Contour")
        dataset_texture = StreamDataset(CONST.dir_texture, "Texture")

        # Load dataset
        train_size = int(0.8 * len(dataset_rgb))
        valid_size = len(dataset_rgb) - train_size
        print(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        train_dataset_rgb, valid_dataset_rgb = random_split(dataset_rgb, [train_size, valid_size])

        train_size = int(0.8 * len(dataset_contour))
        valid_size = len(dataset_contour) - train_size
        print(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        train_dataset_contour, valid_dataset_contour = random_split(dataset_contour, [train_size, valid_size])

        train_size = int(0.8 * len(dataset_texture))
        valid_size = len(dataset_texture) - train_size
        print(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        train_dataset_texture, valid_dataset_texture = random_split(dataset_texture, [train_size, valid_size])

        self.train_data_loader_rgb = DataLoader(train_dataset_rgb, batch_size=cfg.batch_size, shuffle=True)
        self.valid_data_loader_rgb = DataLoader(valid_dataset_rgb, batch_size=cfg.batch_size, shuffle=True)

        self.train_data_loader_contour = DataLoader(train_dataset_contour, batch_size=cfg.batch_size, shuffle=True)
        self.valid_data_loader_contour = DataLoader(valid_dataset_contour, batch_size=cfg.batch_size, shuffle=True)

        self.train_data_loader_texture = DataLoader(train_dataset_texture, batch_size=cfg.batch_size, shuffle=True)
        self.valid_data_loader_texture = DataLoader(valid_dataset_texture, batch_size=cfg.batch_size, shuffle=True)

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
                                          lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Specify scheduler
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=1 / 3)

    def fit(self):
        for epoch in tqdm(range(4), desc="Epochs"):
            running_loss = 0.0

            for loader_rgb, loader_contour, loader_texture in zip(self.train_data_loader_rgb,
                                                                  self.train_data_loader_contour,
                                                                  self.train_data_loader_texture):
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

                    running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_data_loader_rgb)
            print(f"Epoch {epoch}, loss: {epoch_loss}")


if __name__ == "__main__":
    try:
        tm = TrainModel()
        tm.fit()
    except KeyboardInterrupt as kbe:
        print(kbe)