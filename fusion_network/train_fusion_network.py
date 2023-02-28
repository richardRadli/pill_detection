import torch

from torch.utils.data import DataLoader

from config import ConfigStreamNetwork
from const import CONST
from fusion_network import FusionNet
from stream_network.stream_dataset_loader import StreamDataset
from stream_network.stream_network import StreamNetwork
from utils.utils import create_timestamp
from utils.triplet_loss import TripletLoss

cfg = ConfigStreamNetwork().parse()


class TrainFusionNet:
    def __init__(self):
        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possibly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]
        list_of_channels_contour_texture = [1, 32, 48, 64, 128, 192, 256]

        # Load datasets
        dataset_rgb = StreamDataset(CONST.dir_bounding_box, cfg.type_of_network)
        self.train_rgb_data_loader = DataLoader(dataset_rgb, batch_size=cfg.batch_size, shuffle=True)

        dataset_contour = StreamDataset(CONST.dir_contour, cfg.type_of_network)
        self.train_contour_data_loader = DataLoader(dataset_contour, batch_size=cfg.batch_size, shuffle=True)

        dataset_texture = StreamDataset(CONST.dir_texture, cfg.type_of_network)
        self.train_texture_data_loader = DataLoader(dataset_texture, batch_size=cfg.batch_size, shuffle=True)

        # Load the pre-trained weights of the first network
        rgb_stream_net = StreamNetwork(list_of_channels_rgb)
        rgb_stream_net.load_state_dict(torch.load(CONST.dir_stream_rgb_model_weights +
                                                  "/2023-02-28_13-57-01/epoch_190.pt"))

        # Load the pre-trained weights of the second network
        contour_stream_net = StreamNetwork(list_of_channels_contour_texture)
        contour_stream_net.load_state_dict(torch.load(CONST.dir_stream_contour_model_weights +
                                                      "/2023-02-28_14-34-30/epoch_195.pt"))

        # Load the pre-trained weights of the third network
        texture_stream_net = StreamNetwork(list_of_channels_contour_texture)
        texture_stream_net.load_state_dict(torch.load(CONST.dir_stream_texture_model_weights +
                                                      "/2023-02-28_14-50-28/epoch_195.pt"))

        for param in rgb_stream_net.parameters():
            param.requires_grad = False
        for param in contour_stream_net.parameters():
            param.requires_grad = False
        for param in texture_stream_net.parameters():
            param.requires_grad = False

        # Create the fused network
        self.fused_net = FusionNet(rgb_stream_net, contour_stream_net, texture_stream_net)

        # Load model and upload it to the GPU
        self.fused_net.to(self.device)

        # Specify loss function
        self.criterion = TripletLoss().cuda(self.device)

        # Specify optimizer
        self.optimizer = torch.optim.Adam(self.fused_net.parameters(), lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay)

    def fit(self):
        pass


if __name__ == "__main__":
    try:
        fusion_net = TrainFusionNet()
        fusion_net.fit()
    except KeyboardInterrupt as kie:
        print(kie)
