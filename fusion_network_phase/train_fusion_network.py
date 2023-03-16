import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

from config import ConfigStreamNetwork
from const import CONST
from fusion_network import FusionNet
from fusion_dataset_loader import FusionNetDataset
from utils.utils import create_timestamp

cfg = ConfigStreamNetwork().parse()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   M O D E L +++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainModel:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        print(f"The selected network is {cfg.type_of_network}")

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possibly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        list_of_channels_con_tex = [1, 32, 48, 64, 128, 192, 256]
        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]

        # Load dataset
        # hardest_negative_samples = hardest_samples(CONST.dir_hardest_neg_samples)
        # hardest_positive_samples = hardest_samples(CONST.dir_hardest_pos_samples)
        # hardest_anchor_samples = hardest_samples(CONST.dir_hardest_anc_samples)
        #
        # hardest_sample_images = set(hardest_positive_samples) | set(hardest_negative_samples) | \
        #                         set(hardest_anchor_samples)

        # copy_hardest_samples("D:/project/IVM/images/texture_hardest/", hardest_sample_images)
        dataset = FusionNetDataset(CONST.dir_bounding_box, CONST.dir_contour, CONST.dir_texture)

        # # Load dataset
        # train_size = int(0.8 * len(dataset))
        # valid_size = len(dataset) - train_size
        # print(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        # train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        #
        # self.train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        # self.valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)
        #
        # # Initialize the fusion network
        # self.model = FusionNet()
        #
        # # Load the saved state dictionaries of the stream networks
        # stream_con_state_dict = (
        #     torch.load(CONST.dir_stream_contour_model_weights + "/2023-03-14_14-55-27/epoch_19.pt"))
        # stream_rgb_state_dict = (
        #     torch.load(CONST.dir_stream_rgb_model_weights + "/2023-03-14_15-03-15/epoch_19.pt"))
        # stream_tex_state_dict = (
        #     torch.load(CONST.dir_stream_texture_model_weights + "/2023-03-14_09-18-22/epoch_19.pt"))
        #
        # # Update the state dictionaries of the fusion network's stream networks
        # self.model.contour_network.load_state_dict(stream_con_state_dict)
        # self.model.rgb_network.load_state_dict(stream_rgb_state_dict)
        # self.model.texture_network.load_state_dict(stream_tex_state_dict)
        #
        # # Freeze the weights of the stream networks
        # for param in self.model.contour_network.parameters():
        #     param.requires_grad = False
        # for param in self.model.rgb_network.parameters():
        #     param.requires_grad = False
        # for param in self.model.texture_network.parameters():
        #     param.requires_grad = False
        #
        # # Load model and upload it to the GPU
        # self.model.to(self.device)
        # summary(self.model, input_size=[(list_of_channels_con_tex[0], cfg.img_size, cfg.img_size),
        #         (list_of_channels_rgb[0], cfg.img_size, cfg.img_size),
        #         (list_of_channels_con_tex[0], cfg.img_size, cfg.img_size)])
        #
        # # Specify loss function
        # self.criterion = nn.TripletMarginLoss(margin=cfg.margin)
        #
        # # Specify optimizer
        # self.optimizer = torch.optim.Adam(list(self.model.fc1.parameters()) + list(self.model.fc2.parameters()),
        #                                   lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        #
        # # Specify scheduler
        # self.scheduler = StepLR(self.optimizer, step_size=5, gamma=1 / 3)

    def fit_second_stage(self):
        for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
            running_loss = 0.0

            # Training
            for idx, (anchor, positive, negative, _, _, _) in tqdm(enumerate(self.train_data_loader),
                                                          total=len(self.train_data_loader), desc="Train"):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)

                # Compute triplet loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb).to(self.device)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Display loss
                running_loss += loss.item()

            # Print loss for epoch
            print('\nEpoch %d, train loss: %.4f' % (epoch + 1, running_loss / len(self.train_data_loader)))


if __name__ == "__main__":
    try:
        tm = TrainModel()
        # tm.fit_second_stage()
    except KeyboardInterrupt as kbe:
        print(kbe)