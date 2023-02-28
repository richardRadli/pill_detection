import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from config import ConfigStreamNetwork
from const import CONST
from stream_dataset_loader import StreamDataset
from stream_network import StreamNetwork
from utils.utils import create_timestamp

cfg = ConfigStreamNetwork().parse()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++ T R I P L E T   L O S S ++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TripletLoss(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, margin=cfg.margin):
        """
        This function initializes an instance of the TripletLoss class. The constructor takes an optional margin
        argument, which is initialized to the value of the "cfg.margin" constant if it is not provided.
        :param margin: margin that is enforced between positive and negative pairs
        """

        super(TripletLoss, self).__init__()
        self.margin = margin

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- F O R W A R D -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, anchor, positive, negative):
        """
        The forward method takes as input three tensors: anchor, positive, and negative, which represent the embeddings
        of an anchor sample, a positive sample, and a negative sample, respectively.

        The method calculates the Euclidean distance between the anchor and positive examples (pos_dist) and the
        Euclidean distance between the anchor and negative examples (neg_dist). It then calculates the triplet loss as
        the mean of the maximum of 0 and the difference between the pos_dist and neg_dist, with a margin value
        subtracted from the difference.

        :param anchor:
        :param positive:
        :param negative:
        :return: loss
        """

        # Calculate the Euclidean distance between anchor and positive examples
        pos_dist = F.pairwise_distance(anchor, positive, p=2)

        # Calculate the Euclidean distance between anchor and negative examples
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Calculate the triplet loss
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))

        return loss


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ T R A I N   M O D E L +++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TrainModel:
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------- _ I N I T _ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        """
        This function is the constructor of a class. The function initializes the utilized devices, if cuda is available
        it will use the GPU. Next it defines which network it will use (RGB or Texture/Contour). By that definition, it
        set up the corresponding dataloader and network architecture. Finally, it uploads the model to the GPU (if
        available), and set the loss function and optimizer.
        """

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possibly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if cfg.type_of_network == "RGB":
            list_of_channels = [3, 64, 96, 128, 256, 384, 512]
            dataset = StreamDataset(CONST.dir_bounding_box, cfg.type_of_network)
            self.save_path = os.path.join(CONST.dir_stream_rgb_model_weights, self.timestamp)
        elif cfg.type_of_network in ["Texture", "Contour"]:
            list_of_channels = [1, 32, 48, 64, 128, 192, 256]
            dataset = StreamDataset(CONST.dir_texture, cfg.type_of_network) if cfg.type_of_network == "Texture" else \
                StreamDataset(CONST.dir_contour, cfg.type_of_network)
            self.save_path = os.path.join(CONST.dir_stream_texture_model_weights, self.timestamp) \
                if cfg.type_of_network == "Texture" \
                else os.path.join(CONST.dir_stream_contour_model_weights, self.timestamp)
        else:
            raise ValueError("Wrong type was given!")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Load dataset
        self.train_data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        # Load model
        self.model = StreamNetwork(list_of_channels)

        # Load model and upload it to the GPU
        self.model.to(self.device)
        print(self.model)

        # Specify loss function
        self.criterion = TripletLoss().cuda(self.device)
        self.best_valid_loss = float('inf')

        # Specify optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def fit(self):
        """
        This function is responsible for the training of the network.
        :return:
        """

        for epoch in range(cfg.epochs):
            running_loss = 0.0
            for idx, (anchor, positive, negative) in tqdm(enumerate(self.train_data_loader),
                                                          total=len(self.train_data_loader)):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)

                # Compute triplet loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Display loss
                running_loss += loss.item()

            # Print average loss for epoch
            print('\nEpoch %d, loss: %.4f' % (epoch + 1, running_loss / len(self.train_data_loader)))

            if cfg.save:
                torch.save(self.model.state_dict(), self.save_path + "/" + "epoch_" + (str(epoch) + ".pt"))


if __name__ == "__main__":
    try:
        tm = TrainModel()
        tm.fit()
    except KeyboardInterrupt as kbe:
        print(kbe)
