import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config import ConfigStreamNetwork
from const import CONST
from stream_dataset_loader import StreamDataset
from stream_network import StreamNetwork
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
        """
        This function is the constructor of a class. The function initializes the utilized devices, if cuda is available
        it will use the GPU. Next it defines which network it will use (RGB or Texture/Contour). By that definition, it
        set up the corresponding dataloader and network architecture. Finally, it uploads the model to the GPU (if
        available), and set the loss function and optimizer.
        """

        print(f"The selected network is {cfg.type_of_network}")

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possibly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if cfg.type_of_network == "RGB":
            list_of_channels = [3, 64, 96, 128, 256, 384, 512]
            dataset = StreamDataset(CONST.dir_bounding_box, cfg.type_of_network)
            self.save_path = os.path.join(CONST.dir_stream_rgb_model_weights, self.timestamp)
            tensorboard_log_dir = CONST.dir_rgb_logs
        elif cfg.type_of_network in ["Texture", "Contour"]:
            list_of_channels = [1, 32, 48, 64, 128, 192, 256]
            dataset = StreamDataset(CONST.dir_texture, cfg.type_of_network) if cfg.type_of_network == "Texture" else \
                StreamDataset(CONST.dir_contour, cfg.type_of_network)
            self.save_path = os.path.join(CONST.dir_stream_texture_model_weights, self.timestamp) \
                if cfg.type_of_network == "Texture" \
                else os.path.join(CONST.dir_stream_contour_model_weights, self.timestamp)
            tensorboard_log_dir = CONST.dir_texture_logs if cfg.type_of_network == "Texture" \
                else CONST.dir_contour_logs
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
        summary(self.model, (list_of_channels[0], 128, 128))

        # Specify loss function
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2).cuda(self.device)

        # Specify optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        tensorboard_log_dir = os.path.join(tensorboard_log_dir, self.timestamp)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def fit(self):
        """
        This function is responsible for the training of the network.
        :return:
        """

        for epoch in tqdm(range(cfg.epochs), desc="Training epochs"):
            running_loss = 0.0
            for idx, (anchor, positive, negative) in tqdm(enumerate(self.train_data_loader),
                                                          total=len(self.train_data_loader), desc="Batch processing"):
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
                self.writer.add_scalar("Loss/train", loss, epoch)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Display loss
                running_loss += loss.item()

            # Print average loss for epoch
            print('\nEpoch %d, loss: %.4f' % (epoch + 1, running_loss / len(self.train_data_loader)))

            if cfg.save and epoch % cfg.save_freq == 0:
                torch.save(self.model.state_dict(), self.save_path + "/" + "epoch_" + (str(epoch) + ".pt"))

        self.writer.flush()


if __name__ == "__main__":
    try:
        tm = TrainModel()
        tm.fit()
    except KeyboardInterrupt as kbe:
        print(kbe)
