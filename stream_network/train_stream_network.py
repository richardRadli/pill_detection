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
        train_size = int(0.8 * len(dataset))
        valid_size = len(dataset) - train_size
        print(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # self.train_data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        self.train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)

        # Load model
        self.model = StreamNetwork(list_of_channels)

        # Load model and upload it to the GPU
        self.model.to(self.device)
        summary(self.model, (list_of_channels[0], cfg.img_size, cfg.img_size))

        # Specify loss function
        self.criterion = TripletLossWithHardMining(1.0).to(self.device)

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

        for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
            running_loss = 0.0
            running_val_loss = 0.0

            for idx, (anchor, positive, negative) in tqdm(enumerate(self.train_data_loader),
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

            with torch.no_grad():
                for idx, (anchor, positive, negative) in tqdm(enumerate(self.valid_data_loader),
                                                              total=len(self.valid_data_loader), desc="Validation"):
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                    # Forward pass
                    anchor_emb = self.model(anchor)
                    positive_emb = self.model(positive)
                    negative_emb = self.model(negative)

                    # Compute triplet loss
                    val_loss = self.criterion(anchor_emb, positive_emb, negative_emb).to(self.device)

                    # Display loss
                    running_val_loss  += val_loss.item()

                self.writer.add_scalars("Loss", {"train": loss, "validation": val_loss}, epoch)

            # Print loss for epoch
            print('\nEpoch %d, train loss: %.4f, validation loss: %.4f' % (
            epoch + 1, running_loss / len(self.train_data_loader), running_val_loss / len(self.valid_data_loader)))

            if cfg.save and epoch % cfg.save_freq == 0:
                torch.save(self.model.state_dict(), self.save_path + "/" + "epoch_" + (str(epoch) + ".pt"))

        self.writer.close()
        self.writer.flush()


if __name__ == "__main__":
    try:
        tm = TrainModel()
        tm.fit()
    except KeyboardInterrupt as kbe:
        print(kbe)
