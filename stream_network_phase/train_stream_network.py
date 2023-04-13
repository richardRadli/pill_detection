import json
import numpy as np
import os
import torch
# import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from config import ConfigStreamNetwork
from const import CONST
from triplet_loss import TripletLoss
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
        print(f"The selected network is {cfg.type_of_network}")

        # Create time stamp
        self.timestamp = create_timestamp()

        # Select the GPU if possibly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup network config
        if cfg.type_of_network not in ["RGB", "Texture", "Contour"]:
            raise ValueError("Wrong type was given!")

        network_config = {
            "RGB": {
                "channels": [3, 64, 96, 128, 256, 384, 512],
                "dataset_dir": CONST.dir_bounding_box,
                "model_weights_dir": CONST.dir_stream_rgb_model_weights,
                "logs_dir": CONST.dir_rgb_logs
            },
            "Texture": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "dataset_dir": CONST.dir_texture,
                "model_weights_dir": CONST.dir_stream_texture_model_weights,
                "logs_dir": CONST.dir_texture_logs
            },
            "Contour": {
                "channels": [1, 32, 48, 64, 128, 192, 256],
                "dataset_dir": CONST.dir_contour,
                "model_weights_dir": CONST.dir_stream_contour_model_weights,
                "logs_dir": CONST.dir_contour_logs
            }
        }

        network_cfg = network_config.get(cfg.type_of_network)
        if not network_cfg:
            raise ValueError("Wrong type was given!")

        self.dataset = StreamDataset(network_cfg.get('dataset_dir'), cfg.type_of_network)
        self.save_path = os.path.join(network_cfg.get('model_weights_dir'), self.timestamp)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Load dataset
        train_size = int(0.8 * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        print(f"Size of the train set: {train_size}, size of the validation set: {valid_size}")
        train_dataset, valid_dataset = random_split(self.dataset, [train_size, valid_size])
        self.train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)

        # Load model and upload it to the GPU
        self.model = StreamNetwork(network_cfg.get('channels'))
        self.model.to(self.device)
        summary(self.model, (network_cfg.get('channels')[0], cfg.img_size, cfg.img_size))

        # Specify loss function
        self.criterion = torch.nn.TripletMarginLoss(margin=cfg.margin, p=2).to(self.device)

        # Specify optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Scheduler
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=1 / 3)

        # Tensorboard
        tensorboard_log_dir = os.path.join(network_cfg.get('logs_dir'), self.timestamp)
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- G E T   H A R D  S A M P L E S -----------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_hardest_samples(self, hardest_indices, epoch, operation):
        """

        :param hardest_indices:
        :param epoch:
        :param operation:
        :return:
        """

        if operation == "anchor":
            index = 3
            dict_name = os.path.join(CONST.dir_hardest_anc_samples, self.timestamp, cfg.type_of_network)
            os.makedirs(dict_name, exist_ok=True)
            file_name = os.path.join(dict_name, self.timestamp + "_epoch_" + str(epoch) + "_hardest_anc.txt")
        elif operation == "positive":
            index = 4
            dict_name = os.path.join(CONST.dir_hardest_pos_samples, self.timestamp, cfg.type_of_network)
            os.makedirs(dict_name, exist_ok=True)
            file_name = os.path.join(dict_name, self.timestamp + "_epoch_" + str(epoch) + "_hardest_pos.txt")
        elif operation == "negative":
            index = 5
            dict_name = os.path.join(CONST.dir_hardest_neg_samples, self.timestamp, cfg.type_of_network)
            os.makedirs(dict_name, exist_ok=True)
            file_name = os.path.join(dict_name, self.timestamp + "_epoch_" + str(epoch) + "_hardest_neg.txt")
        else:
            raise ValueError("Wrong type!")

        hardest_images = [self.dataset[idx][index] for idx in hardest_indices]
        dict_imgs = {hardest_indices[i]: hardest_images[i] for i in range(len(hardest_indices))}

        with open(file_name, "w") as fp:
            json.dump(dict_imgs, fp)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------ F I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def fit(self):
        """
        This function is responsible for the training of the network.
        :return:
        """

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []

        for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
            for idx, (anchor, positive, negative, _, _, _) in tqdm(enumerate(self.train_data_loader),
                                                                   total=len(self.train_data_loader), desc="Train"):
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
                loss = self.criterion(anchor_emb, positive_emb, negative_emb).to(self.device)

                # Backward pass, optimize and scheduler
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                # Accumulate loss
                train_losses.append(loss.item())

            # Validation
            with torch.no_grad():
                for idx, (anchor, positive, negative, _, _, _) in tqdm(enumerate(self.valid_data_loader),
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
                    val_loss = self.criterion(anchor_emb, positive_emb, negative_emb).to(self.device)

                    # Accumulate loss
                    valid_losses.append(val_loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            # Print loss for epoch
            print(f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')
            print('Learning rate: %e' % self.optimizer.param_groups[0]['lr'])

            # clear lists to track next epoch
            train_losses.clear()
            valid_losses.clear()

            # # Get the hardest positive images
            # self.get_hardest_samples(hardest_indices=self.criterion.hardest_positive_indices, epoch=epoch,
            #                          operation="positive")
            #
            # # Get the hardest negative images
            # self.get_hardest_samples(hardest_indices=self.criterion.hardest_negative_indices, epoch=epoch,
            #                          operation="negative")
            #
            # # Get the hardest anchor images
            # self.get_hardest_samples(hardest_indices=self.criterion.hardest_anchor_indices, epoch=epoch,
            #                          operation="anchor")

            # Save the model and weights
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
