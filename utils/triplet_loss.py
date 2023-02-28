import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ConfigStreamNetwork

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
