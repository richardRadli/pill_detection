"""
File: triplet_loss_hard_mining.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This code implements the triplet loss.
"""

import torch
import torch.nn.functional as functional

from typing import Tuple


class TripletLossWithHardMining(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLossWithHardMining, self).__init__()
        self.margin = margin

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- F O R W A R D -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        This function calculates the loss and selects the hardest samples for each anchor.

        Finally, the loss is computed using the functional.relu function to enforce the margin constraint, and the mean
        of the resulting tensor is taken as the loss value. The loss value, along with the tensors of the
        hardest samples, is returned as a tuple.

        :param anchor: tensor of anchor images
        :param positive: tensor of positive images
        :param negative: tensor of negative images
        :return: a tuple containing the loss tensor, tensor of hardest negative samples and tensor of hardest positive
        samples
        """

        # Compute the Euclidean distances between the embeddings
        dist_pos = functional.pairwise_distance(anchor, positive, 2)
        dist_neg = functional.pairwise_distance(anchor, negative, 2)

        # Select the hardest sample for each anchor
        hard_triplet = []
        for i in range(dist_pos.size(0)):
            # Get the indices of the hard samples that violate the margin
            dist_diff = dist_pos[i] - dist_neg[i] + self.margin
            candidate_indices = torch.where(dist_diff > self.margin)[0]
            if len(candidate_indices) == 0:
                # If there are no hard samples that violate the margin, skip
                hard_triplet.append(None)
                continue
            # Select the hardest sample (i.e., the one with the largest distance to the anchor)
            if len(candidate_indices) == 1:
                hard_sample_idx = candidate_indices[0]
            else:
                hard_sample_idx = torch.argmax(dist_neg[i, candidate_indices])
            hard_triplet.append(negative[candidate_indices[hard_sample_idx]].unsqueeze(0))

        # Combine the hardest samples into a single tensor
        hard_triplet = [x for x in hard_triplet if x is not None and x.shape[0] != 0]
        if hard_triplet:
            hard_triplet = torch.cat(hard_triplet, dim=0)
        else:
            hard_triplet = torch.tensor([])

        # Compute the loss
        loss = functional.relu(self.margin + dist_pos - dist_neg).mean()

        return loss, hard_triplet
