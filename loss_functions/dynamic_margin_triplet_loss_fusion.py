"""
File: dynamic_margin_triplet_loss_fusion.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: March 4, 2024

Description: This code implements the dynamic margin triplet loss in the fusion phase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import List


class DynamicMarginTripletLoss(nn.Module):
    def __init__(self,
                 euc_dist_mtx,
                 upper_norm_limit: int = 3,
                 margin: float = 0.5) -> None:
        """

            euc_dist_mtx:
            upper_norm_limit:
            margin:
        """

        super(DynamicMarginTripletLoss, self).__init__()

        self.euc_dist_mtx = euc_dist_mtx
        self.upper_norm_limit = upper_norm_limit
        self.margin = margin

    def forward(self,
                anchor_tensor: torch.Tensor,
                positive_tensor: torch.Tensor,
                negative_tensor: torch.Tensor,
                anchor_file_names: List[str],
                negative_file_names: List[str]) -> torch.Tensor:
        """
        Perform forward pass of the DynamicMarginTripletLoss module.
        
        Args:
            anchor_tensor: The tensor containing anchor samples.
            positive_tensor: The tensor containing positive samples.
            negative_tensor: The tensor containing negative samples.
            anchor_file_names: The list of anchor file names.
            negative_file_names: The list of negative file names.

        Returns:
             A tuple containing the triplet loss, hardest negative samples, and hardest positive samples.
        """

        anchor_file_names = self.process_file_names(anchor_file_names)
        negative_file_names = self.process_file_names(negative_file_names)

        batch_size = anchor_tensor.size(0)
        losses = []

        for i in range(batch_size):
            ap_dist = functional.pairwise_distance(anchor_tensor[i:i + 1], positive_tensor[i:i + 1], 2)
            an_dist = functional.pairwise_distance(anchor_tensor[i:i + 1], negative_tensor[i:i + 1], 2)

            normalized_margins = self.normalize_row(anchor_file_names, negative_file_names, i)
            margin = self.margin * normalized_margins

            pos_neg_dists = ap_dist - an_dist
            loss = functional.relu(margin + pos_neg_dists)
            losses.append(loss)

        triplet_loss = torch.mean(torch.stack(losses))

        return triplet_loss

    def normalize_row(self,
                      anchor_file_names: List,
                      negative_file_names: List,
                      idx: int) -> torch.Tensor:
        """
        Normalize a row of the Euclidean distance matrix.

        Args:
            anchor_file_names: The list of anchor file names.
            negative_file_names: The list of negative file names.
            idx: The index of the row to normalize.

        Returns:
             The normalized row.
        """

        d_i = self.euc_dist_mtx.loc[anchor_file_names[idx]]
        d_min = d_i.min()
        d_max = d_i.max()
        d_i_norm = 1 + ((self.upper_norm_limit - 1) * (d_i - d_min)) / (d_max - d_min)
        return d_i_norm[negative_file_names[idx]]

    @staticmethod
    def process_file_names(lines: list) -> list:
        """
        Process the file names and extract the texture names.

            lines: A list of file paths.

        Returns:
             A list of texture names extracted from the file paths.
        """

        file_names = []
        for line in lines:
            file_name = line.split("\\")[2]
            file_names.append(file_name)

        return file_names