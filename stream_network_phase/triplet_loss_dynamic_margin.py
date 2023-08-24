import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import List, Tuple

from config.config import ConfigGeneral


class DynamicMarginTripletLoss(nn.Module):
    def __init__(self, euc_dist_mtx, upper_norm_limit: int = 2, margin: float = 0.5):
        """

        :param euc_dist_mtx:
        :param upper_norm_limit:
        :param margin:
        """

        super(DynamicMarginTripletLoss, self).__init__()

        self.euc_dist_mtx = euc_dist_mtx
        self.upper_norm_limit = upper_norm_limit
        self.margin = margin
        self.dataset_type = ConfigGeneral().parse().dataset_type
        self.regexp = {"cure": r'^(?:(texture|contour|lbp)_)?id_\d{3}_([a-zA-Z0-9_]+)_\d{3}\.png$',
                       "ogyei": r'^(?:(texture|contour|lbp)_)?[0-9]+_(bottom|top)_[0-9]+_(bottom|top)_[0-9]+\.png$'}

    def forward(self, anchor_tensor: torch.Tensor, positive_tensor: torch.Tensor, negative_tensor: torch.Tensor,
                anchor_file_names: List[str], negative_file_names: List[str]) -> (
            Tuple)[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass of the DynamicMarginTripletLoss module.

        :param anchor_tensor: The tensor containing anchor samples.
        :param positive_tensor: The tensor containing positive samples.
        :param negative_tensor: The tensor containing negative samples.
        :param anchor_file_names: The list of anchor file names.
        :param negative_file_names: The list of negative file names.
        :return: A tuple containing the triplet loss, hardest negative samples, and hardest positive samples.
        """

        anchor_file_names = self.process_file_names(anchor_file_names)
        negative_file_names = self.process_file_names(negative_file_names)

        batch_size = anchor_tensor.size(0)
        losses = []

        hard_neg = []
        hard_pos = []

        for i in range(batch_size):
            normalized_similarity = self.normalize_row(anchor_file_names, negative_file_names, i)
            margin = normalized_similarity * self.margin
            dist_pos = functional.pairwise_distance(anchor_tensor[i:i + 1], positive_tensor[i:i + 1], 2)
            dist_neg = functional.pairwise_distance(anchor_tensor[i:i + 1], negative_tensor[i:i + 1], 2)
            loss = functional.relu(margin + dist_pos - dist_neg)
            losses.append(loss)

            # Select hardest negative sample for this anchor
            dist_diff_neg = dist_pos - dist_neg + margin
            hard_neg = self.select_hardest_samples(dist_diff_neg, dist_neg, negative_tensor, hard_neg)

            # Select hardest positive sample for this anchor
            dist_diff_pos = dist_neg - dist_pos + margin
            hard_pos = self.select_hardest_samples(dist_diff_pos, dist_pos, positive_tensor, hard_pos)

        triplet_loss = torch.mean(torch.stack(losses))
        hard_neg_samples = torch.cat(hard_neg, dim=0) if hard_neg else torch.tensor([])
        hard_pos_samples = torch.cat(hard_pos, dim=0) if hard_pos else torch.tensor([])

        return triplet_loss, hard_neg_samples, hard_pos_samples

    def normalize_row(self, anchor_file_names: List, negative_file_names: List, idx: int) -> torch.Tensor:
        """
        Normalize a row of the Euclidean distance matrix.

        :param anchor_file_names: The list of anchor file names.
        :param negative_file_names: The list of negative file names.
        :param idx: The index of the row to normalize.
        :return: The normalized row.
        """

        row = self.euc_dist_mtx.loc[anchor_file_names[idx]]
        min_val = row.min()
        max_val = row.max()
        normalized_row = 1 + (self.upper_norm_limit - 1) * (
                (row - min_val) / (max_val - min_val))
        return normalized_row[negative_file_names[idx]]

    @staticmethod
    def select_hardest_samples(
            dist_diff, dist: torch.Tensor, tensor: torch.Tensor, hard_list: List[torch.Tensor]) ->\
            List[torch.Tensor]:
        """
        Select the hardest samples based on the distance differences.

        :param dist_diff: The difference between distances.
        :param dist: The distances.
        :param tensor: The tensor containing the samples.
        :param hard_list: The list to store the hardest samples.

        :return: The updated list of hardest samples.
        """

        violation_mask = dist_diff > 0
        candidate_idxs = torch.where(violation_mask)[0]
        if candidate_idxs.numel() > 0:
            hard_idx = torch.argmin(dist[candidate_idxs])
            hard_sample = tensor[candidate_idxs[hard_idx]].unsqueeze(0)
            hard_list.append(hard_sample)

        return hard_list

    def process_file_names(self, lines: list) -> list:
        """
        Process the file names and extract the texture names.

        :param lines: A list of file paths.

        :return: A list of texture names extracted from the file paths.
        """

        texture_names = []
        for line in lines:
            paths = line.strip('()\n').split(', ')

            for filename in paths:
                filename = filename.strip("'")
                match = re.search(self.regexp.get(self.dataset_type), os.path.basename(filename))
                texture_name = match.group(2) if match else print("Filename doesn't match the pattern")
                texture_names.append(texture_name)

        return texture_names
