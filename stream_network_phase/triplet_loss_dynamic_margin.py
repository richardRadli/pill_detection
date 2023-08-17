import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import Tuple


class DynamicMarginTripletLoss(nn.Module):
    def __init__(self, euc_dist_mtx, upper_norm_limit: int = 10,
                 margin: float = 0.5):
        super(DynamicMarginTripletLoss, self).__init__()

        self.margin = margin
        self.euc_dist_mtx = euc_dist_mtx
        self.upper_norm_limit = upper_norm_limit

    def forward(self, anchor_tensor: torch.Tensor, positive_tensor: torch.Tensor, negative_tensor: torch.Tensor,
                anchor_file_names, negative_file_names) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        anchor_file_names = self.process_file_names(anchor_file_names)
        negative_file_names = self.process_file_names(negative_file_names)

        batch_size = anchor_tensor.size(0)
        losses = []

        hard_neg = []  # List to store hardest negative samples
        hard_pos = []  # List to store hardest positive samples

        for i in range(batch_size):
            row = self.euc_dist_mtx.loc[anchor_file_names[i]]
            min_val = row.min()
            max_val = row.max()
            normalized_row = 1 + (self.upper_norm_limit - 1) * (
                    (row - min_val) / (max_val - min_val))  # Normalize row between 1 and upper_norm_limit
            normalized_similarity = normalized_row[negative_file_names[i]]

            margin = normalized_similarity * self.margin
            dist_pos = functional.pairwise_distance(anchor_tensor[i:i + 1], positive_tensor[i:i + 1], 2)
            dist_neg = functional.pairwise_distance(anchor_tensor[i:i + 1], negative_tensor[i:i + 1], 2)
            loss = functional.relu(margin + dist_pos - dist_neg)
            losses.append(loss)

            # Select hardest negative sample for this anchor
            dist_diff = dist_pos - dist_neg + margin
            violation_mask = dist_diff > 0
            candidate_idxs = torch.where(violation_mask)[0]
            if candidate_idxs.numel() > 0:
                hard_neg_idx = torch.argmin(dist_neg[candidate_idxs])
                hard_neg_sample = negative_tensor[candidate_idxs[hard_neg_idx]].unsqueeze(0)
                hard_neg.append(hard_neg_sample)

            # Select hardest positive sample for this anchor
            dist_diff = dist_neg - dist_pos + margin
            violation_mask = dist_diff > 0
            candidate_idxs = torch.where(violation_mask)[0]
            if candidate_idxs.numel() > 0:
                hard_pos_idx = torch.argmin(dist_pos[candidate_idxs])
                hard_pos_sample = positive_tensor[candidate_idxs[hard_pos_idx]].unsqueeze(0)
                hard_pos.append(hard_pos_sample)

        triplet_loss = torch.mean(torch.stack(losses))
        hard_neg_samples = torch.cat(hard_neg, dim=0) if hard_neg else torch.tensor([])
        hard_pos_samples = torch.cat(hard_pos, dim=0) if hard_pos else torch.tensor([])

        return triplet_loss, hard_neg_samples, hard_pos_samples

    @staticmethod
    def process_file_names(lines):
        texture_names = []

        for line in lines:
            # Remove parentheses and split the line into individual file paths
            paths = line.strip('()\n').split(', ')

            # Process each individual file path
            for filename in paths:
                filename = filename.strip("'")  # Remove single quotes around the path
                match = re.search(r'^(?:(texture|contour|lbp)_)?id_\d{3}_([a-zA-Z0-9_]+)_\d{3}\.png$',
                                  os.path.basename(filename))
                texture_name = match.group(2) if match else print("Filename doesn't match the pattern")
                texture_names.append(texture_name)

        return texture_names
