import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as functional
import pandas as pd


class DynamicMarginTripletLoss(nn.Module):
    def __init__(self, euc_dist_mtx, anchor_file_names, negative_file_names, upper_norm_limit: int = 2,
                 margin: float = 0.5):
        """

        :param euc_dist_mtx:
        :param anchor_file_names:
        :param negative_file_names:
        :param upper_norm_limit:
        :param margin:
        """

        super(DynamicMarginTripletLoss, self).__init__()

        self.margin = margin
        self.euc_dist_mtx = euc_dist_mtx
        self.anchor_file_names = anchor_file_names
        self.negative_file_names = negative_file_names
        self.upper_norm_limit = upper_norm_limit

    def forward(self, anchor_tensor: torch.Tensor, positive_tensor: torch.Tensor, negative_tensor: torch.Tensor):
        anchor_file_names = self.process_file_names(self.anchor_file_names)
        negative_file_names = self.process_file_names(self.negative_file_names)

        batch_size = anchor_tensor.size(0)
        losses = []

        for i in range(batch_size):
            row = self.euc_dist_mtx.loc[anchor_file_names[i]]
            min_val = row.min()
            max_val = row.max()
            normalized_row = 1 + (self.upper_norm_limit - 1) * (
                        (row - min_val) / (max_val - min_val))  # Normalize row between 1 and 2
            normalized_similarity = normalized_row[negative_file_names[i]]

            margin = normalized_similarity * self.margin
            dist_pos = functional.pairwise_distance(anchor_tensor[i:i + 1], positive_tensor[i:i + 1], 2)
            dist_neg = functional.pairwise_distance(anchor_tensor[i:i + 1], negative_tensor[i:i + 1], 2)
            loss = functional.relu(margin + dist_pos - dist_neg)
            losses.append(loss)

        return torch.mean(torch.stack(losses))

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


if __name__ == "__main__":
    # # Example usage
    excel_file_path = \
        'C:/Users/ricsi/Downloads/Vektor_távolságok_szöveg_alapján_korábbi_gyógyszerekre.xlsx'
    sheet_index = 1
    df = pd.read_excel(excel_file_path, sheet_name=sheet_index, index_col=0)

    num_samples = 128
    embedding_dim = 128

    anchor_tensor = torch.randn(num_samples, embedding_dim)
    positive_tensor = torch.randn(num_samples, embedding_dim)
    negative_tensor = torch.randn(num_samples, embedding_dim)

    # Read the content of the text file
    file_path = 'C:/Users/ricsi/Downloads/anchor_filenames.txt'  # Replace with the actual file path
    with open(file_path, 'r') as file:
        anchor_labels = file.readlines()

        # Read the content of the text file
    file_path = 'C:/Users/ricsi/Downloads/negative_filenames.txt'  # Replace with the actual file path
    with open(file_path, 'r') as file:
        negative_labels = file.readlines()

    triplet_loss = DynamicMarginTripletLoss(df, anchor_labels, negative_labels)
    loss = triplet_loss(anchor_tensor, positive_tensor, negative_tensor)
    print("Triplet Loss:", loss.item())
