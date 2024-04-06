"""
File: dynamic_margin_triplet_loss_stream.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: March 04, 2024

Description: The program creates the dynamic margin-triplet loss for the stream phase.
"""

import torch

from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from typing import List


class DynamicMarginTripletLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
        euc_dist_mtx:
        upper_norm_limit:
    """

    def __init__(
        self,
        margin=0.5,
        triplets_per_anchor="all",
        euc_dist_mtx=None,
        upper_norm_limit=3.0,
        mapping_table=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.mapping_table = mapping_table
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.upper_norm_limit = upper_norm_limit
        normalized_euc_dist_mtx = euc_dist_mtx.apply(self.normalize_row, axis=1)
        self.normalized_euc_dist_mtx = self.convert_dataframe_to_dict(normalized_euc_dist_mtx)

    def compute_loss(self,
                     embeddings: torch.Tensor,
                     labels: list,
                     indices_tuple: tuple,
                     ref_emb: torch.Tensor,
                     ref_labels: list) \
            -> dict:
        """
        Compute the triplet loss.

        Args:
            embeddings (torch.Tensor): Embeddings of the current batch.
            labels (list): Labels of the current batch.
            indices_tuple (tuple): Tuple containing anchor, positive, and negative indices.
            ref_emb (torch.Tensor): Reference embeddings.
            ref_labels (list): Reference labels.

        Returns:
            dict: Dictionary containing the computed loss and related information.
        """

        c_f.labels_or_indices_tuple_required(labels, indices_tuple)

        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
        )

        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()

        mat = self.distance(embeddings, ref_emb)    # Tensor(batch_size, batch_size)
        ap_dists = mat[anchor_idx, positive_idx]    # Tensor(x,)
        an_dists = mat[anchor_idx, negative_idx]    # Tensor(x,)

        normalized_margins = self.get_normalized_value(anchor_idx, negative_idx, labels)
        normalized_margins = self.margin * normalized_margins
        normalized_margins = normalized_margins.to("cuda")

        pos_neg_dists = self.distance.margin(ap_dists, an_dists)    # Tensor(x,)

        loss = normalized_margins + pos_neg_dists
        losses = torch.nn.functional.relu(loss)

        return {
            "loss": {
                "losses": losses,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def normalize_row(self, row):
        """

        Args:
            row:

        Returns:

        """

        non_diagonal_elements = row[row.index != row.name]  # Exclude diagonal elements
        min_distance = non_diagonal_elements.min()
        max_distance = non_diagonal_elements.max()
        normalized_row = 1 + ((self.upper_norm_limit - 1) * (row - min_distance)) / (max_distance - min_distance)

        return normalized_row

    @staticmethod
    def convert_dataframe_to_dict(df):
        """

        Args:
            df:

        Returns:

        """

        distance_dict = {}

        for index, row in df.iterrows():
            for col in df.columns:
                distance_dict[(index, col)] = row[col]

        return distance_dict

    def get_normalized_value(self,
                             anchor_file_idx: List[int],
                             negative_file_idx: List[int],
                             labels: List[int]) \
            -> torch.Tensor:
        """
        Normalize rows of the Euclidean distance matrix.

        Args:
            anchor_file_idx (List[int]): List of anchor file indices.
            negative_file_idx (List[int]): List of negative file indices.
            labels (List[int]): List of labels.

        Returns:
            torch.Tensor: Normalized rows.
        """

        # Initialize a tensor to store normalized distances
        normalized_distances = torch.zeros(len(anchor_file_idx))

        # Iterate over each pair of anchor and negative indices
        for i, (anchor_idx, negative_idx) in enumerate(zip(anchor_file_idx, negative_file_idx)):
            # Extract labels for anchor and negative samples
            anchor_label = labels[anchor_idx].item()
            negative_label = labels[negative_idx].item()

            for key, value in self.mapping_table.items():
                if value == anchor_label:
                    anchor_label = key

            for key, value in self.mapping_table.items():
                if value == negative_label:
                    negative_label = key

            normalized_distance = self.normalized_euc_dist_mtx[(anchor_label, negative_label)]
            normalized_distances[i] = normalized_distance

        return normalized_distances

    def get_default_reducer(self) -> AvgNonZeroReducer:
        """
        Get the default reducer.

        Returns:
            AvgNonZeroReducer: The default reducer.
        """

        return AvgNonZeroReducer()
