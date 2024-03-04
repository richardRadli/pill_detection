import numpy as np
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
        margin=0.05,
        triplets_per_anchor="all",
        euc_dist_mtx=None,
        upper_norm_limit=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.euc_dist_mtx = euc_dist_mtx
        self.upper_norm_limit = upper_norm_limit

    def compute_loss(self, embeddings: torch.Tensor, labels: list, indices_tuple: tuple,
                     ref_emb: torch.Tensor, ref_labels: list) -> dict:
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

        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]

        normalized_margins = self.normalize_row(anchor_idx, negative_idx, labels)
        normalized_margins = normalized_margins.to("cuda")
        normalized_margins = normalized_margins * self.margin
        pos_neg_dists = self.distance.margin(ap_dists, an_dists)

        loss = normalized_margins + pos_neg_dists
        losses = torch.nn.functional.relu(loss)

        return {
            "loss": {
                "losses": losses,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def normalize_row(self, anchor_file_idx: List[int], negative_file_idx: List[int], labels: List[int]) \
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

        # Convert tensor labels to integers
        labels = [int(label) for label in labels]

        # Create a dictionary to map labels to indices in the normalize_row function
        label_to_index = {label: idx for idx, label in enumerate(labels)}

        # Handle the case where labels may not start from 0 or may not be continuous
        max_label = max(label_to_index.keys())
        for label in range(max_label + 1):
            label_to_index.setdefault(label, len(label_to_index))

        anchor_file_idx_cpu = [label_to_index[int(idx)] for idx in anchor_file_idx]
        negative_file_idx_cpu = [label_to_index[int(idx)] for idx in negative_file_idx]
        rows = self.euc_dist_mtx.iloc[anchor_file_idx_cpu].to_numpy()
        min_vals = rows.min(axis=1)
        max_vals = rows.max(axis=1)
        normalized_rows = 1 + (self.upper_norm_limit - 1) * (
                (rows - min_vals[:, np.newaxis]) / (max_vals - min_vals)[:, np.newaxis])
        return torch.tensor(normalized_rows[:, negative_file_idx_cpu])

    def get_default_reducer(self) -> AvgNonZeroReducer:
        """
        Get the default reducer.

        Returns:
            AvgNonZeroReducer: The default reducer.
        """

        return AvgNonZeroReducer()
