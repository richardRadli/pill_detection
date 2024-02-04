import torch
import torch.nn as nn


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        distance_positive = torch.pairwise_distance(anchor_embedding, positive_embedding, p=2)
        distance_negative = torch.pairwise_distance(anchor_embedding, negative_embedding, p=2)
        loss = torch.relu(distance_positive - distance_negative + self.margin)

        return torch.mean(loss)
