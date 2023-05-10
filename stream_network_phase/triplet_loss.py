import torch
import torch.nn.functional as F


class TripletLossWithHardMining(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLossWithHardMining, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute the Euclidean distances between the embeddings
        dist_pos = F.pairwise_distance(anchor, positive, 2)
        dist_neg = F.pairwise_distance(anchor, negative, 2)

        # Select the hardest negative sample for each anchor
        hard_neg = []
        for i in range(dist_pos.size(0)):
            # Get the indices of the negative samples that violate the margin
            dist_diff = dist_pos[i] - dist_neg[i] + self.margin
            candidate_idxs = torch.where(dist_diff > 0)[0]
            if len(candidate_idxs) == 0:
                # If there are no negative samples that violate the margin, skip
                hard_neg.append(None)
                continue
            # Select the hardest negative sample (i.e., the one with the largest distance to the anchor)
            if len(candidate_idxs) == 1:
                hard_neg_idx = candidate_idxs[0]
            else:
                hard_neg_idx = torch.argmax(dist_neg[i, candidate_idxs])
            hard_neg.append(negative[candidate_idxs[hard_neg_idx]].unsqueeze(0))
        # hard_neg = torch.cat([x for x in hard_neg if x is not None], dim=0)
        hard_neg = [x for x in hard_neg if x is not None and x.shape[0] != 0]
        if hard_neg:
            hard_neg = torch.cat(hard_neg, dim=0)
        else:
            hard_neg = torch.tensor([])

        # Select the hardest positive sample for each anchor
        hard_pos = []
        for i in range(dist_pos.size(0)):
            # Get the indices of the positive samples that violate the margin
            dist_diff = dist_neg[i] - dist_pos[i] + self.margin
            candidate_idxs = torch.where(dist_diff > 0)[0]
            if len(candidate_idxs) == 0:
                # If there are no positive samples that violate the margin, skip
                hard_pos.append(None)
            else:
                # Select the hardest positive sample (i.e., the one with the smallest distance to the anchor)
                if len(candidate_idxs) == 1:
                    hard_pos_idx = candidate_idxs[0]
                else:
                    hard_pos_idx = torch.argmin(dist_pos[i, candidate_idxs])
                hard_pos.append(positive[candidate_idxs[hard_pos_idx]].unsqueeze(0))

        # Combine the hardest negative and hardest positive samples into a single tensor
        hard_pos = torch.cat([x for x in hard_pos if x is not None], dim=0)

        # Compute the loss
        loss = F.relu(self.margin + dist_pos - dist_neg).mean()

        return loss, hard_neg, hard_pos
