import torch
import torch.nn.functional as F

class TripletLossWithHardMining(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLossWithHardMining, self).__init__()
        self.margin = margin
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):

        d_ap = F.pairwise_distance(anchor_embeddings, positive_embeddings, 2)
        d_an = self.hardest_negative(anchor_embeddings, positive_embeddings, negative_embeddings)
        d_pp = self.hardest_positive(anchor_embeddings, positive_embeddings, negative_embeddings)

        loss = torch.mean(torch.clamp(d_ap - 0.5 * d_pp - 0.5 * d_an + self.margin, min=0))

        return loss

    def hardest_negative(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        d_an = []
        for anchor_embedding, positive_embedding in zip(anchor_embeddings, positive_embeddings):
            dists = F.pairwise_distance(anchor_embedding.unsqueeze(0), negative_embeddings, 2)
            dists = dists.to(self.device)
            if (dists < F.pairwise_distance(anchor_embedding.unsqueeze(0), positive_embedding.unsqueeze(0),
                                            2)).sum() == 0:
                # No negative examples farther from the anchor than the positive example
                d_an.append(torch.tensor(0.0).to(self.device))
            else:
                d_an.append(torch.max(dists[dists < F.pairwise_distance(anchor_embedding.unsqueeze(0),
                                                                        positive_embedding.unsqueeze(0), 2)]).to(
                    self.device))

        return torch.mean(torch.stack(d_an)) if len(d_an) > 0 else torch.tensor(0.0)

    def hardest_positive(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        d_pp = []
        for anchor_embedding, negative_embedding in zip(anchor_embeddings, negative_embeddings):
            dists = F.pairwise_distance(anchor_embedding.unsqueeze(0), positive_embeddings, 2)
            dists = dists.to(self.device)
            if (dists > F.pairwise_distance(anchor_embedding.unsqueeze(0), negative_embedding.unsqueeze(0),
                                            2)).sum() == 0:
                # No positive examples closer to the anchor than the negative example
                d_pp.append(torch.tensor(0.0).to(self.device))
            else:
                d_pp.append(torch.min(dists[dists > F.pairwise_distance(anchor_embedding.unsqueeze(0),
                                                                        negative_embedding.unsqueeze(0), 2)]).to(
                    self.device))

        return torch.mean(torch.stack(d_pp)) if len(d_pp) > 0 else torch.tensor(0.0)
