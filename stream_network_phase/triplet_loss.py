import torch
import torch.nn.functional as func


class TripletLossWithHardMining(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLossWithHardMining, self).__init__()
        self.margin = margin
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hardest_positive_indices = None
        self.hardest_negative_indices = None
        self.hardest_anchor_indices = None

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        d_ap = func.pairwise_distance(anchor_embeddings, positive_embeddings, 2)
        d_an, self.hardest_negative_indices = self.hardest_negative(anchor_embeddings, positive_embeddings,
                                                                    negative_embeddings)
        d_pp, self.hardest_positive_indices = self.hardest_positive(anchor_embeddings, positive_embeddings,
                                                                    negative_embeddings)
        d_aa, self.hardest_anchor_indices = self.hardest_anchor(anchor_embeddings, positive_embeddings,
                                                                negative_embeddings)
        loss = torch.mean(torch.clamp(d_ap - self.margin * d_pp - self.margin * d_an + self.margin, min=0))

        return loss

    def hardest_negative(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        d_an = []
        hardest_negative_indices = []
        for i, (anchor_embedding, positive_embedding) in enumerate(zip(anchor_embeddings, positive_embeddings)):
            dists = func.pairwise_distance(anchor_embedding.unsqueeze(0), negative_embeddings, 2)
            dists = dists.to(self.device)
            if (dists < func.pairwise_distance(anchor_embedding.unsqueeze(0), positive_embedding.unsqueeze(0),
                                               2)).sum() == 0:
                d_an.append(torch.tensor(0.0).to(self.device))
                hardest_negative_indices.append(i)
            else:
                hard_neg_idx = torch.argmax(dists[dists < func.pairwise_distance(anchor_embedding.unsqueeze(0),
                                                                                 positive_embedding.unsqueeze(0), 2)])
                d_an.append(dists[hard_neg_idx].to(self.device))
                hardest_negative_indices.append(hard_neg_idx.item())

        return torch.mean(torch.stack(d_an)), hardest_negative_indices

    def hardest_positive(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        d_pp = []
        hardest_positive_indices = []
        for i, (anchor_embedding, negative_embedding) in enumerate(zip(anchor_embeddings, negative_embeddings)):
            dists = func.pairwise_distance(anchor_embedding.unsqueeze(0), positive_embeddings, 2)
            dists = dists.to(self.device)
            if (dists > func.pairwise_distance(anchor_embedding.unsqueeze(0), negative_embedding.unsqueeze(0),
                                               2)).sum() == 0:
                d_pp.append(torch.tensor(0.0).to(self.device))
                hardest_positive_indices.append(i)
            else:
                hard_pos_idx = torch.argmin(dists[dists > func.pairwise_distance(anchor_embedding.unsqueeze(0),
                                                                                 negative_embedding.unsqueeze(0), 2)])
                d_pp.append(dists[hard_pos_idx].to(self.device))
                hardest_positive_indices.append(hard_pos_idx.item())

        return torch.mean(torch.stack(d_pp)), hardest_positive_indices

    def hardest_anchor(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        d_aa = []
        hardest_anchor_indices = []
        for i, (positive_embedding, negative_embedding) in enumerate(zip(positive_embeddings, negative_embeddings)):
            dists = func.pairwise_distance(anchor_embeddings, positive_embedding.unsqueeze(0), 2) \
                  - func.pairwise_distance(anchor_embeddings, negative_embedding.unsqueeze(0), 2)
            dists = dists.to(self.device)
            hard_anchor_idx = torch.argmax(dists)
            d_aa.append(dists[hard_anchor_idx].to(self.device))
            hardest_anchor_indices.append(hard_anchor_idx.item())

        return torch.mean(torch.stack(d_aa)), hardest_anchor_indices
