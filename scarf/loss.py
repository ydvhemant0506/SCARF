import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NTXent(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss
    

    
'''Improved InfoNCE loss function ass : FairNTXent'''
class FairNTXent(nn.Module):
    def __init__(self, temperature: float = 1.0, lambda_reg: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg

    def forward(self, z_i: Tensor, z_j: Tensor, sensitive_features: Tensor) -> Tensor:
        batch_size = z_i.size(0)

        # Standard NT-Xent contrastive loss
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        positives = torch.cat([torch.diag(similarity, batch_size), torch.diag(similarity, -batch_size)], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        contrastive_loss = -torch.log(numerator / torch.sum(denominator, dim=1))
        contrastive_loss = torch.sum(contrastive_loss) / (2 * batch_size)

        # Fairness penalty
        fairness_penalty = 0.0
        for value in sensitive_features.unique():
            indices = (sensitive_features == value).nonzero(as_tuple=True)[0]
            if len(indices) > 1:
                sub_similarity = F.cosine_similarity(z[indices].unsqueeze(1), z[indices].unsqueeze(0), dim=2)
                fairness_penalty += torch.sum(sub_similarity) / (len(indices) * (len(indices) - 1))

        fairness_penalty /= sensitive_features.unique().size(0)
        fairness_loss = self.lambda_reg * fairness_penalty

        # Total loss
        total_loss = contrastive_loss + fairness_loss
        return total_loss

