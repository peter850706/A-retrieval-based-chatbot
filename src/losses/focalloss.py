import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, input, target):
        """
        Args:
            input (tensor): (batch, n_samples), have been normalized by sigmoid function
            output (tensor): (batch, n_samples)
        """
        p_t = torch.where(target == 1, input, 1 - input)
        alpha = torch.full_like(input, self.alpha)
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        return (-alpha_t * (1 - p_t) ** self.gamma * p_t.log10()).mean()