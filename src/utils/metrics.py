
import torch
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Can be a list/tensor of weights per class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                self.alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            alpha_factor = self.alpha[targets]
            focal_loss = alpha_factor * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss