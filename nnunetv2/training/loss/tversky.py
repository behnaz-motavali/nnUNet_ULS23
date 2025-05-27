from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-5,
                 apply_nonlin: Callable = None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """
        logits: [B, C, D, H, W]
        target: [B, D, H, W] or [B, 1, D, H, W]
        """
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        target = target.long()

        if self.apply_nonlin is not None:
            probs = self.apply_nonlin(logits)
        else:
            probs = F.softmax(logits, dim=1)

        B, C = probs.shape[:2]
        probs_flat = probs.view(B, C, -1)
        target_flat = target.view(B, -1)  # [B, N]

        one_hot = torch.zeros_like(probs_flat).scatter_(1, target_flat.unsqueeze(1), 1)

        if loss_mask is not None:
            if loss_mask.ndim == 5 and loss_mask.shape[1] == 1:
                loss_mask = loss_mask.squeeze(1)
            loss_mask_flat = loss_mask.view(B, 1, -1)
            one_hot = one_hot * loss_mask_flat
            probs_flat = probs_flat * loss_mask_flat

        tp = (probs_flat * one_hot).sum(dim=2)
        fp = (probs_flat * (1 - one_hot)).sum(dim=2)
        fn = ((1 - probs_flat) * one_hot).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky.mean()

        return loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.75, smooth: float = 1e-6,
                 apply_nonlin: Callable = None):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """
        logits: [B, C, D, H, W]
        target: [B, D, H, W] or [B, 1, D, H, W]
        """
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        target = target.long()

        if self.apply_nonlin is not None:
            probs = self.apply_nonlin(logits)
        else:
            probs = F.softmax(logits, dim=1)

        B, C = probs.shape[:2]
        probs_flat = probs.view(B, C, -1)
        target_flat = target.view(B, -1)  # [B, N]

        one_hot = torch.zeros_like(probs_flat).scatter_(1, target_flat.unsqueeze(1), 1)

        if loss_mask is not None:
            if loss_mask.ndim == 5 and loss_mask.shape[1] == 1:
                loss_mask = loss_mask.squeeze(1)
            loss_mask_flat = loss_mask.view(B, 1, -1)
            one_hot = one_hot * loss_mask_flat
            probs_flat = probs_flat * loss_mask_flat

        tp = (probs_flat * one_hot).sum(dim=2)
        fp = (probs_flat * (1 - one_hot)).sum(dim=2)
        fn = ((1 - probs_flat) * one_hot).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1.0 - tversky) ** self.gamma
        loss = focal_tversky.mean()

        return loss

class LogCoshDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, apply_nonlin: Callable = None):
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        """
        logits: [B, C, D, H, W]
        target: [B, D, H, W] or [B, 1, D, H, W]
        """
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        target = target.long()

        if self.apply_nonlin is not None:
            probs = self.apply_nonlin(logits)
        else:
            probs = F.softmax(logits, dim=1)

        B, C = probs.shape[:2]
        probs_flat = probs.view(B, C, -1)
        target_flat = target.view(B, -1)

        one_hot = torch.zeros_like(probs_flat).scatter_(1, target_flat.unsqueeze(1), 1)

        if loss_mask is not None:
            if loss_mask.ndim == 5 and loss_mask.shape[1] == 1:
                loss_mask = loss_mask.squeeze(1)
            loss_mask_flat = loss_mask.view(B, 1, -1)
            one_hot = one_hot * loss_mask_flat
            probs_flat = probs_flat * loss_mask_flat

        intersection = (probs_flat * one_hot).sum(dim=2)
        union = probs_flat.sum(dim=2) + one_hot.sum(dim=2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1. - dice

        log_cosh_loss = torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss)) / 2)
        return log_cosh_loss.mean()    
 

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    torch.manual_seed(0)
    pred = torch.rand((2, 3, 32, 32, 32), requires_grad=True)
    ref = torch.randint(0, 3, (2, 32, 32, 32))
    mask = torch.ones((2, 1, 32, 32, 32))

    loss_fn = TverskyLoss(apply_nonlin=softmax_helper_dim1, alpha=0.7, beta=0.3)
    loss = loss_fn(pred, ref, loss_mask=mask)
    print(f"Tversky Loss: {loss.item():.4f}")
    loss.backward()    