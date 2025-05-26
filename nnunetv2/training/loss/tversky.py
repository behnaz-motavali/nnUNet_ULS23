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

        assert logits.shape[0] == target.shape[0], f"Batch size mismatch: {logits.shape[0]} vs {target.shape[0]}"
        assert logits.shape[2:] == target.shape[1:], f"Spatial shape mismatch: {logits.shape} vs {target.shape}"
        assert target.max().item() < logits.shape[1], f"Target has class index {target.max().item()} >= num_classes {logits.shape[1]}"

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


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, apply_nonlin=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, logits, targets):
        if self.apply_nonlin is not None:
            logits = self.apply_nonlin(logits)

        probs = logits
        targets = F.one_hot(targets.long(), num_classes=probs.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3).float()

        TP = (probs * targets).sum(dim=(2, 3, 4))
        FP = (probs * (1 - targets)).sum(dim=(2, 3, 4))
        FN = ((1 - probs) * targets).sum(dim=(2, 3, 4))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - tversky
        return loss.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.75, smooth=1e-6, apply_nonlin=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, logits, targets):
        if self.apply_nonlin is not None:
            logits = self.apply_nonlin(logits)

        probs = logits
        targets = F.one_hot(targets.long(), num_classes=probs.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3).float()

        TP = (probs * targets).sum(dim=(2, 3, 4))
        FP = (probs * (1 - targets)).sum(dim=(2, 3, 4))
        FN = ((1 - probs) * targets).sum(dim=(2, 3, 4))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = (1 - tversky) ** self.gamma
        return loss.mean() 

class LogCoshDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, apply_nonlin=None):
        super().__init__()
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, logits, targets):
        if self.apply_nonlin is not None:
            logits = self.apply_nonlin(logits)

        probs = logits
        targets = F.one_hot(targets.long(), num_classes=probs.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3).float()

        intersection = (probs * targets).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice

        return torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss)) / 2).mean()    