from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class RCE_L1Loss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, lambda_reg: float = 1.0, ddp: bool = True):
        """
        RCE with L1 region-size regularization.
        Args:
            apply_nonlin: optional non-linearity (e.g. softmax)
            lambda_reg: weighting for the L1 regularization term
            ddp: whether to use DDP-aware aggregation
        """
        super(RCE_L1Loss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.lambda_reg = lambda_reg
        self.ddp = ddp
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, loss_mask=None):
        """
        Args:
            logits: shape (B, C, ...)
            target: shape (B, ...) with int labels
            loss_mask: optional mask of valid voxels, shape (B, 1, ...)
        Returns:
            loss: scalar tensor
        """
        # Fix: remove extra channel dim if present
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)  # from [B, 1, D, H, W] to [B, D, H, W]

        # Apply softmax
        if self.apply_nonlin is not None:
            probs = self.apply_nonlin(logits)
        else:
            probs = F.softmax(logits, dim=1)

        # Standard Cross-Entropy loss
        target = target.long()
        ce_loss = self.ce(logits, target)

        B, C = probs.shape[:2]
        pred_flat = probs.view(B, C, -1)
        target_flat = target.view(B, -1)
        target_flat = target_flat.long()  # Convert to long

        # Target region proportions (ŷk): one-hot then normalize per batch
        with torch.no_grad():
            y_onehot = torch.zeros_like(pred_flat)
            y_onehot.scatter_(1, target_flat.unsqueeze(1), 1)
            if loss_mask is not None:
                loss_mask_flat = loss_mask.view(B, 1, -1)
                y_onehot *= loss_mask_flat
            region_target = y_onehot.sum(dim=2)
            region_target = region_target / (region_target.sum(dim=1, keepdim=True).clamp(min=1e-6))

        # Predicted region proportions (p̂k)
        if loss_mask is not None:
            loss_mask_flat = loss_mask.view(B, 1, -1)
            pred_flat = pred_flat * loss_mask_flat
        region_pred = pred_flat.sum(dim=2)
        region_pred = region_pred / (region_pred.sum(dim=1, keepdim=True).clamp(min=1e-6))

        # Optional DDP aggregation
        if self.ddp:
            region_pred = AllGatherGrad.apply(region_pred).sum(0)
            region_target = AllGatherGrad.apply(region_target).sum(0)

        # L1 penalty on region-size distributions
        l1_penalty = torch.abs(region_pred - region_target).mean()

        return ce_loss + self.lambda_reg * l1_penalty

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    pred = torch.rand((2, 3, 32, 32, 32), requires_grad=True)
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    loss_fn = RCE_L1Loss(apply_nonlin=softmax_helper_dim1, lambda_reg=1.0, ddp=False)
    loss = loss_fn(pred, ref)
    print(f"Loss: {loss.item()}")
    loss.backward()
