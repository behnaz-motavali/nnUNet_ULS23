from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, apply_nonlin: Callable = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.apply_nonlin = apply_nonlin
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=self.alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)  # Convert [B, 1, D, H, W] to [B, D, H, W]

        target = target.long()  # Ensure proper dtype

        assert logits.shape[0] == target.shape[0], f"Batch size mismatch: {logits.shape[0]} vs {target.shape[0]}"
        assert logits.shape[2:] == target.shape[1:], f"Spatial shape mismatch: {logits.shape} vs {target.shape}"
        assert target.max().item() < logits.shape[1], f"Target has class index {target.max().item()} >= num_classes {logits.shape[1]}"

        if self.apply_nonlin is not None:
            probs = self.apply_nonlin(logits)
        else:
            probs = F.softmax(logits, dim=1)

        ce_loss = self.ce(logits, target)  # [B, D, H, W]
        pt = torch.gather(probs, 1, target.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if loss_mask is not None:
            if loss_mask.ndim == 5 and loss_mask.shape[1] == 1:
                loss_mask = loss_mask.squeeze(1)
            loss_mask = loss_mask.float()
            assert loss_mask.shape == loss.shape, f"loss_mask shape {loss_mask.shape} != loss shape {loss.shape}"
            loss = (loss * loss_mask).sum() / loss_mask.sum()
        else:
            loss = loss.mean()

        return loss


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    torch.manual_seed(0)
    pred = torch.rand((2, 3, 32, 32, 32), requires_grad=True)
    ref = torch.randint(0, 3, (2, 32, 32, 32))
    mask = torch.ones((2, 1, 32, 32, 32))

    loss_fn = FocalLoss(apply_nonlin=softmax_helper_dim1, gamma=2.0, alpha=None)
    loss = loss_fn(pred, ref, loss_mask=mask)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()