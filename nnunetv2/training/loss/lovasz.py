from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    gtsum = gt_sorted.sum()
    intersection = gtsum - gt_sorted.cumsum(0)
    union = gtsum + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    grad = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
    return grad


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor):
    """
    probs: [C, N]
    labels: [N]
    """
    C = probs.shape[0]
    losses = []
    for c in range(C):
        fg = (labels == c).float().view(-1)       # [N]
        prob = probs[c].view(-1)                  # [N]
        if fg.sum() == 0:
            continue
        errors = (fg - prob).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append((errors_sorted * grad).sum())
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=probs.device)


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None):
        super().__init__()
        self.apply_nonlin = apply_nonlin

    def forward(self, logits: torch.Tensor, target: torch.Tensor, loss_mask=None):
        """
        logits: [B, C, D, H, W]
        target: [B, D, H, W]
        """
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)  # [B, D, H, W]

        if self.apply_nonlin is not None:
            logits = self.apply_nonlin(logits)

        probs = F.softmax(logits, dim=1)           # [B, C, D, H, W]
        B, C = probs.shape[:2]
        losses = []

        for i in range(B):
            prob_i = probs[i].contiguous().view(C, -1)     # [C, N]
            target_i = target[i].contiguous().view(-1)     # [N]
            losses.append(lovasz_softmax_flat(prob_i, target_i))

        return torch.mean(torch.stack(losses))

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    pred = torch.rand((2, 3, 32, 32, 32), requires_grad=True)
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    loss_fn = LovaszSoftmaxLoss(apply_nonlin=softmax_helper_dim1)
    loss = loss_fn(pred, ref)
    print(f"Loss: {loss.item()}")
    loss.backward()
