from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, ignore_label=None):
        super(CrossEntropyLossWrapper, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):
        # x: (b, c, ...) - raw logits
        # y: (b, ...) - ground truth labels
        # loss_mask: (b, 1, ...) - optional binary mask

        # Flatten predictions and targets if needed
        if loss_mask is not None:
            # Apply loss mask
            loss_mask = loss_mask[:, 0].bool()  # shape: (b, ...)
            x = x.permute(0, *range(2, x.ndim), 1).contiguous()  # (b, ..., c)
            x = x[loss_mask]  # (n, c)
            y = y[loss_mask]  # (n,)
        else:
            x = x.permute(0, *range(2, x.ndim), 1).contiguous().view(-1, x.shape[1])
            y = y.view(-1)

        y = y.long()  # Ensure correct dtype

        if self.ignore_label is not None:
            valid_mask = y != self.ignore_label
            x = x[valid_mask]
            y = y[valid_mask]

        ce_loss = F.cross_entropy(x, y)
        return ce_loss