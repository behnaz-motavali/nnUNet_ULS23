import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss


class ReverseCrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, student_logits: torch.Tensor, teacher_soft_targets: torch.Tensor) -> torch.Tensor:
        student_probs = F.softmax(student_logits, dim=1)
        teacher_soft_targets = torch.clamp(teacher_soft_targets, self.eps, 1.0)
        # sum over classes, mean over batch
        loss = -torch.sum(student_probs * torch.log(teacher_soft_targets), dim=1).mean()
        return loss


class ReverseKLLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, student_logits: torch.Tensor, teacher_soft_targets: torch.Tensor) -> torch.Tensor:
        student_probs = F.softmax(student_logits, dim=1)
        student_probs = torch.clamp(student_probs, self.eps, 1.0)
        teacher_soft_targets = torch.clamp(teacher_soft_targets, self.eps, 1.0)
        # Reverse KL: sum over classes, mean over batch
        loss = torch.sum(teacher_soft_targets * (torch.log(teacher_soft_targets) - torch.log(student_probs)), dim=1).mean()
        return loss


class DiceReverseCELoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_rce=1.0, dice_args=None):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_rce = weight_rce
        self.dice = SoftDiceLoss(**(dice_args if dice_args is not None else {}))
        self.rce = ReverseCrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        weak_soft_labels: torch.Tensor,
        ground_truth: torch.Tensor = None,
        ) -> torch.Tensor:
        device = student_logits.device
        loss = torch.tensor(0.0, device=device)

        if ground_truth is not None:
            loss += self.weight_dice * self.dice(student_logits, ground_truth)

        loss += self.weight_rce * self.rce(student_logits, weak_soft_labels)
        return loss

class DiceReverseKLRCE(nn.Module):
    def __init__(
        self,
        weight_dice=1.0,
        weight_rce=1.0,
        weight_rkl=1.0,
        dice_args=None,
        eps=1e-8,
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_rce = weight_rce
        self.weight_rkl = weight_rkl

        self.dice = SoftDiceLoss(**(dice_args if dice_args is not None else {}))
        self.rce = ReverseCrossEntropyLoss(eps)
        self.rkl = ReverseKLLoss(eps)

    def forward(
        self,
        student_logits: torch.Tensor,
        weak_soft_labels: torch.Tensor,
        ground_truth: torch.Tensor = None,
    ) -> torch.Tensor:
        device = student_logits.device
        loss = torch.tensor(0.0, device=device)

        if ground_truth is not None:
            loss += self.weight_dice * self.dice(student_logits, ground_truth)

        loss += self.weight_rce * self.rce(student_logits, weak_soft_labels)
        loss += self.weight_rkl * self.rkl(student_logits, weak_soft_labels)

        return loss
