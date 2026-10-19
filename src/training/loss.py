"""
Loss functions tailored for Diabetic Retinopathy classification.
Includes Focal Loss and Class-Weighted Cross-Entropy to counteract No-DR dominance.
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples (majority No-DR healthy eyes) and focuses gradient on hard lesions.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([float(alpha)], dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_classes)
        targets: (B,) class indices
        """
        num_classes = logits.size(-1)
        # Compute log-probabilities and probabilities
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Gather target probabilities
        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_p = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - target_p) ** self.gamma
        focal_loss = -modulating_factor * target_log_p

        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_factor = self.alpha.gather(dim=0, index=targets)
            focal_loss = alpha_factor * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(
    loss_type: str = "focal",
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Factory helper to instantiate loss function.
    """
    lt = loss_type.lower().strip()
    if lt == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma, reduction="mean")
    elif lt in ["weighted_ce", "ce", "cross_entropy"]:
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'. Choose 'focal' or 'weighted_ce'.")
