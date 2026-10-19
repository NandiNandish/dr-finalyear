"""
Classical Classification Head for Diabetic Retinopathy Screening.
Direct projection: 1280 -> 5 with Dropout.
Parameter count: 1280 * 5 + 5 = 6,405 parameters.
"""

import torch
import torch.nn as nn


class ClassicalHead(nn.Module):
    """
    Standard linear classification head on 1280-d frozen features.
    """

    def __init__(self, in_features: int = 1280, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 1280)
        Output: Logits (B, 5)
        """
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
