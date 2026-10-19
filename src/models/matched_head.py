"""
Matched Control Dense Head for Diabetic Retinopathy Screening.
Mandatory Ablation Control:
Replaces the quantum circuit with a classical 4-neuron bottleneck (1280 -> 4 -> 5).
Exact Parameter Counts:
W1: 4 * 1280 + 4 = 5,124 params
W2: 5 * 4 + 5 = 25 params
Total parameters = 5,149 classical parameters.
"""

import torch
import torch.nn as nn


class MatchedDenseHead(nn.Module):
    """
    Matched classical control head mirroring the dimensional bottleneck of the quantum circuit.
    Uses tanh activation to bound intermediate activations in [-1, 1],
    strictly matching the expectation range of Pauli-Z measurements.
    """

    def __init__(self, in_features: int = 1280, bottleneck_dim: int = 4, num_classes: int = 5):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes

        # W1 + b1: 1280 -> 4 (5,124 params)
        self.fc1 = nn.Linear(in_features, bottleneck_dim, bias=True)
        self.activation = nn.Tanh()

        # W2 + b2: 4 -> 5 (25 params)
        self.fc2 = nn.Linear(bottleneck_dim, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 1280)
        Intermediate: (B, 4) in [-1, 1]
        Output: Logits (B, 5)
        """
        z = self.activation(self.fc1(x))
        logits = self.fc2(z)
        return logits
