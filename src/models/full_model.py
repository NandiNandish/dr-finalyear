"""
Unified PyTorch Model for Diabetic Retinopathy Screening.
Combines frozen backbone (EfficientNet-B0) with switchable interchangeable heads:
- 'classical' (Dropout + Linear 1280->5, 6,405 params)
- 'hybrid_quantum' (Linear 1280->4 + 4-qubit VQC + Linear 4->5, 5,149 + 4L params)
- 'matched_dense' (Linear 1280->4 + Tanh + Linear 4->5, 5,149 params)
"""

from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_backbone, FundusBackbone
from .classical_head import ClassicalHead
from .matched_head import MatchedDenseHead
from .quantum_head import HybridQuantumHead


class DRScreeningModel(nn.Module):
    """
    End-to-End Diabetic Retinopathy screening model with frozen backbone
    and switchable head architectures.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        active_head: str = "hybrid_quantum",
        num_classes: int = 5,
        num_qubits: int = 4,
        variational_layers: int = 3,
        quantum_backend: str = "default.qubit",
        diff_method: str = "parameter-shift",
        noise_model: Optional[str] = None,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.active_head_type = active_head.lower().strip()
        self.num_classes = num_classes

        # 1. Frozen Feature Extractor
        self.backbone: FundusBackbone = get_backbone(
            name=backbone_name,
            pretrained=True,
            freeze=freeze_backbone,
        )
        self.feature_dim = self.backbone.feature_dim

        # 2. Instantiate interchangeable heads
        self.classical_head = ClassicalHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.matched_dense_head = MatchedDenseHead(
            in_features=self.feature_dim,
            bottleneck_dim=num_qubits,
            num_classes=num_classes,
        )

        self.hybrid_quantum_head = HybridQuantumHead(
            in_features=self.feature_dim,
            num_qubits=num_qubits,
            num_layers=variational_layers,
            num_classes=num_classes,
            backend=quantum_backend,
            diff_method=diff_method,
            noise_model=noise_model,
        )

        # Set default active head
        self.set_active_head(self.active_head_type)

    def set_active_head(self, head_name: str):
        """
        Switches active head: 'classical' | 'hybrid_quantum' | 'matched_dense'.
        """
        h_clean = head_name.lower().strip()
        if h_clean in ["classical", "standard"]:
            self.active_head_type = "classical"
            self.active_head = self.classical_head
        elif h_clean in ["hybrid_quantum", "quantum", "vqc"]:
            self.active_head_type = "hybrid_quantum"
            self.active_head = self.hybrid_quantum_head
        elif h_clean in ["matched_dense", "matched", "control"]:
            self.active_head_type = "matched_dense"
            self.active_head = self.matched_dense_head
        else:
            raise ValueError(f"Unknown head type: '{head_name}'. Choose 'classical', 'hybrid_quantum', or 'matched_dense'.")

    def forward_from_features(self, f: torch.Tensor) -> torch.Tensor:
        """
        Evaluates active head directly on pooled features: (B, feature_dim) -> (B, num_classes).
        """
        return self.active_head(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass:
        - If input is 4D image tensor (B, 3, H, W): passes through backbone, then active head.
        - If input is 2D cached feature tensor (B, 1280): passes directly to active head.
        """
        if x.ndim == 4:
            f = self.backbone(x)
        elif x.ndim == 2:
            f = x
        else:
            raise ValueError(f"Expected 2D features (B, D) or 4D image tensor (B, C, H, W), got shape: {x.shape}")

        logits = self.forward_from_features(f)
        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (predicted_class_index, probabilities, confidence).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
        return pred, probs, conf
