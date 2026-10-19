"""
Feature Extraction Backbones for Fundus Photographs.

Primary: EfficientNet-B0 (ImageNet pre-trained, FROZEN), pooled 1280-d feature vector.
Baselines: ResNet50 (2048-d pooled) and MobileNetV3 / Xception compatible.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class FundusBackbone(nn.Module):
    """
    Frozen feature extractor producing pooled 1D embeddings.
    Also retains references to last convolutional layer for Grad-CAM.
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.architecture = architecture.lower().strip()
        self.pretrained = pretrained
        self.freeze = freeze

        if self.architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            self.features = net.features
            self.target_conv_layer = net.features[-1]  # Last conv block for Grad-CAM
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = 1280

        elif self.architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            # Remove fc and avgpool
            self.conv1 = net.conv1
            self.bn1 = net.bn1
            self.relu = net.relu
            self.maxpool = net.maxpool
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.target_conv_layer = net.layer4[-1]
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = 2048
            # Group into sequential feature extractor
            self.features = nn.Sequential(
                self.conv1, self.bn1, self.relu, self.maxpool,
                self.layer1, self.layer2, self.layer3, self.layer4
            )

        else:
            raise ValueError(f"Unsupported backbone architecture: '{architecture}'. Choose 'efficientnet_b0' or 'resnet50'.")

        # Freeze all parameters if requested
        if self.freeze:
            self.freeze_parameters()

    def freeze_parameters(self):
        """
        Locks all backbone parameters so gradients are never computed.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns spatial feature map (B, C, H', W') from last convolutional layer.
        Used by Grad-CAM explainability module.
        """
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled 1D feature vector: (B, 3, H, W) -> (B, feature_dim).
        """
        spatial_maps = self.forward_features(x)
        pooled = self.pool(spatial_maps)
        flattened = torch.flatten(pooled, 1)
        return flattened


def get_backbone(
    name: str = "efficientnet_b0",
    pretrained: bool = True,
    freeze: bool = True,
) -> FundusBackbone:
    return FundusBackbone(architecture=name, pretrained=pretrained, freeze=freeze)
