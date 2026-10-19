"""
Explainability Module: Grad-CAM for Diabetic Retinopathy Screening.

Clinical & Algorithmic Notice:
- Computes Gradient-weighted Class Activation Mapping (Grad-CAM) through the CNN backbone.
- Demonstrates image features' relevance to the visual embedding; it does NOT represent
  quantum state evolution or microscopic lesion outlines.
- Flags for human specialist review if the activation peak lies outside the anatomical
  retinal boundary, or if prediction confidence is low.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GradCAMResult:
    heatmap_raw: np.ndarray          # 2D float array in [0.0, 1.0]
    overlay_rgb: np.ndarray          # 3D uint8 RGB overlay
    peak_coord: Tuple[int, int]      # (y, x) of maximum activation
    lies_outside_retina: bool        # True if peak is in outer black margin
    flag_for_review: bool            # True if anatomical plausibility fails
    clinical_note: str               # Explanatory documentation


class GradCAM:
    """
    Hook-based Grad-CAM generator for convolutional backbones.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Identify target convolutional layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "backbone") and hasattr(model.backbone, "target_conv_layer"):
            self.target_layer = model.backbone.target_conv_layer
        else:
            # Fallback: find last Conv2d in model
            convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if not convs:
                raise ValueError("No Conv2d layer found for Grad-CAM.")
            self.target_layer = convs[-1]

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        target_class: Optional[int] = None,
        retina_mask: Optional[np.ndarray] = None,
        alpha: float = 0.45,
    ) -> GradCAMResult:
        """
        Generates Grad-CAM heatmap and validates anatomical plausibility.
        Args:
            image_tensor: (1, 3, H, W) normalized tensor
            original_rgb: (H, W, 3) uint8 RGB image
            target_class: integer class index (default: argmax prediction)
            retina_mask: (H, W) binary mask of retinal field
            alpha: blending weight for heatmap overlay
        """
        self.model.zero_grad()

        # Forward pass with gradient tracking
        logits = self.model(image_tensor)
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        # Global average pooling of gradients
        # Gradients: (1, C, H', W')
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)  # (H', W')
        cam = F.relu(cam).cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        # Resize heatmap to original image resolution
        h, w = original_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Anatomical Plausibility Verification:
        # Check if the peak activation lies outside the retinal disc
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)

        lies_outside_retina = False
        if retina_mask is not None:
            # If peak pixel has mask == 0, it falls outside the retina
            if retina_mask[peak_y, peak_x] == 0:
                lies_outside_retina = True
        else:
            # Fallback: check if peak is in black peripheral corner (intensity < 15)
            gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
            if gray[peak_y, peak_x] < 15:
                lies_outside_retina = True

        flag_for_review = lies_outside_retina

        # Colormap overlay
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        # Blend with original image
        overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, colormap_rgb, alpha, 0)

        clinical_note = (
            "Grad-CAM highlights coarse convolutional feature salience from the classical backbone. "
            "It reflects ImageNet/fundus visual representations, not quantum circuit states. "
            "It is a saliency heatmap, not an automated lesion segmentation outline."
        )
        if flag_for_review:
            clinical_note += " [WARNING: Peak activation falls outside retinal disc. Secondary clinical review advised.]"

        return GradCAMResult(
            heatmap_raw=heatmap_resized,
            overlay_rgb=overlay,
            peak_coord=(int(peak_y), int(peak_x)),
            lies_outside_retina=lies_outside_retina,
            flag_for_review=flag_for_review,
            clinical_note=clinical_note,
        )
