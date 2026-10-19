"""
Quality Gate for Fundus Photography.
Evaluates image gradability prior to downstream deep learning / quantum analysis.

Safety Invariant:
An ungradable image must NEVER produce a reassuring "No DR" result.
Rejected images must immediately return 'Ungradable -> Recapture'.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Union, Optional
import numpy as np
import cv2
from PIL import Image


@dataclass
class QualityResult:
    is_gradable: bool
    rejection_code: Optional[str] = None
    reason: str = "Image passed all quality gate criteria."
    recommendation: str = "PROCEED_TO_GRADING"
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_gradable": self.is_gradable,
            "rejection_code": self.rejection_code,
            "reason": self.reason,
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


class QualityGate:
    """
    Quality gate that rejects blurred, underexposed, overexposed,
    artefact-ridden, or insufficient-retina fundus photographs.
    """

    def __init__(
        self,
        min_laplacian_var: float = 80.0,
        min_mean_brightness: float = 25.0,
        max_mean_brightness: float = 220.0,
        min_contrast_std: float = 15.0,
        min_retina_area_ratio: float = 0.55,
        max_saturated_fraction: float = 0.15,
    ):
        self.min_laplacian_var = float(min_laplacian_var)
        self.min_mean_brightness = float(min_mean_brightness)
        self.max_mean_brightness = float(max_mean_brightness)
        self.min_contrast_std = float(min_contrast_std)
        self.min_retina_area_ratio = float(min_retina_area_ratio)
        self.max_saturated_fraction = float(max_saturated_fraction)

    def _to_numpy_rgb(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not load image from path: {image}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3 and image.shape[2] == 3:
                return image
            elif image.ndim == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                raise ValueError(f"Unsupported image array shape: {image.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def assess(self, image_input: Union[str, np.ndarray, Image.Image]) -> QualityResult:
        """
        Runs quality heuristics:
        1. Mean brightness & standard deviation (exposure / contrast).
        2. Saturation fraction (flash flare / white blown out pixels).
        3. Laplacian variance (motion blur / defocus).
        4. Retinal area ratio (percentage of image containing valid fundus tissue).
        """
        img_rgb = self._to_numpy_rgb(image_input)
        h, w, _ = img_rgb.shape

        if h < 64 or w < 64:
            return QualityResult(
                is_gradable=False,
                rejection_code="UNGRA_DIMENSIONS_TOO_SMALL",
                reason=f"Image resolution {w}x{h} is below minimum diagnostic threshold (64x64).",
                recommendation="RECAPTURE",
                metrics={"height": float(h), "width": float(w)},
            )

        # Convert to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Segment retinal tissue from black camera margins (> 15)
        retina_mask = gray > 15
        retina_area = float(np.sum(retina_mask))
        total_area = float(h * w)
        retina_ratio = retina_area / total_area

        # If zero retinal tissue is detected, image is completely dark / blank
        if retina_area == 0:
            return QualityResult(
                is_gradable=False,
                rejection_code="UNGRA_TOO_DARK",
                reason="No retinal illumination detected. Image is completely underexposed or black.",
                recommendation="RECAPTURE",
                metrics={"mean_brightness": 0.0, "retina_area_ratio": 0.0},
            )

        # 2. Exposure & brightness evaluation on retinal tissue
        valid_pixels = gray[retina_mask]
        mean_retina = float(np.mean(valid_pixels))
        std_retina = float(np.std(valid_pixels))

        if mean_retina < self.min_mean_brightness:
            return QualityResult(
                is_gradable=False,
                rejection_code="UNGRA_TOO_DARK",
                reason=f"Fundus photograph is severely underexposed (retinal brightness {mean_retina:.1f} < {self.min_mean_brightness:.1f}). Retinal structures are unobservable.",
                recommendation="RECAPTURE",
                metrics={"mean_brightness": mean_retina, "std_contrast": std_retina, "retina_area_ratio": retina_ratio},
            )

        if mean_retina > self.max_mean_brightness:
            return QualityResult(
                is_gradable=False,
                rejection_code="UNGRA_TOO_BRIGHT",
                reason=f"Fundus photograph is severely overexposed (retinal brightness {mean_retina:.1f} > {self.max_mean_brightness:.1f}). Sensor wash-out detected.",
                recommendation="RECAPTURE",
                metrics={"mean_brightness": mean_retina, "std_contrast": std_retina, "retina_area_ratio": retina_ratio},
            )

        # 3. Retinal area ratio check (field-of-view completeness)
        if retina_ratio < self.min_retina_area_ratio:
            return QualityResult(
                is_gradable=False,
                rejection_code="UNGRA_TOO_LITTLE_RETINA",
                reason=f"Insufficient retinal field visible ({retina_ratio*100:.1f}% < {self.min_retina_area_ratio*100:.1f}%). Retinal disc is clipped or obscured.",
                recommendation="RECAPTURE",
                metrics={"retina_area_ratio": retina_ratio, "mean_brightness": mean_retina},
            )

        # 4. Contrast dynamic range on retinal tissue
        if std_retina < self.min_contrast_std:
            return QualityResult(
                is_gradable=False,
                rejection_code="UNGRA_LOW_CONTRAST",
                reason=f"Image lacks sufficient diagnostic contrast (std dev {std_retina:.1f} < {self.min_contrast_std:.1f}).",
                recommendation="RECAPTURE",
                metrics={"mean_brightness": mean_retina, "std_contrast": std_retina, "retina_area_ratio": retina_ratio},
            )

        # 5. Glare and saturation check (saturated white pixels >= 250)
        saturated_pixels = np.sum(gray >= 250)
        saturated_fraction = float(saturated_pixels / (h * w))
        if saturated_fraction > self.max_saturated_fraction:
            return QualityResult(
                is_gradable=False,
                rejection_code="UNGRA_FLASH_ARTEFACT",
                reason=f"Severe flash glare or optical artefact detected ({saturated_fraction*100:.1f}% saturated pixels > {self.max_saturated_fraction*100:.1f}%).",
                recommendation="RECAPTURE",
                metrics={"saturated_fraction": saturated_fraction, "retina_area_ratio": retina_ratio},
            )

        # 4. Blur estimation via Laplacian variance on valid retinal region
        # Mask out outer black border so border edges don't falsely inflate sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        valid_laplacian = laplacian[retina_mask]
        laplacian_var = float(np.var(valid_laplacian)) if len(valid_laplacian) > 0 else float(np.var(laplacian))

        if laplacian_var < self.min_laplacian_var:
            return QualityResult(
                is_gradable=False,
                rejection_code="UNGRA_BLURRED",
                reason=f"Image failed sharpness check (Laplacian variance {laplacian_var:.1f} < {self.min_laplacian_var:.1f}). Defocus or motion blur obscures microaneurysms.",
                recommendation="RECAPTURE",
                metrics={
                    "laplacian_var": laplacian_var,
                    "mean_brightness": mean_retina,
                    "std_contrast": std_retina,
                    "retina_area_ratio": retina_ratio,
                    "saturated_fraction": saturated_fraction,
                },
            )

        # Passed all checks
        return QualityResult(
            is_gradable=True,
            rejection_code=None,
            reason="Image passed quality gate (gradable).",
            recommendation="PROCEED_TO_GRADING",
            metrics={
                "laplacian_var": round(laplacian_var, 2),
                "mean_brightness": round(mean_retina, 2),
                "std_contrast": round(std_retina, 2),
                "retina_area_ratio": round(retina_ratio, 4),
                "saturated_fraction": round(saturated_fraction, 4),
            },
        )
