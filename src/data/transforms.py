"""
Preprocessing and clinically safe data augmentations for fundus photographs.

Pipeline specifications:
1. Crop black borders.
2. Keep circular field (suppress peripheral scanner artefacts).
3. Resize to 224 px (configurable up to 299).
4. Rescale intensities and apply mild contrast correction (CLAHE on L-channel).
5. Safe training augmentations: small flips, minor rotations (+-10 deg), subtle brightness/zoom (+-10%)
   that strictly preserve clinical lesions.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def crop_black_borders(img: np.ndarray, tol: int = 15) -> np.ndarray:
    """
    Crops extraneous black scanner borders around the retinal circle.
    """
    if img.ndim == 2:
        mask = img > tol
        if not np.any(mask):
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not np.any(mask):
        return img
    row_mask = mask.any(1)
    col_mask = mask.any(0)
    return img[np.ix_(row_mask, col_mask)]


def apply_circular_mask(img: np.ndarray, margin: float = 0.98) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a smooth circular mask inscribed in the image frame to keep the circular field
    and eliminate square corner artefacts.
    Returns: (masked_image, binary_retina_mask)
    """
    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = int(min(center_y, center_x) * margin)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    circular_mask = dist_from_center <= radius

    masked = np.zeros_like(img)
    if img.ndim == 3:
        for c in range(img.shape[2]):
            masked[:, :, c] = img[:, :, c] * circular_mask
    else:
        masked = img * circular_mask

    return masked, circular_mask.astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies mild Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the luminance channel of the LAB color space.
    """
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

    # Convert RGB to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    lab_enhanced = cv2.merge((l_clahe, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced


class FundusPreprocessor:
    """
    Deterministic clinical preprocessor for inference and training evaluation.
    """

    def __init__(
        self,
        image_size: int = 224,
        crop_borders: bool = True,
        keep_circular: bool = True,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        normalize_imagenet: bool = True,
    ):
        self.image_size = int(image_size)
        self.crop_borders = crop_borders
        self.keep_circular = keep_circular
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = tuple(clahe_tile_grid)
        self.normalize_imagenet = normalize_imagenet

        # Standard ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def preprocess_numpy(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs image preprocessing pipeline on NumPy RGB image.
        Returns: (processed_rgb_uint8, retina_mask_uint8)
        """
        img = img_rgb.copy()

        # Step 1: Crop black borders
        if self.crop_borders:
            img = crop_black_borders(img)

        # Step 2: Resize to target dimension
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # Step 3: Keep circular field
        mask = np.ones((self.image_size, self.image_size), dtype=np.uint8)
        if self.keep_circular:
            img, mask = apply_circular_mask(img)

        # Step 4: Mild contrast correction (CLAHE)
        if self.use_clahe:
            img = apply_clahe(img, clip_limit=self.clahe_clip_limit, tile_grid_size=self.clahe_tile_grid)
            # Re-apply circular mask so CLAHE background remains clean black
            if self.keep_circular:
                for c in range(3):
                    img[:, :, c] = img[:, :, c] * mask

        return img, mask

    def __call__(self, img_input: np.ndarray) -> torch.Tensor:
        """
        Converts raw RGB NumPy array to normalized PyTorch tensor (C, H, W).
        """
        processed_rgb, _ = self.preprocess_numpy(img_input)
        
        # Rescale intensities to [0, 1]
        tensor = TF.to_tensor(processed_rgb)  # Shape (3, H, W), float in [0.0, 1.0]

        # Normalize with ImageNet statistics
        if self.normalize_imagenet:
            tensor = TF.normalize(tensor, mean=self.mean, std=self.std)

        return tensor


class SafeClinicalAugmentation:
    """
    Clinically safe data augmentation:
    - Small horizontal and vertical flips (fundus orientation invariance).
    - Minor rotations (+-10 degrees).
    - Subtle brightness adjustments (+-10%) and subtle zoom (+-5%).
    Never introduces cutouts, severe blurs, or color distortions that would fake or erase lesions.
    """

    def __init__(
        self,
        base_preprocessor: FundusPreprocessor,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        max_rotation_deg: float = 10.0,
        brightness_factor_range: Tuple[float, float] = (0.90, 1.10),
    ):
        self.preprocessor = base_preprocessor
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.max_rotation_deg = max_rotation_deg
        self.brightness_factor_range = brightness_factor_range

    def __call__(self, img_rgb: np.ndarray) -> torch.Tensor:
        # Preprocess base image first (crop, resize, CLAHE)
        proc_rgb, _ = self.preprocessor.preprocess_numpy(img_rgb)
        pil_img = Image.fromarray(proc_rgb)

        # Safe horizontal flip
        if np.random.rand() < self.p_hflip:
            pil_img = TF.hflip(pil_img)

        # Safe vertical flip
        if np.random.rand() < self.p_vflip:
            pil_img = TF.vflip(pil_img)

        # Minor rotation
        angle = float(np.random.uniform(-self.max_rotation_deg, self.max_rotation_deg))
        pil_img = TF.rotate(pil_img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)

        # Subtle brightness change
        b_factor = float(np.random.uniform(*self.brightness_factor_range))
        pil_img = TF.adjust_brightness(pil_img, b_factor)

        # Convert to tensor and normalize
        tensor = TF.to_tensor(pil_img)
        if self.preprocessor.normalize_imagenet:
            tensor = TF.normalize(tensor, mean=self.preprocessor.mean, std=self.preprocessor.std)

        return tensor
