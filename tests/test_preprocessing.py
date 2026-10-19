"""
Unit tests for fundus preprocessing and clinically safe data augmentations.
"""

import pytest
import numpy as np
import torch
from src.data.transforms import (
    crop_black_borders,
    apply_circular_mask,
    apply_clahe,
    FundusPreprocessor,
    SafeClinicalAugmentation,
)
from data.synthetic_generator import generate_synthetic_fundus


def test_crop_black_borders():
    # Create image with black outer padding (100 px on all sides)
    inner = np.full((100, 100, 3), 150, dtype=np.uint8)
    padded = np.pad(inner, ((50, 50), (50, 50), (0, 0)), mode="constant", constant_values=0)

    cropped = crop_black_borders(padded, tol=10)
    assert cropped.shape[0] == 100
    assert cropped.shape[1] == 100
    assert cropped.shape[2] == 3


def test_apply_circular_mask():
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    masked, mask = apply_circular_mask(img, margin=0.9)

    assert masked.shape == (200, 200, 3)
    assert mask.shape == (200, 200)

    # Corners must be 0
    assert masked[0, 0, 0] == 0
    assert masked[199, 199, 0] == 0
    assert mask[0, 0] == 0

    # Center must be intact
    assert masked[100, 100, 0] == 200
    assert mask[100, 100] == 1


def test_apply_clahe():
    raw_img = generate_synthetic_fundus(grade=0, image_size=200)
    clahe_img = apply_clahe(raw_img, clip_limit=2.0)

    assert clahe_img.shape == raw_img.shape
    assert clahe_img.dtype == np.uint8
    # Contrast enhancement should change pixel intensities mildly
    diff = np.abs(clahe_img.astype(float) - raw_img.astype(float))
    assert np.mean(diff) > 0.1


def test_fundus_preprocessor_tensor_output():
    preproc = FundusPreprocessor(image_size=224, keep_circular=True, use_clahe=True)
    raw_img = generate_synthetic_fundus(grade=2, image_size=256)

    tensor = preproc(raw_img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    # Verify standard ImageNet normalized range (mean ~ 0, std ~ 1)
    assert -3.0 < tensor.mean().item() < 3.0


def test_safe_clinical_augmentation():
    preproc = FundusPreprocessor(image_size=224)
    aug = SafeClinicalAugmentation(base_preprocessor=preproc, p_hflip=0.5, p_vflip=0.5, max_rotation_deg=10.0)

    raw_img = generate_synthetic_fundus(grade=3, image_size=256)
    aug_tensor = aug(raw_img)

    assert isinstance(aug_tensor, torch.Tensor)
    assert aug_tensor.shape == (3, 224, 224)
