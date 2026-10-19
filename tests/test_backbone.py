"""
Unit tests for backbone feature extraction, parameter freezing, and caching.
"""

import pytest
import torch
from pathlib import Path
from src.models.backbone import get_backbone
from src.models.extract_features import cache_backbone_features
from src.data.dataset import FundusDataset


def test_efficientnet_b0_output_shape_and_freezing():
    backbone = get_backbone(name="efficientnet_b0", pretrained=False, freeze=True)
    backbone.eval()

    # Verify all parameters are frozen
    for name, param in backbone.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"

    # Test forward pass shape
    dummy_input = torch.randn(2, 3, 224, 224)
    features = backbone(dummy_input)

    assert features.shape == (2, 1280)
    assert features.dtype == torch.float32


def test_resnet50_baseline_shape_and_freezing():
    backbone = get_backbone(name="resnet50", pretrained=False, freeze=True)
    backbone.eval()

    for name, param in backbone.named_parameters():
        assert not param.requires_grad, f"ResNet50 parameter {name} is not frozen!"

    dummy_input = torch.randn(1, 3, 224, 224)
    features = backbone(dummy_input)

    assert features.shape == (1, 2048)


def test_feature_caching_and_dataset_integration(tmp_path):
    manifest_p = Path("data/manifests/demo_manifest.csv")
    if not manifest_p.exists():
        pytest.skip("Demo manifest does not exist")

    cache_dir = tmp_path / "test_cache"
    # Run caching for fast subset (first 4 samples)
    cached_count = cache_backbone_features(
        manifest_path=str(manifest_p),
        cache_dir=str(cache_dir),
        backbone_name="efficientnet_b0",
        fast=True,
        max_samples=4,
    )

    assert cached_count >= 1
    cached_files = list(cache_dir.glob("*.pt"))
    assert len(cached_files) >= 1

    # Check cached tensor shape
    tensor = torch.load(cached_files[0], map_location="cpu")
    assert tensor.shape == (1280,)
    assert tensor.dtype == torch.float32

    # Verify FundusDataset seamlessly loads cached features
    ds = FundusDataset(manifest=str(manifest_p), split="train", feature_cache_dir=str(cache_dir))
    # If the first item was cached, dataset item contains 'features'
    first_item = ds[0]
    if "features" in first_item:
        assert first_item["features"].shape == (1280,)
        assert isinstance(first_item["label"], torch.Tensor)
