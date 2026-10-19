"""
Feature extraction and disk-caching pipeline.
Pre-computes and caches 1280-d frozen EfficientNet-B0 embeddings to disk.
This enables instant multi-seed training and hyperparameter search of
the quantum, matched-dense, and classical heads without repeatedly evaluating the CNN.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd
import torch
from tqdm import tqdm
import cv2

from src.data.transforms import FundusPreprocessor
from src.models.backbone import get_backbone


def cache_backbone_features(
    manifest_path: str,
    cache_dir: str = "data/cached_features",
    backbone_name: str = "efficientnet_b0",
    image_size: int = 224,
    device: str = "cpu",
    fast: bool = False,
    max_samples: Optional[int] = None,
) -> int:
    """
    Extracts features for all images in manifest and saves to cache_dir as {stem}.pt
    """
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_p)
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fast or max_samples:
        limit = 10 if fast else max_samples
        df = df.iloc[:limit]

    print(f"Caching features using frozen '{backbone_name}' on device '{device}'...")
    print(f"Total images to process: {len(df)} -> Output: {out_dir}")

    # Initialize backbone and preprocessor
    backbone = get_backbone(name=backbone_name, pretrained=True, freeze=True)
    backbone.to(device)
    backbone.eval()

    preprocessor = FundusPreprocessor(image_size=image_size)

    cached_count = 0
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            img_path = Path(row["path"])
            stem = img_path.stem
            out_file = out_dir / f"{stem}.pt"

            if out_file.exists():
                cached_count += 1
                continue

            if not img_path.exists():
                print(f"Warning: Image path does not exist: {img_path}")
                continue

            # Load and preprocess image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = preprocessor(img_rgb).unsqueeze(0).to(device)

            # Forward pass through frozen backbone
            feat = backbone(tensor).squeeze(0).cpu()  # Shape: (1280,)

            # Save to disk
            torch.save(feat, out_file)
            cached_count += 1

    print(f"Feature caching complete. Total cached vectors: {cached_count}")
    return cached_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and cache backbone features.")
    parser.add_argument("--manifest", default="data/manifests/demo_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--cache_dir", default="data/cached_features", help="Output directory for .pt features")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Backbone architecture")
    parser.add_argument("--image_size", type=int, default=224, help="Preprocessing image resolution")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--fast", action="store_true", help="Smoke test flag (cache first 10 images only)")
    args = parser.parse_args()

    cache_backbone_features(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        backbone_name=args.backbone,
        image_size=args.image_size,
        device=args.device,
        fast=args.fast,
    )
