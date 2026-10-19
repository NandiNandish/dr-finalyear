"""
PyTorch Dataset for Fundus Photography.
Supports patient-disjoint splits, safe augmentations, minority oversampling on train split,
and feature cache integration.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

from .transforms import FundusPreprocessor
from .label_maps import map_dataset_label


class FundusDataset(Dataset):
    """
    Fundus Dataset supporting:
    - Raw image loading with preprocessing / augmentation transforms.
    - Precomputed frozen backbone feature loading (1280-d vectors).
    - Training-only minority oversampling (strictly patient-disjoint post-split).
    """

    def __init__(
        self,
        manifest: Union[str, pd.DataFrame],
        split: str = "train",
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        feature_cache_dir: Optional[str] = None,
        oversample: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split.lower().strip()
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        self.transform = transform
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None

        # Load manifest
        if isinstance(manifest, (str, Path)):
            df = pd.read_csv(manifest)
        else:
            df = manifest.copy()

        # Filter by split if column exists
        if "split" in df.columns:
            df = df[df["split"].str.lower() == self.split].reset_index(drop=True)

        # For standard classification, filter out ungradable images (label == -1) from train/val
        # Ungradable images in test split are evaluated separately by the Quality Gate
        if self.split in ["train", "val"]:
            if "label" in df.columns:
                df = df[df["label"] >= 0].reset_index(drop=True)

        self.df = df

        # Oversampling minority classes: ONLY allowed on the train split
        if oversample and self.split == "train":
            self.df = self._oversample_minority_classes(self.df, seed=seed)

    def _oversample_minority_classes(self, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """
        Oversamples minority classes to match the majority class frequency.
        Crucial for Diabetic Retinopathy where Grade 0 dominates real-world cohorts.
        """
        if len(df) == 0 or "label" not in df.columns:
            return df

        rng = np.random.default_rng(seed)
        class_counts = df["label"].value_counts()
        max_count = class_counts.max()

        resampled_dfs = []
        for cls, count in class_counts.items():
            cls_df = df[df["label"] == cls]
            if count < max_count:
                # Sample with replacement
                replace_indices = rng.choice(cls_df.index, size=max_count, replace=True)
                resampled_dfs.append(df.loc[replace_indices])
            else:
                resampled_dfs.append(cls_df)

        oversampled_df = pd.concat(resampled_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return oversampled_df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        if not img_path.is_absolute() and not img_path.exists():
            img_path = self.root_dir / img_path

        label = int(row["label"]) if "label" in row else -1
        patient_id = str(row.get("patient_id", "UNKNOWN"))
        dataset_name = str(row.get("dataset", "unknown"))

        # Check if precomputed feature vector exists in cache
        if self.feature_cache_dir is not None:
            feature_file = self.feature_cache_dir / f"{img_path.stem}.pt"
            if feature_file.exists():
                features = torch.load(feature_file, map_location="cpu")
                return {
                    "features": features,
                    "label": torch.tensor(label, dtype=torch.long),
                    "patient_id": patient_id,
                    "path": str(img_path),
                    "dataset": dataset_name,
                }

        # Otherwise load raw image from disk
        if not img_path.exists():
            raise FileNotFoundError(f"Fundus image not found: {img_path}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not read fundus image at {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply transformation
        if self.transform is not None:
            img_tensor = self.transform(img_rgb)
        else:
            # Default preprocessor fallback
            preproc = FundusPreprocessor()
            img_tensor = preproc(img_rgb)

        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
            "path": str(img_path),
            "dataset": dataset_name,
        }
