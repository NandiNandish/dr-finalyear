"""
Unit tests for data splitting, patient-disjoint integrity, oversampling isolation,
and label mapping.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.label_maps import (
    DRGrade,
    CLASS_NAMES,
    is_referable_dr,
    map_dataset_label,
)
from src.data.dataset import FundusDataset


@pytest.fixture
def manifest_path():
    path = Path("data/manifests/demo_manifest.csv")
    if not path.exists():
        pytest.skip("Manifest demo_manifest.csv not generated yet")
    return str(path)


def test_patient_disjoint_splits(manifest_path):
    """
    Ensure zero patient leakage across train, validation, and test splits.
    """
    df = pd.read_csv(manifest_path)
    train_patients = set(df[df["split"] == "train"]["patient_id"])
    val_patients = set(df[df["split"] == "val"]["patient_id"])
    test_patients = set(df[df["split"] == "test"]["patient_id"])

    # Overlaps must be completely empty
    train_val_leakage = train_patients.intersection(val_patients)
    train_test_leakage = train_patients.intersection(test_patients)
    val_test_leakage = val_patients.intersection(test_patients)

    assert len(train_val_leakage) == 0, f"Patient leakage between train and val: {train_val_leakage}"
    assert len(train_test_leakage) == 0, f"Patient leakage between train and test: {train_test_leakage}"
    assert len(val_test_leakage) == 0, f"Patient leakage between val and test: {val_test_leakage}"


def test_oversampling_only_on_train_split(manifest_path):
    """
    Verify oversampling ONLY modifies the training split, leaving val/test untouched.
    """
    # Raw dataset without oversampling
    ds_train_raw = FundusDataset(manifest=manifest_path, split="train", oversample=False)
    ds_train_oversampled = FundusDataset(manifest=manifest_path, split="train", oversample=True, seed=42)

    # Oversampling should rebalance minority classes, typically increasing count
    assert len(ds_train_oversampled) >= len(ds_train_raw)

    # Validation dataset must NOT be changed even if oversample=True is passed
    ds_val_raw = FundusDataset(manifest=manifest_path, split="val", oversample=False)
    ds_val_attempt = FundusDataset(manifest=manifest_path, split="val", oversample=True)
    assert len(ds_val_raw) == len(ds_val_attempt), "Oversampling must NEVER alter validation split!"

    # Test dataset must NOT be changed even if oversample=True is passed
    ds_test_raw = FundusDataset(manifest=manifest_path, split="test", oversample=False)
    ds_test_attempt = FundusDataset(manifest=manifest_path, split="test", oversample=True)
    assert len(ds_test_raw) == len(ds_test_attempt), "Oversampling must NEVER alter test split!"


def test_label_mappings_and_referable():
    # 5-class canonical values
    assert DRGrade.NO_DR == 0
    assert DRGrade.MILD_NPDR == 1
    assert DRGrade.MODERATE_NPDR == 2
    assert DRGrade.SEVERE_NPDR == 3
    assert DRGrade.PROLIFERATIVE_DR == 4

    # Referable DR check (Grade >= 2)
    assert is_referable_dr(0) is False
    assert is_referable_dr(1) is False
    assert is_referable_dr(2) is True
    assert is_referable_dr(3) is True
    assert is_referable_dr(4) is True

    # Dataset-specific mapping
    assert map_dataset_label("eyepacs", 3) == 3
    assert map_dataset_label("messidor", 0) == 0
    assert map_dataset_label("messidor", 2) == 2
    assert map_dataset_label("messidor", 3) == 3
