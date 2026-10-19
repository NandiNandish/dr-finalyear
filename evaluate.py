"""
Evaluation Engine for Diabetic Retinopathy Screening.
Evaluates model on locked test split or external test benchmarks.
Calibrates probabilities and tunes referable DR decision threshold
exclusively on the VALIDATION split.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FundusDataset
from src.models.full_model import DRScreeningModel
from src.training.metrics import compute_screening_metrics, find_optimal_threshold
from src.training.calibration import ModelWithTemperature
from src.quality.gate import QualityGate


def evaluate_model(
    checkpoint_path: str,
    manifest_path: str,
    split: str = "test",
    cache_dir: str = "data/cached_features",
    external_manifest: Optional[str] = None,
    calibrate_temperature: bool = True,
    device: str = "cpu",
    output_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"EVALUATION: Split = {split.upper()} | Checkpoint = {checkpoint_path}")
    print(f"==================================================")

    # 1. Load checkpoint
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt_data = torch.load(ckpt_file, map_location=device)

    # Determine head architecture from filename or state_dict
    stem = ckpt_file.stem
    if "quantum" in stem:
        head_type = "hybrid_quantum"
    elif "matched" in stem:
        head_type = "matched_dense"
    else:
        head_type = "classical"

    model = DRScreeningModel(
        active_head=head_type,
        variational_layers=3,
        diff_method="backprop",
        freeze_backbone=True,
    )
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. VALIDATION CALIBRATION: Calibrate temperature & threshold on VAL data only
    val_manifest = manifest_path
    val_ds = FundusDataset(manifest=val_manifest, split="val", feature_cache_dir=cache_dir, oversample=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    print(f"Calibrating on Validation Split ({len(val_ds)} samples)...")
    calibrated_model = ModelWithTemperature(model)
    if calibrate_temperature:
        learned_temp = calibrated_model.calibrate(val_loader, device=device)
    else:
        learned_temp = 1.0

    # Collect val probabilities to pick referable DR threshold (Grade >= 2)
    val_targets = []
    val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["features"].to(device) if "features" in batch else batch["image"].to(device)
            scaled_logits = calibrated_model(x)
            probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(batch["label"].cpu().numpy())

    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)
    val_binary_referable = (val_targets >= 2).astype(int)
    val_prob_referable = np.sum(val_probs[:, 2:], axis=1)

    # Pick threshold for target sensitivity >= 0.85 on validation
    chosen_threshold = find_optimal_threshold(val_binary_referable, val_prob_referable, target_sensitivity=0.85)
    print(f"Optimal Referable-DR decision threshold chosen on validation split: {chosen_threshold:.4f}")

    # 3. EVALUATION ON TARGET SET (LOCKED TEST SPLIT OR EXTERNAL SET)
    if split == "external":
        target_manifest = external_manifest if external_manifest else "data/manifests/external_test.csv"
        target_split = "external"
    else:
        target_manifest = manifest_path
        target_split = split

    test_ds = FundusDataset(manifest=target_manifest, split=target_split, feature_cache_dir=cache_dir, oversample=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    print(f"\nRunning locked single-pass evaluation on {split.upper()} ({len(test_ds)} samples)...")
    test_targets = []
    test_probs = []
    test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device) if "features" in batch else batch["image"].to(device)
            scaled_logits = calibrated_model(x)
            probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            test_probs.extend(probs)
            test_preds.extend(preds)
            test_targets.extend(batch["label"].cpu().numpy())

    test_targets = np.array(test_targets)
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)

    # Compute comprehensive evaluation metrics using the locked validation threshold
    metrics = compute_screening_metrics(
        y_true=test_targets,
        y_pred=test_preds,
        y_probs=test_probs,
        referable_threshold=chosen_threshold,
    )

    metrics["calibration"] = {
        "temperature": round(learned_temp, 4),
        "validation_chosen_threshold": round(chosen_threshold, 4),
    }
    metrics["meta"] = {
        "head_architecture": head_type,
        "split_evaluated": split,
        "sample_count": len(test_ds),
        "checkpoint": str(checkpoint_path),
    }

    # Quality Gate Assessment on the raw manifest (Imageability Rate)
    raw_df = pd.read_csv(target_manifest)
    if "split" in raw_df.columns:
        target_df = raw_df[raw_df["split"].str.lower() == target_split.lower()]
    else:
        target_df = raw_df

    gate = QualityGate()
    gradable_count = 0
    total_checked = 0
    for _, r in target_df.iterrows():
        p = Path(r["path"])
        if p.exists():
            res = gate.assess(str(p))
            if res.is_gradable:
                gradable_count += 1
            total_checked += 1

    imageability_rate = float(gradable_count / total_checked) if total_checked > 0 else 1.0
    metrics["quality_gate"] = {
        "total_images_checked": total_checked,
        "gradable_images": gradable_count,
        "imageability_rate": round(imageability_rate, 4),
    }

    # Print Measured Results
    print("\n---------------- MEASURED EVALUATION RESULTS ----------------")
    print(f"Architecture:              {head_type.upper()}")
    print(f"5-Class Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Macro-F1:                  {metrics['macro_f1']:.4f}")
    print(f"Quadratic Weighted Kappa:  {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"Moderate DR Recall:        {metrics['moderate_dr_recall']:.4f}")
    print(f"Referable Sensitivity:     {metrics['referable_dr']['sensitivity']:.4f}")
    print(f"Referable Specificity:     {metrics['referable_dr']['specificity']:.4f}")
    print(f"Referable ROC-AUC:         {metrics['referable_dr']['auc']:.4f}")
    print(f"Imageability Rate:         {metrics['quality_gate']['imageability_rate']*100:.1f}%")
    print("-------------------------------------------------------------")

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation report to: {output_report_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DR Screening Models.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", default="data/manifests/demo_manifest.csv", help="Internal manifest")
    parser.add_argument("--split", default="test", choices=["test", "val", "external"], help="Target split")
    parser.add_argument("--external_manifest", default="data/manifests/external_test.csv", help="External manifest")
    parser.add_argument("--cache_dir", default="data/cached_features", help="Path to cached feature directory")
    parser.add_argument("--no_calibration", action="store_true", help="Disable temperature scaling")
    parser.add_argument("--output_report", default=None, help="Path to save JSON evaluation report")
    parser.add_argument("--device", default="cpu", help="Compute device")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        split=args.split,
        cache_dir=args.cache_dir,
        external_manifest=args.external_manifest,
        calibrate_temperature=not args.no_calibration,
        device=args.device,
        output_report_path=args.output_report,
    )
