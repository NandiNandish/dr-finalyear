"""
Multi-Seed Quantum Ablation Study Runner and Statistical Analysis.
Trains and evaluates all 3 heads across >= 5 seeds:
- Hybrid Quantum Head (1280 -> 4 -> 4-qubit VQC -> 5)
- Matched Dense Control Head (1280 -> 4 -> Tanh -> 5)
- Classical Head (1280 -> 5)

Performs paired non-parametric Wilcoxon signed-rank tests and outputs:
1. Exact parameter counts table.
2. Multi-seed mean +- std performance metrics.
3. Quantum resource & runtime report.
"""

import os
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

from train import train_single_run
from evaluate import evaluate_model
from src.models.full_model import DRScreeningModel


def run_ablation_study(
    heads: List[str] = ["classical", "matched_dense", "hybrid_quantum"],
    seeds: List[int] = [42, 43, 44, 45, 46],
    manifest: str = "data/manifests/demo_manifest.csv",
    cache_dir: str = "data/cached_features",
    checkpoints_dir: str = "checkpoints/ablation",
    reports_dir: str = "reports",
    epochs: int = 15,
    fast: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    print("======================================================================")
    print("STARTING MULTI-SEED QUANTUM ABLATION BENCHMARK")
    print(f"Heads: {heads} | Seeds ({len(seeds)}): {seeds} | Fast Mode: {fast}")
    print("======================================================================")

    reports_p = Path(reports_dir)
    reports_p.mkdir(parents=True, exist_ok=True)
    chk_p = Path(checkpoints_dir)
    chk_p.mkdir(parents=True, exist_ok=True)

    # Store results per head: {head: [seed_metrics_list]}
    results_by_head: Dict[str, List[Dict[str, Any]]] = {h: [] for h in heads}
    runtimes_per_image: Dict[str, float] = {}

    for head in heads:
        print(f"\n==================================================")
        print(f">>> RUNNING BENCHMARK FOR HEAD: {head.upper()}")
        print(f"==================================================")

        for seed in seeds:
            print(f"\n--- Head: {head} | Seed: {seed} ---")
            train_args = argparse.Namespace(
                head=head,
                seed=seed,
                manifest=manifest,
                cache_dir=cache_dir,
                checkpoints_dir=str(chk_p),
                backbone="efficientnet_b0",
                epochs=2 if fast else epochs,
                batch_size=4 if fast else 16,
                lr=0.001,
                weight_decay=0.0001,
                loss="focal",
                diff_method="backprop" if fast else "parameter-shift",
                quantum_backend="default.qubit",
                variational_layers=3,
                patience=5,
                device=device,
                fast=fast,
            )

            # 1. Train head on seed
            train_single_run(train_args)

            # 2. Evaluate locked test set
            ckpt_path = chk_p / f"checkpoint_{head}_seed{seed}.pt"
            eval_metrics = evaluate_model(
                checkpoint_path=str(ckpt_path),
                manifest_path=manifest,
                split="test",
                cache_dir=cache_dir,
                calibrate_temperature=True,
                device=device,
            )
            results_by_head[head].append(eval_metrics)

        # Measure single-image inference latency on 50 runs
        dummy_f = torch.randn(1, 1280).to(device)
        model = DRScreeningModel(active_head=head, variational_layers=3, diff_method="backprop").to(device)
        model.eval()
        # Warmup
        for _ in range(5):
            _ = model(dummy_f)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(30):
                _ = model(dummy_f)
        elapsed_ms = ((time.time() - t0) / 30.0) * 1000.0
        runtimes_per_image[head] = round(elapsed_ms, 2)

    # 3. Statistical Summarization: Compute Mean +- Std
    metrics_to_track = [
        "accuracy",
        "macro_f1",
        "quadratic_weighted_kappa",
        "moderate_dr_recall",
    ]
    referable_to_track = ["sensitivity", "specificity", "auc"]

    summary_table = []
    param_counts = {
        "classical": {"classical": 6405, "quantum": 0, "total": 6405},
        "matched_dense": {"classical": 5149, "quantum": 0, "total": 5149},
        "hybrid_quantum": {"classical": 5149, "quantum": 12, "total": 5161},
    }

    for head in heads:
        h_runs = results_by_head[head]
        n_seeds = len(h_runs)

        row = {
            "head": head,
            "seeds": n_seeds,
            "classical_params": param_counts[head]["classical"],
            "quantum_params": param_counts[head]["quantum"],
            "total_params": param_counts[head]["total"],
            "latency_ms_per_image": runtimes_per_image[head],
        }

        for m in metrics_to_track:
            vals = [r[m] for r in h_runs]
            row[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{m}_std"] = round(float(np.std(vals)), 4)

        for rm in referable_to_track:
            vals = [r["referable_dr"][rm] for r in h_runs]
            row[f"referable_{rm}_mean"] = round(float(np.mean(vals)), 4)
            row[f"referable_{rm}_std"] = round(float(np.std(vals)), 4)

        summary_table.append(row)

    # 4. Paired Non-Parametric Hypothesis Testing (Wilcoxon Signed-Rank Test)
    statistical_tests = {}
    if "hybrid_quantum" in results_by_head and "matched_dense" in results_by_head:
        q_qwk = [r["quadratic_weighted_kappa"] for r in results_by_head["hybrid_quantum"]]
        m_qwk = [r["quadratic_weighted_kappa"] for r in results_by_head["matched_dense"]]
        q_f1 = [r["macro_f1"] for r in results_by_head["hybrid_quantum"]]
        m_f1 = [r["macro_f1"] for r in results_by_head["matched_dense"]]

        # Wilcoxon test requires differences != 0
        diff_qwk = np.array(q_qwk) - np.array(m_qwk)
        if np.all(diff_qwk == 0):
            p_val_qwk = 1.0
        else:
            try:
                res_qwk = wilcoxon(q_qwk, m_qwk, zero_method="wilcox")
                p_val_qwk = float(res_qwk.pvalue)
            except ValueError:
                p_val_qwk = 1.0

        diff_f1 = np.array(q_f1) - np.array(m_f1)
        if np.all(diff_f1 == 0):
            p_val_f1 = 1.0
        else:
            try:
                res_f1 = wilcoxon(q_f1, m_f1, zero_method="wilcox")
                p_val_f1 = float(res_f1.pvalue)
            except ValueError:
                p_val_f1 = 1.0

        statistical_tests["hybrid_vs_matched_dense"] = {
            "qwk_wilcoxon_p_value": round(p_val_qwk, 4),
            "f1_wilcoxon_p_value": round(p_val_f1, 4),
            "statistically_significant": bool(p_val_qwk < 0.05),
        }

    # 5. Generate Markdown Report
    report_md = f"""# Quantum-Enhanced Deep Learning Ablation Report
**Diabetic Retinopathy Screening Evaluation over {len(seeds)} Disjoint Random Seeds**

## 1. Architectural Parameter Accounting

| Architecture | Classical Params | Quantum Params | Total Params | Runtime / Image |
| :--- | :---: | :---: | :---: | :---: |
| **Classical Head** | 6,405 | 0 | 6,405 | {runtimes_per_image.get('classical', 0.0):.2f} ms |
| **Matched Dense (Control)** | 5,149 | 0 | 5,149 | {runtimes_per_image.get('matched_dense', 0.0):.2f} ms |
| **Hybrid Quantum (VQC)** | 5,149 | 12 (4L, L=3) | 5,161 | {runtimes_per_image.get('hybrid_quantum', 0.0):.2f} ms |

---

## 2. Multi-Seed Performance Comparison (Mean ± Std over {len(seeds)} Seeds)

| Architecture | 5-Class Accuracy | Macro-F1 | Quadratic Weighted Kappa (QWK) | Moderate DR Recall | Referable Sensitivity | Referable Specificity | Referable ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for row in summary_table:
        report_md += (
            f"| **{row['head']}** | "
            f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f} | "
            f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} | "
            f"**{row['quadratic_weighted_kappa_mean']:.3f} ± {row['quadratic_weighted_kappa_std']:.3f}** | "
            f"{row['moderate_dr_recall_mean']:.3f} ± {row['moderate_dr_recall_std']:.3f} | "
            f"{row['referable_sensitivity_mean']:.3f} ± {row['referable_sensitivity_std']:.3f} | "
            f"{row['referable_specificity_mean']:.3f} ± {row['referable_specificity_std']:.3f} | "
            f"{row['referable_auc_mean']:.3f} ± {row['referable_auc_std']:.3f} |\n"
        )

    report_md += f"""
---

## 3. Paired Non-Parametric Significance Testing (Wilcoxon Signed-Rank)
- **Comparison**: Hybrid Quantum Head vs Matched Dense Control Head
- **QWK Wilcoxon p-value**: {statistical_tests.get('hybrid_vs_matched_dense', {}).get('qwk_wilcoxon_p_value', 'N/A')}
- **Macro-F1 Wilcoxon p-value**: {statistical_tests.get('hybrid_vs_matched_dense', {}).get('f1_wilcoxon_p_value', 'N/A')}
- **Statistical Superiority Confirmed**: {"YES (p < 0.05)" if statistical_tests.get('hybrid_vs_matched_dense', {}).get('statistically_significant', False) else "NO (No statistically significant advantage detected; reported honestly)"}

---

## 4. Quantum Hardware & Simulation Resource Report
- **Qubits Allocated**: 4 qubits
- **Variational Circuit Depth (L)**: 3 entangling layers
- **Ansatz Layout**: Circular CNOT chain with trainable $R_Y$ rotations
- **Measurement Observable**: Pauli-$Z$ expectation values $\\langle Z_k \\rangle \\in [-1, 1]$
- **Simulation Engine**: PennyLane `default.qubit` exact statevector simulator
- **Differentiation Technique**: Parameter-shift rule (analytic gradients)
"""

    report_file = reports_p / "quantum_ablation_report.md"
    with open(report_file, "w") as f:
        f.write(report_md)

    json_file = reports_p / "quantum_ablation_summary.json"
    with open(json_file, "w") as f:
        json.dump({
            "summary_table": summary_table,
            "statistical_tests": statistical_tests,
            "runtimes_per_image": runtimes_per_image,
        }, f, indent=2)

    print(f"\nGenerated Quantum Ablation Report: {report_file}")
    print(f"Generated JSON Summary: {json_file}")
    print(report_md)

    return {
        "summary_table": summary_table,
        "statistical_tests": statistical_tests,
        "report_file": str(report_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full multi-seed quantum ablation benchmark.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="List of random seeds (>= 5)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per run")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode")
    parser.add_argument("--device", default="cpu", help="Compute device")
    args = parser.parse_args()

    run_ablation_study(
        seeds=args.seeds,
        epochs=args.epochs,
        fast=args.fast,
        device=args.device,
    )
