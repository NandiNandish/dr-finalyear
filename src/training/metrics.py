"""
Evaluation metrics for Diabetic Retinopathy Screening.
Calculates:
- Quadratic Weighted Kappa (QWK) [Primary selection metric]
- 5-class Accuracy & Macro-F1
- Class-wise Recall (highlighting Moderate NPDR)
- Referable DR (Grade >= 2) Sensitivity, Specificity, ROC-AUC
- Confusion Matrix
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Quadratic Weighted Kappa (QWK) across 5 DR severity grades.
    """
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_referable_metrics(
    y_true_binary: np.ndarray,
    y_prob_referable: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes clinical referable-DR (Grade >= 2) sensitivity, specificity, and ROC-AUC.
    """
    y_pred_bin = (y_prob_referable >= threshold).astype(int)

    # Sensitivity = TP / (TP + FN) = Recall of positive class
    # Specificity = TN / (TN + FP) = Recall of negative class
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_bin, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # AUC calculation
    try:
        auc = float(roc_auc_score(y_true_binary, y_prob_referable))
    except ValueError:
        auc = 0.5  # Fallback if single class present

    return {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "auc": round(auc, 4),
        "threshold": round(threshold, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def find_optimal_threshold(
    y_true_binary: np.ndarray,
    y_prob_referable: np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    """
    Selects the decision threshold on VALIDATION data that achieves target sensitivity
    (e.g., >= 90% for clinical safety) while maximizing specificity (Youden's Index variant).
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_referable)

    # Filter thresholds where sensitivity (tpr) >= target_sensitivity
    valid_indices = np.where(tpr >= target_sensitivity)[0]
    if len(valid_indices) > 0:
        # Choose the one with minimum FPR (maximum specificity)
        best_idx = valid_indices[np.argmin(fpr[valid_indices])]
        chosen_thresh = float(thresholds[best_idx])
    else:
        # Fallback to Youden's J statistic (max tpr - fpr)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        chosen_thresh = float(thresholds[best_idx])

    # Bound to [0.05, 0.95]
    return float(np.clip(chosen_thresh, 0.05, 0.95))


def compute_screening_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    referable_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation suite for 5-class DR screening.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    # 1. 5-Class standard metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = quadratic_weighted_kappa(y_true, y_pred)

    # Class-wise recall
    class_recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    class_recall_dict = {
        "recall_grade_0_no_dr": round(float(class_recalls[0]), 4),
        "recall_grade_1_mild": round(float(class_recalls[1]), 4),
        "recall_grade_2_moderate": round(float(class_recalls[2]), 4),
        "recall_grade_3_severe": round(float(class_recalls[3]), 4),
        "recall_grade_4_pdr": round(float(class_recalls[4]), 4),
    }

    # 2. Referable DR binary metrics (Grade >= 2)
    y_true_referable = (y_true >= 2).astype(int)
    # Probability of referable DR is sum of probabilities for classes 2, 3, 4
    prob_referable = np.sum(y_probs[:, 2:], axis=1)

    ref_metrics = compute_referable_metrics(y_true_referable, prob_referable, threshold=referable_threshold)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "moderate_dr_recall": class_recall_dict["recall_grade_2_moderate"],
        "class_recalls": class_recall_dict,
        "referable_dr": ref_metrics,
        "confusion_matrix": cm,
    }
