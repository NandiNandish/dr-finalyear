# Quantum-Enhanced Deep Learning Ablation Report
**Diabetic Retinopathy Screening Evaluation over 5 Disjoint Random Seeds**

## 1. Architectural Parameter Accounting

| Architecture | Classical Params | Quantum Params | Total Params | Runtime / Image |
| :--- | :---: | :---: | :---: | :---: |
| **Classical Head** | 6,405 | 0 | 6,405 | 0.00 ms |
| **Matched Dense (Control)** | 5,149 | 0 | 5,149 | 0.03 ms |
| **Hybrid Quantum (VQC)** | 5,149 | 12 (4L, L=3) | 5,161 | 9.77 ms |

---

## 2. Multi-Seed Performance Comparison (Mean ± Std over 5 Seeds)

| Architecture | 5-Class Accuracy | Macro-F1 | Quadratic Weighted Kappa (QWK) | Moderate DR Recall | Referable Sensitivity | Referable Specificity | Referable ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **classical** | 0.380 ± 0.040 | 0.276 ± 0.034 | **0.321 ± 0.063** | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.760 ± 0.033 | 0.941 ± 0.027 |
| **matched_dense** | 0.300 ± 0.114 | 0.145 ± 0.082 | **0.270 ± 0.276** | 0.200 ± 0.400 | 0.960 ± 0.080 | 0.760 ± 0.033 | 0.941 ± 0.046 |
| **hybrid_quantum** | 0.200 ± 0.114 | 0.092 ± 0.049 | **0.035 ± 0.273** | 0.200 ± 0.400 | 0.920 ± 0.098 | 0.733 ± 0.112 | 0.883 ± 0.049 |

---

## 3. Paired Non-Parametric Significance Testing (Wilcoxon Signed-Rank)
- **Comparison**: Hybrid Quantum Head vs Matched Dense Control Head
- **QWK Wilcoxon p-value**: 0.4375
- **Macro-F1 Wilcoxon p-value**: 0.4375
- **Statistical Superiority Confirmed**: NO (No statistically significant advantage detected; reported honestly)

---

## 4. Quantum Hardware & Simulation Resource Report
- **Qubits Allocated**: 4 qubits
- **Variational Circuit Depth (L)**: 3 entangling layers
- **Ansatz Layout**: Circular CNOT chain with trainable $R_Y$ rotations
- **Measurement Observable**: Pauli-$Z$ expectation values $\langle Z_k \rangle \in [-1, 1]$
- **Simulation Engine**: PennyLane `default.qubit` exact statevector simulator
- **Differentiation Technique**: Parameter-shift rule (analytic gradients)
