# DRIVE Retinal Dataset: Clinical Screening & Evaluation Report
**Dataset Source**: Dutch Diabetic Retinopathy Screening Program (DRIVE Benchmark)  
**Total Images Ingested**: 40 (20 Train, 20 Test)  
**Resolution**: Native 565  584 RGB (Canon CR5 non-mydriatic 3-CCD camera)  
**Ground Truth**: Expert Ophthalmologist Vessel & FOV Segmentations Included  

---

## 1. Executive Summary & Quality Gate Audit

| Metric | Training Cohort | Test Cohort | Combined Total |
| :--- | :---: | :---: | :---: |
| **Total Images** | 20 | 20 | **40** |
| **Passed Quality Gate (Gradable)** | 13 (65.0%) | 12 (60.0%) | **25 (62.5%)** |
| **Rejected (Ungradable  Recapture)** | 7 | 8 | **15 (37.5%)** |
| **Safety Invariant Compliance** | 100% | 100% | **100%** |

> **Safety Invariant Rule**: No ungradable image produced a reassuring 'No DR' result; all were correctly routed to `MANDATORY_RECAPTURE`.

---

## 2. Test Cohort Predictions & Triage Classification (20 Test Patients)

| Image ID | Split | GT Label | Quality Gate | Predicted Grade | Calibrated Conf | Triage Recommendation | Action Plan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `01_test.png` | `test` | **No DR** | o. Gradable | **No DR** | 34.9% | `ROUTINE_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `02_test.png` | `test` | **No DR** | o. Gradable | **Mild** | 36.8% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `03_test.png` | `test` | **Early DR** | ?O UNGRA_LOW_CONTRAST | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `04_test.png` | `test` | **No DR** | o. Gradable | **Mild** | 34.6% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `05_test.png` | `test` | **No DR** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `06_test.png` | `test` | **No DR** | ?O UNGRA_LOW_CONTRAST | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `07_test.png` | `test` | **No DR** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `08_test.png` | `test` | **Early DR** | o. Gradable | **Mild** | 26.6% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `09_test.png` | `test` | **No DR** | o. Gradable | **Mild** | 34.5% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `10_test.png` | `test` | **No DR** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `11_test.png` | `test` | **No DR** | o. Gradable | **Mild** | 31.7% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `12_test.png` | `test` | **No DR** | o. Gradable | **No DR** | 40.3% | `ROUTINE_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `13_test.png` | `test` | **No DR** | o. Gradable | **No DR** | 36.4% | `ROUTINE_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `14_test.png` | `test` | **Early DR** | o. Gradable | **Proliferative DR** | 26.7% | `IMMEDIATE_EVALUATION_1_WEEK` | Proliferative Diabetic Retinopathy (PDR) dete... |
| `15_test.png` | `test` | **No DR** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `16_test.png` | `test` | **No DR** | o. Gradable | **No DR** | 35.1% | `ROUTINE_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `17_test.png` | `test` | **Early DR** | o. Gradable | **Severe NPDR** | 33.2% | `URGENT_EVALUATION_1_TO_2_WEEKS` | Severe Non-Proliferative Diabetic Retinopathy... |
| `18_test.png` | `test` | **No DR** | o. Gradable | **Severe NPDR** | 57.7% | `URGENT_EVALUATION_1_TO_2_WEEKS` | Severe Non-Proliferative Diabetic Retinopathy... |
| `19_test.png` | `test` | **No DR** | ?O UNGRA_LOW_CONTRAST | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `20_test.png` | `test` | **No DR** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |

---

## 3. Ground-Truth Anatomical FOV Validation
- The DRIVE benchmark provides pixel-level ground-truth binary masks for retinal field boundaries (`mask/`).
- Mean Retinal Area Coverage measured against ground truth masks: **68.8%**.
- Grad-CAM heatmap overlays were exported to `reports/drive_screening/` and validated to fall within anatomical retinal boundaries.

---
*Notice: This system is a Clinical Decision Support Screening AID, not an autonomous diagnostic device. Definitive diagnostic decisions must always be made by a qualified ophthalmologist.*
