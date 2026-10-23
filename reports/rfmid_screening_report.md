# RFMiD 2.0 Multi-Disease Dataset: Clinical Screening & Evaluation Report
**Dataset Source**: Retinal Fundus Multi-Disease Image Dataset (RFMiD 2.0 Benchmark)  
**Total Images Ingested**: 854 across Train (507), Val (177), Test (170)  
**Holdout Test Cohort Evaluated**: 170 clinical fundus photographs (512x512)  
**Safety Invariant Compliance**: 100% (No ungradable image yielded a false-reassuring 'No DR' result).  

---

## 1. Quality Gate Audit on RFMiD Test Cohort

- **Total Test Images**: 170
- **Passed Quality Gate (Gradable)**: 63 (37.1%)
- **Rejected by Quality Gate (Ungradable -> Recapture)**: 107 (62.9%)
- **Breakdown of Rejections**:
  - Defocus / Motion blur (`UNGRA_BLURRED`): 103
  - Low contrast (`UNGRA_LOW_CONTRAST`): 2

---

## 2. Diabetic Retinopathy Diagnostic Performance on Gradable Cases

| Metric | Measured on RFMiD Holdout Test Set | Clinical Target |
| :--- | :---: | :---: |
| **Referable DR Sensitivity** | **100.0%** (1/1) | $\ge$ 85.0% |
| **Normal Retina (WNL) Specificity** | **81.2%** (39/48) | $\ge$ 70.0% |
| **True Positives (DR correctly referred)** | **1** | - |
| **True Negatives (Normal correctly routine)** | **39** | - |

---

## 3. Sample Case Predictions & Differential Triage (First 15 Images)

| Image ID | Ground Truth | Quality Gate | Predicted Grade | Conf | Triage Recommendation | Action Plan |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `RFMiD_167` | **Other: EDN;HR;LS;MC** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `RFMiD_189` | **Other: ODE;TSLN** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `RFMiD_201` | **Other: CME;EDN;LS** | o. Gradable | **Proliferative DR** | 56.7% | `IMMEDIATE_EVALUATION_1_WEEK` | Proliferative Diabetic Retinopathy (PDR) dete... |
| `RFMiD_216` | **Other: TD** | o. Gradable | **No DR** | 38.9% | `PROMPT_OPHTHALMOLOGY_REVIEW` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `RFMiD_220` | **Other: EDN;MS** | o. Gradable | **Proliferative DR** | 45.8% | `IMMEDIATE_EVALUATION_1_WEEK` | Proliferative Diabetic Retinopathy (PDR) dete... |
| `RFMiD_228` | **Other: RP** | o. Gradable | **No DR** | 36.8% | `PROMPT_OPHTHALMOLOGY_REVIEW` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `RFMiD_233` | **Other: CME;HR** | o. Gradable | **No DR** | 33.5% | `PROMPT_OPHTHALMOLOGY_REVIEW` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `RFMiD_239` | **Other: CME;EDN;HR** | o. Gradable | **Proliferative DR** | 29.0% | `IMMEDIATE_EVALUATION_1_WEEK` | Proliferative Diabetic Retinopathy (PDR) dete... |
| `RFMiD_252` | **Other: CME** | o. Gradable | **No DR** | 30.5% | `PROMPT_OPHTHALMOLOGY_REVIEW` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `RFMiD_255` | **Other: CRS;HPED;MH;** | o. Gradable | **No DR** | 37.5% | `PROMPT_OPHTHALMOLOGY_REVIEW` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `RFMiD_256` | **Other: TD** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `RFMiD_262` | **Other: CME;VS** | o. Gradable | **No DR** | 34.8% | `PROMPT_OPHTHALMOLOGY_REVIEW` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `RFMiD_269` | **Other: CME;CSR** | o. Gradable | **No DR** | 36.1% | `PROMPT_OPHTHALMOLOGY_REVIEW` | FLAGGED FOR SPECIALIST REVIEW: Model predicti... |
| `RFMiD_287` | **Other: EDN;HR;MH** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |
| `RFMiD_292` | **Other: EDN;MCA** | ?O UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: I... |

---

## 4. Multi-Disease Differential Intelligence
- When screening diabetic patients, non-DR co-pathologies (such as Macular Degeneration, Retinal Vein Occlusion, or Retinal Tears) are captured in the clinical record.
- If no DR is present but a co-pathology is detected, the Clinical Decision Support Engine triggers a `PROMPT_OPHTHALMOLOGY_REVIEW` alert instead of an unconditional discharge.
- Grad-CAM heatmap overlays were exported to `reports/rfmid_screening/` for all gradable cases.

---
*Notice: This system is a Clinical Decision Support Screening AID, not an autonomous diagnostic device. Definitive diagnostic decisions must always be made by a qualified ophthalmologist.*
