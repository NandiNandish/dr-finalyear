# External Test Dataset Screening Report: ROC Dataset
**Dataset Source**: Retinopathy Online Challenge (ROC) Blind Test Set (`ROCtestImages/images`)  
**Active Inference Head**: CLASSICAL (Trained on ROC Specialist Annotations)  
**Total Images Processed**: 50

---

## 1. Executive Summary & Safety Invariant Breakdown

- **Total Evaluated Photographs**: 50
- **Images Passing Quality Gate (Gradable)**: 20 (40.0%)
- **Images Rejected by Quality Gate (Ungradable)**: 30 (60.0%)
- **Safety Invariant Compliance**: 100% (No ungradable image produced a false reassuring 'No DR' result).
- **Referable Cases Detected (Grade $\ge$ 2)**: 0 (0.0% of gradable cohort)

---

## 2. Predicted Diabetic Retinopathy Severity Distribution

| Severity Grade | Clinical Description | Predicted Cases | Action / Triage Recommendation |
| :---: | :--- | :---: | :--- |
| **Grade 0** | No Apparent Retinopathy | **4** | Routine Annual Follow-up |
| **Grade 1** | Mild Non-Proliferative DR | **16** | Rescreen in 6–12 Months |
| **Grade 2** | Moderate Non-Proliferative DR | **0** | Referable (Ophthalmology review within 4–6 weeks) |
| **Grade 3** | Severe Non-Proliferative DR | **0** | Referable (Prompt referral within 2–4 weeks) |
| **Grade 4** | Proliferative Retinopathy (PDR) | **0** | Urgent Referral (< 1–2 weeks) |
| **Ungradable** | Blur / Low Contrast / Insufficient Retina | **30** | Immediate Recapture |

---

## 3. Sample Case Predictions (First 15 Images)

| Image | Quality Status | Predicted Grade | Confidence | Triage Tier | Action Plan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `image0_test.jpg` | ❌ UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: Image ... |
| `image10_test.jpg` | ✅ Gradable | **Mild** | 33.3% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model prediction co... |
| `image11_test.jpg` | ✅ Gradable | **No DR** | 26.8% | `ROUTINE_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model prediction co... |
| `image12_test.jpg` | ✅ Gradable | **Mild** | 48.9% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model prediction co... |
| `image13_test.jpg` | ❌ UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: Image ... |
| `image14_test.jpg` | ❌ UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: Image ... |
| `image15_test.jpg` | ❌ UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: Image ... |
| `image16_test.jpg` | ✅ Gradable | **Mild** | 32.4% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model prediction co... |
| `image17_test.jpg` | ❌ UNGRA_LOW_CONTRAST | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: Image ... |
| `image18_test.jpg` | ❌ UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: Image ... |
| `image19_test.jpg` | ✅ Gradable | **Mild** | 43.0% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model prediction co... |
| `image1_test.jpg` | ❌ UNGRA_BLURRED | **Ungradable** | N/A | `MANDATORY_RECAPTURE` | Recapture colour fundus photograph. Reason: Image ... |
| `image20_test.jpg` | ✅ Gradable | **Mild** | 35.9% | `ROUTINE_6_TO_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model prediction co... |
| `image21_test.jpg` | ✅ Gradable | **No DR** | 38.2% | `ROUTINE_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model prediction co... |
| `image22_test.jpg` | ✅ Gradable | **No DR** | 34.4% | `ROUTINE_12_MONTHS` | FLAGGED FOR SPECIALIST REVIEW: Model prediction co... |

---

## 4. Explainable AI (XAI)
- Grad-CAM heatmap overlays were generated and exported to `reports/test_screening/` for all gradable cases.
- Clinical review is flagged whenever peak activation falls outside the segmented retinal disc.

---
*Notice: This system is a Clinical Decision Support Screening AID, not an autonomous diagnostic device. Definitive diagnostic decisions must always be made by a qualified ophthalmologist.*
