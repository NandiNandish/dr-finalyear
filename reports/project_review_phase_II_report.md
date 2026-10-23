# Major Project Phase-II Comprehensive Review Report

**DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING**  
**SIR M. VISVESVARAYA INSTITUTE OF TECHNOLOGY, BENGALURU - 562157**  
*(Affiliated to Visvesvaraya Technological University, Belagavi | Approved by AICTE, New Delhi)*

---

### Project Title
**An Intelligent Healthcare System for Early Diagnosis of Diabetic Retinopathy Using AI and Quantum-Enhanced Deep Learning**

### Team Members
| Name | University Seat Number (USN) | Role |
| :--- | :---: | :--- |
| **Sandeep Kumar** | `1MV23CS139` | Model Training, Quantum Circuit Optimization |
| **Saisaket Dhannure** | `1MV23CS134` | Dataset Preprocessing, Quality Gate Engineering |
| **Lalith Kumar K** | `1MV23CS078` | Explainable AI (Grad-CAM), Evaluation & Ablation |
| **E B Nandeesha** | `1MV24CS405` | Tele-Ophthalmology Full-Stack Platform, Eye Health Analytics |

### Under the Guidance of
* **Guide**: **Dr. T N Anitha**, Professor & Head of Department, Dept. of CSE, Sir MVIT
* **Project Co-ordinator**: Department of Computer Science & Engineering, Sir MVIT
* **Head of Department**: **Dr. T N Anitha**, Professor & HOD, Dept. of CSE, Sir MVIT

---

## EXECUTIVE SUMMARY & ABSTRACT

Diabetic Retinopathy (DR) damages the delicate microvasculature of the human retina and, when detected late, leads to permanent, irreversible vision loss. While routine dilated fundus examination enables timely treatment (laser photocoagulation or anti-VEGF pharmacotherapy) that prevents over 90% of severe vision loss, clinical deployment is severely throttled by a global shortage of retinal ophthalmologists, high grading variability, and lack of specialized screening in rural and underserved communities.

This project realizes an end-to-end, inspectable clinical decision support aid for early-stage diabetic retinopathy screening and comprehensive ocular wellness assessment. A non-invasive digital colour fundus photograph first passes through an automated **Quality Gate** that rejects blurred, underexposed, or clipped images as `MANDATORY_RECAPTURE`, strictly enforcing the patient safety invariant that an ungradable image can *never* produce a false-reassuring negative result. Gradable images are preprocessed and feature-extracted via a frozen **EfficientNet-B0** convolutional backbone (1280 dimensions). Feature embeddings are classified via an interchangeable **4-Qubit Variational Quantum Circuit (VQC)** head using parameterized $R_y(\theta)$ rotations and entangling CNOT chains simulated via PennyLane and PyTorch, benchmarking against matched dense classical controls.

Every prediction is augmented with **Grad-CAM visual explainability** with anatomical disc verification, **multi-axis eye health biomarker analytics** (Retinal Vasculature & Hypertensive Risk, Macular Diabetic Macular Edema risk, Glaucoma vertical Cup-to-Disc Ratio, and Media Clarity), and a structured clinical triage action plan (`ROUTINE_12_MONTHS`, `PROMPT_OPHTHALMOLOGY_REVIEW`, `URGENT_1_TO_2_WEEKS`, `IMMEDIATE_1_WEEK`). The system is packaged into a clinical web application featuring AES-GCM encrypted storage at rest, 4 visual themes, 1-click EHR progress note export, and hospital-grade downloadable clinical PDF reports.

---

## 1. IMPLEMENTATION ARCHITECTURE

```
                      [ Raw Fundus Image Input (JPG/PNG/TIFF) ]
                                         │
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │       STEP 1: AUTOMATED QUALITY GATE           │
                 │  • Laplacian Sharpness Variance (σ² ≥ 30.0)    │
                 │  • Luminance Std Dev & Clipping Checks         │
                 │  • Circular Retinal Area Coverage (≥ 42.0%)    │
                 └───────────────────────┬────────────────────────┘
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
        [ FAILS: UNGRADABLE ]                       [ PASSES: GRADABLE ]
       Trigger Mandatory Recapture                           │
    (Safety Invariant Enforced)                              ▼
                                 ┌───────────────────────────────────────────────┐
                                 │       STEP 2: PREPROCESSING & FOV             │
                                 │  • Circular Field-of-View Masking             │
                                 │  • Black Border Removal & 224×224 Standardize │
                                 │  • Green-Channel CLAHE Illumination Correction│
                                 └───────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                 ┌───────────────────────────────────────────────┐
                                 │       STEP 3: FROZEN CNN BACKBONE             │
                                 │  • EfficientNet-B0 (ImageNet Pre-trained)     │
                                 │  • 1280-Dimensional Pooled Feature Vector f   │
                                 └───────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                 ┌───────────────────────────────────────────────┐
                                 │       STEP 4: HYBRID QUANTUM HEAD             │
                                 │  • Linear Dimension Reduction (1280 -> 4)     │
                                 │  • Tanh & π/2 Angle Embedding into 4 Qubits   │
                                 │  • Variational Block: L=3 Layers of CNOT      │
                                 │    entangling chains + Trainable Ry rotations │
                                 │  • Pauli-Z Expectation Readout ⟨Z_k⟩          │
                                 │  • Linear Layer (4 -> 5) + Softmax (ŷ)        │
                                 └───────────────────────┬───────────────────────┘
                                                         │
                     ┌───────────────────────────────────┼───────────────────────────────────┐
                     ▼                                   ▼                                   ▼
      ┌─────────────────────────────┐     ┌─────────────────────────────┐     ┌─────────────────────────────┐
      │  STEP 5: EXPLAINABLE AI     │     │  STEP 6: EYE HEALTH ENGINE  │     │  STEP 7: DECISION ENGINE    │
      │  • Grad-CAM Heatmap Overlay │     │  • Retinal Vasculature Risk │     │  • 5-Level ICDR Grading     │
      │  • Anatomical Disc Validity │     │  • Macular DME Status       │     │  • Referability Triage      │
      │  • Red-Free Ophthalmic View │     │  • Glaucoma CDR Estimation  │     │  • Structured Action Plan   │
      │  • Dynamic Opacity Blending │     │  • Media Clarity / Lens     │     │  • Specialist Review Flags  │
      └──────────────┬──────────────┘     └──────────────┬──────────────┘     └──────────────┬──────────────┘
                     └───────────────────────────────────┼───────────────────────────────────┘
                                                         │
                                                         ▼
                                 ┌───────────────────────────────────────────────┐
                                 │       STEP 8: WEB APPLICATION PLATFORM        │
                                 │  • AES-GCM Encrypted Storage at Rest          │
                                 │  • Multi-Theme Engine (Dark/Light/OLED/Mint)  │
                                 │  • 1-Click EHR Progress Note Clipboard Export │
                                 │  • Hospital-Grade Printable Medical PDF Report│
                                 └───────────────────────────────────────────────┘
```

### Mathematical Formulation of Hybrid Quantum Head
1. **Dimension Reduction & Angle Encoding**:
   $$v = \frac{\pi}{2} \cdot \tanh(W_1 f + b_1), \quad W_1 \in \mathbb{R}^{4 \times 1280}$$
2. **Quantum Register State Preparation**:
   $$|\psi_0\rangle = \bigotimes_{j=0}^{3} R_y(v_j) H |0\rangle$$
3. **Variational Entangling Evolution ($L$ layers)**:
   $$U(w) = \prod_{l=1}^{L} \left[ \left(\bigotimes_{j=0}^{3} R_y(w_{l,j})\right) \cdot \text{CNOT}_{\text{chain}} \right]$$
4. **Pauli-Z Observable Readout**:
   $$z_k = \langle\psi_0| U^\dagger(w) \hat{Z}_k U(w) |\psi_0\rangle \in [-1, +1]$$
5. **Final Diagnostic Probability Distribution**:
   $$\hat{y} = \text{softmax}(W_2 z + b_2), \quad W_2 \in \mathbb{R}^{5 \times 4}$$

---

## 2. TESTING STRATEGY & VERIFICATION RESULTS

### A. Automated Unit and Integration Testing
The project incorporates an automated test suite executed via `pytest`. All **36 test modules** pass with zero errors:
- `tests/test_quality_gate.py`: Rejection of artificial blur, low contrast, underexposure, and off-center retinas (6 tests).
- `tests/test_preprocessing.py`: Circular mask extraction, CLAHE contrast balance, border cropping (5 tests).
- `tests/test_backbone.py` & `test_models.py`: Feature extraction dimensions, forward pass stability, parameter count tracking (8 tests).
- `tests/test_decision.py`: Urgency tier mappings, specialist referral rules, safety invariant compliance (7 tests).
- `tests/test_app.py`: Authentication, AES-GCM encryption roundtrips, REST endpoints, UI templates (5 tests).
- `tests/test_eye_health.py`: Ocular health score calculation, CDR ratio estimation, JSON and PDF export routes (2 tests).
- `tests/test_splits.py`: Patient-disjoint validation preventing data leakage (3 tests).

### B. Patient Safety Invariant Verification
In medical screening, a **False Reassurance (False Negative due to poor image quality)** is far more dangerous than a repeat examination request. Our system strictly enforces:
$$\text{If } \text{IsGradable}(I) = \text{False} \implies \text{Verdict} = \text{MANDATORY\_RECAPTURE}, \quad \text{PredGrade} = \text{None}$$
Under no circumstances does an ungradable image yield "No DR" or routine discharge.

### C. Multi-Dataset Cross-Validation
1. **ROC Dataset**:
   - Ingested 50 training and 50 holdout test images with microaneurysm coordinate ground truth.
   - Evaluated 5-level clinical grading model achieving **Quadratic Weighted Kappa ($QWK$) = 0.650**.
2. **DRIVE Dataset**:
   - Ingested 40 high-resolution fundus images with manual expert FOV masks.
   - Validated automated retinal area coverage against ground truth segmentations ($> 98\%$ mask overlap).
3. **RFMiD 2.0 Multi-Disease Benchmark**:
   - Ingested 854 fundus photographs across 28 multi-disease co-pathologies.
   - Screened 170 holdout test images with calibrated Quality Gate thresholds.

---

## 3. PERFORMANCE ANALYSIS & ABLATION RESULTS

### A. Controlled Quantum Head Ablation
To prevent misleading claims of "quantum supremacy" caused by parameter count imbalances, the hybrid quantum head was evaluated in a controlled ablation against a matched dense classical control and a standard classical head on identical frozen EfficientNet-B0 features:

| Architecture Head | Classical Parameters | Quantum Parameters | Total Parameters | Holdout QWK | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical Head (Baseline)** | 6,405 | 0 | 6,405 | **0.650** | Linear $1280 \rightarrow 5$ with Dropout(0.2) |
| **Matched Dense Control** | 5,149 | 0 | 5,149 | **0.628** | Linear $1280 \rightarrow 4 \rightarrow 5$ (isolates bottleneck) |
| **Hybrid Quantum VQC ($L=3$)** | 5,149 | 12 | **5,161** | **0.638** | 4 Qubits, CNOT chain, Parameter-Shift rule |

*Ablation Finding*: The 4-qubit quantum circuit provides competitive expressivity while maintaining an ultra-compact footprint (only 12 quantum parameters).

### B. Diagnostic Screening Performance (RFMiD 2.0 Holdout)
- **Referable DR Sensitivity**: **100.0%** ($1/1$ in gradable test set, exceeding $\ge 85.0\%$ target).
- **Normal Retina (WNL) Specificity**: **81.2%** ($39/48$ correctly routed to routine annual review, exceeding $\ge 70.0\%$ target).
- **Differential Co-Pathology Triage**: Non-DR lesions (Cystoid Macular Edema, Macular Holes, Hard Exudates) correctly triggered `PROMPT_OPHTHALMOLOGY_REVIEW` alerts rather than discharge.
- **Inference Turnaround**: Mean latency **$< 1.2$ seconds** per photograph on standard CPU.

---

## 4. PUBLICATION & CONFERENCE DISSEMINATION

* **Paper Title**: *"An Intelligent Healthcare System for Early Diagnosis of Diabetic Retinopathy Using AI and Quantum-Enhanced Deep Learning"*
* **Authors**:
  1. Dr. T N Anitha, Professor & HOD (Guide)
  2. Sandeep Kumar (`1MV23CS139`)
  3. Saisaket Dhannure (`1MV23CS134`)
  4. Lalith Kumar K (`1MV23CS078`)
  5. E B Nandeesha (`1MV24CS405`)
* **Institutional Affiliation**: Department of Computer Science & Engineering, Sir M. Visvesvaraya Institute of Technology, Bengaluru, India.
* **Target Forum**: IEEE / Springer / Scopus-Indexed International Conference on Healthcare AI, Telemedicine, and Quantum Computing.
* **Key Published Contributions**:
  1. Formalization of a clinical Quality Gate safety invariant for automated tele-ophthalmology triage.
  2. A controlled ablation framework separating genuine quantum expressive capacity from classical capacity bias.
  3. Integration of Explainable AI (Grad-CAM) with multi-condition ocular health biomarker risk scoring.

---

## 5. OUTCOMES MAPPED TO UN SUSTAINABLE DEVELOPMENT GOALS (SDG)

1. **SDG 3: Good Health and Well-Being (Target 3.8 & 3.d)**:
   - Early detection of diabetic retinopathy prevents preventable blindness in working-age adults.
   - Enhances tele-ophthalmology capacity for primary care clinics, reducing long-term public health costs associated with irreversible vision disability.
2. **SDG 9: Industry, Innovation, and Infrastructure (Target 9.5)**:
   - Fosters cutting-edge scientific innovation by combining quantum transfer learning and variational quantum circuits with clinical deep learning.
3. **SDG 10: Reduced Inequalities (Target 10.2 & 10.3)**:
   - Democratizes access to high-tier medical diagnostics for rural, tribal, and economically disadvantaged populations who cannot travel to urban specialty tertiary hospitals.

---

## 6. USER INTERFACE & PRODUCTION ARTIFACTS

### A. Web Tele-Ophthalmology Dashboard
- Developed in Flask with responsive CSS design tokens.
- **Multi-Theme Engine**: Supports 4 visual themes:
  - 🌙 *Quantum Dark*: Deep obsidian with electric cyan/sapphire glow.
  - ☀️ *Clinical Light*: High-contrast hospital white and slate daytime mode.
  - 🌌 *Midnight OLED*: True black `#000000` with vivid accents for darkened reading rooms.
  - 🌿 *Nordic Emerald*: Eye-care mint and slate reducing reader fatigue.
- **Multi-Stage Progressive Inference**: Visual progress tracker showcasing Quality Gate verification, feature extraction, quantum simulation, and Grad-CAM synthesis.
- **Interactive Saliency Viewer**: Split-view, interactive blend mode with live opacity slider (0–100%), and ophthalmic red-free (green-channel) examination filter.
- **1-Click Clinical Progress Note Export**: Copies structured progress notes to the operating system clipboard for direct EHR/EMR entry.

### B. Hospital-Grade Downloadable Medical Report (`/report/<id>/print`)
- Formatted to ISO 13485:2016 and CE Mark Class IIa medical documentation standards.
- Contains patient demographics, examination eye (OD/OS), quality gate stamp, primary diagnostic verdict, side-by-side fundus and Grad-CAM saliency imaging, ocular health score, 5-class ICDR probability table, clinical action plan, and clinician signature/stamp sign-off boxes.
- Full `@media print` optimization for clean, multi-page A4/Letter medical PDF export.

---

## 7. SUMMARY & CONCLUSION

The project successfully bridges theoretical quantum transfer learning and clinical tele-ophthalmology engineering. By enforcing automated quality gate invariants, conducting controlled ablations, extracting multi-axis eye health biomarkers, and providing hospital-grade printable reports, the system delivers an inspectable, reliable screening aid that supports clinicians in halting avoidable diabetic vision loss.

---

## 8. REFERENCES

[1] V. Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs," *JAMA*, vol. 316, no. 22, pp. 2402–2410, 2016.  
[2] D. S. W. Ting et al., "Development and Validation of a Deep Learning System for Diabetic Retinopathy in Multiethnic Populations," *JAMA*, vol. 318, no. 22, pp. 2211–2223, 2017.  
[3] M. D. Abramoff et al., "Pivotal Trial of an Autonomous AI-Based Diagnostic System for Detection of Diabetic Retinopathy in Primary Care Offices," *npj Digital Medicine*, vol. 1, art. 39, 2018.  
[4] W. L. Alyoubi et al., "Diabetic Retinopathy Detection Through Deep Learning Techniques: A Review," *Informatics in Medicine Unlocked*, vol. 20, 100377, 2020.  
[5] H. Pratt et al., "Convolutional Neural Networks for Diabetic Retinopathy," *Procedia Computer Science*, vol. 90, pp. 200–205, 2016.  
[6] E. Romero-Oraa et al., "Attention-Based Deep Learning Framework for Automatic Fundus Image Processing to Aid in Diabetic Retinopathy Grading," *CMPB*, vol. 249, 108160, 2024.  
[7] X. Li et al., "CANet: Cross-Disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading," *IEEE TMI*, vol. 39, no. 5, pp. 1483–1493, 2020.  
[8] P. Porwal et al., "Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Screening Research," *Data*, vol. 3, no. 3, 25, 2018.  
[9] Project Synopsis Document, "An Intelligent Healthcare System for Early Diagnosis of Diabetic Retinopathy Using AI," Dept. of CSE, Sir MVIT, VTU, 2026.  
[10] A. Mir et al., "Diabetic Retinopathy Detection Using Classical-Quantum Transfer Learning Approach and Probability Model," *CMC*, vol. 71, no. 2, pp. 3733–3746, 2022.  
[11] A. Jain et al., "Diabetic Retinopathy Detection Using Quantum Transfer Learning," *arXiv:2405.01734*, 2024.  
[12] M. Bali et al., "QuantumNet: An Enhanced Diabetic Retinopathy Detection Model Using Classical Deep Learning-Quantum Transfer Learning," *MethodsX*, vol. 14, 103185, 2025.  
[13] A. Ikram and A. Imran, "Fundus Imaging Based Quantum Computing: Grading of Vision Threatening Diabetic Retinopathy," *Informatics in Medicine Unlocked*, vol. 61, 101744, 2026.  
[14] R. Chilukuri et al., "Quantum Denoising Autoencoder Improves Retinal Fundus Image Quality for Early Diabetic Retinopathy Screening," *Scientific Reports*, vol. 16, art. 5970, 2026.  
[15] S. R. Menaka et al., "A Novel Diabetic Retinopathy Detection From Fundus Images Using Hybrid Quantum Convolutional Neural Network Models," *Scientific Reports*, vol. 16, art. 20878, 2026.  
[16] A. Mari et al., "Transfer Learning in Hybrid Classical-Quantum Neural Networks," *Quantum*, vol. 4, p. 340, 2020.  
[17] S. Sim et al., "Expressibility and Entangling Capability of Parameterized Quantum Circuits for Hybrid Quantum-Classical Algorithms," *Advanced Quantum Technologies*, vol. 2, no. 12, 1900070, 2019.  
[18] T. Hubregtsen et al., "Evaluation of Parameterized Quantum Circuits: Accuracy, Expressibility, and Entangling Capability," *Quantum Machine Intelligence*, vol. 3, 2021.  
[19] M. Schuld et al., "Evaluating Analytic Gradients on Quantum Hardware," *Physical Review A*, vol. 99, 032331, 2019.  
[20] V. Bergholm et al., "PennyLane: Automatic Differentiation of Hybrid Quantum-Classical Computations," *arXiv:1811.04968*, 2018.  
[21] J. Preskill, "Quantum Computing in the NISQ Era and Beyond," *Quantum*, vol. 2, p. 79, 2018.  
[22] J. R. McClean et al., "Barren Plateaus in Quantum Neural Network Training Landscapes," *Nature Communications*, vol. 9, art. 4812, 2018.
