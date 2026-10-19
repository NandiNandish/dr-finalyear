# Fundus Datasets: Provenance, Licences, and Label Harmonization

This directory documents the public, de-identified diabetic retinopathy datasets supported by this screening pipeline, their licensing restrictions, provenance, and canonical label mappings.

---

## 1. Supported Datasets

| Dataset | Provenance & Source | Primary Use | Image Count | Licence / Access Terms |
| :--- | :--- | :--- | :--- | :--- |
| **EyePACS** | Kaggle Diabetic Retinopathy Detection Challenge (California Health Care Foundation) | Training / Internal Validation / Internal Test | ~88,702 images | Open access for research via Kaggle challenge terms. Non-commercial, de-identified. |
| **IDRiD** | Indian Diabetic Retinopathy Image Dataset (IEEE Dataport) | Disease Grading + Lesion Reference | 516 fundus images | CC BY 4.0 licence. Clinical acquisition at Dr. Shroff's Charity Eye Hospital, New Delhi. |
| **DDR** | Dataset for Diabetic Retinopathy (BGI, China) | Multi-class Evaluation | 13,673 images | Academic research licence. Multi-hospital fundus screening cohort. |
| **Messidor** | Messidor French National Research Agency (Techno-Vision) | External Test Benchmark | 1,200 images | Research agreement required. 3 ophthalmology departments in France. |
| **Messidor-2** | Messidor-2 Project (LaTIM Laboratory, Brest, France) | External Test Benchmark | 1,748 images | Non-commercial research licence. |
| **APTOS 2019** | Asia Pacific Tele-Ophthalmology Society (Aravind Eye Hospital, India) | External Held-out Benchmark | 3,662 images | Kaggle competition research terms. Real-world screening clinic acquisitions. |

---

## 2. Canonical 5-Level Grading Scale (ICDR Standard)

All datasets are mapped into the International Clinical Diabetic Retinopathy (ICDR) 5-point severity scale:

| Canonical Label | Clinical Name | Description & Key Lesions | Binary Referable Status |
| :---: | :--- | :--- | :---: |
| **0** | **No DR** | No microaneurysms, hemorrhages, or exudates. Healthy retina. | Non-referable |
| **1** | **Mild NPDR** | Microaneurysms only. | Non-referable |
| **2** | **Moderate NPDR** | More than microaneurysms, but less than severe NPDR (cotton-wool spots, hard exudates, moderate hemorrhages). | **Referable** |
| **3** | **Severe NPDR** | 4-2-1 rule: $\ge 20$ intraretinal hemorrhages in each of 4 quadrants, definite venous beading in $\ge 2$ quadrants, or prominent IRMA in $\ge 1$ quadrant. | **Referable** |
| **4** | **Proliferative DR (PDR)** | Neovascularization (NVD/NVE), vitreous or preretinal hemorrhage, or fibrous proliferation. | **Referable** |

---

## 3. Dataset-Specific Mapping Rules

### EyePACS, IDRiD, DDR, APTOS 2019
These datasets natively follow the 0–4 scale and map directly:
- `0 -> 0`, `1 -> 1`, `2 -> 2`, `3 -> 3`, `4 -> 4`

### Messidor Original (4-Class Scale 0–3)
Messidor grades retinopathy severity based on microaneurysm count and hemorrhages:
- Grade 0 (0 MAs, 0 hemorrhages) $\to$ **0 (No DR)**
- Grade 1 (1–5 MAs, 0 hemorrhages) $\to$ **1 (Mild NPDR)**
- Grade 2 (6–14 MAs, $< 5$ hemorrhages, no neovascularization) $\to$ **2 (Moderate NPDR)**
- Grade 3 ($\ge 15$ MAs, or $\ge 5$ hemorrhages, or neovascularization) $\to$ **3 (Severe NPDR)** or **4 (PDR)** if neovascularization is recorded in clinical metadata.

---

## 4. Manifest Format Specification

Every dataset is represented in a unified CSV format with the columns:
- `path`: Relative or absolute path to the fundus image file.
- `label`: Canonical integer label (`0`, `1`, `2`, `3`, `4`).
- `dataset`: Dataset identifier (`eyepacs`, `idrid`, `ddr`, `messidor`, `aptos`, `synthetic`).
- `patient_id`: Unique patient identifier (used to guarantee patient-disjoint splits).
- `eye`: `left` or `right` if known, else `unknown`.
- `split`: `train`, `val`, `test`, or `external`.

---

## 5. Synthetic / Demo Dataset

For instant end-to-end testing, continuous integration, and reproducibility without requiring manual multi-gigabyte downloads, the repository provides:
- `data/synthetic_generator.py`: Generates anatomically-grounded synthetic fundus images across all 5 classes and deliberate ungradable samples, saving them to `data/demo/` and populating `data/manifests/demo_manifest.csv` and `data/manifests/external_test.csv`.
