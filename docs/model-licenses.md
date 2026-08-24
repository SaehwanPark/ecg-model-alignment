# Model Licenses, Data Governance, and Ethics

This document establishes the licensing terms, data governance requirements, ethical constraints, and provenance documentation for all datasets, foundation models, and software utilized in the ECG Model Alignment project.

---

## 1. Datasets

### 1.1 MIMIC-IV v3.1
- **Source:** PhysioNet / MIT Laboratory for Computational Physiology
- **URL:** [https://physionet.org/content/mimiciv/3.1/](https://physionet.org/content/mimiciv/3.1/)
- **License:** [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/)
- **Requirements & Governance:**
  - Mandatory CITI "Data or Specimens Only Research" training completion.
  - Signed Data Use Agreement (DUA) strictly prohibiting re-identification attempts.
  - No redistribution or public hosting of row-level clinical or patient records.
  - Protected Health Information (PHI) firewall: Clinical variables (diagnoses, labs, medications, notes) may only be used for cohort construction, linkage, outcome definition, and evaluation stratification—never as predictor features for Model `A` or Model `B`.

### 1.2 MIMIC-IV-ECG v1.0
- **Source:** PhysioNet / MIT Laboratory for Computational Physiology
- **URL:** [https://physionet.org/content/mimic-iv-ecg/1.0/](https://physionet.org/content/mimic-iv-ecg/1.0/)
- **License:** [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/)
- **Requirements & Governance:**
  - Credentialed PhysioNet researcher access.
  - Raw WFDB signals (`.dat` / `.hea`) and machine measurement tables (`machine_measurements.csv`) stored locally outside version control.
  - Waveform data serve as the sole permitted patient-specific predictor input for both Model `A` and Model `B`.

---

## 2. Foundation Models and Transformers (Model `B`)

### 2.1 D-BETA (Primary Multimodal Transformer)
- **Reference:** Pham Hung M, Saeed A, Ma D. *Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners.* ICML 2025.
- **Hugging Face Hub:** [`Manhph2211/D-BETA`](https://huggingface.co/Manhph2211/D-BETA)
- **Exact Revision:** `20ff3ccce1759d7d629171e15befafa9a424d2ca`
- **License:** Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
- **Terms of Use:** Non-commercial, research and academic use only.
- **Pretraining Contamination Disclosure:** Pretrained multimodally on MIMIC-IV-ECG matched text reports. All MIMIC-IV evaluations represent **in-domain representation probing**, not independent external validation.

### 2.2 CarDSLab ECG-CLIP (Engineering & Secondary Transformer Prototype)
- **Reference:** Oikonomou EK, et al. *TARGET-AI: a foundational approach for the targeted deployment of artificial intelligence electrocardiography in the electronic health record.* medRxiv 2025.
- **Hugging Face Hub:** [`CarDSLab/ecg-clip-beit-base-384`](https://huggingface.co/CarDSLab/ecg-clip-beit-base-384)
- **Exact Revision:** `80131ef06310dd1c8f2efe9082709c82433dc66e`
- **License:** Open Research / MIT License via Hugging Face Hub
- **Terms of Use:** Academic and research evaluation.
- **Pretraining Contamination Disclosure:** Pretrained on large-scale rendered 12-lead ECG images. Used for rendering validation and secondary sensitivity comparison.

### 2.3 PULSE-7B (Secondary Multimodal Large Language Model)
- **Reference:** Liu R, Bai Y, Yue X, et al. *Teaching multimodal LLMs to comprehend 12-lead electrocardiographic images.* npj Digital Medicine. 2026;9:349.
- **Repository:** [`PULSE-7B`](https://github.com/PULSE-ECG/PULSE)
- **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Terms of Use:** Open source for academic and commercial use with attribution.
- **Pretraining Contamination Disclosure:** Pretrained on ECGInstruct which incorporates MIMIC-IV-ECG.

---

## 3. Traditional Model `A`

### 3.1 Cardiac Infarction/Injury Score (CIIS)
- **Reference:** Richardson RT, et al. *A new ECG scoring system for the detection of myocardial infarction and ischemic injury: the Cardiac Infarction Injury Score (CIIS).* Circulation. 1981;64(2):292-302.
- **Intellectual Property:** Classical public domain clinical scoring system.
- **Implementation:** Re-implemented in pure Python from published formulas and feature weight tables under `src/ecg_alignment/scoring/traditional.py`.

### 3.2 Sokolow-Lyon & Cornell Voltage Criteria
- **References:**
  - Sokolow M, Lyon TP. *The ventricular complex in left ventricular hypertrophy as obtained by unipolar precordial and limb leads.* Am Heart J. 1949.
  - Casale PN, et al. *Electrocardiographic detection of left ventricular hypertrophy: development and clinical validation of the Cornell voltage criterion.* Am J Cardiol. 1985.
- **Implementation:** Public domain standard clinical criteria implemented in `src/ecg_alignment/scoring/traditional.py`.

---

## 4. Software Dependencies and Tooling

| Software | Version | License | Purpose |
| :--- | :--- | :--- | :--- |
| **Python** | `>=3.13` | PSF License | Runtime environment |
| **uv** | `>=0.5.0` | Apache 2.0 / MIT | Environment & package manager |
| **Polars** | `>=1.44.0` | MIT | High-performance dataframe processing |
| **PyTorch** | `>=2.13.0` | Modified BSD | Deep learning runtime & tensor ops |
| **Transformers** | `>=5.15.1` | Apache 2.0 | Foundation model loading & inference |
| **Scikit-learn** | `>=1.9.0` | BSD-3-Clause | Linear probing & evaluation metrics |
| **SciPy** | `>=1.18.1` | BSD-3-Clause | Statistical tests & bootstrap CI |
| **WFDB** | `>=4.3.1` | MIT | PhysioNet waveform file reading |
| **Matplotlib** | `>=3.11.1` | PSF | Figure generation |
| **Pytest** | `>=9.1.1` | MIT | Test framework |
| **Basedpyright** | `>=1.39.10` | MIT | Static type analysis |
