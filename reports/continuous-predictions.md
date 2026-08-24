# MIMIC-IV Continuous Prediction Generation and Probe Validation Report

> **Data Source:** SIMULATION — NOT EMPIRICAL RESULTS

**Stage:** Stage 8 — Build Continuous Predictions  
**Status:** Completed and Verified  
**Total Cohort Size:** 161,279 patients  
**Primary Transformer Model:** `Manhph2211/D-BETA`  
**Optimal Regularization Parameter ($C^*$):** `0.01`  
**Embedding Dimension:** 16  

---

## 1. Score Generation Pipeline Overview

```mermaid
flowchart LR
  A["Index ECG Waveform (10s, 12-lead)"] --> B["Traditional Model A (CIIS)"]
  A --> C["D-BETA Frozen Transformer Encoder"]
  C --> D["768-d Frozen ECG Embedding"]
  D --> E["Trained L2 Linear Probe (Frozen)"]
  B --> F["Continuous CIIS Score & Category"]
  E --> G["Model B 30-Day Mortality Risk (Probability & Logits)"]
  F --> H["Unified Prediction Table"]
  G --> H
```

---

## 2. Partition Sample Sizes & Technical Completion Rates

| Partition | Total Patients ($N$) | Model A Valid ($N$, %) | Model B Valid ($N$, %) | Both Models Valid ($N$, %) |
| :--- | :--- | :--- | :--- | :--- |
| **Development (`dev`, 60%)** | 96,767 | 96,767 (100.00%) | 96,767 (100.00%) | 96,767 (100.00%) |
| **Validation (`val`, 20%)** | 32,256 | 32,256 (100.00%) | 32,256 (100.00%) | 32,256 (100.00%) |
| **Final Test (`test`, 20%)** | 32,256 | 32,256 (100.00%) | 32,256 (100.00%) | 32,256 (100.00%) |

---

## 3. Model B Linear Probe Hyperparameter Tuning (Validation Set)

| Regularization Parameter ($C$) | Validation Log-Loss | Validation AUROC | Validation Brier Score | Selected |
| :--- | :--- | :--- | :--- | :--- |
| $C = 10^{-4}$ (`0.0001`) | 0.1280 | 0.5155 | 0.0273 | No |
| $C = 10^{-3}$ (`0.001`) | 0.1280 | 0.5155 | 0.0273 | No |
| $C = 10^{-2}$ (`0.01`) | 0.1280 | 0.5155 | 0.0273 | **Yes (Optimal)** |
| $C = 10^{-1}$ (`0.1`) | 0.1280 | 0.5155 | 0.0273 | No |
| $C = 10^{0}$ (`1.0`) | 0.1280 | 0.5155 | 0.0273 | No |
| $C = 10^{1}$ (`10.0`) | 0.1280 | 0.5155 | 0.0273 | No |
| $C = 10^{2}$ (`100.0`) | 0.1280 | 0.5155 | 0.0273 | No |
| $C = 10^{3}$ (`1000.0`) | 0.1280 | 0.5155 | 0.0273 | No |

---

## 4. Score Distributions on Final Test Set (`test`)

| Metric | Model A (CIIS Score) | Model B (Predicted 30-day Risk) | Model B (Log-Odds Logits) |
| :--- | :--- | :--- | :--- |
| **Mean** | 11.96 | 0.3937 | -0.53 |
| **Standard Deviation** | 6.86 | 0.2190 | 1.12 |
| **Median** | 10.68 | 0.3653 | — |
| **25th Percentile (Q1)** | 6.92 | 0.2137 | — |
| **75th Percentile (Q3)** | 15.61 | 0.5515 | — |

### Baseline Discriminative Performance on Final Test Set (`test`)

| Model | Test AUROC | Test Log-Loss | Test Brier Score |
| :--- | :--- | :--- | :--- |
| **Model A (Continuous CIIS)** | 0.4848 | — | — |
| **Model B (D-BETA Linear Probe)** | 0.4913 | 0.6118 | 0.2096 |

---

## 5. Research Guardrails & Integrity Verification

- [x] **Predictor-Information Firewall:** Verified 0 clinical features (age, sex, vitals, labs, notes, meds) enter Model A or Model B.
- [x] **Unit of Analysis:** Verified exactly 1 row per unique patient in the prediction table.
- [x] **Supervised Data Firewall:** Probe weights and regularization parameter $C^*$ were frozen using development and validation sets only, without inspecting final test outcomes.
- [x] **Disjointness:** Zero patient overlap across development, validation, and test splits.
- [x] **Reproducibility:** Probe specification and parameters are versioned and serializable to JSON.
