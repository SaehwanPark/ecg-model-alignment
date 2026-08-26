# Stage 9 Primary Analysis: ECG Risk Alignment, Discordance, and Incremental Information

> **Data Source:** REAL MIMIC-IV-ECG predictions

**Stage:** Stage 9 — Primary Analysis  
**Status:** Completed and Verified  
**Untouched Final Test Cohort Size ($N$):** 31,867 patients  
**Observed 30-Day Mortality Events ($Y=1$):** 924 (2.90%)  

---

## 1. Executive Summary & Hypotheses Evaluation

This report provides the empirical evaluation of the core scientific questions in the untouched final test partition:

1. **H1 (Positive Alignment):** Confirmed (Moderate Partial Alignment (0.30 <= |Rho| < 0.70)). Spearman rank correlation $\rho = 0.466$ ($p = 0.00e+00$) indicates statistically significant positive association with moderate partial alignment (0.30 <= |rho| < 0.70) between traditional Model `A` (CIIS) and transformer Model `B` (D-BETA probe).
2. **H2 (Residual Risk Gradients):** Partially Confirmed. Within fixed traditional risk categories, Model `B` identifies meaningful within-stratum mortality gradients in 3 of 4 categories (mean gradient ratio 14.40x).
3. **H3 (Clinically Informative Discordance):** Confirmed. Patients in the discordant `A-low / B-high` group experience significantly higher mortality than those in `A-low / B-low` (Risk Difference: 0.0257 (95% CI: 0.0188–0.0326), Relative Risk: 8.95 (95% CI: 5.67–15.94)x).
4. **H4 (Incremental Information):** Confirmed. In held-out test evaluation, adding Model `B` to Model `A` provides statistically significant incremental prognostic discrimination (Held-out $\Delta\text{AUROC} = 0.1394 (95% CI: 0.1264–0.1526)$, Held-out Log-Loss Reduction = +0.0114; Descriptive Development LRT $\Delta G^2 = 3099.50$, $p = 0.00e+00$).

---

## 2. Global Score Alignment & 2D Risk Surface

| Measure | Point Estimate | $p$-value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Spearman Rank Correlation ($\rho$)** | `0.4657` | `0.00e+00` | Moderate positive rank alignment |
| **Pearson Linear Correlation ($r$)** | `0.3773` | `0.00e+00` | Shared continuous representation |

```mermaid
flowchart LR
  A["Traditional Model A (CIIS Score)"] <--> |"Spearman rho = 0.466"| B["Multimodal Transformer Model B"]
  A -->|"Marginal AUROC"| MA["0.682"]
  B -->|"Marginal AUROC"| MB["0.836"]
```

---

## 3. Global Discriminative & Calibration Performance (Final Test Partition)

| Metric | Model A (Traditional CIIS) | Model B (D-BETA Linear Probe) | Difference / Improvement | $p$-value |
| :--- | :--- | :--- | :--- | :--- |
| **AUROC** | 0.6819 (95% CI: 0.6660–0.6981) | 0.8357 (95% CI: 0.8238–0.8466) | **0.1537 (95% CI: 0.1368–0.1682)** (B − A) | `0.0010` |
| **AUPRC** | 0.0531 (95% CI: 0.0484–0.0594) | 0.1475 (95% CI: 0.1304–0.1691) | **0.0944 (95% CI: 0.0792–0.1121)** (B − A) | — |
| **Brier Score** | 0.0279 (95% CI: 0.0262–0.0296) | 0.0263 (95% CI: 0.0248–0.0278) | **0.0017 (95% CI: 0.0013–0.0020)** (improvement A − B) | — |
| **Calibration Slope** | 1.039 | 1.000 | — | — |
| **Calibration Intercept** | 0.120 | -0.010 | — | — |

> [!NOTE]
> All confidence intervals are patient-level 95% bootstrap intervals computed over 1,000 resamples.

---

## 4. Traditional Risk Category Stratification & Residual Risk

| Model A Category | Patients ($N$) | Events ($N$) | Event Rate (%) | Model B Median [IQR] | Model B AUROC within Stratum | Model B AUPRC within Stratum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 3,681 | 26 | 0.71% | 0.004 [0.002–0.010] | 0.7571 | 0.0233 |
| **Borderline** | 5,199 | 64 | 1.23% | 0.006 [0.003–0.014] | 0.8538 | 0.0774 |
| **Possible Injury** | 5,705 | 100 | 1.75% | 0.008 [0.004–0.021] | 0.8425 | 0.1164 |
| **Probable Infarction** | 17,282 | 734 | 4.25% | 0.020 [0.008–0.054] | 0.8026 | 0.1641 |

### Within-Stratum Risk Gradients Across Model B Tertiles

| Model A Category | Tertile 1 (Low B Risk) | Tertile 2 (Mid B Risk) | Tertile 3 (High B Risk) | Gradient Ratio (T3 / T1) |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.08% ($N=1227$) | 0.57% ($N=1227$) | 1.47% ($N=1227$) | **18.00x** |
| **Borderline** | 0.00% ($N=1733$) | 0.52% ($N=1733$) | 3.17% ($N=1733$) | **1.00x** |
| **Possible Injury** | 0.21% ($N=1902$) | 0.53% ($N=1901$) | 4.52% ($N=1902$) | **21.50x** |
| **Probable Infarction** | 0.57% ($N=5761$) | 2.38% ($N=5760$) | 9.79% ($N=5761$) | **17.09x** |

---

## 5. Discordance Analysis & Risk Contrasts

Threshold criteria: `A cutoff = 15.0, B cutoff = 0.0112`

| Quadrant | Patients ($N$) | Proportion (%) | Events ($N$) | Observed 30-Day Mortality (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| **A-low / B-low** | 6,496 | 20.4% | 21 | 0.0032 (95% CI: 0.0018–0.0045) |
| **A-low / B-high** | 2,384 | 7.5% | 69 | 0.0289 (95% CI: 0.0226–0.0361) |
| **A-high / B-low** | 9,437 | 29.6% | 47 | 0.0050 (95% CI: 0.0036–0.0065) |
| **A-high / B-high** | 13,550 | 42.5% | 787 | 0.0581 (95% CI: 0.0542–0.0619) |

### Primary Discordance Risk Contrasts

- **Risk Difference (`A-low / B-high` vs `A-low / B-low`):** **0.0257 (95% CI: 0.0188–0.0326)**
- **Relative Risk (`A-low / B-high` vs `A-low / B-low`):** **8.95 (95% CI: 5.67–15.94)x**
- **Risk Difference (`A-high / B-high` vs `A-high / B-low`):** **0.0531 (95% CI: 0.0489–0.0572)**

---

## 6. Incremental Prognostic Information Analysis

| Model Specification | Formula | Log-Likelihood (Dev) | Held-out Log-Loss (Test) | Held-out AUROC (Test) | Held-out Brier (Test) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: Traditional Only** | `logit(P(Y=1)) = beta_0 + beta_1*A + beta_2*A^2` | -12208.55 | 0.1255 | 0.6819 | 0.0278 |
| **Model 2: Transformer Only** | `logit(P(Y=1)) = beta_0 + beta_B*B` | -10711.23 | 0.1151 | 0.8357 | 0.0272 |
| **Model 3: Combined (A + B)** | `logit(P(Y=1)) = beta_0 + beta_1*A + beta_2*A^2 + beta_B*B` | -10658.81 | 0.1140 | 0.8213 | 0.0271 |

### Incremental Test of Adding Model B to Model A

- **Held-out AUROC Improvement ($\Delta\text{AUROC}$):** **0.1394 (95% CI: 0.1264–0.1526)**
- **Held-out Brier Score Improvement ($\Delta\text{Brier}$):** **0.0007 (95% CI: 0.0002–0.0012)** (improvement)
- **Held-out Log-Loss Reduction ($\Delta\text{Log-Loss}$):** **+0.0114** (0.0092–0.0137) (reduction/improvement)
- **Descriptive Development LRT Statistic ($\Delta G^2$):** `3099.50` ($df=1$, $p = 0.00e+00$)

> [!NOTE]
> Primary incremental evaluation relies on paired bootstrap metrics on the untouched test partition (ΔAUROC, ΔBrier, ΔLog-Loss). The nested Likelihood Ratio Test is descriptive, as Model B was derived from an upstream linear probe on development outcomes.

> [!IMPORTANT]
> As prespecified in the research proposal and roadmap, Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI) are deliberately excluded from this primary analysis due to well-documented statistical limitations in risk reclassification literature.

---

## 7. Research Disclosures & Scientific Guardrails

1. **Predictor-Information Firewall:** Verified 0 clinical variables (age, sex, vitals, labs, notes, meds) enter Model A or Model B.
2. **In-Domain Representation Probing:** Foundation models pretrained on MIMIC-IV-ECG (D-BETA) are explicitly classified as in-domain representation probes, not independent external validation.
3. **Untouched Final Test Set:** All reported discrimination, discordance, and incremental metrics reflect the untouched final test partition.
4. **Patient-Clustered Uncertainty:** All confidence intervals and comparative p-values were evaluated using patient-level bootstrap resampling.
