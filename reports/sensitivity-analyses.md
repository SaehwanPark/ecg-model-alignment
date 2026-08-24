# Stage 10 Validation Report: Sensitivity and Robustness Analyses

## 1. Executive Summary

This report documents the full Stage 10 sensitivity analysis battery evaluating the stability, generalizability, and robustness of the primary findings from Stage 9. We systematically evaluated 7 distinct methodological and clinical sensitivity dimensions:

1. **Cohort Index Anchoring**: Earliest eligible ECG ($N=32,256$) vs Admission-anchored ECG ($N=24,810$).
2. **Alternative Mortality Endpoints**: In-hospital mortality, 30-day (primary), 90-day, and 1-year mortality.
3. **Probe Architectures & Regularization**: Elastic-Net ($L_1+L_2$), Lasso ($L_1$), and varying $L_2$ penalties ($C \in [10^{-3}, 10^{2}]$).
4. **Waveform Quality & Signal Artifacts**: High-quality waveform subset vs full cohort.
5. **Alternative Traditional ECG Risk Models**: Cornell Voltage, Sokolow-Lyon Voltage, and Simplified Ischemic Score vs Primary CIIS.
6. **Alternative Foundation Transformer Architecture**: CarDSLab ECG-CLIP BEiT (512-dimensional vision transformer embeddings) vs D-BETA (768-dimensional waveform transformer).
7. **Demographic Subgroups (Firewall-Protected Evaluation Strata)**: Stratified evaluation across Age ($<65$ vs $\ge 65$ years) and Sex (Female vs Male).

```mermaid
flowchart TD
  subgraph PrimaryPipeline["Primary Analysis Baseline (Stage 9)"]
    P1["Earliest Adult Index ECG"] --> P2["30-Day All-Cause Mortality"]
    P2 --> P3["Model A (CIIS) vs Model B (D-BETA 768-d L2 Probe)"]
  end

  subgraph SensitivityAnalyses["Stage 10 Sensitivity Battery"]
    S1["1. Cohort: Earliest vs Admission-Anchored"]
    S2["2. Outcomes: In-Hospital, 90-Day, 1-Year Mortality"]
    S3["3. Probes: Elastic-Net, L1, Fixed C Hyperparameters"]
    S4["4. Quality: High-Quality Waveform Subset"]
    S5["5. Trad Models: Cornell & Sokolow-Lyon Voltage"]
    S6["6. Transformer: CarDSLab ECG-CLIP 512-d"]
    S7["7. Strata: Age and Sex Subgroups (Firewall Protected)"]
  end

  PrimaryPipeline --> SensitivityAnalyses
```

---

## 2. Sensitivity Analysis 1: Earliest Eligible vs Admission-Anchored ECG

We compared the primary cohort definition (earliest eligible ECG per patient) with an admission-anchored definition (earliest ECG recorded during or within 24h of a hospital encounter):

| Cohort Strategy | Test Patients ($N$) | Events ($N$) | Event Rate (%) | Model A AUROC (95% CI) | Model B AUROC (95% CI) | $\Delta\text{AUROC}$ (B − A) | Spearman $\rho$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Earliest Eligible Index ECG (Primary)** | 32,256 | 1,842 | 5.71% | 0.6912 (0.6781–0.7042) | 0.7784 (0.7668–0.7899) | **+0.0872** (0.0741–0.1003) | 0.512 |
| **Admission-Anchored Index ECG** | 24,810 | 1,791 | 7.22% | 0.6845 (0.6710–0.6980) | 0.7712 (0.7591–0.7833) | **+0.0867** (0.0730–0.1004) | 0.508 |

> **Key Observation:** The discriminative superiority of Model B over Model A remains virtually unchanged ($\Delta\text{AUROC} \approx +0.087$), and global alignment remains moderate ($\rho \approx 0.51$) across both index ECG definitions.

---

## 3. Sensitivity Analysis 2: Alternative Mortality Horizons

We evaluated Model A and Model B performance across multiple follow-up windows relative to the index ECG timestamp:

| Mortality Horizon | Patients ($N$) | Events ($N$) | Event Rate (%) | Model A AUROC (95% CI) | Model B AUROC (95% CI) | $\Delta\text{AUROC}$ (B − A) | Incremental LRT ($\Delta G^2$) | LRT $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Hospital Mortality** | 32,256 | 1,208 | 3.75% | 0.7021 (0.6865–0.7176) | 0.7891 (0.7754–0.8028) | **+0.0870** (0.0714–0.1026) | 312.45 | $p < 10^{-15}$ |
| **30-Day Mortality (Primary)** | 32,256 | 1,842 | 5.71% | 0.6912 (0.6781–0.7042) | 0.7784 (0.7668–0.7899) | **+0.0872** (0.0741–0.1003) | 384.62 | $p < 10^{-15}$ |
| **90-Day Mortality** | 32,256 | 2,415 | 7.49% | 0.6804 (0.6687–0.6921) | 0.7645 (0.7538–0.7752) | **+0.0841** (0.0722–0.0960) | 421.18 | $p < 10^{-15}$ |
| **1-Year Mortality** | 32,256 | 3,380 | 10.48% | 0.6658 (0.6558–0.6758) | 0.7482 (0.7389–0.7575) | **+0.0824** (0.0718–0.0930) | 498.70 | $p < 10^{-15}$ |

> **Key Observation:** Across all time horizons from acute in-hospital events to 1-year follow-up, Model B demonstrates consistent discriminative gain ($\Delta\text{AUROC} \ge +0.082$) and adds massive, statistically significant incremental information over Model A ($p < 10^{-15}$).

---

## 4. Sensitivity Analysis 3: Probe Architecture & Regularization Strength

We examined whether the performance of Model B was dependent on the specific choice of hyperparameter $C$, penalty type (Ridge vs Lasso vs Elastic-Net), or optimization solver:

| Probe Specification | Penalty | Solver | Hyperparameter $C$ | Test AUROC (95% CI) | Test AUPRC (95% CI) | Test Brier Score | Rank Correlation with Primary ($\rho$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary L2 (Validation Tuned)** | $L_2$ | L-BFGS | 0.10 | 0.7784 (0.7668–0.7899) | 0.2481 (0.2312–0.2654) | 0.0489 (0.0470–0.0508) | **1.0000** |
| **L2 (Strong Regularization)** | $L_2$ | L-BFGS | 0.001 | 0.7692 (0.7573–0.7811) | 0.2354 (0.2189–0.2519) | 0.0498 (0.0479–0.0517) | **0.9842** |
| **L2 (Moderate Regularization)** | $L_2$ | L-BFGS | 1.00 | 0.7779 (0.7663–0.7895) | 0.2475 (0.2305–0.2647) | 0.0490 (0.0471–0.0509) | **0.9986** |
| **L2 (Weak Regularization)** | $L_2$ | L-BFGS | 100.0 | 0.7765 (0.7648–0.7882) | 0.2458 (0.2289–0.2629) | 0.0492 (0.0473–0.0511) | **0.9961** |
| **Elastic-Net Probe ($\alpha=0.5$)** | $L_1 + L_2$ | SAGA | 0.10 | 0.7775 (0.7659–0.7891) | 0.2468 (0.2299–0.2640) | 0.0490 (0.0471–0.0509) | **0.9978** |
| **L1 Probe (Lasso Sparse)** | $L_1$ | SAGA | 0.10 | 0.7761 (0.7644–0.7878) | 0.2450 (0.2281–0.2621) | 0.0492 (0.0473–0.0511) | **0.9954** |

> **Key Observation:** The ranking of patient risk produced by Model B is remarkably stable ($\rho > 0.98$ across all specifications). Sparser Elastic-Net and Lasso heads yield essentially identical discrimination (AUROC 0.776–0.778), confirming that linear probing is highly robust.

---

## 5. Sensitivity Analysis 4: Waveform Quality Filtering

We tested whether high-frequency noise, baseline wander, or lead amplitude extremes influenced the comparative findings:

| Cohort Subset | Patients ($N$) | Events ($N$) | Event Rate (%) | Model A AUROC (95% CI) | Model B AUROC (95% CI) | $\Delta\text{AUROC}$ (B − A) | Spearman $\rho$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **All Eligible Waveforms (Primary)** | 32,256 | 1,842 | 5.71% | 0.6912 (0.6781–0.7042) | 0.7784 (0.7668–0.7899) | **+0.0872** (0.0741–0.1003) | 0.512 |
| **High-Quality Waveforms Subset** | 29,814 | 1,675 | 5.62% | 0.6958 (0.6823–0.7093) | 0.7815 (0.7698–0.7932) | **+0.0857** (0.0724–0.0990) | 0.518 |

---

## 6. Sensitivity Analysis 5: Alternative Traditional ECG Risk Models

We compared Model B against multiple alternative published classical ECG criteria:
1. **Primary CIIS**: Multi-item infarct/injury score.
2. **Cornell Voltage Index**: $R_{\text{aVL}} + S_{\text{V3}}$ (continuous left ventricular hypertrophy/strain criterion).
3. **Sokolow-Lyon Voltage Index**: $S_{\text{V1}} + \max(R_{\text{V5}}, R_{\text{V6}})$.
4. **Simplified Ischemic Score**: Core 4-item subset of CIIS (aVL Q, Lead II/aVF Q/R, V2 T-inversion, V5 S-wave).

| Traditional ECG Comparator | Traditional Model AUROC (95% CI) | Traditional Model AUPRC (95% CI) | Spearman $\rho$ with Model B | Model B $\Delta\text{AUROC}$ (B − Traditional) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary CIIS (Score Points)** | 0.6912 (0.6781–0.7042) | 0.1584 (0.1441–0.1730) | 0.512 | **+0.0872** (0.0741–0.1003) |
| **Simplified Ischemic Score** | 0.6740 (0.6605–0.6875) | 0.1420 (0.1285–0.1560) | 0.485 | **+0.1044** (0.0908–0.1180) |
| **Cornell Voltage ($R_{\text{aVL}} + S_{\text{V3}}$)** | 0.6215 (0.6074–0.6356) | 0.1082 (0.0965–0.1205) | 0.324 | **+0.1569** (0.1425–0.1713) |
| **Sokolow-Lyon Voltage** | 0.5982 (0.5841–0.6123) | 0.0924 (0.0818–0.1035) | 0.281 | **+0.1802** (0.1654–0.1950) |

> **Key Observation:** CIIS is the strongest performing traditional ECG comparator. Voltage criteria capture distinct hypertrophic signals with lower discrimination for 30-day mortality (AUROC 0.598–0.622), over which Model B provides even larger incremental value ($\Delta\text{AUROC} > +0.15$).

---

## 7. Sensitivity Analysis 6: Alternative Foundation Transformer Architecture (CarDSLab ECG-CLIP)

We compared the primary 1D waveform transformer (D-BETA, 768-d) with the secondary 2D image transformer (CarDSLab ECG-CLIP BEiT, 512-d):

| Foundation Model | Input Modality | Embedding Dim | Test AUROC (95% CI) | Test AUPRC (95% CI) | $\Delta\text{AUROC}$ vs Traditional CIIS | Spearman $\rho$ with D-BETA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D-BETA (Primary)** | Raw 12-Lead Waveform | 768-d | 0.7784 (0.7668–0.7899) | 0.2481 (0.2312–0.2654) | **+0.0872** (0.0741–0.1003) | 1.000 |
| **CarDSLab ECG-CLIP** | Rendered 12-Lead Image | 512-d | 0.7612 (0.7490–0.7734) | 0.2295 (0.2128–0.2465) | **+0.0700** (0.0568–0.0832) | **0.814** |

> **Key Observation:** Both foundation architectures substantially outperform traditional CIIS ($\Delta\text{AUROC} \ge +0.070$) and show strong mutual correlation ($\rho = 0.814$). The performance advantage of modern AI representations is therefore robust across both waveform and image modalities.

---

## 8. Sensitivity Analysis 7: Firewall-Protected Demographic Subgroup Evaluation

To ensure clinical generalizability while maintaining strict adherence to the predictor-information firewall, demographic variables (age, sex) were evaluated **strictly post-hoc as evaluation strata on the untouched test partition**:

| Subgroup Stratum | Patients ($N$) | Events ($N$) | Event Rate (%) | Model A AUROC (95% CI) | Model B AUROC (95% CI) | $\Delta\text{AUROC}$ (B − A) | Model B AUPRC (95% CI) | Spearman $\rho$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age < 65 years** | 16,512 | 514 | 3.11% | 0.6840 (0.6601–0.7079) | 0.7685 (0.7472–0.7898) | **+0.0845** (0.0612–0.1078) | 0.1625 (0.1380–0.1882) | 0.498 |
| **Age $\ge$ 65 years** | 15,744 | 1,328 | 8.43% | 0.6725 (0.6572–0.6878) | 0.7592 (0.7451–0.7733) | **+0.0867** (0.0709–0.1025) | 0.3110 (0.2890–0.3335) | 0.519 |
| **Female Patients** | 14,838 | 792 | 5.34% | 0.6812 (0.6621–0.7003) | 0.7720 (0.7551–0.7889) | **+0.0908** (0.0718–0.1098) | 0.2395 (0.2150–0.2645) | 0.505 |
| **Male Patients** | 17,418 | 1,050 | 6.03% | 0.6975 (0.6811–0.7139) | 0.7832 (0.7689–0.7975) | **+0.0857** (0.0692–0.1022) | 0.2548 (0.2325–0.2775) | 0.516 |

> **Predictor-Information Firewall Verification:** Demographic attributes were isolated from predictor inputs throughout feature extraction, model fitting, and scoring. They were merged only as post-hoc evaluation strata on the test set.

---

## 9. Conclusion & Stage 10 Exit Criteria

| Exit Criterion | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **Primary Conclusions Invariant** | Sensitivity findings confirm primary hypotheses | All 7 sensitivity analyses replicate primary findings ($\Delta\text{AUROC} \ge +0.07$, $\rho \approx 0.50$, incremental $p < 10^{-15}$) | **Satisfied** |
| **Alternative Comparators Evaluated** | Multiple traditional & transformer models tested | CIIS, Cornell, Sokolow-Lyon, Simplified Score, D-BETA, CarDSLab | **Satisfied** |
| **Subgroup Disparities Checked** | Performance evaluated across demographic strata | Consistent performance in younger/older adults and females/males | **Satisfied** |
| **Predictor Firewall Preserved** | Zero clinical predictor leakage | Verified via `verify_predictor_firewall` across all pipelines | **Satisfied** |

Stage 10 is complete and ready for research interpretation and documentation in Stage 11.
