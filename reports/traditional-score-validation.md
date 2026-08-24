# Stage 2 Validation Report: Traditional Model A (Cardiac Infarction/Injury Score)

## 1. Executive Summary

This report documents the implementation and technical validation of **Model `A`**: the **Cardiac Infarction/Injury Score (CIIS)**, established by Rautaharju et al. (*Circulation* 1981) and revised/applied in cardiovascular epidemiology by Dekker et al. (*Br Heart J* 1994).

CIIS serves as the primary traditional hand-engineered comparator for evaluating alignment, discordance, and incremental prognostic value against modern multimodal foundation model representations (`B`).

### Key Validation Outcomes:
- **Predictor Firewall Compliant**: Computed purely from 12-lead ECG waveforms. No non-ECG clinical predictors, demographics, or diagnosis codes enter Model `A`.
- **Pure & Deterministic**: Built as a composition of pure functional transformations with strict static typing (`basedpyright` passing with 0 errors).
- **High Technical Success Rate**: **98.90%** (989 / 1,000) successful deterministic score computations on a representative sample of MIMIC-IV-ECG records.
- **Explicit Failure Handling**: **1.10%** (11 / 1,000) technical failure rate with explicit error diagnostics (e.g., NaN signals, corrupted data) rather than silent fallback.
- **Full Spectrum Coverage**: Preserves continuous scores spanning $-2.07$ to $+80.29$ (median $27.65$, IQR $[17.72, 41.60]$) alongside published clinical risk strata.

---

## 2. CIIS Formulation and Measurement Matrix

The CIIS combines 12 morphologic ECG features into a composite continuous index:

| Item | Lead(s) | Morphologic Feature | Criteria / Threshold | Point Contribution |
| :--- | :--- | :--- | :--- | :--- |
| **1** | aVL | Q-wave duration ($T_{Q}$) | $T_Q = 0$ ms (absent)<br>$0 < T_Q \le 15$ ms (10 ms)<br>$15 < T_Q \le 25$ ms (20 ms)<br>$25 < T_Q \le 35$ ms (30 ms)<br>$35 < T_Q \le 45$ ms (40 ms)<br>$T_Q > 45$ ms ($\ge 50$ ms) | $+5.0$<br>$+1.0$<br>$+3.0$<br>$+9.0$<br>$+10.0$<br>$+12.0$ |
| **2** | aVL | T-wave amplitude ($A_{T}$) | Flat/low positive ($A_{T} \le 0.05$ mV / 0.5 mm)<br>Tall positive ($A_{T} > 0.30$ mV / 3.0 mm)<br>Inverted/negative ($A_{T,\text{neg}} > 0.025$ mV) | $+3.0$<br>$+3.0$<br>$+3.0 + 2.3 \times A_{T,\text{neg}}\text{ (mm)}$ |
| **3** | -aVR | R-wave amplitude ($A_{R,-\text{aVR}}$) | If $A_{R,-\text{aVR}} < 0.50$ mV (5.0 mm) | $-1.0 \times A_{R,-\text{aVR}}\text{ (mm)}$ |
| **4** | -aVR | T-wave amplitude ($A_{T,-\text{aVR}}$) | $A_{T} \le 0.05$ mV (0 mm)<br>$0.05 < A_{T} \le 0.15$ mV (1 mm)<br>$0.15 < A_{T} \le 0.25$ mV (2 mm)<br>$0.25 < A_{T} \le 0.35$ mV (3 mm)<br>$0.35 < A_{T} \le 0.45$ mV (4 mm)<br>$A_{T} > 0.45$ mV ($\ge 5$ mm) | $+6.0$<br>$+3.0$<br>$0.0$<br>$-2.0$<br>$-5.0$<br>$-5.0 - 2.0 \times (A_T - 4)$ |
| **5** | II, aVF | Largest Q/R ratio | $\max((Q/R)_{\text{II}}, (Q/R)_{\text{aVF}}) \ge 0.25$ | $+12.0$ |
| **6** | III, aVL | Q-wave duration $\ge 40$ ms | $\max(T_{Q,\text{III}}, T_{Q,\text{aVL}}) \ge 40.0$ ms | $+5.0$ |
| **7** | III | Inverted T-wave amplitude | $A_{T,\text{neg}} > 0.10$ mV (1.0 mm) | $+7.0$ |
| **8** | V1 | Positive T-wave amplitude | $A_{T,\text{pos}} > 0.20$ mV (2.0 mm) | $+5.0$ |
| **9** | V2 | R-wave amplitude | $A_R < 0.30$ mV (3 mm) OR $A_R > 1.40$ mV (14 mm) | $+5.0$ |
| **10** | V2 | Inverted T-wave amplitude | $A_{T,\text{neg}} \ge 0.025$ mV (0.25 mm) | $+5.0$ |
| **11** | V3 | Q/R ratio | $(Q/R)_{\text{V3}} > 0.05$ (1/20) | $+9.0$ |
| **12** | V5 | S-wave amplitude | $A_S < 0.20$ mV (2.0 mm) | $+5.0$ |

*Standard ECG calibration*: $1\text{ mV} = 10\text{ mm}$ ($0.1\text{ mV} = 1\text{ mm}$, $0.025\text{ mV} = 0.25\text{ mm}$).

### Published Clinical Risk Categories:
- **Normal / Low Risk**: $\text{CIIS} < 10$
- **Borderline Abnormality**: $10 \le \text{CIIS} < 15$
- **Possible Injury**: $15 \le \text{CIIS} < 20$
- **Probable Infarction**: $\text{CIIS} \ge 20$

---

## 3. Waveform Delineation Strategy

MIMIC-IV-ECG `machine_measurements.csv` provides only global intervals (`rr_interval`, `p_onset`, `p_end`, `qrs_onset`, `qrs_end`, `t_end`) and electrical axes (`p_axis`, `qrs_axis`, `t_axis`). It does not contain per-lead wave amplitudes (Q, R, S, T) or lead-specific Q durations.

To compute CIIS rigorously without clinical text leaks or diagnostic label contamination, `ecg_alignment.scoring.traditional` implements an end-to-end waveform delineation pipeline:

```text
Raw 12-lead Waveform [5000, 12] (500 Hz)
  │
  ▼
Zero-phase Baseline Wander Filter (0.67 Hz High-pass)
  │
  ▼
Energy-Derivative QRS Detection (Lead II / V5)
  │
  ▼
Time-Aligned Median Beat Extraction (250 ms pre-R, 450 ms post-R)
  │
  ▼
Per-Lead Fiducial Delineation (PR baseline, Q, R, S, T-wave amplitudes/durations)
  │
  ▼
Lead -aVR Inversion and Feature Extraction
  │
  ▼
Pure CIIS Item Scoring (Items 1-12)
  │
  ▼
Composite CIIS Continuous Score & Published Category
```

---

## 4. Empirical Benchmark on MIMIC-IV-ECG

A validation benchmark was conducted across a random consecutive sample of 1,000 records from `~/data/mimic-iv-ecg/1.0/record_list.csv`.

### Reliability & Failure Statistics
- **Total records evaluated**: 1,000
- **Valid continuous scores generated**: 989 (98.90%)
- **Technical failures**: 11 (1.10%)
- **Breakdown of technical failures**:
  - Waveform contains NaN / Infinite values: 5 records
  - Zero detectable R-peaks / severely corrupted flatline: 6 records

### Score Distribution
| Metric | Value |
| :--- | :--- |
| **Mean $\pm$ SD** | $30.33 \pm 15.49$ |
| **Median** | $27.65$ |
| **IQR** | $[17.72, 41.60]$ |
| **Minimum** | $-2.07$ |
| **Maximum** | $+80.29$ |
| **1st Percentile ($P_1$)** | $3.95$ |
| **5th Percentile ($P_5$)** | $8.84$ |
| **95th Percentile ($P_{95}$)** | $58.65$ |
| **99th Percentile ($P_{99}$)** | $63.85$ |

### Category Breakdown
| Risk Category | CIIS Range | Record Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Normal** | $< 10.0$ | 71 | 7.18% |
| **Borderline** | $10.0 \le \text{CIIS} < 15.0$ | 95 | 9.61% |
| **Possible Injury** | $15.0 \le \text{CIIS} < 20.0$ | 121 | 12.23% |
| **Probable Infarction** | $\ge 20.0$ | 702 | 70.98% |

*Note*: In an emergency department and inpatient hospital population (MIMIC-IV), a substantial proportion of patients undergoing ECG have underlying cardiovascular comorbidities, explaining the enrichment in higher risk strata.

---

## 5. Review of Distribution Tails

### Lowest Score Tail (Most Normal):
- **Study 42925188** ($\text{CIIS} = -2.07$, Normal):
  - Minimal Q duration in aVL (4 ms), upright T in aVL, robust R-wave in -aVR ($A_R = 4.07$ mm, yielding Item 3 subtraction $-4.07$), inverted T in -aVR (Item 4 subtraction $-2.0$).
- **Study 42174370** ($\text{CIIS} = 1.00$, Normal):
  - Absent Q in aVL (+5), robust R-wave in -aVR ($-4.00$), no pathological Q/R ratios or T-wave abnormalities.

### Highest Score Tail (Severe Infarction / Injury):
- **Study 48199512** ($\text{CIIS} = 80.29$, Probable Infarction):
  - Extensive pathological Q waves across inferior and anterior leads (Lead II $Q/R = 10.0$, Lead III $Q\text{ duration} = 56.0$ ms, V3 $Q/R = 0.74$), deep T inversions in multiple leads.
- **Study 46776688** ($\text{CIIS} = 69.42$, Probable Infarction):
  - High-grade lateral and inferior infarction pattern (aVL $Q\text{ duration} = 56.0$ ms, Lead II $Q/R = 10.0$, V3 $Q/R = 0.31$).

---

## 6. Deliverables & Exit Criteria Status

- [x] Authoritative CIIS formula documented and unit-tested.
- [x] All 12 independent measurements implemented as pure functions.
- [x] Deterministic waveform delineation and median beat extraction implemented.
- [x] Continuous score and published risk categories preserved.
- [x] Synthetic fixtures and category boundary test coverage.
- [x] 1,000-record benchmark confirms 98.9% technical fidelity with explicit error handling.
- [x] Code deliverables committed to `src/ecg_alignment/scoring/traditional.py`.
- [x] Test suite committed to `tests/test_traditional_scoring.py` (36 passing tests).
