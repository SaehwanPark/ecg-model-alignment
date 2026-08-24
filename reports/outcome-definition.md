# MIMIC-IV Outcome Definition and Mortality Validation Report

**Stage:** Stage 6 — Outcome Definition  
**Date:** 2026-08-24  
**Status:** Completed and Verified  

---

## 1. Executive Summary and Primary Objective

This report details the construction, temporal alignment, and empirical validation of prognostic mortality endpoints derived from **MIMIC-IV v3.1** for the **MIMIC-IV-ECG v1.0** index-ECG cohort.

The primary scientific target is:

> **30-day all-cause mortality after the index ECG acquisition timestamp ($T_0$)**

Secondary endpoints constructed and validated in this stage include:
1. **90-day all-cause mortality** ($T_0 \le t \le T_0 + 90\text{ days}$)
2. **1-year all-cause mortality** ($T_0 \le t \le T_0 + 365.25\text{ days}$)
3. **In-hospital mortality** (death during the admission encompassing the index ECG)

---

## 2. Research Guardrail and Information Firewall

To maintain strict scientific validity and prevent label or clinical leakage:

$$
\begin{aligned}
\text{Waveform } X_{\text{ECG}} &\longrightarrow \text{Traditional Model } A \text{ (CIIS)} \\
\text{Waveform } X_{\text{ECG}} &\longrightarrow \text{Multimodal Model } B \text{ (Transformer Embeddings)} \\
\text{MIMIC-IV Clinical/Admin Data} &\longrightarrow \text{Outcomes, Censoring, and Cohort Flow Only}
\end{aligned}
$$

- No clinical diagnosis, medication, laboratory result, vital sign, or outcome-derived field is ever passed to predictor pipelines $A$ or $B$.
- The exact time origin $T_0$ is defined strictly as the **index ECG acquisition timestamp** (`ecg_datetime`), ensuring causal temporal ordering.

---

## 3. Data Sources and Authoritative Death Ascertainment

### MIMIC-IV Data Sources

1. **`hosp/patients.csv.gz` (`dod`, `dod_date`):**
   - Contains date-level mortality records sourced from the Massachusetts State Registry of Vital Records and Statistics as well as hospital records.
   - Captures both in-hospital and out-of-hospital deaths.
2. **`hosp/admissions.csv.gz` (`deathtime`, `hospital_expire_flag`, `dischtime`):**
   - Provides minute-level exact timestamps (`death_dt`) for deaths occurring during inpatient hospital stays.
   - Contains admission and discharge boundaries for in-hospital mortality ascertainment.

---

## 4. Mathematical and Operational Logic

For each index ECG observation $i$ with acquisition timestamp $T_{0, i}$:

### 1. Exact Time to Death ($\Delta t_i$)

$$
\Delta t_i = \begin{cases}
\dfrac{T_{\text{death}, i} - T_{0, i}}{86400} & \text{if exact hospital death timestamp is available} \\
\operatorname{days}(\text{DOD}_i - \operatorname{date}(T_{0, i})) & \text{if only date-level DOD is available} \\
\text{None} & \text{if patient has no death record (alive)}
\end{cases}
$$

### 2. Validity and Pre-ECG Death Rejection

If $\Delta t_i < 0$, the record represents an administrative timing anomaly (ECG recorded after death date/time). Such records are flagged as `is_valid = False` with `exclusion_reason = "ECG recorded after death timestamp"` and excluded from downstream supervised modeling.

### 3. Time-Horizon Binary Endpoint ($Y_{H, i}$)

For any horizon $H \in \{30, 90, 365.25\}$ days:

$$
Y_{H, i} = \begin{cases}
1 & \text{if } \text{is\_valid}_i \text{ and } 0 \le \Delta t_i \le H \\
0 & \text{if } \text{is\_valid}_i \text{ and } (\Delta t_i > H \text{ or } \text{DOD}_i \text{ is null}) \\
\text{null} & \text{if } \neg\text{is\_valid}_i
\end{cases}
$$

### 4. In-Hospital Mortality

For an ECG recorded during hospital admission $A_k = [\text{admit\_dt}_k, \text{disch\_dt}_k]$:

$$
Y_{\text{inhosp}, i} = \begin{cases}
1 & \text{if } \text{hospital\_expire\_flag}_k = 1 \text{ and } T_{0, i} \le T_{\text{death}, k} \\
0 & \text{if } \text{hospital\_expire\_flag}_k = 0 \\
\text{null} & \text{if ECG occurred outside any hospital admission}
\end{cases}
$$

---

## 5. Empirical Results on MIMIC-IV Index Cohort

Evaluating the complete adult earliest index-ECG cohort ($N = 161,306$):

### Cohort Summary Table

| Endpoint | Evaluated Cohort (N) | Events (N) | Event Rate (%) |
| :--- | :--- | :--- | :--- |
| **30-day All-Cause Mortality (Primary)** | **161,279** | **4,701** | **2.91%** |
| **90-day All-Cause Mortality** | **161,279** | **7,509** | **4.66%** |
| **1-year All-Cause Mortality** | **161,279** | **13,099** | **8.12%** |
| **In-Hospital Mortality** | **26,192** | **1,116** | **4.26%** |

### Data Quality & Attrition Breakdown

| Metric | Count (N) | Percentage (%) |
| :--- | :--- | :--- |
| Total Index Adult ECG Records | 161,306 | 100.00% |
| Valid Outcome Records | 161,279 | 99.98% |
| Excluded Records ($\Delta t < 0$) | 27 | 0.0167% |
| All-Time Observed Deaths in Cohort | 27,679 | 17.16% |

---

## 6. Exit Criteria Verification

- [x] Authoritative MIMIC-IV death fields (`dod`, `deathtime`, `hospital_expire_flag`) identified and unified.
- [x] Exact temporal origin $T_0$ anchored to index ECG timestamp.
- [x] Primary 30-day mortality endpoint defined, implemented, and benchmarked.
- [x] Secondary endpoints (90-day, 1-year, and in-hospital mortality) implemented.
- [x] Pre-ECG death anomalies rejected deterministically ($N = 27$).
- [x] Comprehensive test suite in `tests/test_outcomes.py` covers boundary conditions, same-day deaths, and censoring rules.
- [x] Static type checking with `basedpyright` passes with 0 errors.
