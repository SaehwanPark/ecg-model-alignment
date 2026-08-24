# ECG Risk Alignment

> **Investigating alignment and prognostic complementarity between traditional rule-based ECG risk models and modern multimodal transformer foundation representations in MIMIC-IV.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: 2 spaces](https://img.shields.io/badge/code%20style-2_spaces-brightgreen.svg)](docs/development.md)
[![License](https://img.shields.io/badge/license-governance-green.svg)](docs/model-licenses.md)

---

## 1. Overview & Research Question

When given the **exact same 12-lead ECG**, how much prognostic risk information is shared between:
- **Model `A`**: A classical, explicitly engineered, continuous ECG score (Cardiac Infarction/Injury Score [CIIS]); and
- **Model `B`**: A modern multimodal transformer representation (D-BETA 768-d frozen embeddings + linear probe)?

Rather than solely asking if foundation models achieve a higher overall AUROC, this project investigates whether transformer representations uncover **outcome-relevant risk heterogeneity within conventional clinical categories defined by Model `A`**.

```text
               12-Lead Index ECG Waveform
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Traditional Model A         Transformer Model B
   (Deterministic CIIS)        (D-BETA Frozen Emb)
             │                           │
             └─────────────┬─────────────┘
                           ▼
               Alignment & Discordance
                           │
                           ▼
          30-Day All-Cause Mortality (MIMIC-IV)
```

---

## 2. Empirical Findings & Provenance Status

> [!NOTE]
> **Empirical full-cohort run pending regeneration after pipeline provenance audit.**
>
> The core pipeline is fail-closed, data-firewalled, and provenance-verified. Formal numerical findings in [`reports/primary-results.md`](reports/primary-results.md) and [`reports/research-interpretation.md`](reports/research-interpretation.md) are generated dynamically through `ecg-alignment pipeline` and require genuine MIMIC-IV WFDB waveforms (or explicit `--simulate` mode).
>
> To execute the reproducible pipeline on local data:
> ```bash
> uv run ecg-alignment pipeline --mimic-root ~/data/mimiciv/3.1 --ecg-root ~/data/mimic-iv-ecg/1.0
> ```
>
> Key hypotheses under empirical evaluation:
> - **H1 (Partial Alignment):** Positive rank correlation ($\rho > 0$) between traditional CIIS score and transformer representation.
> - **H2 (Residual Risk Gradients):** Meaningful within-stratum mortality gradients across transformer tertiles within fixed CIIS categories.
> - **H3 (Informative Discordance):** Elevated acute mortality among patients in the discordant $A_{\text{low}} / B_{\text{high}}$ quadrant relative to $A_{\text{low}} / B_{\text{low}}$.
> - **H4 (Incremental Prognosis):** Statistically robust improvement in held-out discrimination ($\Delta\text{AUROC}$) and log-loss reduction when adding Model B to flexible $f(A)$.

---

## 3. Interactive Walkthrough & Notebooks

We provide an end-to-end interactive Jupyter notebook demonstrating the complete research flow, interactive figures, and empirical evaluations:

- 📓 [`notebooks/01_research_flow_and_findings.ipynb`](notebooks/01_research_flow_and_findings.ipynb): Interactive walkthrough covering data cohorting, model scoring, 2D risk surfaces, stratified risk gradients, discordance analysis, and likelihood ratio testing.

Launch locally:
```bash
uv run jupyter lab notebooks/01_research_flow_and_findings.ipynb
```

---

## 4. Methodological Architecture & Guardrails

### The Predictor-Information Firewall

To ensure rigorous causal boundaries, patient-specific predictor spaces are restricted exclusively to ECG information:

```text
ECG Waveform / Measurements  ───►  Model A (Traditional CIIS)
ECG Waveform / Rendered Img  ───►  Model B (D-BETA / CarDSLab)

MIMIC-IV Clinical EHR Data   ───►  Cohort Linkage / 30-Day Mortality / Subgroups ONLY
```

- **Permitted Predictor Inputs:** Raw 12-lead waveforms, deterministic voltage/interval measurements, rendered ECG images.
- **Strictly Prohibited Predictors:** Demographics (age, sex, race), diagnoses (ICD codes), medications, lab values, vitals, clinical notes, encounter history.

### Model Comparators

1. **Model `A` (Traditional Baseline):** [Cardiac Infarction/Injury Score (CIIS)](reports/traditional-score-validation.md) — an established deterministic rule-based score mapping 12-lead features to continuous points and 4 clinical categories (Normal, Borderline, Possible Injury, Probable Infarction).
2. **Model `B` (Multimodal Foundation):** [D-BETA](reports/dbeta-smoke-test.md) (ICML 2025) — frozen 768-d transformer encoder paired with an $L_2$-regularized linear probe trained on development-split mortality outcomes.
3. **Secondary Engineering Prototype:** [CarDSLab ECG-CLIP BEiT](reports/cards-clip-smoke-test.md) — 2D image transformer extracting 512-d representations from rendered 12-lead grid tracings.

### Research Guardrail: In-Domain Probing Disclosure

Candidate foundation models (D-BETA, ECG-CLIP) included MIMIC-IV-ECG during pretraining. In accordance with strict scientific integrity standards:
- All findings are classified as **In-Domain Representation Probing**.
- No claims of independent external validation are made.
- Multi-center external validation (e.g. PTB-XL, CODE, UK Biobank) is outlined in [`reports/research-interpretation.md`](reports/research-interpretation.md).

---

## 5. Quickstart & CLI Orchestration

This repository provides a unified, reproducible CLI `ecg-alignment` registered via `uv`.

### 1. Installation

Clone the repository and sync dependencies:

```bash
git clone https://github.com/SaehwanPark/ecg-model-alignment.git
cd ecg-model-alignment
uv sync
```

### 2. Configure Local Data Paths (Optional)

By default, paths resolve to `~/data/mimiciv/3.1` and `~/data/mimic-iv-ecg/1.0`. You can override them via environment variables or CLI flags:

```bash
export MIMIC_ROOT="/path/to/mimiciv/3.1"
export MIMIC_ECG_ROOT="/path/to/mimic-iv-ecg/1.0"
```

### 3. Run the Research Pipeline

Execute all stages end-to-end:

```bash
uv run ecg-alignment pipeline --output-dir ./reports
```

Or run individual research stages:

```bash
# Data linkage inventory
uv run ecg-alignment inventory --report-out ./reports/data-inventory.md

# Cohort construction & patient-disjoint split
uv run ecg-alignment cohort --seed 42 --report-out ./reports/cohort-flow.md

# Linear probe training on frozen embeddings
uv run ecg-alignment probe --seed 42 --report-out ./reports/continuous-predictions.md

# Primary statistical analysis & figure generation
uv run ecg-alignment analyze --output-dir ./reports --generate-figures

# Comprehensive sensitivity battery (horizons, probes, strata)
uv run ecg-alignment sensitivity --report-out ./reports/sensitivity-analyses.md

# Research interpretation synthesis & validation roadmap
uv run ecg-alignment interpret --report-out ./reports/research-interpretation.md
```

### 4. Run Quality Checks

```bash
# Run test suite
uv run pytest

# Run static type checking
uv run basedpyright
```

---

## 6. Repository Structure

```text
ecg-model-alignment/
├── docs/                        # Specifications, roadmap, and guides
│   ├── development.md           # Coding standards, functional design, PR workflow
│   ├── model-licenses.md        # Foundation model licensing and governance
│   ├── research-proposal.md     # Full theoretical formulation & study design
│   └── roadmap.md               # 12-stage milestone roadmap & exit criteria
├── notebooks/                   # Interactive research walkthroughs
│   ├── 01_research_flow_and_findings.ipynb
│   └── README.md
├── reports/                     # Staged validation reports and aggregate summaries
│   ├── cohort-flow.md           # Stage 7 cohort flow and patient-disjoint splits
│   ├── continuous-predictions.md# Stage 8 probe weights and calibration
│   ├── data-inventory.md        # Stage 1 MIMIC linkage statistics
│   ├── primary-results.md       # Stage 9 primary statistical analysis findings
│   ├── research-interpretation.md # Stage 11 scientific synthesis and guardrails
│   └── sensitivity-analyses.md  # Stage 10 sensitivity and robustness checks
├── src/
│   └── ecg_alignment/           # Core library
│       ├── analysis.py          # Alignment, risk surfaces, LRT, bootstrap CIs
│       ├── cli.py               # Unified CLI orchestration
│       ├── cohort.py            # Cohort eligibility and index ECG selection
│       ├── data.py              # Pure data loading and linkage
│       ├── interpretation.py    # Guardrail synthesis and validation roadmap
│       ├── outcomes.py          # 30-day mortality ascertainment
│       ├── probe.py             # Frozen-embedding linear probe training
│       ├── scoring/             # Traditional and transformer model adapters
│       │   ├── cards_clip.py    # CarDSLab ECG-CLIP 2D vision adapter
│       │   ├── dbeta.py         # D-BETA 1D waveform transformer adapter
│       │   ├── preprocess.py    # Lead ordering, normalization, image rendering
│       │   └── traditional.py   # Cardiac Infarction/Injury Score (CIIS)
│       ├── sensitivity.py       # Multi-horizon and subgroup sensitivity suite
│       └── split.py             # Deterministic patient-disjoint partitioner
├── tests/                       # Comprehensive test suite (152+ tests)
└── pyproject.toml               # Project dependencies and tool configurations
```

---

## 7. Documentation & Governance Index

- 📘 [Contributor & Development Guide](docs/development.md): Indentation rules, functional style, type safety, and PR process.
- 📋 [Research Proposal](docs/research-proposal.md): Complete scientific design, hypothesis formulation, and statistical plan.
- 🗺️ [Staged Project Roadmap](docs/roadmap.md): Detailed 12-stage implementation record with audit gates.
- ⚖️ [Model Licenses & Governance](docs/model-licenses.md): Compliance audit for D-BETA, ECG-CLIP, and PULSE.
- 📊 [Validation Reports Index](reports/README.md): Master index of all empirical findings and sensitivity checks.

---

## 8. Key References

1. **D-BETA:** Pham Hung M, Saeed A, Ma D. *Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners.* ICML 2025. [Link](https://proceedings.mlr.press/v267/pham-hung25a.html)
2. **CIIS:** Rautaharju PM, et al. *Cardiac Infarction Injury Score: An electrocardiographic coding scheme for ischemic heart disease.* Circulation. 1981.
3. **CarDSLab ECG-CLIP:** TARGET-AI consortium. *ECG-CLIP BEiT Base.* [Hugging Face](https://huggingface.co/CarDSLab/ecg-clip-beit-base-384) (2025).
4. **PULSE:** Liu R, Bai Y, Yue X, et al. *Teaching multimodal LLMs to comprehend 12-lead electrocardiographic images.* npj Digital Medicine. 2026.
5. **MIMIC-IV-ECG:** Gow B, et al. *MIMIC-IV-ECG: Diagnostic Electrocardiography Database.* PhysioNet. 2023.
