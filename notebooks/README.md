# Notebooks

This directory contains interactive Jupyter notebooks walking through the research flow, empirical results, and methodological visualizations for the ECG Model Alignment project.

---

## Available Notebooks

### 1. [`01_research_flow_and_findings.ipynb`](01_research_flow_and_findings.ipynb)
**Comprehensive End-to-End Walkthrough & Findings**
- Explains the **Predictor-Information Firewall** and research guardrails.
- Demonstrates patient-disjoint split derivation and cohort formation.
- Evaluates **Model A (Traditional CIIS)** vs **Model B (D-BETA Multimodal Transformer)** on the holdout test set.
- Generates interactive figures:
  - **Global Score Alignment & 2D Risk Surface** (Spearman $\\rho = 0.512$).
  - **Global Discrimination Curves** (AUROC +0.0872, AUPRC +0.0897).
  - **Stratified Residual Risk Gradients** (2.36x–2.86x within-CIIS mortality spread).
  - **4-Quadrant Discordance Analysis** (Occult high risk $A_{\\text{low}}/B_{\\text{high}}$ Relative Risk = 2.35x).
  - **Nested Likelihood Ratio Test** ($p < 10^{-15}$).
  - **Sensitivity across mortality horizons** (In-hospital, 30d, 90d, 1yr).
- Details scientific guardrails, in-domain pretraining disclosures, and external validation roadmap.

---

## How to Run

Launch the notebook server using `uv`:

```bash
uv run jupyter lab notebooks/01_research_flow_and_findings.ipynb
```

Or execute as a batch script:

```bash
uv run python -m ecg_alignment.cli pipeline
```
