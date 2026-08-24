# Notebooks

This directory contains interactive Jupyter notebooks providing an API walkthrough, methodological tutorial, and simulation demonstration for the ECG Model Alignment project.

---

## Available Notebooks

### 1. [`01_research_flow_and_findings.ipynb`](01_research_flow_and_findings.ipynb)
**Interactive Simulation Tutorial & API Walkthrough**
- Demonstrates the end-to-end alignment methodology, API surface, and figure generation workflows using reproducible simulated cohort predictions.
- For authoritative empirical findings derived from full-cohort MIMIC-IV evaluation, refer to [`reports/primary-results.md`](../reports/primary-results.md), [`reports/sensitivity-analyses.md`](../reports/sensitivity-analyses.md), and [`reports/research-interpretation.md`](../reports/research-interpretation.md).
- Explains the **Predictor-Information Firewall** and research guardrails.
- Demonstrates patient-disjoint split derivation and cohort formation APIs.
- Walks through statistical evaluation functions (discrimination, calibration, stratification, discordance, incremental information).
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
