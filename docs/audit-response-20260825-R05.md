# Audit Response Report: Reviewer Feedback Round 5 (R05)

**Date:** August 25, 2026  
**Auditor / Reviewer Document:** `docs/audit-feedback-20260825-R05.md`  
**Status:** All 8 Audit Items Fully Resolved & Verified  
**Branch:** `fix/audit-feedback-round-5`  
**Target:** `main`

---

## Executive Summary

This report documents the systematic investigation, code modifications, documentation updates, and test suite verifications implemented to address the 8 concerns raised in the Round 5 pre-run audit review (`docs/audit-feedback-20260825-R05.md`). 

All 8 concerns have been resolved in strict accordance with the repository conventions (`AGENTS.md`), the research proposal (`docs/research-proposal.md`), and the staged roadmap (`docs/roadmap.md`). All 170 unit and integration tests pass with 100% success rate, and `basedpyright` reports 0 errors and 0 warnings.

---

## Summary Matrix of Audit Items & Resolutions

| # | Audit Item | Severity / Area | Resolution Summary | Key Code / Doc References |
|---|---|---|---|---|
| **1** | Manifest Fail-Closed Integrity | Reliability & Integrity | Malformed JSON, non-dictionary JSON, missing `data_mode`, and missing `predictions_artifact_sha256` now raise explicit `ValueError` in real mode unless `--allow-unverified-artifact` is provided. | `src/ecg_alignment/probe.py` (lines 1276–1334) |
| **2** | Stage 11 Incremental vs Marginal Metrics | Methodological Precision | Stage 11 clinical utility evaluation and external validation recommendations now strictly evaluate incremental metrics ($A+B$ vs $A$, e.g. `incremental_res.auroc_improvement`, `incremental_res.brier_improvement`, `incremental_res.held_out_loss_reduction`) rather than standalone marginal metrics ($B$ vs $A$). | `src/ecg_alignment/interpretation.py` (lines 492–558, 805–875) |
| **3** | Descriptive LRT Gating of H4 | Inferential Soundness | Removed inferential gating on nested dev LRT $p < 0.05$. Held-out test $\Delta\text{AUROC} > 0$ (and held-out loss reduction) now governs H4 verdicts and validation recommendations, while LRT $\Delta G^2$ and $p$-value are reported alongside as descriptive supporting metrics. | `src/ecg_alignment/analysis.py` (lines 1575–1590), `src/ecg_alignment/interpretation.py` (lines 504–510, 848–855) |
| **4** | Prespecified vs Descriptive Thresholds | Statistical Transparency | Prespecified descriptive correlation bands ($|\rho| < 0.30$ weak, $[0.30, 0.70)$ moderate, $\ge 0.70$ strong) and within-stratum gradient conventions ($T_3/T_1 \ge 1.50$) are explicitly formalized in the research proposal. When $\rho \ge 0.70$, H1 evaluates to "Confirmed (Strong Alignment)". | `docs/research-proposal.md` (lines 52–88), `src/ecg_alignment/analysis.py` (lines 1501–1525) |
| **5** | Bootstrap Resample Counts | Statistical Power | Removed hardcoded `min(ctx.n_bootstraps, 20)` cap in CLI commands (`run_sensitivity`, `run_interpret`, `run_pipeline`). CLI context now defaults to 1,000 bootstraps in real mode and 50 in simulation mode. | `src/ecg_alignment/cli.py` (lines 259–265, 400–406, 928–934, 986–992, 1088–1094) |
| **6** | Transformer Checkpoint Resumption | Engineering Efficiency | `extract_transformer_embeddings()` now verifies checkpoint metadata (`cohort_hash`, `model_name`, `embedding_dim`, `processed_count`) and resumes extraction from the saved record count instead of starting from zero. | `src/ecg_alignment/probe.py` (lines 676–798) |
| **7** | Mode-Aware Demonstration Wording | Provenance & Transparency | Markdown generators across Stage 9 and Stage 11 explicitly state "analysis demonstration" and synthetic record counts during simulation mode, reserving "empirical evaluation" for genuine real waveform runs. | `src/ecg_alignment/analysis.py` (lines 1607–1620), `src/ecg_alignment/interpretation.py` (lines 1055–1060, 1178–1185) |
| **8** | Roadmap Status Reconciliation | Project Tracking | Updated `docs/roadmap.md` to explicitly delineate `[code]` implemented checklist items vs `[data]` items pending empirical execution on the full MIMIC-IV dataset. | `docs/roadmap.md` (Stages 8–12) |

---

## Detailed Audit Point Breakdown

### Item 1: Manifest Fail-Closed Integrity
- **Issue:** In `load_and_verify_prediction_artifact()`, invalid JSON or missing `data_mode` caused `manifest_data` to remain `None`, falling through to an unverified load in real execution mode without raising an error.
- **Fix:** In `src/ecg_alignment/probe.py`, `json.JSONDecodeError` or non-dict payloads now immediately raise `ValueError` when `requested_mode == "real"`. Missing `data_mode` or missing `predictions_artifact_sha256` also raise explicit `ValueError` unless `--allow-unverified-artifact` is supplied.
- **Tests:** Added `test_load_and_verify_malformed_and_incomplete_manifests` in `tests/test_probe.py` covering syntax errors, non-object JSON, missing `data_mode`, and missing SHA-256 hashes.

### Item 2: Stage 11 Incremental vs Marginal Metrics
- **Issue:** `evaluate_clinical_utility_distinction()` and `synthesize_external_validation_recommendation()` referenced marginal performance comparison metrics (`comp_res.delta_auroc`, `comp_res.delta_brier` measuring $B$ vs $A$) for statements regarding adding Model $B$ to Model $A$.
- **Fix:** Updated `src/ecg_alignment/interpretation.py` to use `incremental_res.auroc_improvement`, `incremental_res.brier_improvement`, and `incremental_res.held_out_loss_reduction` for all incremental evaluation statements and recommendation logic.
- **Tests:** Added `test_evaluate_clinical_utility_distinction` assertions verifying priority of incremental over marginal metrics.

### Item 3: Descriptive LRT Gating of H4
- **Issue:** H4 verdict and validation recommendation gated on $p < 0.05$ of the nested likelihood ratio test evaluated on the development set, creating tension with the proposal principle that held-out test evaluation governs.
- **Fix:** In `src/ecg_alignment/analysis.py` and `src/ecg_alignment/interpretation.py`, H4 confirmation and external validation recommendation now gate strictly on held-out test $\Delta\text{AUROC}$ bootstrap 95% CI lower bound $> 0$. The development LRT statistic $\Delta G^2$ and $p$-value are reported alongside as descriptive supporting metrics.

### Item 4: Prespecified vs Descriptive Thresholds
- **Issue:** Proposal defined H1 as $\operatorname{cor}(A, B) > 0$, while code classified $|\rho| \ge 0.70$ as "Not Confirmed (Strong Alignment)".
- **Fix:** Formally documented descriptive interpretation bands in `docs/research-proposal.md` under Core Hypotheses. Updated `generate_primary_results_markdown()` so that $\rho > 0$ and $p < 0.05$ with $\rho \ge 0.70$ evaluates to `Confirmed (Strong Alignment)`.

### Item 5: Bootstrap Resample Counts
- **Issue:** CLI commands artificially capped sensitivity and interpretation bootstrap iterations at 20 (`min(ctx.n_bootstraps, 20)`).
- **Fix:** In `src/ecg_alignment/cli.py`, removed `min(ctx.n_bootstraps, 20)` and passed `ctx.n_bootstraps` directly. Configured default CLI context to use 1,000 resamples for real mode and 50 for simulation mode.

### Item 6: Transformer Checkpoint Resumption
- **Issue:** `extract_transformer_embeddings()` accepted `checkpoint_path` and saved intermediate `.npz` files, but did not resume from them on re-run.
- **Fix:** In `src/ecg_alignment/probe.py`, implemented true checkpoint resumption: computes a deterministic `cohort_hash` from index ECG IDs, validates `model_name`, `embedding_dim`, and `processed_count`, restores completed embeddings/flags, and iterates only over remaining batches.
- **Tests:** Added `test_extract_transformer_embeddings_checkpoint_resume` verifying that identical re-runs skip waveform reads.

### Item 7: Mode-Aware Demonstration Wording
- **Issue:** Simulation outputs contained phrasing implying empirical execution on all 161,279 waveforms.
- **Fix:** Updated report generation logic in `src/ecg_alignment/analysis.py` and `src/ecg_alignment/interpretation.py` to distinguish "analysis demonstration" (simulation) from "empirical evaluation" (real execution).

### Item 8: Roadmap Status Reconciliation
- **Issue:** Roadmap checklists marked full pipeline stages as complete without distinguishing implemented code infrastructure from pending empirical dataset execution.
- **Fix:** Updated `docs/roadmap.md` across Stages 8 through 12 to explicitly tag items as `[code]` implemented vs `[data]` pending empirical execution.

---

## Verification & Quality Checks

1. **Pytest Suite:**
   ```bash
   uv run pytest
   ```
   *Result:* `170 passed, 5 warnings in 26.53s` (100% pass rate).

2. **Type Checking:**
   ```bash
   uv run basedpyright
   ```
   *Result:* `0 errors, 0 warnings, 0 notes`.

3. **End-to-End Pipeline Demonstration:**
   ```bash
   uv run ecg-alignment pipeline --simulate
   ```
   *Result:* Completed all stages (inventory -> cohort -> probe -> primary -> sensitivity -> interpret) with zero errors, writing updated demonstration reports to `reports/`.
