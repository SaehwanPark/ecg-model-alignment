---
title: "ECG Risk Alignment Project Roadmap"
description: "Staged implementation roadmap with actionable checklists and exit criteria."
date: 2026-08-24
status: "Working roadmap"
---

# ECG Risk Alignment Project Roadmap

## Goal

Build a reproducible experiment comparing:

- a traditional ECG-only risk score `A`;
- a modern multimodal transformer-derived ECG score `B`;

while using MIMIC-IV clinical data only for cohort construction, outcome ascertainment, follow-up, and secondary evaluation.

The roadmap is intentionally staged. Each stage should leave behind reviewable code, tests, and artifacts before the next stage begins.

## Project Rules

- [ ] Patient-specific predictor inputs come from ECG only.
- [ ] MIMIC-IV clinical features never enter `A` or `B`.
- [ ] One patient never crosses supervised train/validation/test boundaries.
- [ ] Raw MIMIC data are never committed.
- [ ] Hugging Face tokens or other credentials are never committed.
- [ ] All reusable code lives under `src/`.
- [ ] `uv` manages Python, dependencies, and commands.
- [ ] `pytest` covers reusable logic and data contracts.
- [ ] `basedpyright` must pass before merge.
- [ ] Repository indentation/tab size is 2 spaces.
- [ ] Prefer functional transformations and explicit side-effect boundaries.
- [ ] Keep functions topologically ordered where practical.
- [ ] Every substantive change is implemented on a branch and reviewed through a pull request.

## Stage 0 — Repository and Environment

### Objective

Create a reproducible development environment before touching model or outcome logic.

### Checklist

- [x] Initialize repository with `uv`.
- [x] Add Python version constraint.
- [x] Create `src/ecg_alignment/`.
- [x] Create `tests/`.
- [x] Create `docs/`.
- [x] Create `reports/`.
- [x] Add `.gitignore` entries for MIMIC data, model caches, credentials, outputs, and local environments.
- [x] Add `.editorconfig` with `indent_size = 2`.
- [x] Configure editor settings for Python tab size 2.
- [x] Add `pytest` development dependency.
- [x] Add `basedpyright` development dependency.
- [x] Configure `basedpyright` in `pyproject.toml`.
- [x] Configure `pytest` in `pyproject.toml`.
- [x] Add a minimal import smoke test.
- [x] Confirm `uv run pytest`.
- [x] Confirm `uv run basedpyright`.

### Exit Criteria

- [x] Fresh clone can be initialized with documented `uv` commands.
- [x] Test suite passes.
- [x] Static type checking passes.
- [x] No protected data or credentials are tracked.

## Stage 1 — Data Inventory and Linkage

### Objective

Understand exactly what can be linked between local MIMIC-IV-ECG v1.0 and MIMIC-IV v3.1.

### Inputs

```text
~/data/mimic-iv-ecg/1.0/
~/data/mimiciv/3.1/
```

### Checklist

- [x] Load `record_list.csv`.
- [x] Inspect ECG identifiers, subject identifiers, timestamps, paths, sampling rate, and waveform duration.
- [x] Verify a sample of WFDB records under `files/`.
- [x] Load relevant MIMIC-IV patient/admission tables.
- [x] Document join keys.
- [x] Quantify ECGs linkable to a MIMIC-IV subject.
- [x] Quantify ECGs linkable to a hospital admission under a prespecified timing rule.
- [x] Count ECGs per patient.
- [x] Characterize the calendar distribution of ECGs.
- [x] Define technical waveform eligibility.
- [x] Implement pure linkage functions.
- [x] Add synthetic unit tests for linkage edge cases.
- [x] Save only non-sensitive aggregate inventory results under `reports/`.

### Deliverables

- [x] `src/ecg_alignment/data.py`
- [x] `src/ecg_alignment/cohort.py`
- [x] `tests/test_data.py`
- [x] `tests/test_cohort.py`
- [x] `reports/data-inventory.md`

### Exit Criteria

- [x] Linkage logic is deterministic and tested.
- [x] Eligible ECG count and unique-patient count are known.
- [x] Index-ECG timestamp semantics are documented.

## Stage 2 — Traditional Model `A`

### Objective

Implement the traditional ECG-only comparator independently from the AI pipeline.

### Primary candidate

Cardiac Infarction/Injury Score (CIIS).

### Checklist

- [x] Retrieve and document the authoritative CIIS formula.
- [x] Enumerate every required ECG measurement.
- [x] Determine which measurements can be obtained from existing MIMIC machine measurements.
- [x] Identify measurements that must be derived from the waveform.
- [x] Select or implement a deterministic delineation/measurement strategy.
- [x] Keep machine-generated diagnostic interpretation text out of `A`.
- [x] Implement one pure function per logically independent measurement.
- [x] Implement the final score as a pure composition of measurements.
- [x] Preserve the continuous score.
- [x] Implement published CIIS risk categories separately from score computation.
- [x] Create fixture ECGs or synthetic measurement fixtures with known expected values.
- [x] Test boundary conditions at every published category threshold.
- [x] Quantify score-computation failure rate on a sample of MIMIC ECGs.
- [x] Review implausible score tails manually.

### Deliverables

- [x] `src/ecg_alignment/scoring/traditional.py`
- [x] `tests/test_traditional_scoring.py`
- [x] `reports/traditional-score-validation.md`

### Exit Criteria

- [x] CIIS can be computed reproducibly for a substantial fraction of eligible ECGs.
- [x] Category boundaries are covered by tests.
- [x] Technical failures are explicit rather than silently imputed.

### Stop/Go Gate

If CIIS cannot be recovered with acceptable technical fidelity:

- [ ] stop full-cohort scoring;
- [ ] document the failure;
- [ ] select the next traditional ECG-only model with reproducible continuous measurements;
- [ ] preserve the same analysis interface.

## Stage 3 — Common Model-`B` Adapter Interface

### Objective

Build infrastructure that lets multiple transformer models produce standardized outputs without leaking model-specific assumptions into the analysis code.

### Target interface

Each adapter should map an ECG to either:

```text
continuous score
```

or:

```text
frozen embedding
```

plus explicit metadata about preprocessing and failures.

### Checklist

- [x] Define immutable model configuration.
- [x] Define canonical lead order.
- [x] Define waveform units.
- [x] Define resampling policy.
- [x] Define padding/cropping policy.
- [x] Define deterministic ECG-image rendering for image models.
- [x] Define model-output schema.
- [x] Define batch inference interface.
- [x] Keep model loading outside pure transformation functions.
- [x] Add tests for lead order and shape validation.
- [x] Add tests for deterministic rendering.
- [x] Add tests for missing/duplicate lead failures.
- [x] Add smoke-test mode using a handful of ECGs.

### Deliverables

- [x] `src/ecg_alignment/scoring/base.py`
- [x] `src/ecg_alignment/scoring/preprocess.py`
- [x] `tests/test_preprocess.py`

### Exit Criteria

- [x] Model-specific adapters can share the same downstream scoring pipeline.
- [x] Identical input produces identical preprocessing output.

## Stage 4 — CarDSLab ECG-CLIP Engineering Prototype

### Objective

Use currently granted access to validate image-model infrastructure immediately.

### Model

```text
CarDSLab/ecg-clip-beit-base-384
```

### Scientific status

Engineering/prototyping model under the current strict criteria because the associated TARGET-AI manuscript is presently a preprint.

### Checklist

- [x] Verify authenticated local model download.
- [x] Record exact Hugging Face revision/commit (`80131ef06310dd1c8f2efe9082709c82433dc66e`).
- [x] Inspect model card and reference inference code.
- [x] Reproduce official demo on supplied example images.
- [x] Implement deterministic MIMIC waveform-to-image rendering.
- [x] Match supported ECG layouts.
- [x] Extract 512-dimensional ECG embeddings.
- [x] Reproduce a documented zero-shot example if feasible.
- [x] Run on a small MIMIC ECG sample.
- [x] Record inference failures and GPU memory requirements.
- [x] Cache only derived non-PHI artifacts permitted by the local environment.

### Deliverables

- [x] `src/ecg_alignment/scoring/cards_clip.py`
- [x] `tests/test_cards_clip.py`
- [x] `reports/cards-clip-smoke-test.md`

### Exit Criteria

- [x] End-to-end MIMIC ECG -> rendered image -> embedding works.
- [x] Output shape and determinism are tested.
- [x] The common model adapter is validated.

## Stage 5 — D-BETA Primary Transformer Path

### Objective

Once Hugging Face access is granted, establish the leading peer-reviewed multimodal transformer representation.

### Checklist

- [x] Confirm access to `Manhph2211/D-BETA`.
- [x] Record license and research-use restrictions.
- [x] Record exact model revision (`20ff3ccce1759d7d629171e15befafa9a424d2ca`).
- [x] Reproduce official `AutoModel` loading example.
- [x] Verify expected input shape `[batch, 12, 5000]`.
- [x] Verify lead order and waveform scaling requirements from source.
- [x] Verify 768-dimensional `pooler_output`.
- [x] Implement D-BETA adapter.
- [x] Run on a small MIMIC sample.
- [x] Confirm deterministic inference in evaluation mode.
- [x] Benchmark batch size and GPU memory.
- [x] Document that MIMIC-IV-ECG was used during model pretraining.
- [x] Label all subsequent MIMIC analysis as in-domain representation probing.

### Deliverables

- [x] `src/ecg_alignment/scoring/dbeta.py`
- [x] `tests/test_dbeta.py`
- [x] `reports/dbeta-smoke-test.md`

### Exit Criteria

- [x] Frozen ECG embeddings can be generated reproducibly for the cohort.
- [x] No MIMIC-IV clinical feature is involved in embedding generation.

### Stop/Go Gate

If access remains unavailable or the released checkpoint cannot be reproduced:

- [ ] preserve the CarDSLab prototype;
- [ ] evaluate PULSE as the next peer-reviewed multimodal candidate;
- [ ] document why the primary model changed.

## Stage 6 — Outcome Definition

### Objective

Construct outcome labels from MIMIC-IV independently from predictor generation.

### Primary target

30-day all-cause mortality after index ECG.

### Checklist

- [x] Identify authoritative MIMIC-IV fields for death ascertainment.
- [x] Define exact time origin as index ECG timestamp.
- [x] Define 30-day event logic.
- [x] Define minimum follow-up requirements.
- [x] Identify records with ambiguous or insufficient follow-up.
- [x] Decide whether ambiguous follow-up is excluded or censored.
- [x] Implement in-hospital mortality as a secondary endpoint.
- [x] Implement 90-day mortality as a secondary endpoint if supportable.
- [x] Assess feasibility of 1-year mortality.
- [x] Test timestamp edge cases.
- [x] Test death exactly at time-window boundaries.
- [x] Test ECGs occurring after the recorded outcome and reject them.
- [x] Keep all outcome code independent from predictor code.

### Deliverables

- [x] `src/ecg_alignment/outcomes.py`
- [x] `tests/test_outcomes.py`
- [x] `reports/outcome-definition.md`

### Exit Criteria

- [x] Every eligible patient has a reproducible event/censoring status.
- [x] Outcome construction is fully tested.
- [x] No outcome-derived field enters predictor preprocessing.

## Stage 7 — Freeze the Primary Cohort and Split

### Objective

Create a patient-level analytic cohort before fitting the transformer probe.

### Checklist

- [x] Apply adult eligibility.
- [x] Apply waveform technical eligibility.
- [x] Apply outcome follow-up eligibility.
- [x] Select earliest eligible ECG per patient.
- [x] Quantify exclusions at each step.
- [x] Produce a cohort flow table.
- [x] Generate deterministic patient-disjoint split.
- [x] Save split assignments separately from model outputs.
- [x] Freeze the random seed.
- [x] Verify no `subject_id` overlap across splits.
- [x] Add a test that fails on split overlap.

### Suggested initial split

```text
60% development
20% validation
20% final test
```

### Deliverables

- [x] `src/ecg_alignment/split.py`
- [x] `tests/test_split.py`
- [x] `reports/cohort-flow.md`

### Exit Criteria

- [x] Final test subject IDs are frozen.
- [x] No final-test outcome has been used for model or hyperparameter selection.

## Stage 8 — Build Continuous Predictions

### Objective

Generate `A` and `B` on exactly the same index ECG cohort.

### `A`

- [x] Compute continuous CIIS.
- [x] Compute prespecified CIIS category.
- [x] Record technical failure reasons.

### `B`

For D-BETA:

- [x] Extract frozen embedding on development set.
- [x] Extract frozen embedding on validation set.
- [x] Do not inspect final-test outcomes.
- [x] Fit L2-regularized logistic regression using development outcomes only.
- [x] Select regularization on validation data only.
- [x] Freeze probe coefficients.
- [x] Generate final-test `B` scores.

### Joint checks

- [x] Confirm one row per patient.
- [x] Confirm identical ECG ID for `A` and `B`.
- [x] Confirm no clinical predictors were accidentally merged.
- [x] Confirm all score generation can be reproduced from IDs and configuration.

### Deliverables

- [x] `src/ecg_alignment/probe.py`
- [x] `tests/test_probe.py`
- [x] `reports/continuous-predictions.md`
- [x] versioned analysis configuration
- [x] derived score table in the protected analysis environment

### Exit Criteria

- [x] Continuous `A` and `B` exist for the common test cohort.
- [x] Probe specification is frozen.

## Stage 9 — Primary Analysis

### Objective

Answer alignment, residual-risk, discordance, and incremental-information questions.

### Global alignment

- [x] Compute Spearman correlation.
- [x] Plot `A` vs `B`.
- [x] Plot smooth $E(B \mid A)$.
- [x] Create two-dimensional outcome-risk surface.

### Global performance

- [x] Compute AUROC for `A`.
- [x] Compute AUROC for `B`.
- [x] Compute AUPRC for `A`.
- [x] Compute AUPRC for `B`.
- [x] Evaluate Brier score where meaningful.
- [x] Generate calibration plots where meaningful.
- [x] Bootstrap patient-level confidence intervals.

### `A`-stratified analysis

- [x] Apply published `A` categories.
- [x] Report `N` and event rate per category.
- [x] Report `B` distribution per category.
- [x] Split `B` into prespecified quantiles within or across `A` categories.
- [x] Estimate event rates across `B` quantiles.
- [x] Compute `B` AUROC/AUPRC within `A` categories where estimable.
- [x] Avoid treating range-restricted `A` performance as a fair within-category comparison.

### Discordance

- [x] Define prespecified low/high score rules.
- [x] Create `A-low/B-low`.
- [x] Create `A-low/B-high`.
- [x] Create `A-high/B-low`.
- [x] Create `A-high/B-high`.
- [x] Compare observed event rates.
- [x] Bootstrap risk differences and ratios.

### Incremental information

- [x] Fit `Y ~ f(A)`.
- [x] Fit `Y ~ B`.
- [x] Fit `Y ~ f(A) + B`.
- [x] Compare held-out log loss.
- [x] Compare AUROC.
- [x] Compare Brier score.
- [x] Test incremental contribution of `B`.
- [x] Keep NRI/IDI out of the primary analysis.

### Deliverables

- [x] `src/ecg_alignment/analysis.py`
- [x] `tests/test_analysis.py`
- [x] `reports/primary-results.md`
- [x] reproducible figure outputs

### Exit Criteria

- [x] Every primary result comes from the untouched final test set.
- [x] Every reported uncertainty estimate is patient-level.
- [x] Analysis can be rerun from a single documented command or workflow.

## Stage 10 — Sensitivity Analyses

### Checklist

- [ ] Earliest eligible ECG vs admission-anchored ECG.
- [ ] In-hospital mortality.
- [ ] 90-day mortality.
- [ ] 1-year mortality if follow-up is supportable.
- [ ] Alternative regularization strength within the prespecified validation procedure.
- [ ] Alternative simple probe such as elastic-net logistic regression.
- [ ] Exclusion of low-quality waveforms.
- [ ] Alternative traditional score implementation if justified.
- [ ] CarDSLab embedding analysis as an engineering/secondary comparator.
- [ ] PULSE secondary analysis if technically worthwhile.
- [ ] Optional demographic subgroup evaluation using demographics only as evaluation strata.

### Exit Criteria

- [ ] Primary conclusions are clearly separated from sensitivity findings.
- [ ] Any material conclusion change is documented.

## Stage 11 — Research Interpretation

### Checklist

- [ ] State whether alignment is strong, moderate, or weak using prespecified descriptive criteria.
- [ ] Identify whether `B` reveals meaningful within-`A` outcome gradients.
- [ ] Interpret discordant groups.
- [ ] Distinguish statistical incremental information from clinical utility.
- [ ] Explicitly disclose foundation-model MIMIC pretraining contamination.
- [ ] Avoid external-validation language for contaminated models.
- [ ] Document technical failure rates for both models.
- [ ] Document any post hoc choices separately from prespecified choices.
- [ ] Decide whether findings justify a larger external-validation study.

## Stage 12 — Repository Hardening

### Checklist

- [ ] `uv run pytest` passes.
- [ ] `uv run basedpyright` passes.
- [ ] No hard-coded user-specific absolute paths in reusable code.
- [ ] Local MIMIC paths are configured through CLI/config/environment.
- [ ] No PHI or row-level protected data in Git.
- [ ] No Hugging Face token in Git history.
- [ ] README setup is tested from a clean environment.
- [ ] Research proposal matches implemented design.
- [ ] Roadmap reflects completed and deferred items.
- [ ] Model licenses are documented.
- [ ] Reproducibility commands are documented.

## Recommended GitHub Issue Sequence

1. **Initialize typed `uv` project and CI-quality local checks**
2. **Inventory MIMIC-IV-ECG and MIMIC-IV linkage**
3. **Define and test index-ECG cohort logic**
4. **Implement traditional CIIS measurements**
5. **Validate CIIS categories and edge cases**
6. **Create common transformer model adapter**
7. **Run CarDSLab ECG-CLIP smoke test**
8. **Integrate D-BETA after model access**
9. **Define and test 30-day mortality outcome**
10. **Freeze patient-disjoint analytic split**
11. **Train frozen-embedding linear probe**
12. **Generate common `A`/`B` score table**
13. **Implement alignment analysis**
14. **Implement `A`-stratified residual-risk analysis**
15. **Implement discordance analysis**
16. **Implement incremental-information analysis**
17. **Run sensitivity analyses**
18. **Generate reproducible research report**

Each issue should be implemented on its own branch and submitted through a pull request with:

```text
## What problem does this PR solve?

## What approach did you take?

## What assumptions did you make?

## How did you test it?

## Example output
```

## Definition of Done for a Pull Request

- [ ] Scope matches the GitHub issue.
- [ ] Reusable logic is under `src/`.
- [ ] Functions are small and typed.
- [ ] Side effects are isolated.
- [ ] New behavior has tests.
- [ ] `uv run pytest` passes.
- [ ] `uv run basedpyright` passes.
- [ ] No protected data or secrets are included.
- [ ] Documentation is updated when behavior or assumptions change.
- [ ] Reviewer can reproduce the example output.
