## Re-audit verdict

There has been **substantial improvement in the implementation**, especially around the primary Stage 8→9 path. I would now characterize the repository as:

> **A credible real-data primary-analysis pipeline is now present, but the committed scientific findings are still not trustworthy as empirical results.**

The largest remaining problem is that the repository still mixes **real-data code, automatic simulation fallbacks, stale static reports, and synthetic sensitivity analyses** without sufficiently hard boundaries between them.

| Previous finding                           | Current status           | Assessment                                                                   |
| ------------------------------------------ | ------------------------ | ---------------------------------------------------------------------------- |
| Primary CLI uses only synthetic A/B        | **Mostly fixed in code** | A real CIIS + D-BETA path now exists                                         |
| Test-set leakage in A calibration          | **Fixed**                | A calibration is now fit on development data                                 |
| Empty risk-surface cells                   | **Fixed**                | Empty cells use `NaN` + count matrix                                         |
| A-high discordance CI missing              | **Fixed**                | Proper bootstrap added                                                       |
| Prediction artifact / run provenance       | **Partially fixed**      | Artifact + manifest added, but provenance metadata is insufficient           |
| Accidental simulation                      | **Still critical**       | Real failures automatically fall back to simulation                          |
| Stage 10 sensitivity uses synthetic inputs | **Still critical**       | Essentially unchanged                                                        |
| Stage 11 hard-coded findings               | **Still high severity**  | Optional dynamic inputs exist, but the pipeline does not pass them           |
| Naïve 1-df LRT                             | **Not fixed**            | Same inferential issue remains                                               |
| Committed empirical reports inconsistent   | **Still critical**       | README was edited; underlying reports were not regenerated consistently      |
| CIIS measurement validation                | **Not fixed**            | Technical completion is still being conflated with validation                |
| Research notebook                          | **Not fixed**            | Still explicitly generates synthetic data while calling itself authoritative |

### What they fixed well

The biggest positive change is real. `build_real_predictions_for_cohort()` now does what I originally wanted the pipeline to do: it runs `score_traditional_cohort()` over actual waveforms, extracts D-BETA embeddings through `DbetaAdapter`, fits the probe using development outcomes and validation tuning, applies the frozen probe, and constructs the unified prediction table.

The Stage 9 leakage problem is also genuinely corrected. Model A's mortality calibration is now trained on `a_dev, y_dev` and only applied to test CIIS scores. The 2-D risk surface now represents empty cells as `NaN` and records their sample counts, and the A-high discordance contrast receives an actual bootstrap interval.

The new persisted prediction artifact and manifest are also the right architectural direction. `generate_run_manifest()` records cohort/event counts, validity counts, probe configuration, D-BETA revision and a checksum.

So I would no longer say that the repository simply lacks a real research pipeline. **It now has one.**

## The new biggest problem: real mode does not fail closed

This is the most important remaining issue.

`run_probe()` does not require explicit `--simulate` before producing synthetic predictions. If waveforms are not detected, it silently chooses simulation; more seriously, if actual CIIS/D-BETA scoring throws **any exception**, it catches that exception and says:

> `Real scoring failed ... Falling back to simulation.`

It then persists those simulated predictions to the same `predictions.parquet` pathway used for real predictions. `_get_unified_table()` similarly auto-simulates when waveforms are unavailable.

This means a command presented as:

```bash
uv run ecg-alignment pipeline
```

can finish successfully without having run the intended experiment.

That is worse than an explicit simulation mode from a provenance perspective, because downstream code can subsequently load the cached artifact and has no reliable way to know whether it contains real or synthetic scores.

The manifest does not solve this yet. It has no `data_mode: real|simulation`, and the CLI does not populate `git_sha`; it also lacks the requested `uv.lock` checksum and does not hash the serialized prediction artifact itself.

I would make the rule extremely simple:

> **Without `--simulate`, inability to run the real models must terminate the command with nonzero status.**

The ordinary integration test currently reinforces the opposite behavior: the fixture contains no real WFDB signals, yet `probe`, `analyze`, `sensitivity`, and the complete pipeline are all expected to succeed without `--simulate`.

That test should be inverted.

## The committed “findings” are still not reconciled

This remains a blocking problem.

The authoritative cohort report still gives the frozen test partition as:

[
N=32{,}256,\qquad D_{30}=941,\qquad 2.92%.
]

The README was edited to match that count: 941 deaths / 2.92%. But all of the old effect estimates immediately below it remain unchanged.

More importantly, `reports/primary-results.md` **still says 1,842 deaths / 5.71%**. Its CIIS strata contain 582 + 405 + 368 + 487 = 1,842 events, and its discordance quadrants likewise sum to 1,842. It also retains the original internal H4 conflict: the executive table reports LRT 384.62 and ΔAUROC 0.0542, while the detailed section reports 649.40 and 0.0886.

This is especially notable because PR #15 says it:

> “Synchronized test cohort numbers ... across `README.md`, `reports/primary-results.md`, and `reports/research-interpretation.md`.”

But `reports/primary-results.md` was not actually part of the changed-file set in that repair, and its current contents demonstrate that the synchronization did not happen.

Therefore my previous conclusion remains:

> **The numerical scientific results in the repository have still not been regenerated from one authoritative real prediction artifact.**

I would not try to manually repair individual numbers. Delete/regenerate the report after a real run.

There is another inconsistency: Stage 8 says only about **98.3%** of each partition has both models valid, while Stage 9 evaluates exactly all 32,256 test subjects even though `run_primary_analysis()` filters to `model_a_valid & model_b_valid`. Those two descriptions cannot both come from the same prediction artifact.

## Stage 10 is still synthetic

This part was not substantively repaired.

Even after `_get_unified_table()` returns potentially real Stage 9 predictions, `run_sensitivity()` creates:

```python
dev_emb = rng.normal(...)
test_emb = rng.normal(...)
```

then constructs fake demographic strata by array position and random Cornell/Sokolow-Lyon scores. It also passes the exact same `test_df` as both the earliest-ECG cohort and admission-anchored cohort.

So the supposed admission-anchored sensitivity isn't admission-anchored.

At the sensitivity-library level, missing CarDSLab embeddings are still replaced with random 512-dimensional vectors, and a missing waveform-quality filter defaults to an all-`True` mask.

Yet the committed sensitivity report continues to describe these as substantive empirical robustness analyses, including precise admission-anchored, Cornell, Sokolow-Lyon, ECG-CLIP, age and sex estimates. It still uses the old 1,842-death primary cohort as well.

This should be changed from “fallback” semantics to **optional-analysis semantics**:

* real inputs available → run analysis;
* unavailable → report “not run”;
* never fabricate replacements.

There is also a separate statistical leakage issue inside the sensitivity code. `evaluate_outcome_horizons()` fits the A-only and A+B logistic models **directly on the test outcome being evaluated** and calculates its LRT there. It also min-max scales A using the same test data for Brier evaluation.

So Stage 10 needs more than just plugging in real embeddings.

## Stage 11 is only superficially dynamic

The API has improved: `run_interpret()` now accepts `primary_result` and `sensitivity_result`.

But the actual `run_pipeline()` does this:

```text
run_analyze(...)
run_sensitivity(...)
run_interpret(ctx)
```

Both preceding functions return status codes, and no result object is passed into `run_interpret()`.

Therefore the interpretation module falls back to its hard-coded “default Stage 9 empirical findings”: (\rho=0.512), the old CIIS stratum counts, old discordance counts and rates, etc.

Technical completeness is also hard-coded separately as 161,427 waveforms, 129 Model A failures and zero Model B failures.

That conflicts with the frozen cohort report's 161,279 subjects.

The committed interpretation consequently still calls itself an “authoritative” empirical synthesis and contains these fixed values.

I would remove all empirical defaults from `synthesize_research_interpretation()`. If no real `PrimaryAnalysisResult` is supplied, it should refuse to generate an empirical interpretation.

## The LRT issue remains

The original statistical concern has not been addressed.

The code still computes a conventional nested likelihood-ratio test between:

[
Y\sim f(A)
]

and

[
Y\sim f(A)+B
]

on development data and assigns exactly one additional degree of freedom:

```python
df_diff = 1
p_val_lrt = chi2.sf(lrt_stat, df_diff)
```

But (B) is the prediction from a 768-dimensional supervised probe already estimated using those same development outcomes. Treating it as a fixed one-dimensional covariate for classical LRT inference does not account for that upstream estimation.

I would therefore still make these the primary incremental-information quantities:

[
\Delta AUROC,\quad
\Delta AUPRC,\quad
\Delta\text{log-loss},\quad
\Delta\text{Brier},
]

all on untouched test patients with paired bootstrap CIs.

The LRT could be retained as descriptive/exploratory, with an explicit caveat, or replaced by a cross-fitted procedure.

## Model A probability metrics need another correction

The leakage fix revealed another inconsistency.

For the standalone Model A performance table, `compute_global_performance(..., is_probability=False)` takes CIIS, min-max scales it using the test-set score range, computes a pseudo-Brier score, and simply assigns:

```python
slope = 1.0
intercept = 0.0
```

But the paired ΔBrier calculation uses the separate development-fitted logistic calibration that was just added.

Thus:

* “Model A Brier” in the table uses probability transformation (P_A^{(1)});
* ΔBrier uses another transformation (P_A^{(2)}).

Those quantities should not appear in the same comparison.

The development-fitted CIIS→mortality calibration should be constructed once and then used consistently for Model A Brier, log loss, calibration intercept/slope, and ΔBrier. Raw CIIS should remain the input for AUROC/AUPRC.

## The reporting generator still has a reproducibility bug

The CLI default is now only:

```text
--n-bootstraps 50
```

But `generate_primary_results_markdown()` always says:

> “All confidence intervals ... computed over 1,000 resamples.”

That text needs to use `result.*.n_bootstraps`, or the research-mode default should truly be 1,000 while unit/integration tests explicitly request smaller counts.

There is a related p-value concern: with 50 empirical bootstrap replications, the current bootstrap-difference implementation cannot support an empirical (p<0.001) claim.

## CIIS validation still deserves attention

The traditional-model validation report has not changed materially. It now uses the safer phrase “technical success rate” in parts, which is good, but still concludes that the 1,000-record benchmark demonstrates “98.9% technical fidelity.”

It still reports a striking distribution:

[
70.98% \text{ probable infarction},\qquad
\text{median CIIS}=27.65.
]

The existence of the real primary scoring path makes this more important, not less. Before trusting A-stratified scientific conclusions, I would validate the **waveform-derived CIIS measurements**, not merely score computability.

A useful validation sample would compare Q duration, Q/R ratios, T amplitudes, etc. against an independent implementation or manually reviewed ECG measurements. Until then I would call the 98.9% figure **technical completion**, not fidelity.

## The notebook remains particularly misleading

This should be fixed before sharing the repository externally.

The notebook introduces itself as:

> **“Authoritative Research Walkthrough & Empirical Findings”**

and says the holdout contains 32,256 real patients.

Then its own code defines:

```python
get_cohort_dataset(seed=42, n_total=30000)
```

and generates gamma-distributed CIIS scores, synthetic correlated transformer scores, and simulated mortality. Its execution output is 30,000 synthetic patients with 6,000 test patients and 8.5% mortality. It even produces (\rho=0.4936), while adjacent prose says (\rho=0.512) and still repeats the old “shared variance ~26%” interpretation.

This notebook should either become:

> **Simulation tutorial demonstrating the analysis API**

or load `predictions.parquet` + verified manifest and become the actual empirical notebook.

Not both.

## One engineering concern with the new real path

The real D-BETA path is correct conceptually, but currently not very efficient.

`build_real_predictions_for_cohort()` constructs a D-BETA adapter with `batch_size=32`, but `extract_transformer_embeddings()` iterates through every cohort record and calls `adapter.embed_single()` one at a time.

So `batch_size=32` isn't actually benefiting the full-cohort extraction.

For ~161k ECGs, I would use `embed_batch()`, expose `--device cpu|mps|cuda`, and checkpoint embeddings periodically. This also makes the unchanged Stage 5 claim of 3,414 ECGs/sec less relevant to how the actual full pipeline now operates.

## What I would require before reviewing the findings themselves

The repair sequence is now shorter than before:

1. **Make real mode fail closed.** Only `--simulate` may create synthetic data; a D-BETA/WFDB error must terminate. Add `source_mode` to artifacts and manifests.
2. **Perform one real Stage 8 run** and persist both the unified predictions and the underlying D-BETA embeddings. Manifest the actual Git SHA, `uv.lock` hash, model revision/weight hash, dataset versions and full artifact checksum.
3. **Regenerate Stage 9 from that artifact**. No manual README numbers. Verify cohort/event/category/quadrant identities automatically.
4. **Quarantine Stage 10 until wired to real inputs.** Real admission-anchored ECGs, embeddings, quality masks, Cornell/Sokolow measurements, CarDSLab embeddings and actual MIMIC age/sex evaluation strata; missing components should say “not run.”
5. **Make Stage 11 require actual Stage 9/10 result objects** and remove all hard-coded empirical fallbacks.
6. **Resolve the statistical issues**: consistent development-fitted A calibration, paired held-out incremental metrics as primary, LRT either cross-fitted or explicitly secondary, and an honest bootstrap count.
7. **Validate CIIS measurement fidelity** before interpreting the CIIS risk categories.
8. **Replace or relabel the notebook** so a reader cannot confuse a synthetic demonstration with empirical findings.
9. **Add provenance regression tests**: no-waveform/non-`--simulate` must fail; real-scoring exceptions must never produce synthetic artifacts; simulated artifacts must be visibly marked; report counts must reconcile automatically.
10. **Batch D-BETA inference and checkpoint it** before the full real run.

### Bottom line

PR #15 is **not cosmetic**. Several important pieces were genuinely repaired, and I now think the **primary real-data architecture is close to usable**.

But I still would not review the stated (\rho=0.512), AUROC 0.778, 2.35× discordance RR, or within-CIIS gradients as scientific findings, because there is still no coherent evidence on `main` that those numbers came from the newly implemented real pipeline. The current committed reports actually demonstrate the opposite through their unresolved count contradictions.

The next milestone should therefore be very narrow:

> **Produce one unquestionably real, provenance-locked Stage 8 prediction artifact and regenerate Stage 9 entirely from it.**

Once that exists, a re-audit can shift from “is this result real?” to the much more interesting question: **“what do the actual alignment findings mean?”**
