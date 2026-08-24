## Re-audit verdict

This update is **materially better**. The core real-data pipeline is now close to something I would trust operationally.

However, I still would **not trust the numerical findings committed in the repository**. The main blocker has shifted from “the code is secretly synthetic” to:

> **The code is increasingly correct, but the committed reports/README still contain old, internally inconsistent results that were not regenerated from the corrected pipeline.**

### Status of the previous blockers

| Issue                                             | Current status              |
| ------------------------------------------------- | --------------------------- |
| Real mode silently falls back to simulation       | **Fixed**                   |
| Explicit provenance / `data_mode`                 | **Mostly fixed**            |
| Model A test-set calibration leakage              | **Fixed**                   |
| Model A Brier/calibration inconsistency           | **Fixed**                   |
| Empty risk-surface cells / discordance CI         | **Fixed previously**        |
| LRT presented as primary formal inference         | **Improved substantially**  |
| Stage 10 fabricates missing analyses              | **Fixed in core code**      |
| Stage 11 requires primary results                 | **Fixed in API**            |
| Notebook masquerades as empirical                 | **Fixed**                   |
| CIIS “fidelity” wording                           | **Fixed appropriately**     |
| Committed empirical reports consistent            | **Still critically broken** |
| Cached simulation artifact provenance enforcement | **Still broken**            |
| Stage 11 entirely derived from current results    | **Still incomplete**        |
| Stage 10 reporting entirely data-driven           | **Still incomplete**        |

---

# 1. Fail-closed execution is genuinely fixed

This was the most important coding change.

Without `--simulate`, missing waveforms now produce a nonzero exit rather than silently switching modes. Exceptions during real scoring likewise terminate instead of substituting random data.

The tests were also changed appropriately:

```python
rc = main([
  "probe",
  ...
])
assert rc == 1
```

when waveforms are absent, while simulation tests now explicitly pass `--simulate`.

This is a **real fix**, not documentation-only.

The manifest is also considerably better: it now records `data_mode`, Git SHA, `uv.lock` SHA-256, and the serialized prediction artifact checksum. That is exactly the right provenance direction.

---

# 2. But cached prediction artifacts can still bypass the mode firewall

This is now the most important **code-level provenance issue**.

`_get_unified_table()` does:

```python
if ctx.predictions_path is not None and ctx.predictions_path.exists():
  return load_unified_prediction_table(ctx.predictions_path)
```

without checking whether that artifact came from `"real"` or `"simulation"` mode. Likewise, `run_probe()` simply accepts an existing prediction file.

And the test suite explicitly demonstrates the problem:

1. generate `custom_preds.parquet` using `--simulate`;
2. verify the companion manifest says `"data_mode": "simulation"`;
3. call `analyze --predictions-path custom_preds.parquet` **without `--simulate`**;
4. expect analysis to succeed.

So:

[
\text{real scoring path is fail-closed}
]

but

[
\text{artifact consumption is not fail-closed}.
]

I would bind prediction artifact + manifest tightly. `load_unified_prediction_table()` or a higher-level loader should verify:

```text
artifact checksum == manifest checksum
manifest.data_mode == requested execution mode
```

A simulation artifact should require either `--simulate` or an explicit `--allow-simulated-artifact`.

More importantly, **every generated report should carry the provenance mode in its header**:

> Data source: REAL MIMIC-IV-ECG predictions

or

> Data source: SIMULATION — NOT EMPIRICAL RESULTS

That would make accidental mislabeling much harder.

---

# 3. The committed Stage 9 report is still decisively inconsistent

This remains the largest repository-level problem.

They changed one line to say:

[
N=32,256,\quad D=941,\quad 2.92%.
]

But the rest of `reports/primary-results.md` still describes the old 1,842-event dataset.

For example, the CIIS strata report:

* Normal: 582 deaths
* Borderline: 405
* Possible Injury: 368
* Probable Infarction: 487

which gives

[
582+405+368+487=\boxed{1,842}.
]

The discordance quadrants likewise give:

[
442+688+98+614=\boxed{1,842}.
]

Yet immediately above those tables, the report says there were only **941 total deaths**.

This is not a rounding problem. It proves that the report was **edited rather than regenerated**.

The old H4 contradiction also survives:

* executive summary: LRT = 384.62, ΔAUROC = 0.0542;
* detailed section: LRT = 649.40, ΔAUROC = 0.0886.

So I would still classify the committed Stage 9 findings as:

> **not scientifically interpretable yet.**

The updated `generate_primary_results_markdown()` is substantially better. The problem is that the committed `primary-results.md` clearly isn't its authoritative regenerated output.

---

# 4. The README therefore remains misleading

The README still prominently labels these numbers:

## “Key Empirical Findings”

and reports:

* (\rho=0.512)
* Model B AUROC = 0.778
* RR = 2.35
* ΔAUROC = 0.0886
* LRT = 384.62

while pairing them with the newly corrected 941-event denominator.

Because the underlying Stage 9 tables are incompatible with 941 deaths, the README cannot currently substantiate those values.

I would temporarily replace that whole section with something like:

> **Empirical full-cohort run pending regeneration after pipeline provenance audit.**

Then restore results automatically after the authoritative real run.

---

# 5. Stage 10 code is much better, but the committed Stage 10 report is still old

This is another substantial coding improvement.

`run_sensitivity_analyses()` now treats unavailable analyses as unavailable rather than generating random substitutes:

* no admission cohort → anchoring analysis omitted;
* no embeddings → probe sensitivity omitted;
* no quality mask → quality analysis omitted;
* no alternative ECG scores → omitted;
* no CarDSLab embeddings → omitted;
* no demographic strata → omitted.

That is exactly the right design.

The current pipeline realistically supplies:

* actual primary/test predictions;
* development predictions;
* MIMIC-derived age/sex evaluation strata;

but does **not** yet supply admission-anchored predictions, raw D-BETA embeddings, alternative ECG scores, quality masks, or CarDSLab embeddings.

So many Stage 10 sections should correctly say **“not run.”**

But the committed `reports/sensitivity-analyses.md` still claims all seven sensitivity dimensions were actually evaluated, including precise admission-anchored, Cornell, Sokolow-Lyon, quality-filter, CarDSLab and probe-architecture results. It also still uses **1,842 deaths / 5.71%**.

That report is therefore obsolete and should probably just be deleted and regenerated.

---

# 6. There are still hard-coded claims in the new Stage 10 report generator

Even after the optionality fix, a few reporting strings remain too opinionated.

For example, the outcome-horizon table always writes:

```python
"$p < 0.001$"
```

instead of using `oh.incremental_pvalue`.

It also unconditionally states:

> Model B consistently demonstrates significant discriminative performance over Model A.

That conclusion needs to depend on the actual confidence intervals/results.

Similarly, when cohort anchoring happens to be available, its prose hardcodes:

> ΔAUROC > +0.07 and ρ ≈ 0.50

rather than deriving the interpretation from the object.

I would make result generators almost boringly mechanical:

```text
results object → formatting
```

and put substantive interpretation in a separate, explicitly data-driven interpretation layer.

---

# 7. Stage 11 is improved, but is still not truly dynamic

Removing the fallback `rho=0.512` object was a very good fix.

`synthesize_research_interpretation()` now requires `PrimaryAnalysisResult` and refuses to synthesize empirical conclusions without it. The pipeline also now passes its actual `primary_result` into Stage 11.

But several hard-coded empirical values survived elsewhere.

For example, `synthesize_external_validation_recommendation()` still says, regardless of the supplied result:

> “LRT p < 10^-15, Delta AUROC +0.0872”

and:

> “29.7% ... with 2.35x mortality elevation”

and claims invariance across a “comprehensive sensitivity battery.”

Those exact numbers need to be computed from `primary_result`.

Even more visibly, `generate_research_interpretation_markdown()` still hardcodes its Mermaid diagram:

```text
rho = 0.512
2.36x - 2.86x
RR = 2.35x
Delta AUROC +0.0872
99.92% Scored
```

regardless of the supplied synthesis object.

So Stage 11 is now **dynamically driven in its main tables but still statically contaminated in its prose/figures**.

---

# 8. Technical failure reporting has a new provenance problem

`synthesize_research_interpretation()` defaults to:

```python
total_waveforms=161279
model_a_failures=0
model_b_failures=0
```

And `run_interpret()` does not override those counts from the unified prediction table.

That means a genuine real run in which CIIS fails on, say, 1% of ECGs can still generate a Stage 11 report claiming zero failures.

This is especially easy to fix because the information already exists:

```python
model_a_failures = (~unified_df["model_a_valid"]).sum()
model_b_failures = (~unified_df["model_b_valid"]).sum()
```

There should be **no defaults for empirical failure counts**.

---

# 9. The LRT treatment is now acceptable for this preliminary study

This part is much better.

The primary-results generator now puts held-out metrics first:

* ΔAUROC
* ΔBrier
* held-out log-loss reduction

and explicitly describes the LRT as:

> **Descriptive Development LRT**

with a note explaining that B comes from an upstream probe trained on development outcomes.

That resolves my main objection. I would still consider cross-fitting if this becomes a methods-heavy manuscript, but for a preliminary opportunity-probing experiment, the current framing is reasonable.

One small extension: I'd add a paired bootstrap interval for **Δlog-loss**, not only a point reduction.

---

# 10. Model A calibration is now handled correctly

Another genuine fix.

The current flow fits:

[
CIIS \rightarrow P(Y=1)
]

using development data, then passes those frozen probabilities into `compute_global_performance()` for:

* Brier score;
* log loss;
* calibration slope/intercept.

Raw CIIS remains the ranking score for AUROC/AUPRC.

That is exactly the separation I wanted.

---

# 11. Notebook labeling is fixed

The notebook now clearly identifies itself as:

> **Simulation Tutorial & API Walkthrough**

and explicitly says its scores are simulated.

So that previous issue is resolved.

The only remaining problem is that it directs readers to the current `primary-results.md` as the “authoritative empirical results,” when that report is still internally inconsistent.

---

# 12. CIIS wording is now appropriate

They also incorporated the distinction I wanted:

> **98.9% technical completion rate**

rather than “98.9% fidelity,” and explicitly mark external fiducial/measurement validation as future work.

That is an appropriate scientific boundary for this preliminary experiment.

---

# 13. Small implementation bugs remain

Two are worth fixing.

### `--batch-size` doesn't actually control extraction batching

`build_real_predictions_for_cohort()` creates:

```python
DbetaAdapter(
  DbetaConfig(
    device=device,
    batch_size=batch_size,
  )
)
```

but then calls:

```python
extract_transformer_embeddings(
  cohort_df,
  adapter=adapter,
  ecg_data_dir=ecg_root,
)
```

without passing `batch_size`. The extraction function therefore uses its own default of 32.

So:

```bash
--batch-size 128
```

currently changes adapter configuration but not the outer batching loop.

Pass:

```python
batch_size=batch_size
```

explicitly.

The newly implemented checkpoint functionality is likewise not yet wired into the CLI.

### Simulation class-repair typo

There is:

```python
if len(np.unique(dev_y)) < 2 and len(dev_y) >= 2:
  dev_y[0] = 1
  dev_y[0] = 0
```

The second line presumably should be:

```python
dev_y[1] = 0
```

This only affects simulation mode, but is an obvious bug.

---

# Revised overall assessment

My assessment has meaningfully improved:

### Research software

**Now largely credible.**

The real pipeline is present, fail-closed behavior is implemented, provenance information is much stronger, analysis leakage was corrected, Stage 10 optionality is sensible, and the primary statistical design is now much more defensible.

### Scientific results currently displayed on GitHub

**Still not credible.**

The repository's README, Stage 9, Stage 10 and Stage 11 reports continue to contain legacy result values that demonstrably cannot all arise from the stated 941-event cohort. The Stage 9 stratum and quadrant tables alone prove that.

So I would now describe the state as:

> **Research pipeline nearly ready for an authoritative empirical run; previously committed numerical findings must be discarded and regenerated.**

That is a significant improvement from the earlier state.

## What I would do next

I would stop changing methodology for the moment and do one focused cleanup/run cycle:

1. Fix cached-artifact mode validation.
2. Remove remaining hard-coded empirical values from Stage 10/11 generators.
3. Derive technical failure counts from the unified table.
4. Fix the batching and simulation typos.
5. Temporarily remove the README “Key Empirical Findings.”
6. Delete or archive the existing Stage 8–11 empirical Markdown reports.
7. Run the **real** full pipeline once on the local MIMIC-IV/MIMIC-IV-ECG datasets.
8. Preserve `predictions.parquet` + manifest + embedding checkpoints locally.
9. Regenerate Stage 8–11 exclusively from that artifact.
10. Check simple invariants automatically before publishing:

[
\sum_{\text{CIIS strata}}N=N_{\text{test-valid}},
]

[
\sum_{\text{CIIS strata}}D=D_{\text{test-valid}},
]

[
\sum_{\text{quadrants}}N=N_{\text{test-valid}},
]

[
\sum_{\text{quadrants}}D=D_{\text{test-valid}}.
]

After that, I think the next audit can finally focus mostly on the **actual scientific result** rather than provenance reconstruction.
