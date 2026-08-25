This round is another strong improvement. I would now rate the repository as **ready for an authoritative real-data run after a small number of targeted fixes**.

The main problems from the previous audit are genuinely resolved: hypothesis verdicts are now null-safe, the simulation negative control produces H2–H4 “Not Confirmed,” discordance language respects confidence intervals, quadrant definitions are correct, Brier sign semantics are clearer, external-validation recommendations are no longer automatically positive, and simulated reports are explicitly labeled as non-empirical.

PR #18 also adds the missing-manifest firewall, preserves actual artifact mode through reporting, and reports 168 passing tests with clean `basedpyright`.

## Current assessment

| Area                                                | Status                                      |
| --------------------------------------------------- | ------------------------------------------- |
| Simulation vs real provenance                       | **Strong**                                  |
| Missing-manifest fail-closed behavior               | **Strong, one edge case remains**           |
| Artifact checksum validation                        | **Strong**                                  |
| Report provenance banners                           | **Strong**                                  |
| Null-safe H2/H3/H4 interpretation                   | **Strong**                                  |
| Discordance interpretation                          | **Strong**                                  |
| Sensitivity report optionality                      | **Strong**                                  |
| Brier sign labeling                                 | **Fixed**                                   |
| Incremental-information semantics across Stage 9/11 | **Needs correction**                        |
| H4 decision criterion                               | **Needs correction**                        |
| Prespecification of hypothesis thresholds           | **Needs clarification before real run**     |
| Bootstrap settings for final analysis               | **Too small**                               |
| Long-running embedding recovery                     | **Checkpointing exists, resume does not**   |
| Roadmap completion status                           | **Does not reflect actual empirical state** |

### 1. One provenance edge case remains: malformed manifests fail open

The new artifact loader correctly fails if a manifest is entirely missing in real mode. That's good.

But if a companion manifest **exists but is malformed**, the loader does:

```python
except Exception:
  pass
```

and proceeds to load the artifact. Likewise, if the JSON parses but lacks `data_mode` or `predictions_artifact_sha256`, real mode can still accept it.

For real mode, I'd make a manifest valid only if it is:

```text
valid JSON object
+ data_mode == "real"
+ predictions_artifact_sha256 present
+ checksum matches
```

Anything else should be treated exactly like “manifest missing” unless `--allow-unverified-artifact` is explicitly supplied.

That's the last meaningful provenance hole I see.

## 2. Stage 11 still confuses **marginal superiority** with **incremental value**

This is now the most important scientific-code issue.

There are two different comparisons:

$$
\Delta AUROC_{\text{marginal}}
=
AUROC(B)-AUROC(A),
$$

versus

$$
\Delta AUROC_{\text{incremental}}
=
AUROC(A+B)-AUROC(A).
$$

The project question H4 is clearly the **second** one.

Stage 9 correctly distinguishes them. In the current simulation:

$$
AUROC_B-AUROC_A=0.0065,
$$

while:

$$
AUROC_{A+B}-AUROC_A=0.0115.
$$

But `evaluate_clinical_utility_distinction()` uses `comp_res.delta_auroc` and `comp_res.delta_brier`, i.e. the first comparison, while talking about **incremental prognostic information**.

`synthesize_external_validation_recommendation()` has the same problem:

```python
delta_auc = primary_result.comparison.delta_auroc
```

rather than:

```python
primary_result.incremental.auroc_improvement
```

That's why Stage 11 currently reports ΔAUROC = 0.0065 and ΔBrier = −0.1813 when discussing incremental information, whereas the actual combined-vs-A Stage 9 quantities are ΔAUROC = 0.0115 and essentially ΔBrier = 0.0000.

I would make Stage 11 use:

```text
incremental.auroc_improvement
incremental.brier_improvement
incremental.held_out_loss_reduction
incremental.held_out_loss_reduction_ci
```

for every statement about **adding B to A**.

Keep `comparison.delta_*` only for statements explicitly comparing standalone B against standalone A.

## 3. The descriptive LRT is still controlling H4

The code now correctly warns that the LRT is only descriptive because B was produced by a supervised upstream probe.

But H4 is still declared confirmed only if:

```python
inc_auc_ci.ci_lower > 0 and lrt_p < 0.05
```

That means the supposedly descriptive LRT still acts as a decision gate.

The external-validation recommendation does the same thing.

I'd remove it from the verdict criterion entirely.

Something like this is cleaner:

$$
H4\text{ supported}
\iff
CI_{95\%}(\Delta AUROC)>0
$$

possibly strengthened by:

$$
CI_{95\%}(\Delta\text{LogLoss reduction})>0.
$$

Then report the development LRT alongside those metrics as descriptive supporting information.

That makes the inferential hierarchy internally consistent.

## 4. Some newly called “prespecified” thresholds aren't actually in the proposal

This is worth fixing **before** the real run, because you still have the opportunity to freeze them prospectively.

The research proposal defines H1 simply as:

$$
\operatorname{cor}(A,B)>0
$$

and H2 as:

$$
P(Y=1\mid C_A,B_{\text{high}})
>
P(Y=1\mid C_A,B_{\text{low}}).
$$

But the implementation now defines:

* H1 confirmed only if \(0.30\le|\rho|<0.70\);
* H2 “meaningful” only if the T3/T1 ratio is at least **1.50** in every category.

The roadmap says strong/moderate/weak uses “prespecified descriptive criteria,” but doesn't actually document these numerical thresholds, nor the 1.50 H2 threshold.

Two clean choices exist now:

* explicitly add these thresholds to `docs/research-proposal.md` **before the real run**, with rationale; or
* stop calling them prespecified and treat them as descriptive interpretation conventions.

I slightly prefer the second for H1. A correlation of 0.75 wouldn't really mean H1 “failed”; it would mean **more alignment than hypothesized**.

For H2, I would additionally report uncertainty around the T3/T1 gradient or risk difference rather than relying only on a point-estimate cutoff.

## 5. Increase the bootstrap count before the real run

This is operationally important.

The committed simulation report correctly says its Stage 9 CIs use only **50 bootstrap samples**.

More importantly, Stage 10 is hard-capped in the CLI at:

```python
n_bootstraps=min(ctx.n_bootstraps, 20)
```

So even:

```bash
--n-bootstraps 1000
```

would leave sensitivity analyses at **20 resamples**.

That's appropriate for tests and demonstration runs, not an authoritative research report.

Before the real run I'd make:

```text
real mode:       1000 bootstrap resamples
simulation/test: 20–50
```

or provide separate CLI options.

With only 20 sensitivity bootstraps, percentile CIs are extremely coarse.

## 6. Checkpointing still doesn't actually resume

Batch inference is now correctly wired, and checkpoints are written periodically. Good improvement.

But `extract_transformer_embeddings()` initializes a fresh zero matrix and loops from ECG 0 every time. It never reads an existing checkpoint or resumes from `processed_count`.

For ~161k ECGs and a large transformer, I'd implement actual resume before the full run:

```text
checkpoint exists
→ validate cohort/model revision
→ restore embeddings + validity
→ start at processed_count
```

Also include enough metadata to ensure a checkpoint can't accidentally be resumed against a different cohort ordering or D-BETA revision.

This isn't a scientific blocker, but it can save a very painful rerun.

## 7. Simulation reporting still has a few semantic oddities

The provenance banner is clear enough that I don't consider these dangerous, but they are worth polishing.

Stage 9 says:

> “This report provides the **empirical evaluation**...”

while immediately above it says:

> **SIMULATION — NOT EMPIRICAL RESULTS**

Similarly, the simulated Stage 11 report says:

> “Model A completed scoring on 161,279 waveforms”

and:

> “Model B (D-BETA) completed scoring...”

even though those are simulated prediction rows, not actual waveform inference.

I'd make these mode-aware:

```text
simulation → "analysis demonstration"
real       → "empirical evaluation"
```

and omit technical model-completion claims entirely for simulation mode.

## 8. The roadmap currently overstates empirical completion

The README correctly says:

> **Empirical full-cohort run pending**

But the roadmap has Stage 8/9 fully checked, including:

* final-test B scores generated;
* derived score table;
* all primary results;
* every Stage 10 sensitivity;
* CarDSLab analysis;
* admission-anchored ECG analysis;
* quality-filter analysis.

Meanwhile the actual Stage 10 report correctly says those latter analyses **were not run**.

I would split roadmap status into:

```text
[code] implemented
[data] empirically executed
```

or simply uncheck the empirical items until the real run.

That will be especially useful for agents following `AGENTS.md`.

# What I think the current simulation tells us

As a software negative control, it's now doing its job very well.

It produces:

$$
AUROC_A\approx0.485,\qquad AUROC_B\approx0.491,
$$

no consistent within-CIIS gradients, no meaningful discordance effect, and no held-out incremental performance. The reports now mostly interpret that as null/inconclusive rather than forcing the intended hypotheses to succeed.

That substantially increases my confidence that an eventual positive real-data result won't simply be an artifact of the reporting templates.

## Recommendation

I would make **one final small pre-run PR**, rather than another broad audit cycle:

1. Make malformed/incomplete manifests fail closed.
2. Change all Stage 11 “incremental” quantities to **combined A+B vs A**, not B vs A.
3. Remove descriptive LRT p-values from H4/external-validation decision gates.
4. Freeze or relabel the H1/H2 interpretation thresholds.
5. Set research-grade bootstrap counts.
6. Add actual checkpoint resume.
7. Reconcile roadmap empirical checkboxes.

After that, I would stop polishing and run:

```bash
uv run ecg-alignment pipeline \
  --mimic-root ~/data/mimiciv/3.1 \
  --ecg-root ~/data/mimic-iv-ecg/1.0 \
  --n-bootstraps 1000 \
  --checkpoint-path <protected-local-path>/dbeta-embeddings.npz
```

At that point, the next audit should finally be an audit of the **real CIIS/D-BETA findings themselves**, rather than the framework generating them.
