Yes — this round is a significant improvement. The repository is now **much closer to being research-ready**, and several previous blockers are genuinely resolved.

My current assessment is:

> **The computational/provenance pipeline is now largely sound, but the scientific interpretation layer is still unsafe because it can convert null results into affirmative claims.**

That is a narrower and much easier problem to fix than what we started with.

## Current status

| Area                                              | Status                        |
| ------------------------------------------------- | ----------------------------- |
| Real-vs-simulation fail-closed execution          | **Fixed**                     |
| Cached-artifact mode checking                     | **Mostly fixed**              |
| Artifact SHA-256 verification                     | **Fixed**                     |
| Manifest records mode/Git/lock/artifact hash      | **Fixed**                     |
| `--batch-size` actually propagated                | **Fixed**                     |
| Embedding checkpoint CLI wiring                   | **Fixed**                     |
| Simulation class-repair typo                      | **Fixed**                     |
| Stage 9 count invariants                          | **Fixed/tested**              |
| README claiming premature empirical findings      | **Fixed**                     |
| Committed reports clearly marked simulation       | **Fixed**                     |
| Old 941-vs-1,842 count contradictions             | **Fixed**                     |
| Stage 10 fake sensitivity inputs                  | **Fixed**                     |
| Technical-failure counts passed from actual table | **Fixed**                     |
| Interpretation conditional on actual statistics   | **Still problematic**         |
| Artifact provenance fully fail-closed             | **One important gap remains** |
| Metric sign/label consistency                     | **One important bug remains** |

PR #17 explicitly tackles the previous audit items and reports 165 passing tests plus clean `basedpyright`.

# What is now genuinely good

The README now does exactly what I wanted: it removes the old headline effect sizes and says that the **empirical full-cohort run is pending regeneration**, while documenting how the real pipeline should be executed.

The committed Stage 9 report is also finally mathematically coherent. It is prominently labeled:

> **SIMULATION — NOT EMPIRICAL RESULTS**

and the CIIS stratum event counts now sum to 941:

[
467+230+133+111=941,
]

while the discordance quadrant events also sum to 941:

[
410+287+58+186=941.
]

The patient counts likewise reconcile to 32,256.

They added a regression test specifically enforcing those invariants, which is exactly the right response to the earlier failure.

Stage 10 is also now behaving properly. Unavailable analyses say “not run” rather than fabricating Cornell scores, CarDSLab embeddings, quality masks, etc. The current simulated report runs only the available mortality-horizon and demographic-strata analyses.

And the notebook is now appropriately described as a simulation/API tutorial rather than empirical evidence.

So the **data lineage problem is largely under control**.

---

# The new critical problem: the simulation has exposed an interpretation bug

Ironically, the new simulation report is now very useful as a negative control.

Its results are approximately null:

* Model A AUROC: 0.485
* Model B AUROC: 0.491
* ΔAUROC: (0.0065), CI (-0.0056,0.0257)
* discordance RD: (0.0010), CI (-0.0035,0.0058)
* discordance RR: (1.03), CI (0.89,1.21)
* incremental AUROC CI crosses zero
* descriptive LRT (p=0.165)

Yet the executive summary says:

> **H2: Confirmed**

> **H3: Confirmed ... significantly higher mortality**

> **H4: Confirmed ... significant incremental prognostic information**

Those statements are plainly contradicted by the statistics directly below them.

The cause is clear in `generate_primary_results_markdown()`: the words **“Confirmed”**, **“meaningful”**, **“significantly higher”**, and **“significant incremental prognostic information”** are unconditional string literals.

This must be fixed before the real run, because otherwise the report is structurally biased toward confirming the hypotheses regardless of outcome.

## H2 has the same problem

The simulated within-CIIS gradient ratios are:

[
1.09,\quad0.94,\quad1.26,\quad0.86.
]

Those are essentially null and even reverse direction in two strata. The Stage 11 table correctly labels every category **Sub-threshold**.

But immediately above it, Stage 11 says:

> “Model B consistently uncovers substantial residual risk gradients across all 4 traditional CIIS risk categories ... Even among ... Normal patients, Model B identifies a near 3-fold mortality spread.”

That narrative comes from `assess_within_a_gradients()`, where the summary is hard-coded irrespective of `all_meaningful` or the actual ratios.

This is probably the single most important remaining scientific-software bug.

A negative-control simulation should produce something like:

> No consistent residual-risk gradient was detected; none of the four strata met the prespecified meaningful-gradient threshold.

---

# Discordance interpretation is also significance-blind

The current simulated contrast is:

[
RD=+0.10%,\qquad95%,CI=(-0.35%,+0.58%)
]

and

[
RR=1.03,\qquad95%,CI=(0.89,1.21).
]

Yet Stage 11 says:

> “This represents a **statistically significant** risk difference ... identifying ... hidden risk”

and calls the result:

> “actionable risk reclassification.”

`interpret_discordant_groups()` literally inserts “statistically significant,” “more than double,” “actionable,” and similar language regardless of the interval estimates.

That should instead be conditional:

```text
CI excludes null + effect direction correct
    -> evidence of informative discordance

CI includes null
    -> no clear evidence of discordance

effect reversed
    -> discordance in opposite direction
```

And I would avoid **“actionable”** entirely until DCA/prospective evidence exists.

---

# There is also a quadrant-description bug

This line constructs quadrant criteria based on:

```python
'low' in q.label.lower()
```

for **both A and B**.

Consequently, the Stage 11 report describes:

`A-high / B-low`

as:

> `A < 15.0, B < 0.365`

which is wrong; it should be:

[
A\ge15,\qquad B<0.365.
]

And `A-low / B-high` similarly gets the B direction wrong. You can see those incorrect criteria in the committed report.

Parse the A and B components independently rather than searching for `"low"` anywhere in the combined label.

---

# The “clinical utility distinction” function still hard-codes significance

This is another clear negative-control failure.

The actual simulated LRT is:

[
p=0.165.
]

But Stage 11 says:

> “highly significant incremental prognostic information”

and even:

> “p < 10^-15”

plus:

> “Model B adds undeniable variance”

The source shows why:

```python
statistical_summary = (
  "Statistical evaluation demonstrates highly significant ..."
  f"... p < 10^-15 ..."
)
```

The numeric `lrt_pval` is computed but isn't actually used in that string.

Likewise its formal statement always says the study establishes:

> “strong, statistically robust incremental prognostic information.”

That must be conditional on the results.

This is especially important because the real outcome may genuinely be null. The software needs to regard **failure to confirm H2–H4 as a scientifically valid outcome**.

---

# External-validation recommendation is still biased toward positivity

This function is more dynamic now, but not sufficiently so.

With the current null simulation, it produces:

> “Significant incremental prognostic signal confirmed (LRT p = 1.65e-01...)”

which is self-contradictory.

It then concludes:

> “Formal external validation is STRONGLY JUSTIFIED. The consistent, large effect size...”

despite ΔAUROC ≈ 0.0065 with a CI containing zero.

There are really three possible recommendations:

* **Strong positive result:** external validation is strongly justified.
* **Suggestive/uncertain result:** external validation may be worthwhile to resolve uncertainty.
* **Clear null result:** external validation of this exact formulation is not yet strongly motivated; revisit representation, endpoint, or comparator first.

The contamination issue alone does not make external validation scientifically worthwhile if the in-domain probe shows no signal.

---

# Brier-score semantics need correction

There is also a real metric-label bug.

The code defines:

[
\Delta Brier =
Brier_A-Brier_B
]

so **positive means B improves upon A**.

But the primary results table labels its difference column:

> **Difference (Model B − Model A)**

That is the opposite sign convention.

The current simulation makes this obvious:

[
Brier_A=0.0283,\qquad
Brier_B=0.2096.
]

So B is dramatically worse. Yet the table reports:

> `-0.1813 ... (improvement)`

Numerically,

[
0.0283-0.2096=-0.1813,
]

so the implementation is behaving as designed, but the label and unconditional word **“improvement”** are wrong.

I would call it:

> **Brier improvement (A − B)**

and render:

* positive → improvement;
* zero → no change;
* negative → deterioration.

The same applies to log-loss reduction.

---

# Artifact provenance still has one important escape hatch

The artifact firewall is substantially improved. The loader now checks a companion manifest's `data_mode` and the artifact SHA-256.

But it does **not fail when the manifest is absent**.

If:

```python
manifest_path = find_companion_manifest(p)
```

returns `None`, the function simply loads the prediction table.

So an arbitrary unmanifested Parquet file can still be consumed in `"real"` mode.

I would change real-mode behavior to:

```text
requested_mode == real
AND manifest missing
    -> fail
```

unless there is an explicit flag such as:

```text
--allow-unverified-artifact
```

There is an additional subtlety: if `--allow-simulated-artifact` is used while `ctx.simulate == False`, the loader correctly allows the simulated artifact, **but the generated report receives `data_mode="real"` because report provenance is derived from the CLI flag rather than from the artifact manifest**.

So this command conceptually could:

```text
real mode
+ allow simulated artifact
→ load simulation
→ label report REAL
```

The correct design is for the verified loader to return something like:

```python
VerifiedPredictionArtifact(
  dataframe=df,
  data_mode="simulation",
  manifest=...,
  sha256=...,
)
```

and downstream report provenance must come from that object, never from `ctx.simulate`.

Also, if `--predictions-path` points outside `output_dir`, the current manifest is saved to `output_dir/run_manifest.json`, whereas companion-manifest discovery searches next to the artifact. Saving a sidecar such as:

```text
predictions.parquet.manifest.json
```

would make this much more robust.

---

# Stage 10 has one remaining unconditional conclusion

The Stage 10 generator is now mostly data-driven, but its conclusion always says:

> “Evaluated sensitivity analyses confirmed directional consistency with primary findings.”

The current simulation includes:

* 1-year ΔAUROC = (-0.0025)
* age <65 ΔAUROC = (-0.0008)
* female ΔAUROC = (-0.0053)

so even as a literal direction statement, it isn't uniformly true.

That should be calculated, not asserted.

---

# One wording regression: “fidelity”

Earlier they correctly changed CIIS validation to **technical completion rate**.

But Stage 11 now says:

> “Technical scoring achieved high **fidelity** across the cohort...”

That reintroduces the distinction we deliberately removed. Successful execution is not measurement fidelity.

Use:

> **technical completion**

until the waveform-derived CIIS components have been externally validated.

---

## What I would do now

At this point I would **not make another broad architecture pass**. The architecture is good enough.

I would create one focused PR called something like:

> `fix: make scientific interpretation null-safe and provenance-strict`

with these checks:

1. Missing manifest fails in real mode.
2. Actual artifact mode, not CLI mode, determines report banner.
3. H1–H4 verdicts are computed from prespecified criteria rather than hard-coded `Confirmed`.
4. H2 summary respects `is_meaningful` and `all_categories_meaningful`.
5. H3 “significant” language requires CI exclusion of 0/1.
6. Discordance quadrant criteria are rendered correctly.
7. H4 wording respects held-out CIs; descriptive LRT never overrides a null held-out result.
8. Brier/log-loss sign conventions are explicit and consistent.
9. Stage 10 invariance claim is derived from results.
10. Stage 11 external-validation recommendation is effect-dependent.
11. Replace “technical fidelity” with “technical completion.”

Most importantly, add **negative-control tests**. Feed the current near-null simulation into every report generator and assert that the output does **not** contain phrases such as:

```text
Confirmed
statistically significant
actionable
substantial residual risk
highly significant
undeniable
large effect size
strongly justified
```

unless the statistics actually meet the corresponding prespecified criteria.

That would be a powerful research-software safeguard.

## Overall trajectory

The progression across these audits is encouraging:

**first audit:** reported findings were effectively synthetic but presented as empirical.

**second audit:** real model path existed, but provenance and reports were still mixed.

**third audit:** provenance was mostly secured, but reports were stale.

**current audit:** **provenance and numerical consistency are largely fixed; the remaining failure is chiefly scientific interpretation logic.**

Once that interpretation layer becomes **null-safe**, I would be comfortable recommending they run the first authoritative full real-data pipeline and then we can finally audit the actual MIMIC/D-BETA/CIIS findings rather than the machinery around them.
