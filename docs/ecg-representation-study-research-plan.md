---
title: "Research Plan: Incremental Prognostic Value, Population Heterogeneity, and Deployment Trade-offs of ECG Representations"
author: "Sae-Hwan Park"
date: 2026-09-02
---

**Status:** Canonical working plan  
**Date:** September 2026  
**Primary comparison:** CIIS-derived conventional ECG representation vs. D-BETA learned ECG representation  
**Primary outcome:** 30-day all-cause mortality after index ECG  
**Primary data source:** MIMIC-IV + MIMIC-IV-ECG  
**Planned validation:** At least one pretraining-disjoint external cohort if a compatible outcome can be constructed

---

## 1. Motivation

Modern ECG foundation models can encode substantially richer waveform information than conventional ECG measurements. However, demonstrating that a learned representation has a higher AUROC than a traditional ECG score is no longer, by itself, a strong scientific contribution. Recent work has increasingly benchmarked ECG foundation models, evaluated subgroup performance, and examined staged AI-ECG deployment.

The central goal of this study is therefore not simply to ask whether D-BETA predicts 30-day mortality better than CIIS. Instead, the study asks:

> **What incremental prognostic information does a modern ECG representation recover beyond conventional ECG features and ordinary clinical context, is that gain shared consistently across patient populations, and is the gain worth the additional computational burden?**

The design intentionally keeps the primary comparison focused on two contrasting ECG representations rather than expanding immediately to a large model benchmark. This allows deeper investigation of representation value, overlap, subgroup heterogeneity, and deployment implications.

A key limitation must be recognized from the outset: **D-BETA was pretrained on MIMIC-IV-ECG**. Therefore, analyses performed on MIMIC-IV constitute an **in-domain downstream representation study**, not clean external validation of generalizability. A pretraining-disjoint external validation cohort is highly desirable for the strongest version of the study.

---

## 2. Scientific Questions

### RQ1 — Representation value

Under identical clinical context, outcome definition, patient splits, and downstream modeling framework, how much incremental predictive value does D-BETA provide relative to conventional CIIS-derived ECG features?

### RQ2 — Information overlap and complementarity

Do CIIS-derived features and D-BETA encode largely overlapping prognostic information, or does each contain information not recoverable from the other?

Of particular interest is **asymmetric information containment**:

- Does D-BETA add substantial information beyond CIIS?
- Does CIIS still add meaningful information beyond D-BETA?

### RQ3 — Clinical-context attenuation

How much of D-BETA's advantage persists after adding ordinary clinical context?

This asks whether D-BETA is primarily recovering latent proxies for known disease burden or whether it retains prognostic information beyond structured clinical history.

### RQ4 — Population heterogeneity

Does the incremental value of D-BETA over CIIS differ across clinically important subgroups such as sex, age, and race/ethnicity?

The target is **heterogeneity of representation gain**, not merely subgroup-specific performance of one model.

### RQ5 — Computational efficiency

What additional memory, compute time, storage, and inference cost are required to obtain D-BETA's predictive gain?

### RQ6 — Marginal deployment value

Can a cheaper CIIS-based system safely screen patients for selective D-BETA evaluation, or does representation discordance cause such a cascade to miss patients whom D-BETA uniquely identifies as high risk?

---

## 3. Overall Study Architecture

The study will proceed in three conceptual layers.

### Layer 1 — ECG-only representation comparison

Compare conventional and learned ECG representations without clinical context.

This preserves continuity with the preliminary study and answers:

> How different are the representations when both receive only the ECG?

### Layer 2 — Controlled clinical-context comparison

Add the same clinical context to both representations and use the same downstream learner.

This answers:

> Does the learned ECG representation still add more prognostic information when ordinary clinical information is available?

### Layer 3 — Heterogeneity and deployment analysis

Evaluate whether the incremental gain varies across populations and whether the gain can be recovered efficiently under computational or care-management constraints.

---

## 4. Cohort and Index Time

### 4.1 Primary cohort

Use MIMIC-IV-ECG linked to MIMIC-IV clinical data.

Primary eligibility principles:

- adults only;
- one index ECG per patient;
- sufficient waveform quality for both CIIS feature extraction and D-BETA embedding extraction;
- sufficient outcome follow-up for 30-day mortality;
- patient-level disjoint development, validation, and final test sets.

The preferred index ECG is the **earliest eligible ECG per patient**, unless a different rule is justified before final analysis.

### 4.2 Index time

The ECG acquisition time defines the prediction index.

All predictor variables must be available before or at this time according to a prespecified availability policy.

### 4.3 Outcome

Primary outcome:

$$
Y = \mathbf{1}(\text{all-cause death within 30 days after index ECG})
$$

Secondary outcomes may be explored later, but the primary study should remain centered on one prespecified endpoint to avoid dilution.

---

## 5. ECG Representations

### 5.1 Canonical CIIS score

Retain the canonical Cardiac Infarction/Injury Score as a clinical reference and continuity analysis.

This is not the primary fair comparator because it was not originally trained as a 30-day mortality model.

### 5.2 Representation A: CIIS-derived conventional ECG features

Use the 15 underlying ECG measurements/components used by CIIS as the conventional ECG representation:

$$
Z_A = \text{CIIS-derived ECG feature vector}
$$

Fit a mortality-specific regularized linear probe to these features.

This makes the comparison with D-BETA substantially fairer because both representations receive the same downstream outcome model.

Preferred manuscript label:

> **CIIS-feature linear probe**

rather than "CIIS model."

### 5.3 Representation B: D-BETA embedding

Use the frozen D-BETA ECG encoder:

$$
Z_B \in \mathbb{R}^{768}
$$

No outcome-specific fine-tuning of the D-BETA encoder will be performed in the primary analysis.

### 5.4 Small representation controls

These are sensitivity analyses, not additional headline models.

#### Dimension-matched D-BETA

Apply PCA learned only on development data:

$$
Z_B^{(15)} = PCA_{15}(Z_B)
$$

then fit the same ridge-logistic probe.

Purpose: assess whether D-BETA's advantage persists after reducing representational dimensionality to approximately match the conventional feature space.

#### Random-encoder control

If technically feasible, generate features from a randomly initialized encoder with the same or closely matched architecture, freeze them, and fit the same downstream probe.

Purpose: address the contemporary concern that high-dimensional random nonlinear features can sometimes perform surprisingly well under linear probing.

---

## 6. Clinical Context

### 6.1 Primary structured context

Define:

$$
C_0 =
\text{Age}
+
\text{Sex}
+
\text{prior-recorded Elixhauser comorbidities}
$$

Race/ethnicity should generally be reserved for subgroup evaluation rather than automatically entered as a predictor unless there is a clear scientific rationale.

### 6.2 Elixhauser construction

Use ICD-based Elixhauser binary indicators, potentially implemented with `risk-compose`.

Important leakage rule:

MIMIC-IV hospital diagnosis codes are discharge/billing-derived and do not provide reliable diagnosis timestamps within an admission. Therefore, the primary predictor set should **not** use index-admission discharge diagnoses.

Preferred definition:

$$
X_2 =
\text{Elixhauser conditions derived from hospitalizations completed before index ECG}
$$

Possible primary history rule:

$$
\text{prior admission discharge time} < \text{index ECG time}
$$

A fixed lookback window can be used if clinically justified; otherwise all observable prior MIMIC history can be used with explicit acknowledgment that MIMIC does not represent complete longitudinal care.

### 6.3 Clinical-note context: secondary analysis

Clinical text should not be required for the primary study.

Define a secondary enriched context:

$$
C_1 = C_0 + X_1
$$

where $X_1$ is a frozen clinical-note embedding.

The scientific question is:

> Does D-BETA still provide superior ECG representation when rich narrative clinical context is already available?

#### Note eligibility policy

Only notes genuinely available before the index ECG may be used.

Exclude:

- discharge summaries completed after the index ECG;
- ECG interpretation reports linked to the index ECG;
- notes containing later clinical events;
- documentation whose timing cannot be established sufficiently for the intended prediction setting.

The note encoder should be selected for clinical-note representation quality; it need not be architecturally matched to D-BETA's text encoder.

---

## 7. Model Specification

### 7.1 Downstream learner

Use the same intentionally simple learner for all learned representations:

- frozen input representation;
- development-set feature standardization;
- $L_2$-penalized logistic regression;
- regularization strength selected independently for each representation using validation log-loss;
- final coefficients and preprocessing frozen before final-test evaluation.

The goal is to isolate differences in **representation information**, not downstream modeling sophistication.

### 7.2 Core model set

For shared context $C$, estimate:

$$
M_C: Y \sim C
$$

$$
M_A: Y \sim C + Z_A
$$

$$
M_B: Y \sim C + Z_B
$$

$$
M_{AB}: Y \sim C + Z_A + Z_B
$$

Primary analyses should use $C=C_0$.

Secondary text-enriched analyses may repeat the comparison with $C=C_1$.

### 7.3 ECG-only models

Also fit:

$$
M_A^{ECG}: Y \sim Z_A
$$

$$
M_B^{ECG}: Y \sim Z_B
$$

These preserve the original ECG-only scientific question.

---

## 8. Primary Estimands

### 8.1 Representation gain without clinical context

For performance measure $P$:

$$
\Delta_{ECG}
=
P(M_B^{ECG}) - P(M_A^{ECG})
$$

### 8.2 Context-adjusted representation gain

$$
\Delta_C
=
P(M_B) - P(M_A)
$$

### 8.3 Attenuation after clinical context

Conceptually:

$$
A_{context}
=
\Delta_{ECG} - \Delta_C
$$

Interpretation:

- large attenuation: much of D-BETA's advantage overlaps with ordinary clinical information;
- little attenuation: D-BETA retains substantial prognostic information beyond known clinical context.

This should be interpreted descriptively across performance measures rather than as a single universal scalar.

### 8.4 Bidirectional incremental value

Evaluate:

$$
M_A \rightarrow M_{AB}
$$

and

$$
M_B \rightarrow M_{AB}
$$

The asymmetry of these two increments is a key target.

If D-BETA adds substantially to A while CIIS features add little to B, this supports approximate asymmetric information containment.

---

## 9. Predictive Performance Evaluation

Report:

- AUROC;
- AUPRC;
- log-loss;
- Brier score;
- calibration intercept;
- calibration slope;
- calibration plots.

Because 30-day mortality is uncommon, AUPRC and calibration should receive substantial emphasis alongside AUROC.

Use patient-level bootstrap intervals for paired performance differences where appropriate.

The scientific interpretation should prioritize effect magnitude, stability, and clinical meaning rather than relying on significance thresholds.

---

## 10. Representation Alignment and Complementarity

### 10.1 Total prediction alignment

Report:

$$
\rho(\hat p_A,\hat p_B)
$$

using Spearman correlation and graphical displays.

### 10.2 ECG-specific contribution alignment

Because Models A and B share clinical context, raw prediction correlation partly reflects common predictors.

For the linear-logit models, define ECG-specific contributions:

$$
S_A = \hat\beta_A^\top Z_A
$$

$$
S_B = \hat\beta_B^\top Z_B
$$

Then examine:

$$
\rho(S_A,S_B)
$$

This asks:

> Do the two ECG representations assign similar ECG-derived prognostic risk after the downstream model has accounted for the shared clinical context?

### 10.3 Reciprocal residual-risk gradients

Analyze both directions.

#### B within A strata

Within fixed quantiles or clinically meaningful strata of Model A risk, examine outcome gradients across Model B risk.

#### A within B strata

Within fixed Model B strata, examine outcome gradients across Model A risk.

A strongly asymmetric pattern would support unequal information containment.

### 10.4 Discordant groups

Study clinically interpretable discordant populations such as:

- A-low / B-high;
- A-high / B-low;
- concordant low;
- concordant high.

Compare observed event rates and clinical characteristics.

These analyses are particularly relevant to understanding whether selective A→B deployment is safe.

---

## 11. Population Heterogeneity

### 11.1 Primary subgroup dimensions

Prespecify a limited number of clinically important subgroups:

- sex;
- age categories;
- race/ethnicity.

Potential additional dimension:

- comorbidity burden.

Intersectional analyses should be exploratory unless event counts are sufficient.

### 11.2 Target estimand

For subgroup $s$:

$$
\Delta_s
=
P(M_B \mid s)
-
P(M_A \mid s)
$$

The central question is whether:

$$
\Delta_{s_1}
\neq
\Delta_{s_2}
$$

This is **heterogeneity of incremental representation value**.

It is distinct from simply asking whether Model B performs differently across groups.

### 11.3 Metrics

Evaluate subgroup-specific differences in:

- AUROC;
- AUPRC;
- calibration;
- capacity-constrained event capture;
- threshold-specific sensitivity/specificity where clinically justified;
- decision-analytic utility where appropriate.

### 11.4 Interpretation

Avoid automatically labeling every subgroup difference as algorithmic unfairness.

Possible mechanisms include:

- disease phenotype heterogeneity;
- representation/training-data heterogeneity;
- case-mix differences;
- prevalence differences;
- label quality;
- measurement quality.

Risk-standardized or adjusted-metric analyses can be added as focused sensitivity analyses if raw subgroup results are scientifically important.

---

## 12. Capacity-Constrained Clinical Evaluation

Clinical care-management systems often operate under fixed capacity rather than a universal probability threshold.

For:

$$
q \in \{1\%, 2.5\%, 5\%, 10\%, 20\%\}
$$

select the top $q\%$ of predicted-risk patients.

Report:

- patients flagged;
- deaths captured;
- sensitivity/event capture;
- PPV;
- false positives per event captured;
- number needed to evaluate/intervene.

This answers:

> With identical clinical-management capacity, which ECG representation identifies more subsequent events?

Capacity thresholds should be distinguished clearly from clinical probability thresholds.

---

## 13. Decision Curve Analysis

Use calibrated predicted probabilities and a prespecified clinically plausible threshold range.

Report:

- model-specific net benefit curves;
- incremental net benefit where useful;
- treat-all and treat-none references.

DCA should remain a decision-analytic layer, not a mass hypothesis-testing exercise.

Avoid:

- dozens of pointwise $P$-values;
- interpreting lack of statistical significance as equivalence of clinical utility.

When comparing subgroups, remember that raw net benefit depends on event prevalence. Cross-group comparisons should therefore be interpreted cautiously, with standardized net benefit considered as a complementary analysis if needed.

---

## 14. Computational Analysis

Measure incremental computational cost under fixed hardware and software conditions.

### 14.1 Feature extraction

Record:

- wall time;
- throughput (ECGs/second);
- per-ECG latency;
- peak CPU memory;
- peak GPU memory;
- representation storage size.

### 14.2 Downstream fitting and inference

Record:

- training wall time;
- memory;
- prediction latency;
- throughput.

### 14.3 Measurement principles

Separate:

$$
\text{one-time representation extraction}
$$

from:

$$
\text{downstream prediction cost}
$$

Exclude common preprocessing cost from incremental comparisons where possible.

Do not treat original D-BETA pretraining cost as part of deployment inference cost unless explicitly performing a broader lifecycle analysis.

Use repeated benchmark runs after warm-up.

---

## 15. Selective Deployment / Cascade Analysis

### 15.1 Motivation

A practical strategy might be:

$$
A \rightarrow \text{screen}
\rightarrow B \rightarrow \text{allocate resources}
$$

However, preliminary evidence suggests D-BETA may identify high-risk patients whom CIIS rates as low risk. Therefore, CIIS may not be a safe gatekeeper for D-BETA.

### 15.2 Compute budget

Let:

$$
q \in \{5\%,10\%,20\%,40\%,100\%\}
$$

represent the proportion of patients receiving D-BETA evaluation.

### 15.3 Compare policies

1. **A-only**
2. **A→B selective cascade**
3. **B-for-all**

### 15.4 Outcomes

Evaluate:

- event capture at fixed care-management capacity;
- PPV;
- net benefit;
- fraction of full-B attainable utility recovered;
- compute used;
- clinically important events missed because A did not pass the patient to B.

A useful summary curve is:

$$
x = \text{fraction receiving D-BETA}
$$

versus

$$
y = \text{fraction of full-B utility/performance recovered}
$$

### 15.5 Key possible finding

A negative result is scientifically meaningful:

> Selective D-BETA invocation after conventional ECG screening may be computationally efficient but clinically lossy because the two representations are sufficiently discordant that the conventional screen cannot safely gate the richer representation.

---

## 16. Coefficients and Clinical-Context Dependence

Raw coefficient comparisons across penalized high-dimensional models should not be a primary analysis.

Reasons include:

- different regularization strengths;
- feature correlation;
- different representation dimensionality;
- shrinkage-induced changes in coefficients.

Instead, evaluate **block dependence**.

Examples:

$$
P(C+B)-P(B)
$$

versus

$$
P(C+A)-P(A)
$$

or remove specific clinical blocks, such as Elixhauser indicators, and measure held-out degradation.

This asks:

> How much does each ECG representation still depend on explicit clinical history?

Shared coefficients may be reported descriptively but should not receive causal interpretation.

---

## 17. Statistical Analysis Principles

- Patient-level disjoint train/validation/test splits.
- No final-test outcome inspection during model selection.
- Same index ECG and outcome for all representations.
- Paired comparisons on the same test patients.
- Bootstrap uncertainty at the patient level.
- Validation log-loss for regularization selection.
- Emphasize magnitude and clinical relevance over dichotomous $P<0.05$ decisions.
- Do not use a simple likelihood-ratio-test framework as the main evidence for incremental value in high-dimensional penalized models.
- Prespecify subgroup definitions and capacity thresholds before final-test evaluation where feasible.

---

## 18. Leakage Controls

### ECG

- Same raw index ECG for A and B.
- No clinical labels incorporated into ECG preprocessing.
- D-BETA encoder frozen.

### Clinical diagnoses

- Do not use index-admission discharge diagnoses in the primary predictor context.
- Use only eligible historical encounters completed before the ECG.

### Clinical notes

If used:

- note time must precede index ECG;
- exclude index-ECG interpretation text;
- exclude discharge summaries or later-event documentation;
- construct all note-selection rules before final evaluation.

### Data splitting

- subject-level separation;
- all preprocessing transformations learned on development data only;
- validation used only for model/hyperparameter selection;
- final test locked until analysis specification is frozen.

---

## 19. External Validation

### 19.1 Why it is important

D-BETA was pretrained on MIMIC-IV-ECG. Consequently, the MIMIC analysis is an in-domain downstream probe even though mortality labels were not used in D-BETA pretraining.

For stronger claims of transportability, validate the core findings in a dataset that was not used in D-BETA pretraining.

### 19.2 Minimum external-validation target

The strongest version of the study should reproduce, where data permit:

1. D-BETA vs. CIIS-feature representation gain;
2. persistence or attenuation after shared clinical context;
3. asymmetric information containment;
4. major subgroup heterogeneity patterns.

The external dataset need not reproduce every secondary computational or note-enrichment analysis.

### 19.3 Interpretation

If external validation is unavailable, claims should be restricted to:

> comparative downstream representation value within the MIMIC-IV environment.

Do not describe MIMIC results as external validation of D-BETA.

---

## 20. Primary Figures and Tables

### Figure 1 — Study design

Flow diagram:

$$
ECG \rightarrow
\begin{cases}
CIIS\text{-derived features}\\
D\text{-BETA embedding}
\end{cases}
+
C
\rightarrow
\text{same ridge-logistic probe}
\rightarrow
30d\ mortality
$$

### Figure 2 — Performance and clinical-context attenuation

Show:

- ECG-only A vs. B;
- context-only C;
- C+A;
- C+B;
- C+A+B.

Include AUROC, AUPRC, and log-loss/Brier.

### Figure 3 — Information overlap

- total prediction alignment;
- ECG-specific contribution alignment;
- reciprocal residual-risk gradients;
- discordant group event rates.

### Figure 4 — Population heterogeneity

Forest-style display of:

$$
\Delta_s = P(B|s)-P(A|s)
$$

for prespecified subgroups.

### Figure 5 — Capacity and decision utility

- event capture at fixed capacity;
- optional DCA.

### Figure 6 — Compute/utility frontier

D-BETA utilization fraction vs. recovered clinical/predictive utility.

### Table 1 — Cohort characteristics

### Table 2 — Overall predictive performance

### Table 3 — Bidirectional incremental value and ablations

### Table 4 — Subgroup representation gain

### Table 5 — Computational cost

---

## 21. Interpretation Framework

Possible result patterns should be anticipated before analysis.

### Pattern A — B dominates A even after clinical context

Interpretation:

D-BETA contains substantial prognostic information not captured by conventional ECG features or ordinary recorded disease burden.

### Pattern B — B advantage strongly attenuates after clinical context

Interpretation:

Much of D-BETA's ECG-derived prognostic signal may correspond to latent manifestations of known clinical disease burden.

### Pattern C — B adds strongly to A; A adds little to B

Interpretation:

D-BETA approximately subsumes conventional ECG prognostic information while adding additional signal.

### Pattern D — Both add reciprocally

Interpretation:

Hand-engineered and learned representations capture complementary information.

### Pattern E — Incremental gain differs substantially by subgroup

Interpretation:

The value of richer ECG representation is population-dependent; mechanisms should be investigated rather than automatically labeled bias.

### Pattern F — A→B cascade loses important events

Interpretation:

A cheaper conventional ECG representation cannot safely gate access to D-BETA because the representations disagree in clinically meaningful ways.

---

## 22. Novelty Positioning

The study should **not** claim novelty from:

- showing that a foundation model beats conventional ECG features;
- subgroup-specific AUROCs alone;
- DCA alone;
- a generic staged-screening architecture;
- adding many ECG foundation models to a benchmark.

The stronger novelty claim is the integrated framework:

> **Under controlled downstream modeling and shared clinical context, quantify what additional prognostic information a modern ECG representation contributes beyond conventional ECG features, whether that incremental value varies across patient populations, how much the representations overlap asymmetrically, and whether representation discordance permits computationally efficient selective deployment.**

Potential novelty pillars:

1. **Controlled representation comparison** with the downstream learner and clinical context held constant.
2. **Clinical-context attenuation** of representation gain.
3. **Bidirectional/asymmetric information containment.**
4. **Heterogeneity of incremental representation value** rather than absolute subgroup performance alone.
5. **Linkage of representation discordance to selective-deployment safety and compute trade-offs.**
6. **External replication in a pretraining-disjoint cohort**, if achieved.

---

## 23. Study Priorities

### Essential

1. Rebuild fair A and B representations.
2. Freeze cohort and splits.
3. Construct leakage-safe structured clinical context.
4. Fit $C$, $C+A$, $C+B$, $C+A+B$.
5. Evaluate overall performance, calibration, and incremental value.
6. Perform reciprocal alignment/residual-risk analyses.
7. Estimate subgroup heterogeneity of B-over-A gain.

### Strongly recommended

8. PCA-15 D-BETA sensitivity analysis.
9. Random-encoder negative control if technically feasible.
10. Capacity-constrained evaluation.
11. Computational benchmarking.
12. Selective A→B deployment experiment.
13. Pretraining-disjoint external validation.

### Secondary / optional

14. Clinical-note enriched context.
15. Broader subgroup intersections.
16. Additional outcomes.
17. Additional ECG foundation models.

---

## 24. Planned Manuscript Message

A successful study should be able to make a statement substantially richer than "D-BETA had higher AUROC."

A target conclusion is:

> **Modern ECG representations can provide prognostic information beyond conventional ECG features and routine clinical context, but the value of that additional representation may vary across patient populations and may not be safely recoverable through conventional-score-gated deployment.**

If asymmetric information containment is strong:

> **D-BETA captures much of the prognostic information available from conventional ECG features while also identifying clinically important residual risk that the conventional representation misses.**

If confirmed in a pretraining-disjoint cohort, these findings would support a broader conclusion about the transportable value of richer ECG representations rather than an in-domain MIMIC-specific result.

---

## 25. Selected Literature Context

The study should be positioned relative to four rapidly developing areas:

1. **ECG foundation-model benchmarking:** recent work shows that foundation-model superiority is task-dependent and that representation evaluation requires stronger downstream tasks and baselines.
2. **AI-ECG subgroup evaluation:** sex, age, and race/ethnicity performance comparisons are already well represented in the literature.
3. **Incremental clinical value of AI-ECG:** several studies combine AI-ECG with conventional clinical risk models, but systematic heterogeneity of the *incremental representation gain* is much less developed.
4. **Cost-aware/staged AI-ECG deployment:** staged screening has precedents, so the distinctive question here is whether a conventional representation can safely gate a richer representation given patient-level discordance.

Important methodological context:

- D-BETA was pretrained using MIMIC-IV-ECG, so MIMIC-only evaluation is not clean external validation.
- Decision-curve analysis should emphasize expected clinical utility rather than threshold-by-threshold significance testing.
- MIMIC-IV discharge/billing diagnoses require careful temporal handling to avoid leakage when constructing comorbidity predictors.

---

## 26. Scope Control

The project should resist premature expansion to many additional ECG models.

The immediate goal is to answer the CIIS-versus-D-BETA comparison correctly and deeply.

Additional representations should be introduced only after the analysis framework is stable and the primary scientific questions have been resolved. At that point, the same framework can naturally support a broader multi-model study without sacrificing interpretability.
