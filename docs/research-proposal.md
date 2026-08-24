---
title: "Traditional ECG Risk vs Multimodal Transformer ECG Representations"
description: "Research proposal for probing alignment, discordance, and incremental prognostic information between traditional ECG-only risk models and modern multimodal transformer-based ECG models in MIMIC-IV."
date: 2026-08-24
status: "Working proposal"
---

# Traditional ECG Risk vs Multimodal Transformer ECG Representations

## Executive Summary

This project will examine whether modern multimodal transformer-based ECG models encode the same risk information as traditional ECG models, and whether they identify clinically meaningful risk heterogeneity that traditional ECG stratification does not capture.

The experiment deliberately restricts **patient-specific predictor inputs to ECG data only**. Traditional model `A` and modern model `B` must both derive their predictions from the index ECG. MIMIC-IV clinical data may be used to define cohorts, ascertain outcomes, determine follow-up, and perform descriptive or stratified evaluation, but must not contribute predictor features.

The central comparison is therefore not simply "traditional model versus AI." It is:

$$
\text{hand-engineered ECG representation}
\quad\text{vs.}\quad
\text{multimodally learned transformer ECG representation}.
$$

The primary scientific target is the structure of **alignment and discordance** between these two representations. In particular, we want to determine whether model `B` identifies outcome-relevant heterogeneity among patients assigned to the same conventional risk category by model `A`.

This is intended as a preliminary opportunity-probing experiment. It prioritizes interpretability, reproducibility, and clean feature control over developing a new state-of-the-art prediction model.

## Scientific Motivation

Traditional ECG interpretation compresses the waveform into explicit measurements and rules: intervals, amplitudes, axes, morphologies, and combinations of these quantities. Modern ECG foundation models can instead learn high-dimensional representations directly from raw waveforms or rendered ECG images.

Multimodal transformer models are particularly interesting because their representations are learned by aligning ECGs with another modality such as clinical reports, structured descriptions, or other cardiac information. At inference time, however, many such models can operate from the ECG alone.

This creates a useful controlled comparison:

- both models observe the same physiological measurement;
- neither model receives demographics, diagnoses, medications, laboratory values, or other patient-specific clinical predictors;
- the main difference lies in how ECG information is represented and compressed.

If modern models largely reproduce traditional ECG knowledge, their outputs should align strongly with established ECG risk measures. If they encode additional information, clinically meaningful outcome gradients may remain within traditional risk strata.

## Primary Research Questions

1. **Alignment:** How strongly is the continuous score from a traditional ECG model associated with a modern multimodal transformer-derived ECG score?
2. **Residual risk:** Within conventional risk categories defined by model `A`, does model `B` identify additional outcome-relevant risk heterogeneity?
3. **Discordance:** Are patients with low traditional risk but high transformer-derived risk meaningfully different from patients with low risk according to both models?
4. **Incremental information:** Does the transformer-derived score retain prognostic information after flexibly accounting for the traditional ECG score?
5. **Robustness:** Are these findings stable across reasonable cohort definitions, index-ECG rules, prediction horizons, and model implementations?

## Core Hypotheses

### H1: Partial alignment

The two scores will be positively associated because both operate on the same ECG and should recover some shared electrophysiologic information.

$$
\operatorname{cor}(A, B) > 0.
$$

We do not expect near-perfect agreement.

### H2: Residual heterogeneity within traditional risk strata

Within a fixed category of `A`, model `B` will show an outcome gradient:

$$
P(Y=1 \mid C_A=k, B_{\text{high}})
>
P(Y=1 \mid C_A=k, B_{\text{low}}).
$$

### H3: Clinically informative discordance

The `A-low / B-high` group will have a higher outcome rate than the `A-low / B-low` group.

### H4: Incremental information

A flexible outcome model containing both scores will fit better than a model using `A` alone:

$$
g\{E(Y)\}
=
\beta_0 + f(A) + \beta_B B.
$$

The main inferential quantity is whether `B` contributes information after conditioning on `A`, not whether one marginal AUROC is numerically larger.

## Operational Definitions

### Traditional model `A`

`A` must:

- use ECG-derived information only;
- be established in the clinical or ECG literature;
- produce a continuous or ordinal score;
- preferably have published thresholds that define clinically meaningful categories;
- be reproducible from the raw ECG or validated ECG measurements.

The leading candidate is the **Cardiac Infarction/Injury Score (CIIS)** because it is a classical multivariable ECG score, produces a continuous value, and has established score ranges that can be used without data-driven cut-point selection.

Candidate alternatives include traditional voltage or conduction criteria when a more diagnosis-focused experiment is desired.

### Modern model `B`

For this project, "AI" is deliberately narrow. `B` should be:

- transformer-centric;
- pretrained using multiple modalities or explicit cross-modal alignment;
- peer-reviewed;
- published in a credible scientific venue;
- runnable from ECG alone at inference time;
- capable of producing a continuous score directly or a frozen ECG embedding from which a prespecified simple prediction head can be fit.

A fixed non-patient-specific text prompt or class description is permissible for zero-shot inference. Patient-specific non-ECG information is not.

### Predictor-information firewall

Allowed for `A` and `B`:

- raw 12-lead ECG waveform;
- deterministic transformations of the same ECG;
- rendered ECG image derived from the waveform;
- fixed prompts, class descriptions, or model constants that are identical across patients.

Not allowed as predictor features:

- age;
- sex;
- race or ethnicity;
- diagnoses;
- medications;
- laboratory values;
- vital signs outside the ECG signal;
- clinical notes;
- encounter history;
- ICU variables;
- admission characteristics.

These fields may be used only for cohort construction, outcome ascertainment, follow-up, descriptive summaries, or secondary stratified evaluation.

## Data Sources

### MIMIC-IV-ECG v1.0

Expected local location:

```text
~/data/mimic-iv-ecg/1.0/
```

Key resources include:

```text
files/
record_list.csv
machine_measurements.csv
waveform_note_links.csv
RECORDS
```

The waveform is the common patient-specific predictor source for both `A` and `B`.

### MIMIC-IV v3.1

Expected local location:

```text
~/data/mimiciv/3.1/
```

Relevant modules:

```text
hosp/
icu/
```

MIMIC-IV is used downstream of prediction for:

- linkage;
- encounter timing;
- cohort eligibility;
- mortality or other outcome ascertainment;
- follow-up;
- optional descriptive and subgroup analyses.

No MIMIC-IV clinical variable may enter the predictor vector for either `A` or `B`.

## Candidate Modern Models

### D-BETA — leading strict-criteria candidate

**Publication:** ICML 2025.

D-BETA learns aligned ECG-text representations using a multimodal masked/contrastive architecture. Its ECG encoder includes a transformer stack, and the released model can produce a 768-dimensional ECG representation from a 12-lead waveform.

The official example supports:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
  "Manhph2211/D-BETA",
  trust_remote_code=True,
)
```

The project Hugging Face repository is gated under CC BY-NC 4.0. Access has been requested and is currently pending.

For this project, the preferred use is:

$$
X_{ECG}
\xrightarrow{F_B}
z \in \mathbb{R}^{768}
\xrightarrow{\text{simple linear probe}}
B.
$$

The encoder is frozen. The outcome-specific prediction layer should be deliberately simple and prespecified, such as regularized logistic regression.

**Important limitation:** D-BETA pretraining used MIMIC-IV-ECG ECG-report pairs. Evaluation on MIMIC-IV-ECG must therefore be described as an **in-domain representation probe**, not clean external validation.

### CarDSLab ECG-CLIP BEiT — available engineering candidate

Model:

```text
CarDSLab/ecg-clip-beit-base-384
```

Access has already been granted.

The released model is a CLIP-style ECG-image/text system with a BEiT vision encoder and a 512-dimensional projection space. It can produce ECG-image embeddings and zero-shot similarity scores.

This checkpoint is highly useful for:

- validating the end-to-end image-rendering pipeline;
- testing model adapters;
- extracting continuous embeddings;
- testing zero-shot scoring infrastructure.

However, the associated TARGET-AI manuscript is currently indexed as a medRxiv preprint. Under the project's strict requirement that primary `B` be peer-reviewed, this model should be treated as an **engineering/prototyping candidate**, not the definitive primary scientific comparator unless publication status changes or the criterion is relaxed.

### PULSE-7B — secondary MLLM candidate

PULSE is a 7B multimodal large language model for ECG-image interpretation published in *npj Digital Medicine* in 2026. Model weights and code are public.

It is attractive because it closely reflects current concerns about multimodal transformer/LLM-based ECG interpretation.

A continuous score could be derived from a fixed closed-form question using token likelihoods rather than free-text parsing.

However, ECGInstruct includes MIMIC-IV-ECG-derived samples, so MIMIC-IV-ECG evaluation is again in-domain rather than clean external validation.

### CSFM — scientifically relevant but operationally secondary

The Cardiac Sensing Foundation Model was published in *Nature Machine Intelligence* in 2026 and is explicitly multimodal. It also included MIMIC-IV-ECG in pretraining, and access to pretrained weights is more restrictive.

It is not a first-line implementation target for this pilot.

## Proposed Primary Experiment

### Outcome

Initial outcome:

> **30-day all-cause mortality after the index ECG**

Reasons:

- clear temporal interpretation;
- ascertainable from linked MIMIC-IV data when follow-up is available;
- independent from ECG machine interpretation;
- applicable across a broad ECG population;
- suitable for a preliminary prognostic experiment.

Secondary horizons may include:

- in-hospital mortality;
- 90-day mortality;
- 1-year mortality where follow-up support is adequate.

All outcome windows must be defined relative to the index ECG timestamp.

### Model `A`

Primary candidate:

> **Cardiac Infarction/Injury Score (CIIS)**

The score will be implemented from ECG-derived measurements only. Published score ranges will be used for risk stratification whenever the required measurements can be reproduced with acceptable fidelity.

Before full-scale analysis, the implementation must pass a technical validation stage demonstrating that required ECG features can be extracted reproducibly from MIMIC-IV-ECG waveforms.

### Model `B`

Primary candidate after access approval:

> **Frozen D-BETA ECG encoder + prespecified regularized linear probe**

The transformer encoder will remain frozen. The probe will use only the ECG embedding and the development-set outcome.

This makes `B` an operational prediction pipeline while preserving the central representation-learning contrast.

Until D-BETA access is available, the CarDSLab ECG-CLIP checkpoint will be used to build and validate common infrastructure.

## Cohort Design

### Unit of analysis

The primary analysis will use **one index ECG per patient**.

This avoids:

- unequal weighting from repeated ECGs;
- implicit severity weighting caused by frequent ECG acquisition;
- within-patient dependence in the primary analysis;
- optimistic uncertainty estimates.

A secondary repeated-ECG analysis may be considered later using patient-clustered inference.

### Candidate index rule

Initial rule:

1. adult patient;
2. ECG can be linked to MIMIC-IV;
3. required waveform leads are present and pass basic technical quality checks;
4. sufficient outcome follow-up exists;
5. select the earliest eligible ECG per patient.

Alternative clinically anchored rules may be evaluated later.

### Train/development/test partitioning

All partitions must be **patient-disjoint**.

Suggested initial split:

- 60% development/training;
- 20% validation/model selection;
- 20% final test.

The final test set must remain untouched until the analysis pipeline, model specification, risk categories, and metrics are frozen.

Because candidate foundation models may already have seen MIMIC-IV-ECG during self-supervised or multimodal pretraining, this split prevents supervised leakage but does not create true external validation. That distinction must remain explicit.

## Prediction Construction

### Traditional score

For each eligible ECG:

$$
A_i = f_A(X_i).
$$

The implementation should preserve the raw continuous score before categorization.

### Transformer score

For a frozen embedding model:

$$
z_i = F_B(X_i).
$$

Then:

$$
B_i = \alpha + \beta^T z_i.
$$

The probe should be simple, regularized, and trained only on the development data.

Recommended first choice:

- logistic regression;
- L2 regularization;
- no demographics or clinical covariates;
- regularization strength chosen only on the validation set.

### Risk categories

Categories are defined from `A`, not `B`.

Prefer established published cutoffs:

$$
C_{A,i}=g(A_i).
$$

If a candidate `A` lacks established cutoffs, quantile-defined groups may be used only as explicitly labeled statistical strata.

## Analysis Plan

### 1. Descriptive score distributions

Report:

- `A` distribution;
- `B` distribution;
- outcome prevalence;
- missingness or technical failure rates.

### 2. Global score alignment

Primary measures:

- Spearman correlation;
- rank agreement;
- smooth estimate of $E(B \mid A)$;
- scatter or hexbin visualization;
- outcome risk over the two-dimensional `A`-`B` space.

The two-dimensional risk surface is especially informative. If both models encode nearly the same information, outcome risk should vary mainly along a common diagonal. Vertical risk gradients at fixed `A` suggest information unique to `B`.

### 3. Global predictive performance

Evaluate `A` and `B` on the untouched test set using:

- AUROC;
- AUPRC;
- Brier score where score calibration is meaningful;
- calibration plots for models producing interpretable probabilities.

The main analysis should not be reduced to an AUROC contest.

### 4. Transformer performance within traditional risk strata

Within each `A` category:

- sample size;
- outcome rate;
- distribution of `B`;
- outcome rates by `B` quantile;
- AUROC/AUPRC of `B` where estimable.

Do **not** interpret poor within-stratum discrimination of `A` as evidence against `A`; conditioning on categories constructed from `A` restricts its range by design.

### 5. Discordance analysis

Define rank-based or prespecified high/low partitions for both scores.

Key groups:

- `A-low / B-low`;
- `A-low / B-high`;
- `A-high / B-low`;
- `A-high / B-high`.

Primary contrast:

$$
P(Y=1 \mid A_{low},B_{high})
\quad\text{vs.}\quad
P(Y=1 \mid A_{low},B_{low}).
$$

This is one of the most interpretable tests of whether the transformer identifies risk hidden within conventional ECG stratification.

### 6. Incremental information

Fit:

$$
g\{E(Y)\}
=
\beta_0 + f(A) + \beta_B B.
$$

Use a flexible representation of `A` where appropriate.

Compare:

- `A` only;
- `B` only;
- `A + B`.

Possible measures:

- likelihood-ratio improvement;
- AUROC change;
- Brier-score change;
- calibration;
- cross-validated or held-out log loss.

NRI/IDI should not be primary metrics.

### 7. Uncertainty

Use patient-level bootstrap confidence intervals for primary performance differences and risk contrasts.

All resampling must occur at the patient level.

## Sensitivity Analyses

Planned sensitivity analyses may include:

- earliest eligible ECG vs ECG anchored to an admission;
- in-hospital vs 30-day vs 90-day mortality;
- alternative technically valid implementations of `A`;
- alternative simple probes for frozen `B` embeddings;
- exclusion of low-quality ECGs;
- complete-case vs explicitly handled technical failures;
- analysis restricted to ECGs linked closely to a hospital encounter;
- alternative `B` models as secondary probes.

## Leakage and Contamination Policy

Three different forms of leakage must be distinguished.

### Patient-level supervised leakage

No patient may appear in more than one supervised development/validation/test split.

### Clinical-feature leakage

No non-ECG clinical feature may enter `A` or `B`.

### Foundation-model pretraining contamination

Several modern multimodal ECG models, including D-BETA and PULSE, used MIMIC-IV-ECG during pretraining.

This does not make the experiment invalid, but it changes the claim.

Allowed interpretation:

> The experiment probes how a pretrained multimodal transformer represents MIMIC-IV ECGs and whether those representations add outcome-relevant information beyond traditional ECG scoring.

Not allowed:

> The experiment provides independent external validation of the foundation model on unseen MIMIC-IV-ECG data.

A future external dataset or a model pretrained without MIMIC-IV-ECG would be required for that stronger claim.

## Reproducibility and Engineering Principles

The project uses:

- `uv` for Python and dependency management;
- `pytest` for tests;
- `basedpyright` for static type checking;
- 2-space indentation/tab size throughout the repository;
- functional-style Python where practical.

Implementation principles:

- prefer pure functions for transformations and scoring;
- isolate filesystem, model-loading, and GPU side effects at clear boundaries;
- avoid hidden mutable global state;
- pass configuration explicitly;
- keep data transformations deterministic;
- use immutable/frozen configuration objects when useful;
- order functions topologically where practical so dependencies appear before callers;
- test numerical edge cases and data-contract assumptions;
- keep model-specific code behind small adapters with common interfaces.

## Expected Repository Architecture

```text
.
├── README.md
├── pyproject.toml
├── docs/
│   ├── research-proposal.md
│   └── roadmap.md
├── src/
│   └── ecg_alignment/
│       ├── cohort.py
│       ├── data.py
│       ├── outcomes.py
│       ├── scoring/
│       │   ├── traditional.py
│       │   ├── dbeta.py
│       │   ├── cards_clip.py
│       │   └── pulse.py
│       ├── analysis.py
│       └── cli.py
├── tests/
│   ├── test_cohort.py
│   ├── test_outcomes.py
│   └── test_scoring.py
├── scripts/
└── reports/
```

Raw MIMIC data and model credentials must never be committed.

## Success Criteria for the Preliminary Experiment

The pilot will be considered technically successful if:

1. at least one traditional ECG score can be reproduced reliably from raw MIMIC-IV-ECG data;
2. at least one eligible multimodal transformer can be run reproducibly from the same index ECG;
3. patient-disjoint outcome labels can be constructed from MIMIC-IV without predictor leakage;
4. both models produce continuous scores for a sufficiently large common cohort;
5. the full analysis can be regenerated from code;
6. test and static-type-check suites pass.

The pilot will be scientifically informative if at least one of the following occurs:

- strong score alignment suggests the transformer largely recovers traditional ECG risk information;
- weak alignment but similar outcome discrimination suggests different representations of comparable risk;
- substantial `B` outcome gradients within `A` strata suggest additional information;
- structured discordance reveals clinically interpretable subgroups;
- no incremental signal is found, which would itself constrain claims about added information from the modern representation.

## Interpretation Boundaries

This study is an opportunity probe, not a clinical deployment study.

It will not establish:

- clinical utility;
- causal benefit;
- safety of AI-supported decisions;
- external generalizability when the foundation model was pretrained on MIMIC-IV-ECG;
- fairness simply from overall performance comparisons.

It can establish whether a modern multimodal transformer representation appears to encode risk information that is aligned with, redundant with, or complementary to a conventional ECG representation under a controlled ECG-only input design.

## References

1. Richardson RT, et al. *A new ECG scoring system for the detection of myocardial infarction and ischemic injury: the Cardiac Infarction Injury Score (CIIS).* Circulation. 1981.
2. Pham Hung M, Saeed A, Ma D. *Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners.* Proceedings of the 42nd International Conference on Machine Learning. PMLR 267:49277-49291; 2025. https://proceedings.mlr.press/v267/pham-hung25a.html
3. D-BETA code repository. https://github.com/manhph2211/D-BETA
4. D-BETA model repository. https://huggingface.co/Manhph2211/D-BETA
5. Oikonomou EK, et al. *TARGET-AI: a foundational approach for the targeted deployment of artificial intelligence electrocardiography in the electronic health record.* medRxiv. 2025. https://doi.org/10.1101/2025.08.25.25334266
6. CarDSLab ECG-CLIP BEiT model repository. https://huggingface.co/CarDSLab/ecg-clip-beit-base-384
7. Liu R, Bai Y, Yue X, et al. *Teaching multimodal LLMs to comprehend 12-lead electrocardiographic images.* npj Digital Medicine. 2026;9:349. https://doi.org/10.1038/s41746-026-02551-3
8. Tang J, Pham Hung M, De Lathauwer I, et al. *Interpretable multimodal zero shot ECG diagnosis via structured clinical knowledge alignment.* npj Cardiovascular Health. 2026;3:1. https://doi.org/10.1038/s44325-025-00099-x
9. MIMIC-IV-ECG documentation. https://mimic.mit.edu/docs/iv/modules/ecg/
10. MIMIC-IV documentation. https://mimic.mit.edu/docs/iv/
