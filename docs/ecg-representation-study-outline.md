---
title: "Enhanced Outline: CIIS vs. D-BETA Study"
author: "Sae-Hwan Park"
date: 2026-09-02
description: "Ultimate Q: What additional prognostic information does a modern ECG representation recover beyond conventional ECG features, does that gain persist after ordinary clinical information is known, and is the gain shared consistently across patient populations?"
---

## Keep the comparison focused on CIIS and D-BETA

- Recap: The preliminary results already suggest that D-BETA contains substantially more prognostic information than the original CIIS score, with only moderate alignment and strong residual mortality gradients within CIIS strata.
- The next step should be to make the comparison as fair and interpretable as possible.

---

## Fair representation comparison

- Instead of comparing the original CIIS score directly with a mortality-trained D-BETA probe, use the same downstream model for both.

### Conventional representation A

Use the 15 ECG features underlying CIIS:

$$
Z_A = \text{CIIS-derived ECG features}
$$

### Modern representation B

Use the frozen 768-dimensional D-BETA ECG embedding:

$$
Z_B = \text{D-BETA embedding}
$$

For both, fit the same $L_2$-regularized logistic regression for 30-day mortality.

This gives a cleaner interpretation:

> Same patients, same outcome, same clinical context, same downstream learner; only the ECG representation differs.

The original canonical CIIS score can remain as a clinical reference.

---

## Add shared clinical context

Primary context:

$$
C =
\text{Age}
+
\text{Sex}
+
\text{prior-recorded Elixhauser comorbidities}
$$

Then fit four core models:

$$
M_C: Y\sim C
$$

$$
M_A: Y\sim C+Z_A
$$

$$
M_B: Y\sim C+Z_B
$$

$$
M_{AB}: Y\sim C+Z_A+Z_B
$$

This lets us ask:

1. How much do conventional ECG features add beyond clinical history?
2. How much does D-BETA add beyond clinical history?
3. Does D-BETA still add information after CIIS features?
4. Do CIIS features still add information after D-BETA?

The last two give us **bidirectional information complementarity**.

---

## A potentially important result: information containment

Suppose:

$$
M_A\rightarrow M_{AB}
$$

produces a substantial gain, but:

$$
M_B\rightarrow M_{AB}
$$

produces little gain.

That would suggest:

> D-BETA captures much of the prognostic information represented by conventional ECG features, while also capturing additional information that CIIS-derived features miss.

The preliminary residual-risk results already suggest this may happen, but the new design can test it much more rigorously.

---

## Does clinical context explain D-BETA's advantage?

We can compare the representation gap before and after adding clinical context.

For performance measure $P$,

ECG only:

$$
\Delta_{ECG}
=
P(B)-P(A)
$$

After clinical context:

$$
\Delta_C
=
P(C+B)-P(C+A)
$$

- If the gap becomes much smaller, D-BETA may largely be recovering latent information about comorbidity or systemic disease burden from the ECG.
- If the gap remains large, that suggests D-BETA captures prognostic physiology not summarized by routine clinical history.

Either result would be interesting.

---

## Subgroup analysis => Heterogeneity analysis

Instead of only asking whether D-BETA performs differently in women and men, define:

$$
\Delta_s
=
P(M_B\mid s)-P(M_A\mid s)
$$

and compare $\Delta_s$ across:

- sex;
- age;
- race/ethnicity;
- possibly comorbidity burden.

The question becomes:

> **Does the incremental value of the modern ECG representation depend on the population?**

- Different from a standard fairness analysis of one model and appears less systematically studied
- Let's describe this primarily as **heterogeneity of representation gain**, with fairness implications where appropriate.

---

## Evaluation

### Predictive performance

- AUROC
- AUPRC
- log-loss
- Brier score
- calibration

### Fixed care-management capacity

For the top 1%, 2.5%, 5%, 10%, and 20% of predicted-risk patients:

- deaths captured;
- sensitivity;
- PPV;
- false positives per event captured.

This asks:

> With the same care-management capacity, which representation identifies more events?

### Decision curve analysis

- Use DCA as a clinical-utility analysis over plausible risk thresholds, 
- But should avoid treating threshold-by-threshold significance testing as the main inference.

---

## Alignment and residual risk

Because Models A and B share clinical context, raw prediction correlation is not enough.

We should compare both:

1. overall predicted-risk alignment;
2. alignment of the ECG-specific linear predictor contributions.

We should also make the residual-risk analysis reciprocal:

- D-BETA risk gradients within fixed A-risk strata;
- A-risk gradients within fixed B-risk strata.

This will show whether one representation effectively subsumes the other.

---

## Computational and deployment analysis

Measure how much additional compute D-BETA requires:

- wall time;
- peak RAM/VRAM;
- throughput;
- latency;
- representation storage.

Then test a selective policy:

$$
A\rightarrow\text{screen}\rightarrow B
$$

for different D-BETA compute budgets.

The important question is not simply whether staged screening saves compute, because such approaches already exist.

The more interesting question is:

> **Can CIIS safely gate access to D-BETA?**

Our preliminary discordance results suggest that some high-risk patients may have low CIIS but high D-BETA risk. If so, using CIIS as the gatekeeper could lose exactly the patients for whom D-BETA is most useful.

---

## Clinical notes: secondary analysis

We can later add pre-index clinical-note embeddings to both A and B:

$$
C_1 = C + X_1
$$

and ask whether D-BETA still adds value when rich narrative information is already available.

Better to remain secondary because of:

- temporal leakage risk;
- unequal note availability;
- additional computational complexity;
- potential contamination from ECG interpretation or later hospital-course documentation.

---

## Unseen Data Validation

D-BETA was pretrained on MIMIC-IV-ECG.

Therefore:

> The MIMIC analysis is an **in-domain downstream representation study**, not clean external validation of D-BETA.

- Good thing: 30-day mortality was not the D-BETA pretraining target. But it limits generalizability claims.
- For the strongest paper, we'd add at least one **pretraining-disjoint external validation cohort** capable of supporting a comparable ECG prognosis outcome.
- Collborators may help.

---

## Small robustness controls

Two inexpensive controls could strengthen the representation claim:

### PCA-matched D-BETA

- Reduce the D-BETA embedding to 15 development-learned principal components before fitting the same ridge probe.
- Purpose: check whether the advantage is merely due to 768 vs. 15 dimensions.

### Random-encoder negative control

- If feasible, compare with frozen random features from a similar encoder architecture.
- Purpose: address recent concerns that random high-dimensional feature maps can sometimes look surprisingly strong under linear probing.

---

## Main contribution we should aim for

Possible framing:

- Under controlled downstream modeling and shared clinical context,
- how much prognostic information does a modern ECG representation add beyond conventional ECG features, 
- who benefits from that additional representation,
- and can the gain be obtained efficiently without missing patients identified only by the richer representation?

That gives us a coherent study around:

1. representation gain;
2. information overlap;
3. clinical-context attenuation;
4. subgroup heterogeneity;
5. computational/deployment trade-offs.

If the central findings replicate in a pretraining-disjoint cohort, this can become a substantially stronger contribution.
