# ECG Risk Alignment

Probe how traditional ECG risk models align with modern multimodal transformer-based ECG representations using MIMIC-IV-ECG.

## Research Question

Given the **same ECG**, how much risk information is shared between:

- a traditional, explicitly engineered ECG score (`A`); and
- a modern multimodal transformer-derived ECG score (`B`)?

The main goal is not simply to ask whether `B` has a larger AUROC. We are especially interested in whether `B` identifies outcome-relevant heterogeneity **within conventional risk categories defined by `A`**.

```text
MIMIC-IV-ECG waveform
        |
        +--------------------+
        |                    |
        v                    v
 traditional ECG A     transformer ECG B
        |                    |
        +---------+----------+
                  |
                  v
        alignment / discordance
                  |
                  v
     outcomes from MIMIC-IV only
```

## Predictor-Input Rule

Patient-specific predictor inputs are restricted to ECG data.

Allowed:

- raw 12-lead waveform;
- deterministic ECG measurements;
- ECG image rendered from the waveform;
- fixed prompts or class descriptions shared by all patients.

Not allowed as predictor features:

- demographics;
- diagnoses;
- medications;
- laboratory values;
- non-ECG vital signs;
- notes;
- encounter history;
- ICU variables.

MIMIC-IV clinical data may be used for:

- cohort definition;
- linkage;
- outcome ascertainment;
- follow-up;
- descriptive summaries;
- secondary evaluation strata.

## Current Model Plan

### Traditional model `A`

Leading candidate:

**Cardiac Infarction/Injury Score (CIIS)**

Why it is attractive:

- ECG-only;
- continuous;
- classical and interpretable;
- published risk categories;
- appropriate for studying whether a modern representation adds information beyond conventional ECG measurements.

### Transformer model `B`

Leading strict-criteria candidate:

**D-BETA** — ICML 2025

Planned use:

```text
12-lead ECG
  -> frozen D-BETA ECG encoder
  -> 768-dimensional embedding
  -> simple regularized linear outcome probe
  -> continuous B score
```

Hugging Face access is currently pending.

Important interpretation limitation: D-BETA was pretrained using MIMIC-IV-ECG, so MIMIC evaluation is an **in-domain representation probe**, not clean external validation.

### Available engineering model

**CarDSLab/ecg-clip-beit-base-384**

Hugging Face access has been granted.

This model is useful immediately for developing:

- ECG rendering;
- model adapters;
- embedding extraction;
- batched inference;
- continuous similarity-score infrastructure.

The associated TARGET-AI manuscript is currently a preprint, so this checkpoint is not the primary scientific `B` under the project's strict peer-review criterion.

### Secondary candidate

**PULSE-7B**

PULSE is a peer-reviewed multimodal ECG-image LLM published in *npj Digital Medicine* in 2026. It is public and downloadable.

PULSE also used MIMIC-IV-ECG-derived training data, so the same in-domain caveat applies.

## Data

Expected local datasets:

```text
~/data/mimic-iv-ecg/1.0/
~/data/mimiciv/3.1/
```

Example MIMIC-IV-ECG layout:

```text
1.0/
├── files/
├── machine_measurements.csv
├── machine_measurements_data_dictionary.csv
├── record_list.csv
├── waveform_note_links.csv
└── RECORDS
```

Raw MIMIC data must stay outside this repository.

## Proposed Initial Outcome

Primary pilot endpoint:

**30-day all-cause mortality after the index ECG**

Secondary endpoints may include:

- in-hospital mortality;
- 90-day mortality;
- 1-year mortality where follow-up is supportable.

Outcome data come from MIMIC-IV and are never used as predictor inputs.

## Analysis Sketch

For each patient:

```text
A = traditional continuous ECG score
B = transformer-derived continuous ECG score
Y = clinical outcome
```

Primary analyses:

1. global correlation and score alignment;
2. global AUROC/AUPRC;
3. distribution of `B` within published `A` risk categories;
4. outcome gradients across `B` within fixed `A` categories;
5. `A-low/B-high` versus `A-low/B-low` discordance;
6. incremental outcome information from `B` after conditioning on `A`.

See [`docs/research-proposal.md`](docs/research-proposal.md) for the full design.

## Repository Layout

Planned structure:

```text
.
├── README.md
├── pyproject.toml
├── docs/
│   ├── research-proposal.md
│   └── roadmap.md
├── src/
│   └── ecg_alignment/
│       ├── analysis.py
│       ├── cohort.py
│       ├── data.py
│       ├── outcomes.py
│       ├── probe.py
│       ├── split.py
│       └── scoring/
│           ├── base.py
│           ├── cards_clip.py
│           ├── dbeta.py
│           ├── preprocess.py
│           ├── pulse.py
│           └── traditional.py
├── tests/
├── scripts/
└── reports/
```

## Command-Line Interface & Execution

This project provides a unified entrypoint `ecg-alignment` registered via `pyproject.toml` (and runnable directly via `uv run ecg-alignment` or `uv run python -m ecg_alignment.cli`).

### Environment Variables & Path Resolution

Paths can be configured via flags or environment variables:
- `MIMIC_ROOT`: Path to MIMIC-IV root directory (default: `~/data/mimiciv/3.1`)
- `MIMIC_ECG_ROOT` or `ECG_ROOT`: Path to MIMIC-IV-ECG root directory (default: `~/data/mimic-iv-ecg/1.0`)

### Available Subcommands

```bash
# 1. Stage 1 Data Inventory & Linkage Statistics
uv run ecg-alignment inventory --report-out ./reports/inventory.md

# 2. Stage 7 Primary Patient Cohort & Patient-Disjoint Split
uv run ecg-alignment cohort --seed 42 --report-out ./reports/cohort-flow.md

# 3. Stage 8 Frozen-Embedding Linear Probe
uv run ecg-alignment probe --seed 42 --report-out ./reports/continuous-predictions.md

# 4. Stage 9 Primary Statistical Analysis & Figure Generation
uv run ecg-alignment analyze --output-dir ./reports --generate-figures

# 5. Stage 10 Comprehensive Sensitivity & Robustness Battery
uv run ecg-alignment sensitivity --report-out ./reports/sensitivity-analyses.md

# 6. Stage 11 Research Interpretation & Translation Roadmap
uv run ecg-alignment interpret --report-out ./reports/research-interpretation.md

# 7. End-to-End Pipeline Execution (Stages 1-11)
uv run ecg-alignment pipeline --output-dir ./reports
```

## Python Environment

This project uses [`uv`](https://docs.astral.sh/uv/) for Python and dependency management.

Initialize or synchronize the environment:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Run static type checking:

```bash
uv run basedpyright
```

Do not create ad hoc Conda environments for project code. When upstream model repositories document Conda-based setup, translate the required Python dependencies into this project's `uv` environment or isolate the upstream reproduction separately.

## Hugging Face Models

Model files are not committed to this repository.

Authenticate with Hugging Face using your normal local Hugging Face credential mechanism. Never place access tokens in source files, notebooks, committed `.env` files, or GitHub issues.

Current access status:

```text
CarDSLab/ecg-clip-beit-base-384    granted
Manhph2211/D-BETA                  pending
```

Always record the exact model repository revision used for an experiment.

## Development Standards

### Dependency management

Use `uv` for:

- Python versions;
- project dependencies;
- development dependencies;
- command execution.

Prefer:

```bash
uv add <package>
uv add --dev <package>
uv run <command>
```

over direct `pip install` commands.

### Indentation

Use **2 spaces** for indentation/tab size throughout the repository, including Python.

Recommended `.editorconfig`:

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
indent_style = space
indent_size = 2
```

Do not run formatters that silently rewrite Python indentation to a conflicting style.

### Functional-style Python

Prefer functional composition over stateful object hierarchies.

Good defaults:

- pure functions for transformations;
- explicit function inputs and outputs;
- immutable configuration where practical;
- no hidden mutable global state;
- side effects isolated to I/O/model-loading boundaries;
- deterministic transformations;
- small model adapters instead of deep inheritance;
- functions ordered topologically where practical, with dependencies before callers.

Example:

```python
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def select_leads(
  signal: npt.NDArray[np.float32],
  indices: Sequence[int],
) -> npt.NDArray[np.float32]:
  return signal[np.asarray(indices)]


def normalize_signal(
  signal: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
  scale = np.maximum(np.std(signal, axis=1, keepdims=True), 1e-8)
  return signal / scale


def prepare_ecg(
  signal: npt.NDArray[np.float32],
  indices: Sequence[int],
) -> npt.NDArray[np.float32]:
  selected = select_leads(signal, indices)
  return normalize_signal(selected)
```

### Testing

Reusable code requires tests.

Use:

```bash
uv run pytest
```

Prioritize tests for:

- cohort boundary conditions;
- time-window logic;
- patient-disjoint splitting;
- ECG shape and lead order;
- scoring thresholds;
- deterministic preprocessing;
- model-output dimensions;
- technical failure handling.

Tests should use synthetic or openly shareable fixtures. Do not commit row-level MIMIC data as test fixtures.

### Static typing

Use:

```bash
uv run basedpyright
```

Type public functions and data contracts.

Avoid using `Any` as an easy escape from unclear interfaces. Narrow external-library types at adapter boundaries when necessary.

## Pull Request Workflow

Every substantive task should be implemented on a branch and opened as a pull request.

PR descriptions should answer:

```text
## What problem does this PR solve?

## What approach did you take?

## What assumptions did you make?

## How did you test it?

## Example output
```

A PR should be reviewable without requiring the reviewer to infer paths, assumptions, or expected output.

Before opening a PR:

```bash
uv run pytest
uv run basedpyright
```

## Data and Security

Never commit:

- MIMIC-IV data;
- MIMIC-IV-ECG data;
- derived row-level patient datasets;
- Hugging Face tokens;
- local credentials;
- model access secrets.

Reusable code should receive dataset roots from explicit configuration or CLI arguments rather than hard-coded user-specific paths.

For example:

```bash
uv run python -m ecg_alignment.cli   --ecg-root ~/data/mimic-iv-ecg/1.0   --mimic-root ~/data/mimiciv/3.1
```

## Reproducibility Policy

Every analysis should record:

- Git commit;
- Python version;
- dependency lock state;
- model repository and exact revision;
- model license/access status;
- cohort definition;
- random seed;
- outcome definition;
- traditional-score implementation version;
- analysis configuration.

The final test set must remain untouched until the model probe and analysis specification are frozen.

## Interpretation Boundary

Several candidate transformer models were pretrained using MIMIC-IV-ECG.

Accordingly, this repository initially supports an **in-domain probing experiment**:

> Does a multimodally pretrained transformer representation encode MIMIC ECG risk information that is aligned with or complementary to a traditional ECG score?

It should not be described as independent external validation unless the selected model is confirmed not to have used the evaluation ECGs during pretraining.

## Project Documents

- [Research proposal](docs/research-proposal.md)
- [Staged roadmap](docs/roadmap.md)
- [Model licenses & data governance](docs/model-licenses.md)

## Key References

- Pham Hung M, Saeed A, Ma D. *Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners.* ICML 2025. https://proceedings.mlr.press/v267/pham-hung25a.html
- D-BETA: https://github.com/manhph2211/D-BETA
- CarDSLab ECG-CLIP BEiT: https://huggingface.co/CarDSLab/ecg-clip-beit-base-384
- TARGET-AI: https://doi.org/10.1101/2025.08.25.25334266
- Liu R, Bai Y, Yue X, et al. *Teaching multimodal LLMs to comprehend 12-lead electrocardiographic images.* npj Digital Medicine. 2026. https://doi.org/10.1038/s41746-026-02551-3
- MIMIC-IV-ECG: https://mimic.mit.edu/docs/iv/modules/ecg/
- MIMIC-IV: https://mimic.mit.edu/docs/iv/
