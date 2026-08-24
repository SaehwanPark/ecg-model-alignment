# Development & Contributor Guide

This document outlines coding standards, environment conventions, testing policies, and pull request workflow for the ECG Model Alignment project.

---

## 1. Environment & Dependency Management

We use [`uv`](https://docs.astral.sh/uv/) for all Python version management, package dependencies, and command execution.

### Environment Setup

Initialize or update the local development environment:

```bash
uv sync
```

### Dependency Commands

Prefer:

```bash
uv add <package>
uv add --dev <package>
uv run <command>
```

> **Note**: Do not create ad hoc Conda environments for project code. When upstream model repositories document Conda-based setup, translate the required Python dependencies into this project's `uv` environment or isolate the upstream reproduction separately.

---

## 2. Python Conventions & Code Architecture

### Indentation

- Use **2 spaces** for indentation across all files, including Python source, tests, and configuration.
- Recommended `.editorconfig` setting:
  ```ini
  root = true

  [*]
  charset = utf-8
  end_of_line = lf
  insert_final_newline = true
  indent_style = space
  indent_size = 2
  ```

### Functional-Style Design

- Prefer functional composition and pure transformations over stateful, mutable object hierarchies.
- Maintain pure functions with explicit inputs and outputs.
- Isolate side effects (file I/O, GPU compute, model weights loading) to adapter boundaries.
- Keep configuration objects immutable (e.g., `@dataclass(frozen=True)`).
- Avoid hidden mutable global state.
- Keep functions topologically ordered where practical (dependencies defined before callers).

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

---

## 3. Quality Assurance: Testing & Type Checking

### Running Tests

Run the full pytest suite:

```bash
uv run pytest
```

Prioritize tests for:
- Cohort boundary conditions and time-window logic;
- Patient-disjoint split isolation (no `subject_id` overlap);
- ECG shape, lead ordering, and physical units;
- Scoring thresholds and deterministic preprocessing;
- Predictor-information firewall verification;
- Technical failure handling and edge cases.

> **Important**: Tests should use synthetic fixtures or openly shareable mock data. Never commit row-level MIMIC records as test fixtures.

### Static Type Checking

Run basedpyright:

```bash
uv run basedpyright
```

- Type public functions, methods, and data contracts.
- Avoid using `Any` as an escape hatch; narrow external-library types at adapter boundaries when necessary.

---

## 4. Predictor-Information Firewall & Data Safety

### Predictor Firewall Rules

```text
ECG Waveform / Measurements -> Model A (CIIS)
ECG Waveform / Image       -> Model B (Transformer)

MIMIC-IV Clinical Data     -> Cohort Definition / Outcomes / Secondary Strata ONLY
```

- **Allowed predictor inputs:** Raw 12-lead waveforms, deterministic ECG measurements, rendered ECG images, fixed shared prompts.
- **Prohibited predictor inputs:** Demographics (age, sex, race), diagnoses (ICD-9/10), medications, lab values, vital signs, clinical notes, ICU charts, admission history.

### Local Data Protection

- Never commit raw or row-level MIMIC-IV or MIMIC-IV-ECG data.
- Never commit credentials, Hugging Face access tokens, or private keys.
- Read from `~/data/` directly without modifying source repositories.
- Keep derived row-level analytic datasets in non-tracked directories.

---

## 5. Pull Request Workflow

Every substantive change must be developed on a dedicated branch and submitted via a pull request.

### PR Description Structure

PR descriptions should answer:

```markdown
## What problem does this PR solve?

## What approach did you take?

## What assumptions did you make?

## How did you test it?

## Example output
```

### Pre-Merge Checklist

Before opening or merging a PR, verify:
- [ ] `uv run pytest` passes.
- [ ] `uv run basedpyright` passes with zero errors/warnings.
- [ ] Code follows 2-space indentation and functional conventions.
- [ ] No PHI, credentials, or row-level data are committed.
- [ ] Documentation and reports are synchronized with code changes.
