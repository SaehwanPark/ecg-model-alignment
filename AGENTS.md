# AGENTS.md

## Project Context

This repository studies alignment between:

- `A`: a traditional ECG-only risk model; and
- `B`: a modern multimodal transformer-derived ECG model.

Both predictors must use **patient-specific ECG information only**. MIMIC-IV clinical data may be used for cohort definition, linkage, outcome ascertainment, follow-up, and evaluation, but never as predictor features.

Before substantial work, read:

- `docs/research-proposal.md`
- `docs/roadmap.md`

Keep implementation consistent with those documents. If code and documentation diverge, update the documentation or explicitly flag the discrepancy.

## Environment

Use `uv` for Python, dependency management, and command execution.

Agents are authorized to:

- access files under `~/data/`, including local MIMIC-IV and MIMIC-IV-ECG datasets;
- inspect local dataset structure and metadata;
- install dependencies necessary to complete the task;
- download permitted model artifacts and supporting resources.

Prefer:

```bash
uv add <package>
uv add --dev <package>
uv run <command>
```

Do not introduce Conda or ad hoc project environments unless required to reproduce an upstream dependency and clearly isolated from the main project.

## Data Safety

Treat all MIMIC data as protected local data.

- Never commit raw or row-level MIMIC data.
- Never copy protected data into tests, examples, issues, or documentation.
- Never commit credentials, access tokens, or model secrets.
- Prefer synthetic fixtures for tests.
- Read from `~/data/` directly; do not reorganize or modify source datasets unless explicitly required.
- Derived row-level analytic files should remain outside Git.

## Python Conventions

- Use 2 spaces for indentation.
- Prefer functional-style Python.
- Favor pure functions and explicit inputs/outputs.
- Isolate filesystem, model-loading, GPU, and other side effects.
- Avoid hidden mutable global state.
- Prefer immutable configuration objects where practical.
- Keep functions small, typed, and topologically ordered where practical.
- Prefer composition over deep class hierarchies.
- Keep model-specific behavior behind small adapters.

## Quality Checks

Use:

```bash
uv run pytest
uv run basedpyright
```

For substantive changes:

- add or update tests;
- test boundary conditions and failure paths;
- preserve deterministic preprocessing where possible;
- verify patient-disjoint train/validation/test logic;
- verify that no non-ECG clinical feature enters `A` or `B`.

Do not silence type errors or failing tests without a documented reason.

## Research Guardrails

Maintain the predictor-information firewall:

```text
ECG -> A
ECG -> B

MIMIC-IV clinical data -> outcomes / cohort / evaluation only
```

Allowed predictor inputs include:

- raw ECG waveforms;
- deterministic ECG-derived measurements;
- ECG images rendered from the waveform;
- fixed prompts or class descriptions shared across patients.

Not allowed as predictor features include:

- demographics;
- diagnoses;
- medications;
- laboratory values;
- non-ECG vital signs;
- notes;
- encounter history;
- ICU variables.

Several candidate transformer models were pretrained on MIMIC-IV-ECG. When applicable, describe results as **in-domain representation probing**, not independent external validation.

## Working Style

When implementing a roadmap item:

1. inspect the relevant existing code and tests;
2. make the smallest coherent change that solves the task;
3. add tests alongside reusable logic;
4. run `pytest` and `basedpyright`;
5. update documentation when assumptions, interfaces, or research decisions change.

Prefer reproducible commands and explicit configuration over machine-specific hard-coded paths. Local defaults such as `~/data/mimic-iv-ecg/1.0` and `~/data/mimiciv/3.1` are acceptable for developer convenience when overrideable.

## Git Hygiene

Keep commits focused and reviewable.

Do not commit:

- raw data;
- generated caches;
- model checkpoints;
- credentials;
- large derived artifacts.

Before considering work complete, ensure the repository remains reproducible from tracked code, configuration, documented local data paths, and permitted external model downloads.

## Tooling

- You may use `gh` for GitHub CLI interactions. Token: `$GH_TOKEN`
- You may use `hf` to interact with Hugging Face models. Token: `$HF_TOKEN`
