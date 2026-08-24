# Reports

This directory stores reproducible research reports, aggregate inventories, model validation summaries, and non-sensitive analytical findings.

## Data Safety Rules

- Never commit raw or row-level MIMIC-IV or MIMIC-IV-ECG data.
- Never commit patient identifiers (`subject_id`, `study_id`, `hadm_id`) or protected health information (PHI).
- Only aggregate, anonymized summaries and reproducible metrics are permitted here.
