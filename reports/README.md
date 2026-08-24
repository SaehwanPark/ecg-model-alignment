# Reports

This directory stores reproducible research reports, aggregate inventories, model validation summaries, and non-sensitive analytical findings.

## Data Safety Rules

- Never commit raw or row-level MIMIC-IV or MIMIC-IV-ECG data.
- Never commit patient identifiers (`subject_id`, `study_id`, `hadm_id`) or protected health information (PHI).
- Only aggregate, anonymized summaries and reproducible metrics are permitted here.

## Staged Reports Index

- `reports/data-inventory.md`: Stage 1 MIMIC-IV-ECG and MIMIC-IV linkage inventory.
- `reports/traditional-score-validation.md`: Stage 2 traditional CIIS validation and category checks.
- `reports/cards-clip-smoke-test.md`: Stage 4 CarDSLab ECG-CLIP adapter and embedding prototype.
- `reports/dbeta-smoke-test.md`: Stage 5 D-BETA transformer adapter and benchmark.
- `reports/outcome-definition.md`: Stage 6 MIMIC-IV mortality endpoint definition and validation.
- `reports/cohort-flow.md`: Stage 7 cohort flow attrition and patient-disjoint split.
- `reports/continuous-predictions.md`: Stage 8 continuous prediction pipeline and linear probe.
- `reports/primary-results.md`: Stage 9 primary statistical analysis (alignment, performance, stratified risk, discordance, incremental information).
- `reports/sensitivity-analyses.md`: Stage 10 sensitivity and robustness analyses (cohort index anchoring, alternative horizons, probe variants, quality filtering, alternative traditional models, secondary transformer architectures, demographic subgroups).

