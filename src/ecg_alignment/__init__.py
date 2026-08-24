"""ECG Model Alignment package.

Studies alignment and discordance between traditional ECG-only risk models
and modern multimodal transformer-derived ECG representations.
"""

from ecg_alignment.cohort import (
  CohortStep,
  compute_age_at_ecg,
  compute_linkage_statistics,
  filter_adult_records,
  generate_cohort_flow,
  link_records_to_admissions,
  select_index_ecgs,
)
from ecg_alignment.data import (
  CANONICAL_12_LEADS,
  DEFAULT_DURATION_SECONDS,
  DEFAULT_SAMPLING_RATE_HZ,
  DEFAULT_SIGNAL_LENGTH_SAMPLES,
  DataPaths,
  EcgHeaderInfo,
  WaveformEligibilityCriteria,
  load_admissions,
  load_machine_measurements,
  load_patients,
  load_record_list,
  read_ecg_header,
  read_ecg_waveform,
  validate_waveform_eligibility,
)

__version__ = "0.1.0"

__all__ = [
  "__version__",
  "CANONICAL_12_LEADS",
  "DEFAULT_DURATION_SECONDS",
  "DEFAULT_SAMPLING_RATE_HZ",
  "DEFAULT_SIGNAL_LENGTH_SAMPLES",
  "DataPaths",
  "EcgHeaderInfo",
  "WaveformEligibilityCriteria",
  "load_admissions",
  "load_machine_measurements",
  "load_patients",
  "load_record_list",
  "read_ecg_header",
  "read_ecg_waveform",
  "validate_waveform_eligibility",
  "CohortStep",
  "compute_age_at_ecg",
  "compute_linkage_statistics",
  "filter_adult_records",
  "generate_cohort_flow",
  "link_records_to_admissions",
  "select_index_ecgs",
]
