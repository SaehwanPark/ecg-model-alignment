"""Data loading and waveform inspection for MIMIC-IV-ECG and MIMIC-IV datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import numpy as np
import numpy.typing as npt
import polars as pl
import wfdb

CANONICAL_12_LEADS: tuple[str, ...] = (
  "I",
  "II",
  "III",
  "aVR",
  "aVF",
  "aVL",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
)

DEFAULT_SAMPLING_RATE_HZ: int = 500
DEFAULT_SIGNAL_LENGTH_SAMPLES: int = 5000
DEFAULT_DURATION_SECONDS: float = 10.0


@dataclass(frozen=True)
class DataPaths:
  """Immutable configuration for dataset filesystem locations."""

  mimic_iv_ecg_dir: Path = Path.home() / "data" / "mimic-iv-ecg" / "1.0"
  mimic_iv_dir: Path = Path.home() / "data" / "mimiciv" / "3.1"

  @property
  def record_list_path(self) -> Path:
    return self.mimic_iv_ecg_dir / "record_list.csv"

  @property
  def machine_measurements_path(self) -> Path:
    return self.mimic_iv_ecg_dir / "machine_measurements.csv"

  @property
  def patients_path(self) -> Path:
    return self.mimic_iv_dir / "hosp" / "patients.csv.gz"

  @property
  def admissions_path(self) -> Path:
    return self.mimic_iv_dir / "hosp" / "admissions.csv.gz"


@dataclass(frozen=True)
class EcgHeaderInfo:
  """Metadata extracted from a WFDB header."""

  record_name: str
  n_sig: int
  fs: int
  sig_len: int
  duration_seconds: float
  sig_name: tuple[str, ...]
  units: tuple[str, ...]
  adc_gain: tuple[float, ...]
  baseline: tuple[int, ...]
  comments: tuple[str, ...]


@dataclass(frozen=True)
class WaveformEligibilityCriteria:
  """Criteria defining technical validity for an ECG waveform."""

  required_leads: tuple[str, ...] = CANONICAL_12_LEADS
  expected_fs: int = DEFAULT_SAMPLING_RATE_HZ
  expected_sig_len: int = DEFAULT_SIGNAL_LENGTH_SAMPLES
  expected_units: str = "mV"


def _clean_record_path(record_path: Path | str) -> str:
  """Strip .hea or .dat suffix from record path for WFDB compatibility."""
  p = str(record_path)
  if p.endswith(".hea"):
    return p[:-4]
  if p.endswith(".dat"):
    return p[:-4]
  return p


def read_ecg_header(record_path: Path | str) -> EcgHeaderInfo:
  """Read header metadata for a single WFDB ECG record.

  Args:
    record_path: Path to the WFDB record (with or without .hea suffix).

  Returns:
    EcgHeaderInfo dataclass containing parsed signal metadata.
  """
  clean_path = _clean_record_path(record_path)
  header = wfdb.rdheader(clean_path)
  fs = int(header.fs) if header.fs is not None else 0
  sig_len = int(header.sig_len) if header.sig_len is not None else 0
  duration = sig_len / fs if fs > 0 else 0.0

  sig_name_list: list[str] = [str(n) for n in (header.sig_name or [])]
  units_list: list[str] = [str(u) for u in (getattr(header, "units", None) or [])]
  adc_gain_list: list[float] = [float(g) for g in (getattr(header, "adc_gain", None) or [])]
  baseline_list: list[int] = [int(b) for b in (getattr(header, "baseline", None) or [])]
  comments_list: list[str] = [str(c) for c in (header.comments or [])]

  return EcgHeaderInfo(
    record_name=str(header.record_name),
    n_sig=int(header.n_sig) if header.n_sig is not None else 0,
    fs=fs,
    sig_len=sig_len,
    duration_seconds=duration,
    sig_name=tuple(sig_name_list),
    units=tuple(units_list),
    adc_gain=tuple(adc_gain_list),
    baseline=tuple(baseline_list),
    comments=tuple(comments_list),
  )


def read_ecg_waveform(
  record_path: Path | str,
) -> tuple[npt.NDArray[np.float64], list[str], int]:
  """Read waveform signals and channel names for a single WFDB record.

  Args:
    record_path: Path to the WFDB record.

  Returns:
    Tuple of (signal_array [sig_len, n_sig], lead_names, sampling_rate_hz).
  """
  clean_path = _clean_record_path(record_path)
  signal_raw, fields = wfdb.rdsamp(clean_path)
  if signal_raw is None:
    raise FileNotFoundError(f"Could not load signal for record {record_path}")
  signal_array = np.asarray(signal_raw, dtype=np.float64)
  fields_dict: dict[str, Any] = cast(dict[str, Any], fields)
  lead_names: list[str] = [str(name) for name in fields_dict.get("sig_name", [])]
  fs: int = int(fields_dict.get("fs", DEFAULT_SAMPLING_RATE_HZ))
  return signal_array, lead_names, fs


def validate_waveform_eligibility(
  header: EcgHeaderInfo,
  criteria: WaveformEligibilityCriteria | None = None,
) -> tuple[bool, str | None]:
  """Check whether an ECG record meets technical waveform eligibility criteria.

  Args:
    header: Parsed WFDB header info.
    criteria: Custom criteria; defaults to standard 12-lead 500 Hz 10s.

  Returns:
    Tuple of (is_eligible, failure_reason). failure_reason is None if eligible.
  """
  crit = criteria or WaveformEligibilityCriteria()

  if header.fs != crit.expected_fs:
    return False, f"Unexpected sampling rate: {header.fs} Hz (expected {crit.expected_fs} Hz)"

  if header.sig_len != crit.expected_sig_len:
    return (
      False,
      f"Unexpected signal length: {header.sig_len} samples (expected {crit.expected_sig_len} samples)",
    )

  missing_leads = [lead for lead in crit.required_leads if lead not in header.sig_name]
  if missing_leads:
    return False, f"Missing required leads: {missing_leads}"

  if crit.expected_units:
    invalid_units = [u for u in header.units if u != crit.expected_units]
    if invalid_units:
      return False, f"Non-standard signal units found: {set(invalid_units)}"

  return True, None


def load_record_list(path: Path | str) -> pl.DataFrame:
  """Load and type-cast MIMIC-IV-ECG record_list.csv.

  Args:
    path: Path to record_list.csv.

  Returns:
    DataFrame with columns: subject_id (i64), study_id (i64), file_name (i64),
    ecg_time (str), path (str), ecg_datetime (Datetime), ecg_year (i64).
  """
  df = pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "study_id": pl.Int64,
      "file_name": pl.Int64,
      "ecg_time": pl.String,
      "path": pl.String,
    },
  )
  return df.with_columns(
    pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ecg_datetime"),
    pl.col("ecg_time").str.slice(0, 4).cast(pl.Int64).alias("ecg_year"),
  )


def load_machine_measurements(path: Path | str) -> pl.DataFrame:
  """Load and parse MIMIC-IV-ECG machine_measurements.csv.

  Args:
    path: Path to machine_measurements.csv.

  Returns:
    DataFrame with machine measurements and parsed ecg_datetime.
  """
  df = pl.read_csv(
    path,
    infer_schema_length=10000,
    schema_overrides={
      "subject_id": pl.Int64,
      "study_id": pl.Int64,
      "cart_id": pl.Int64,
      "ecg_time": pl.String,
      "rr_interval": pl.Int64,
      "p_onset": pl.Int64,
      "p_end": pl.Int64,
      "qrs_onset": pl.Int64,
      "qrs_end": pl.Int64,
      "t_end": pl.Int64,
      "p_axis": pl.Int64,
      "qrs_axis": pl.Int64,
      "t_axis": pl.Int64,
    },
  )
  return df.with_columns(
    pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ecg_datetime")
  )


def load_patients(path: Path | str) -> pl.DataFrame:
  """Load and parse MIMIC-IV hosp/patients.csv.gz.

  Args:
    path: Path to patients.csv.gz or uncompressed patients.csv.

  Returns:
    DataFrame with columns: subject_id (i64), gender (str), anchor_age (i64),
    anchor_year (i64), anchor_year_group (str), dod (Date).
  """
  df = pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "gender": pl.String,
      "anchor_age": pl.Int64,
      "anchor_year": pl.Int64,
      "anchor_year_group": pl.String,
      "dod": pl.String,
    },
  )
  return df.with_columns(
    pl.col("dod").str.to_date("%Y-%m-%d", strict=False).alias("dod_date")
  )


def load_admissions(path: Path | str) -> pl.DataFrame:
  """Load and parse MIMIC-IV hosp/admissions.csv.gz.

  Args:
    path: Path to admissions.csv.gz or uncompressed admissions.csv.

  Returns:
    DataFrame with typed identifiers, timestamps, and encounter metadata.
  """
  df = pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "hadm_id": pl.Int64,
      "admittime": pl.String,
      "dischtime": pl.String,
      "deathtime": pl.String,
      "admission_type": pl.String,
      "admission_location": pl.String,
      "discharge_location": pl.String,
      "insurance": pl.String,
      "language": pl.String,
      "marital_status": pl.String,
      "race": pl.String,
      "edregtime": pl.String,
      "edouttime": pl.String,
      "hospital_expire_flag": pl.Int64,
    },
  )
  return df.with_columns(
    pl.col("admittime").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("admit_dt"),
    pl.col("dischtime").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("disch_dt"),
    pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("death_dt"),
    pl.col("edregtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("edreg_dt"),
    pl.col("edouttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("edout_dt"),
  )
