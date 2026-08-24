"""Tests for data loading, WFDB parsing, and waveform validation."""

import gzip
from pathlib import Path
import numpy as np
import polars as pl
import pytest
import wfdb

from ecg_alignment.data import (
  CANONICAL_12_LEADS,
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


def test_data_paths_defaults() -> None:
  paths = DataPaths()
  assert "mimic-iv-ecg" in str(paths.mimic_iv_ecg_dir)
  assert "mimiciv" in str(paths.mimic_iv_dir)
  assert paths.record_list_path.name == "record_list.csv"
  assert paths.machine_measurements_path.name == "machine_measurements.csv"
  assert paths.patients_path.name == "patients.csv.gz"
  assert paths.admissions_path.name == "admissions.csv.gz"


def test_data_paths_custom(tmp_path: Path) -> None:
  ecg_dir = tmp_path / "ecg"
  mimic_dir = tmp_path / "mimic"
  paths = DataPaths(mimic_iv_ecg_dir=ecg_dir, mimic_iv_dir=mimic_dir)
  assert paths.record_list_path == ecg_dir / "record_list.csv"
  assert paths.patients_path == mimic_dir / "hosp" / "patients.csv.gz"


def test_validate_waveform_eligibility_standard() -> None:
  header = EcgHeaderInfo(
    record_name="12345",
    n_sig=12,
    fs=500,
    sig_len=5000,
    duration_seconds=10.0,
    sig_name=CANONICAL_12_LEADS,
    units=("mV",) * 12,
    adc_gain=(200.0,) * 12,
    baseline=(0,) * 12,
    comments=("Comment",),
  )
  is_valid, reason = validate_waveform_eligibility(header)
  assert is_valid is True
  assert reason is None


def test_validate_waveform_eligibility_failures() -> None:
  # Bad fs
  h_fs = EcgHeaderInfo(
    record_name="1",
    n_sig=12,
    fs=250,
    sig_len=5000,
    duration_seconds=20.0,
    sig_name=CANONICAL_12_LEADS,
    units=("mV",) * 12,
    adc_gain=(200.0,) * 12,
    baseline=(0,) * 12,
    comments=(),
  )
  valid, reason = validate_waveform_eligibility(h_fs)
  assert valid is False
  assert reason is not None
  assert "sampling rate" in reason

  # Bad length
  h_len = EcgHeaderInfo(
    record_name="1",
    n_sig=12,
    fs=500,
    sig_len=4000,
    duration_seconds=8.0,
    sig_name=CANONICAL_12_LEADS,
    units=("mV",) * 12,
    adc_gain=(200.0,) * 12,
    baseline=(0,) * 12,
    comments=(),
  )
  valid, reason = validate_waveform_eligibility(h_len)
  assert valid is False
  assert reason is not None
  assert "signal length" in reason

  # Missing lead
  leads_missing = CANONICAL_12_LEADS[:-1]  # missing V6
  h_leads = EcgHeaderInfo(
    record_name="1",
    n_sig=11,
    fs=500,
    sig_len=5000,
    duration_seconds=10.0,
    sig_name=leads_missing,
    units=("mV",) * 11,
    adc_gain=(200.0,) * 11,
    baseline=(0,) * 11,
    comments=(),
  )
  valid, reason = validate_waveform_eligibility(h_leads)
  assert valid is False
  assert reason is not None
  assert "Missing required leads" in reason
  assert "V6" in reason

  # Bad units
  h_units = EcgHeaderInfo(
    record_name="1",
    n_sig=12,
    fs=500,
    sig_len=5000,
    duration_seconds=10.0,
    sig_name=CANONICAL_12_LEADS,
    units=("uV",) * 12,
    adc_gain=(200.0,) * 12,
    baseline=(0,) * 12,
    comments=(),
  )
  valid, reason = validate_waveform_eligibility(h_units)
  assert valid is False
  assert reason is not None
  assert "units" in reason


def test_read_wfdb_record(tmp_path: Path) -> None:
  # Create synthetic 12-lead signal
  record_name = "test_ecg"
  record_dir = tmp_path / "ecg_files"
  record_dir.mkdir(parents=True, exist_ok=True)
  full_record_path = record_dir / record_name

  sig = np.zeros((5000, 12), dtype=np.float64)
  sig[:, 0] = np.sin(np.linspace(0, 10, 5000))

  wfdb.wrsamp(
    record_name=record_name,
    fs=500,
    units=["mV"] * 12,
    sig_name=list(CANONICAL_12_LEADS),
    p_signal=sig,
    fmt=["16"] * 12,
    write_dir=str(record_dir),
    comments=["Synthetic ECG record"],
  )

  # Read header
  header = read_ecg_header(full_record_path)
  assert header.record_name == record_name
  assert header.fs == 500
  assert header.sig_len == 5000
  assert header.n_sig == 12
  assert header.duration_seconds == 10.0
  assert header.sig_name == CANONICAL_12_LEADS
  assert header.units == ("mV",) * 12

  # Also test reading with .hea extension
  header_with_ext = read_ecg_header(str(full_record_path) + ".hea")
  assert header_with_ext.record_name == record_name

  # Read waveform
  data, lead_names, fs = read_ecg_waveform(full_record_path)
  assert data.shape == (5000, 12)
  assert lead_names == list(CANONICAL_12_LEADS)
  assert fs == 500


def test_load_record_list(tmp_path: Path) -> None:
  csv_content = (
    "subject_id,study_id,file_name,ecg_time,path\n"
    "10001,20001,20001,2150-05-10 14:30:00,files/p10/p10001/s20001/20001\n"
    "10002,20002,20002,2160-01-01 00:00:00,files/p10/p10002/s20002/20002\n"
  )
  file_path = tmp_path / "record_list.csv"
  file_path.write_text(csv_content)

  df = load_record_list(file_path)
  assert len(df) == 2
  assert df["subject_id"].to_list() == [10001, 10002]
  assert df["study_id"].to_list() == [20001, 20002]
  assert df["ecg_year"].to_list() == [2150, 2160]
  assert df["ecg_datetime"].dt.year().to_list() == [2150, 2160]


def test_load_machine_measurements(tmp_path: Path) -> None:
  csv_content = (
    "subject_id,study_id,cart_id,ecg_time,report_0,bandwidth,filtering,rr_interval,p_onset,p_end,qrs_onset,qrs_end,t_end,p_axis,qrs_axis,t_axis\n"
    "10001,20001,101,2150-05-10 14:30:00,Normal ECG,0.005-150 Hz,60 Hz,800,40,120,160,240,480,60,60,60\n"
  )
  file_path = tmp_path / "machine_measurements.csv"
  file_path.write_text(csv_content)

  df = load_machine_measurements(file_path)
  assert len(df) == 1
  assert df["subject_id"][0] == 10001
  assert df["rr_interval"][0] == 800
  assert df["qrs_axis"][0] == 60
  assert df["ecg_datetime"].dt.year()[0] == 2150


def test_load_patients(tmp_path: Path) -> None:
  csv_content = (
    "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod\n"
    "10001,F,60,2150,2014 - 2016,2152-01-15\n"
    "10002,M,45,2160,2017 - 2019,\n"
  )
  gz_path = tmp_path / "patients.csv.gz"
  with gzip.open(gz_path, "wt") as f:
    f.write(csv_content)

  df = load_patients(gz_path)
  assert len(df) == 2
  assert df["subject_id"].to_list() == [10001, 10002]
  assert df["gender"].to_list() == ["F", "M"]
  assert df["anchor_age"].to_list() == [60, 45]
  assert df["dod_date"].to_list() is not None


def test_load_admissions(tmp_path: Path) -> None:
  csv_content = (
    "subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,insurance,language,marital_status,race,edregtime,edouttime,hospital_expire_flag\n"
    "10001,30001,2150-05-10 12:00:00,2150-05-15 16:00:00,,EMERGENCY,EMERGENCY ROOM,HOME,Medicare,English,MARRIED,WHITE,2150-05-10 09:00:00,2150-05-10 13:00:00,0\n"
  )
  gz_path = tmp_path / "admissions.csv.gz"
  with gzip.open(gz_path, "wt") as f:
    f.write(csv_content)

  df = load_admissions(gz_path)
  assert len(df) == 1
  assert df["subject_id"][0] == 10001
  assert df["hadm_id"][0] == 30001
  assert df["hospital_expire_flag"][0] == 0
  assert df["admit_dt"].dt.day()[0] == 10
