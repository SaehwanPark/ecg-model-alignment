"""Tests for patient-disjoint cohort splitting, attrition tracking, and split verification."""

from datetime import datetime
from pathlib import Path
import polars as pl
import pytest

from ecg_alignment.split import (
  CohortAttritionStep,
  CohortSplitResult,
  SplitRatios,
  assign_patient_disjoint_split,
  build_primary_cohort_and_split,
  compute_split_summary_statistics,
  derive_primary_cohort,
  generate_cohort_flow_markdown,
  save_split_assignments,
  verify_split_disjointness,
)


@pytest.fixture
def synthetic_records() -> pl.DataFrame:
  return pl.DataFrame({
    "subject_id": [1, 1, 1, 2, 2, 3, 4, 5, 6, 7],
    "study_id": [101, 102, 103, 201, 202, 301, 401, 501, 601, 701],
    "file_name": [101, 102, 103, 201, 202, 301, 401, 501, 601, 701],
    "ecg_time": [
      "2150-01-01 10:00:00",
      "2150-01-01 10:00:00",  # Duplicate timestamp for tie break
      "2152-06-01 12:00:00",
      "2160-03-15 08:00:00",
      "2165-04-20 16:00:00",
      "2140-11-10 14:00:00",
      "2155-09-01 09:00:00",  # Unlinked subject (not in patients)
      "2170-05-12 11:00:00",  # Invalid outcome (ECG after recorded death)
      "2180-01-01 00:00:00",  # Underage subject (age < 18)
      "2190-07-04 12:00:00",  # Valid adult
    ],
    "path": [f"files/{i}" for i in range(10)],
  }).with_columns(
    pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ecg_datetime"),
    pl.col("ecg_time").str.slice(0, 4).cast(pl.Int64).alias("ecg_year"),
  )


@pytest.fixture
def synthetic_patients() -> pl.DataFrame:
  return pl.DataFrame({
    "subject_id": [1, 2, 3, 5, 6, 7],
    "gender": ["M", "F", "F", "M", "F", "M"],
    "anchor_age": [50, 15, 80, 60, 10, 45],
    "anchor_year": [2150, 2160, 2140, 2170, 2180, 2190],
    "anchor_year_group": [
      "2014 - 2016",
      "2017 - 2019",
      "2011 - 2013",
      "2014 - 2016",
      "2017 - 2019",
      "2014 - 2016",
    ],
    "dod": [
      "2153-01-01",
      None,
      "2141-05-01",
      "2169-12-31",  # Died before ECG in 2170 -> invalid outcome
      None,
      None,
    ],
  })


@pytest.fixture
def synthetic_admissions() -> pl.DataFrame:
  return pl.DataFrame({
    "subject_id": [1, 1, 2],
    "hadm_id": [10001, 10002, 20001],
    "admit_dt": [
      datetime(2150, 1, 1, 8, 0, 0),
      datetime(2152, 6, 1, 10, 0, 0),
      datetime(2160, 3, 15, 6, 0, 0),
    ],
    "disch_dt": [
      datetime(2150, 1, 5, 12, 0, 0),
      datetime(2152, 6, 10, 18, 0, 0),
      datetime(2160, 3, 20, 10, 0, 0),
    ],
    "hospital_expire_flag": [0, 0, 0],
    "death_dt": [None, None, None],
  })


def test_split_ratios_validation() -> None:
  # Default valid ratios
  r = SplitRatios()
  assert r.dev == 0.6
  assert r.val == 0.2
  assert r.test == 0.2

  # Custom valid ratios
  custom = SplitRatios(dev=0.7, val=0.15, test=0.15)
  assert custom.dev == 0.7

  # Invalid: sum != 1.0
  with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
    SplitRatios(dev=0.5, val=0.2, test=0.2)

  # Invalid: non-positive ratio
  with pytest.raises(ValueError, match="All split ratios must be strictly positive"):
    SplitRatios(dev=0.8, val=0.2, test=0.0)


def test_assign_patient_disjoint_split_properties() -> None:
  # Create a cohort with 100 distinct subjects and multiple records per subject
  subject_ids = list(range(1, 101))
  # 2 records per subject
  repeated_subjects = [s for s in subject_ids for _ in range(2)]
  study_ids = list(range(1000, 1000 + len(repeated_subjects)))

  df = pl.DataFrame({
    "subject_id": repeated_subjects,
    "study_id": study_ids,
  })

  split_df = assign_patient_disjoint_split(df, ratios=SplitRatios(dev=0.6, val=0.2, test=0.2), seed=42)

  assert "split" in split_df.columns
  assert len(split_df) == 200

  # Check disjointness
  dev_subjs = set(split_df.filter(pl.col("split") == "dev")["subject_id"].to_list())
  val_subjs = set(split_df.filter(pl.col("split") == "val")["subject_id"].to_list())
  test_subjs = set(split_df.filter(pl.col("split") == "test")["subject_id"].to_list())

  assert len(dev_subjs.intersection(val_subjs)) == 0
  assert len(dev_subjs.intersection(test_subjs)) == 0
  assert len(val_subjs.intersection(test_subjs)) == 0

  # Check proportions on subjects
  assert len(dev_subjs) == 60
  assert len(val_subjs) == 20
  assert len(test_subjs) == 20


def test_assign_patient_disjoint_split_deterministic() -> None:
  df = pl.DataFrame({"subject_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

  res1 = assign_patient_disjoint_split(df, seed=42)
  res2 = assign_patient_disjoint_split(df, seed=42)
  res3 = assign_patient_disjoint_split(df, seed=999)

  assert res1["split"].to_list() == res2["split"].to_list()
  # Different seed should permute differently
  assert res1["split"].to_list() != res3["split"].to_list()


def test_verify_split_disjointness_raises_on_leakage() -> None:
  # Valid disjoint DataFrame
  valid_df = pl.DataFrame({
    "subject_id": [1, 1, 2, 3],
    "split": ["dev", "dev", "val", "test"],
  })
  assert verify_split_disjointness(valid_df) is True

  # Invalid DataFrame where subject 1 is in both dev and test
  leaky_df = pl.DataFrame({
    "subject_id": [1, 1, 2, 3],
    "split": ["dev", "test", "val", "test"],
  })
  with pytest.raises(ValueError, match="Supervised patient split violation"):
    verify_split_disjointness(leaky_df)

  # Missing column
  with pytest.raises(ValueError, match="must contain 'subject_id' and 'split'"):
    verify_split_disjointness(pl.DataFrame({"subject_id": [1, 2]}))


def test_derive_primary_cohort(
  synthetic_records: pl.DataFrame,
  synthetic_patients: pl.DataFrame,
  synthetic_admissions: pl.DataFrame,
) -> None:
  cohort_df, steps = derive_primary_cohort(
    records_df=synthetic_records,
    patients_df=synthetic_patients,
    admissions_df=synthetic_admissions,
    min_age=18,
    index_strategy="earliest",
  )

  # Check attrition steps
  assert len(steps) == 5
  assert steps[0].step_name == "All MIMIC-IV-ECG Records"
  assert steps[0].records_remaining == 10
  assert steps[0].subjects_remaining == 7  # subjects 1, 2, 3, 4, 5, 6, 7

  assert steps[1].step_name == "Linked to MIMIC-IV Patients"
  # Subject 4 is unlinked -> 9 records, 6 subjects
  assert steps[1].records_remaining == 9
  assert steps[1].subjects_remaining == 6
  assert steps[1].records_excluded == 1
  assert steps[1].subjects_excluded == 1

  assert steps[2].step_name == "Adult Patients (Age >= 18)"
  # Subject 2 at 2160 (age 15) excluded; subject 6 (age 10) excluded
  # Subject 2 at 2165 (age 20) included
  # Remaining adult subjects: 1, 2, 3, 5, 7
  assert steps[2].subjects_remaining == 5
  assert steps[2].records_excluded == 2  # record for subj 2 in 2160 and subj 6 in 2180
  assert steps[2].subjects_excluded == 1  # subj 6

  assert steps[3].step_name == "Valid Outcome Follow-Up"
  # Subject 5 has ECG after death -> excluded
  # Remaining subjects: 1, 2, 3, 7 (4 subjects)
  assert steps[3].subjects_remaining == 4
  assert steps[3].subjects_excluded == 1  # subj 5

  assert steps[4].step_name == "Index ECG Selection (Earliest)"
  # Single earliest ECG per remaining subject: 1 (study 101), 2 (study 202), 3 (study 301), 7 (study 701)
  assert steps[4].records_remaining == 4
  assert steps[4].subjects_remaining == 4
  assert len(cohort_df) == 4

  # Check tie breaker on subject 1 (studies 101 and 102 had exact same timestamp, 101 should be chosen)
  p1 = cohort_df.filter(pl.col("subject_id") == 1)
  assert p1["study_id"][0] == 101


def test_build_primary_cohort_and_split_end_to_end(
  synthetic_records: pl.DataFrame,
  synthetic_patients: pl.DataFrame,
  synthetic_admissions: pl.DataFrame,
) -> None:
  result = build_primary_cohort_and_split(
    records_df=synthetic_records,
    patients_df=synthetic_patients,
    admissions_df=synthetic_admissions,
    ratios=SplitRatios(dev=0.5, val=0.25, test=0.25),
    seed=42,
    min_age=18,
    index_strategy="earliest",
  )

  assert isinstance(result, CohortSplitResult)
  assert len(result.cohort_df) == 4
  assert "split" in result.cohort_df.columns
  assert result.seed == 42
  assert sum(result.split_record_counts.values()) == 4
  assert sum(result.split_subject_counts.values()) == 4


def test_save_split_assignments(tmp_path: Path) -> None:
  df = pl.DataFrame({
    "subject_id": [1, 2, 3],
    "study_id": [101, 201, 301],
    "split": ["dev", "val", "test"],
    "extra_feature": [1.0, 2.0, 3.0],
  })

  csv_path = tmp_path / "split_assignments.csv"
  save_split_assignments(df, csv_path)
  assert csv_path.exists()
  loaded_csv = pl.read_csv(csv_path)
  assert loaded_csv.columns == ["subject_id", "study_id", "split"]
  assert len(loaded_csv) == 3

  parquet_path = tmp_path / "split_assignments.parquet"
  save_split_assignments(df, parquet_path)
  assert parquet_path.exists()
  loaded_parquet = pl.read_parquet(parquet_path)
  assert loaded_parquet.columns == ["subject_id", "study_id", "split"]
  assert len(loaded_parquet) == 3


def test_compute_split_summary_and_markdown() -> None:
  cohort_df = pl.DataFrame({
    "subject_id": [1, 2, 3, 4, 5],
    "study_id": [101, 102, 103, 104, 105],
    "split": ["dev", "dev", "dev", "val", "test"],
    "mortality_30d": [False, True, False, False, True],
    "mortality_90d": [False, True, True, False, True],
    "mortality_1yr": [True, True, True, False, True],
  })

  stats = compute_split_summary_statistics(cohort_df)
  assert stats["total_patients"] == 5
  assert stats["dev_n"] == 3
  assert stats["val_n"] == 1
  assert stats["test_n"] == 1
  assert stats["dev_mortality_30d_events"] == 1
  assert stats["test_mortality_30d_events"] == 1

  steps = (
    CohortAttritionStep(1, "All", "All", 10, 10, 0, 0),
    CohortAttritionStep(2, "Index", "Index", 5, 5, 5, 5),
  )
  result = CohortSplitResult(
    cohort_df=cohort_df,
    attrition_steps=steps,
    split_record_counts={"dev": 3, "val": 1, "test": 1},
    split_subject_counts={"dev": 3, "val": 1, "test": 1},
    seed=42,
    ratios=SplitRatios(dev=0.6, val=0.2, test=0.2),
  )

  md = generate_cohort_flow_markdown(result, summary_stats=stats)
  assert "# MIMIC-IV Primary Cohort Flow and Analytic Split Report" in md
  assert "flowchart TD" in md
  assert "Development (`dev`)" in md
  assert "Zero `subject_id` overlap verified" in md
