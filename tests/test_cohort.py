"""Tests for linkage, age computation, and cohort flow derivation."""

from datetime import datetime
import polars as pl
import pytest

from ecg_alignment.cohort import (
  CohortStep,
  compute_age_at_ecg,
  compute_linkage_statistics,
  filter_adult_records,
  generate_cohort_flow,
  link_records_to_admissions,
  select_index_ecgs,
)


@pytest.fixture
def sample_records() -> pl.DataFrame:
  return pl.DataFrame({
    "subject_id": [1, 1, 1, 2, 2, 3, 4],
    "study_id": [101, 102, 103, 201, 202, 301, 401],
    "file_name": [101, 102, 103, 201, 202, 301, 401],
    "ecg_time": [
      "2150-01-01 10:00:00",
      "2150-01-01 10:00:00",  # Exact duplicate timestamp for tie-breaker test
      "2152-06-01 12:00:00",
      "2160-03-15 08:00:00",
      "2165-04-20 16:00:00",
      "2140-11-10 14:00:00",
      "2155-09-01 09:00:00",  # Unlinked subject (not in patients)
    ],
    "path": [f"files/{i}" for i in range(7)],
  }).with_columns(
    pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ecg_datetime"),
    pl.col("ecg_time").str.slice(0, 4).cast(pl.Int64).alias("ecg_year"),
  )


@pytest.fixture
def sample_patients() -> pl.DataFrame:
  return pl.DataFrame({
    "subject_id": [1, 2, 3],
    "gender": ["M", "F", "F"],
    "anchor_age": [50, 15, 80],
    "anchor_year": [2150, 2160, 2140],
    "anchor_year_group": ["2014 - 2016", "2017 - 2019", "2011 - 2013"],
    "dod": ["2153-01-01", None, "2141-05-01"],
  })


@pytest.fixture
def sample_admissions() -> pl.DataFrame:
  return pl.DataFrame({
    "subject_id": [1, 1, 2],
    "hadm_id": [10001, 10002, 20001],
    "admit_dt": [
      datetime(2150, 1, 1, 8, 0, 0),
      datetime(2152, 6, 1, 10, 0, 0),
      datetime(2160, 3, 15, 12, 0, 0),  # ECG is 4h before this admission
    ],
    "disch_dt": [
      datetime(2150, 1, 5, 12, 0, 0),
      datetime(2152, 6, 10, 18, 0, 0),
      datetime(2160, 3, 20, 10, 0, 0),
    ],
    "admission_type": ["EMERGENCY", "ELECTIVE", "EW EMER."],
  })


def test_compute_age_at_ecg(
  sample_records: pl.DataFrame,
  sample_patients: pl.DataFrame,
) -> None:
  joined = compute_age_at_ecg(sample_records, sample_patients)

  # Unlinked subject 4 should be omitted in inner join
  assert set(joined["subject_id"].to_list()) == {1, 2, 3}

  # Patient 1: anchor_age 50 in 2150. ECG in 2150 -> age 50; ECG in 2152 -> age 52
  p1 = joined.filter(pl.col("subject_id") == 1)
  assert p1.filter(pl.col("ecg_year") == 2150)["age_at_ecg"].to_list() == [50, 50]
  assert p1.filter(pl.col("ecg_year") == 2152)["age_at_ecg"].to_list() == [52]

  # Patient 2: anchor_age 15 in 2160. ECG in 2160 -> age 15; ECG in 2165 -> age 20
  p2 = joined.filter(pl.col("subject_id") == 2)
  assert p2.filter(pl.col("ecg_year") == 2160)["age_at_ecg"].to_list() == [15]
  assert p2.filter(pl.col("ecg_year") == 2165)["age_at_ecg"].to_list() == [20]


def test_filter_adult_records(
  sample_records: pl.DataFrame,
  sample_patients: pl.DataFrame,
) -> None:
  joined = compute_age_at_ecg(sample_records, sample_patients)
  adults = filter_adult_records(joined, min_age=18)

  # Patient 2 at 2160 (age 15) must be excluded; Patient 2 at 2165 (age 20) must be included
  assert len(adults.filter((pl.col("subject_id") == 2) & (pl.col("age_at_ecg") == 15))) == 0
  assert len(adults.filter((pl.col("subject_id") == 2) & (pl.col("age_at_ecg") == 20))) == 1

  # Check missing column error
  with pytest.raises(ValueError, match="DataFrame must contain 'age_at_ecg' column"):
    filter_adult_records(sample_records)


def test_link_records_to_admissions(
  sample_records: pl.DataFrame,
  sample_admissions: pl.DataFrame,
) -> None:
  # With pre_admit_hours=0:
  # Subject 1 has ECGs on 2150-01-01 10:00 (during adm 10001) and 2152-06-01 12:00 (during adm 10002)
  # Subject 2 has ECG on 2160-03-15 08:00 (4h before adm 20001 at 12:00) -> not matched with 0h
  matched_0h = link_records_to_admissions(sample_records, sample_admissions, pre_admit_hours=0)
  assert len(matched_0h) == 3  # Two duplicate timestamp ECGs in adm 10001 + one in adm 10002
  assert set(matched_0h["hadm_id"].to_list()) == {10001, 10002}

  # With pre_admit_hours=24:
  # Subject 2's ECG is now within the 24h pre-admission window
  matched_24h = link_records_to_admissions(sample_records, sample_admissions, pre_admit_hours=24)
  assert len(matched_24h) == 4
  assert 20001 in matched_24h["hadm_id"].to_list()


def test_select_index_ecgs_earliest(sample_records: pl.DataFrame) -> None:
  index_df = select_index_ecgs(sample_records, strategy="earliest")

  # Must have exactly 1 record per unique subject
  assert len(index_df) == sample_records["subject_id"].n_unique()
  assert index_df["subject_id"].n_unique() == len(index_df)

  # For subject 1 with duplicate timestamp (study_id 101 vs 102), tie-break selects 101
  p1_index = index_df.filter(pl.col("subject_id") == 1)
  assert p1_index["study_id"][0] == 101

  # For subject 2, earliest is study_id 201 in 2160
  p2_index = index_df.filter(pl.col("subject_id") == 2)
  assert p2_index["study_id"][0] == 201


def test_select_index_ecgs_latest(sample_records: pl.DataFrame) -> None:
  index_df = select_index_ecgs(sample_records, strategy="latest")

  assert len(index_df) == sample_records["subject_id"].n_unique()

  # For subject 1, latest is study_id 103 in 2152
  p1_index = index_df.filter(pl.col("subject_id") == 1)
  assert p1_index["study_id"][0] == 103

  # For subject 2, latest is study_id 202 in 2165
  p2_index = index_df.filter(pl.col("subject_id") == 2)
  assert p2_index["study_id"][0] == 202


def test_compute_linkage_statistics(
  sample_records: pl.DataFrame,
  sample_patients: pl.DataFrame,
  sample_admissions: pl.DataFrame,
) -> None:
  stats = compute_linkage_statistics(sample_records, sample_patients, sample_admissions)

  assert stats["total_ecgs"] == 7
  assert stats["unique_ecg_subjects"] == 4
  assert stats["unique_study_ids"] == 7
  assert stats["linked_patient_subjects"] == 3
  assert stats["linked_patient_subject_pct"] == 75.0
  assert stats["linked_ecgs"] == 6
  assert stats["adult_ecgs"] == 5
  assert stats["adult_patients"] == 3
  assert stats["index_adult_ecgs"] == 3
  assert stats["subjects_with_admissions"] == 2
  assert stats["ecgs_per_patient_min"] == 1


def test_generate_cohort_flow(
  sample_records: pl.DataFrame,
  sample_patients: pl.DataFrame,
) -> None:
  flow = generate_cohort_flow(sample_records, sample_patients)

  assert len(flow) == 4
  assert isinstance(flow[0], CohortStep)

  # Step 1: All records
  assert flow[0].step_name == "All MIMIC-IV-ECG Records"
  assert flow[0].record_count == 7
  assert flow[0].subject_count == 4

  # Step 2: Linked
  assert flow[1].step_name == "Linked to MIMIC-IV Patients"
  assert flow[1].record_count == 6
  assert flow[1].subject_count == 3
  assert flow[1].records_excluded == 1
  assert flow[1].subjects_excluded == 1

  # Step 3: Adult
  assert flow[2].step_name == "Adult Patients (Age >= 18)"
  assert flow[2].record_count == 5
  assert flow[2].subject_count == 3
  assert flow[2].records_excluded == 1
  assert flow[2].subjects_excluded == 0

  # Step 4: Index
  assert flow[3].step_name == "Earliest Index ECG per Patient"
  assert flow[3].record_count == 3
  assert flow[3].subject_count == 3
  assert flow[3].records_excluded == 2
  assert flow[3].subjects_excluded == 0
