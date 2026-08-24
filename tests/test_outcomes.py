"""Unit tests for Stage 6 Outcome Definition module."""

from datetime import date, datetime
import polars as pl
import pytest

from ecg_alignment.outcomes import (
  DEFAULT_HORIZONS,
  DEFAULT_HORIZON_1YR,
  DEFAULT_HORIZON_30D,
  DEFAULT_HORIZON_90D,
  FollowUpPolicy,
  OutcomeHorizon,
  compute_inhospital_mortality,
  compute_mortality_outcomes,
  compute_outcome_statistics,
  compute_patient_death_table,
  filter_valid_outcomes,
  generate_outcome_report_markdown,
)


@pytest.fixture
def synthetic_records_df() -> pl.DataFrame:
  """Synthetic ECG records for testing outcome temporal logic."""
  return pl.DataFrame(
    {
      "subject_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "study_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
      "ecg_datetime": [
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 1: dies at day 10
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 2: dies at exact 30 days (2150-01-31)
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 3: dies at day 31 (2150-02-01)
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 4: dies at exact 90 days (2150-04-01)
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 5: dies at day 100 (2150-04-11)
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 6: dies at day 365 (2151-01-01)
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 7: dies at day 500
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 8: alive (no death record)
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 9: death before ECG (2149-12-31) -> INVALID
        datetime(2150, 1, 1, 12, 0, 0),  # Patient 10: same day death (2150-01-01 18:00:00)
      ],
    }
  )


@pytest.fixture
def synthetic_patients_df() -> pl.DataFrame:
  """Synthetic MIMIC-IV patients DataFrame."""
  return pl.DataFrame(
    {
      "subject_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "dod": [
        "2150-01-11",  # 1: 10 days
        "2150-01-31",  # 2: 30 days
        "2150-02-01",  # 3: 31 days
        "2150-04-01",  # 4: 90 days
        "2150-04-11",  # 5: 100 days
        "2151-01-01",  # 6: 365 days
        "2151-05-16",  # 7: 500 days
        None,          # 8: alive
        "2149-12-31",  # 9: death before ECG
        "2150-01-01",  # 10: same day death
      ],
      "dod_date": [
        date(2150, 1, 11),
        date(2150, 1, 31),
        date(2150, 2, 1),
        date(2150, 4, 1),
        date(2150, 4, 11),
        date(2151, 1, 1),
        date(2151, 5, 16),
        None,
        date(2149, 12, 31),
        date(2150, 1, 1),
      ],
    }
  )


@pytest.fixture
def synthetic_admissions_df() -> pl.DataFrame:
  """Synthetic MIMIC-IV admissions DataFrame."""
  return pl.DataFrame(
    {
      "subject_id": [1, 2, 8, 10],
      "hadm_id": [2001, 2002, 2008, 2010],
      "admittime": [
        "2150-01-01 08:00:00",
        "2150-01-01 08:00:00",
        "2150-01-01 08:00:00",
        "2150-01-01 08:00:00",
      ],
      "dischtime": [
        "2150-01-15 18:00:00",
        "2150-02-05 18:00:00",
        "2150-01-10 12:00:00",
        "2150-01-01 18:00:00",
      ],
      "deathtime": [
        "2150-01-11 14:30:00",  # Subject 1 died in hospital
        None,                  # Subject 2 discharged alive (died later)
        None,                  # Subject 8 discharged alive
        "2150-01-01 18:00:00",  # Subject 10 died same day in hospital
      ],
      "hospital_expire_flag": [1, 0, 0, 1],
    }
  )


def test_compute_patient_death_table(
  synthetic_patients_df: pl.DataFrame,
  synthetic_admissions_df: pl.DataFrame,
) -> None:
  """Test unified death table creation."""
  table = compute_patient_death_table(synthetic_patients_df, synthetic_admissions_df)
  assert len(table) == 10
  assert "dod_date" in table.columns
  assert "first_hosp_death_dt" in table.columns
  assert "in_hosp_death_flag" in table.columns

  subj1 = table.filter(pl.col("subject_id") == 1).to_dicts()[0]
  assert subj1["in_hosp_death_flag"] == 1
  assert subj1["first_hosp_death_dt"] == datetime(2150, 1, 11, 14, 30, 0)

  subj8 = table.filter(pl.col("subject_id") == 8).to_dicts()[0]
  assert subj8["dod_date"] is None
  assert subj8["in_hosp_death_flag"] is None or subj8["in_hosp_death_flag"] == 0


def test_compute_mortality_outcomes_primary_30d(
  synthetic_records_df: pl.DataFrame,
  synthetic_patients_df: pl.DataFrame,
  synthetic_admissions_df: pl.DataFrame,
) -> None:
  """Test primary 30-day mortality endpoint classifications."""
  outcomes = compute_mortality_outcomes(
    synthetic_records_df,
    synthetic_patients_df,
    synthetic_admissions_df,
  )

  # Check invalid subject (death before ECG)
  subj9 = outcomes.filter(pl.col("subject_id") == 9).to_dicts()[0]
  assert subj9["is_valid"] is False
  assert subj9["exclusion_reason"] == "ECG recorded after death timestamp"
  assert subj9["mortality_30d"] is None

  # Check subject 1: died at 10 days -> 30d True
  subj1 = outcomes.filter(pl.col("subject_id") == 1).to_dicts()[0]
  assert subj1["is_valid"] is True
  assert subj1["mortality_30d"] is True

  # Check subject 2: died at 30 days exact boundary -> 30d True
  subj2 = outcomes.filter(pl.col("subject_id") == 2).to_dicts()[0]
  assert subj2["is_valid"] is True
  assert subj2["mortality_30d"] is True

  # Check subject 3: died at 31 days -> 30d False, 90d True
  subj3 = outcomes.filter(pl.col("subject_id") == 3).to_dicts()[0]
  assert subj3["is_valid"] is True
  assert subj3["mortality_30d"] is False
  assert subj3["mortality_90d"] is True

  # Check subject 8: alive -> 30d False, 90d False, 1yr False
  subj8 = outcomes.filter(pl.col("subject_id") == 8).to_dicts()[0]
  assert subj8["is_valid"] is True
  assert subj8["mortality_30d"] is False
  assert subj8["mortality_90d"] is False
  assert subj8["mortality_1yr"] is False

  # Check subject 10: same-day death -> 30d True
  subj10 = outcomes.filter(pl.col("subject_id") == 10).to_dicts()[0]
  assert subj10["is_valid"] is True
  assert subj10["mortality_30d"] is True


def test_compute_mortality_outcomes_all_horizons(
  synthetic_records_df: pl.DataFrame,
  synthetic_patients_df: pl.DataFrame,
  synthetic_admissions_df: pl.DataFrame,
) -> None:
  """Test 90-day and 1-year mortality horizon boundaries."""
  outcomes = compute_mortality_outcomes(
    synthetic_records_df,
    synthetic_patients_df,
    synthetic_admissions_df,
  )

  # Subj 4: 90 days exact -> 30d False, 90d True, 1yr True
  subj4 = outcomes.filter(pl.col("subject_id") == 4).to_dicts()[0]
  assert subj4["mortality_30d"] is False
  assert subj4["mortality_90d"] is True
  assert subj4["mortality_1yr"] is True

  # Subj 5: 100 days -> 30d False, 90d False, 1yr True
  subj5 = outcomes.filter(pl.col("subject_id") == 5).to_dicts()[0]
  assert subj5["mortality_30d"] is False
  assert subj5["mortality_90d"] is False
  assert subj5["mortality_1yr"] is True

  # Subj 6: 365 days -> 1yr True
  subj6 = outcomes.filter(pl.col("subject_id") == 6).to_dicts()[0]
  assert subj6["mortality_1yr"] is True

  # Subj 7: 500 days -> 1yr False
  subj7 = outcomes.filter(pl.col("subject_id") == 7).to_dicts()[0]
  assert subj7["mortality_1yr"] is False


def test_compute_inhospital_mortality(
  synthetic_records_df: pl.DataFrame,
  synthetic_admissions_df: pl.DataFrame,
) -> None:
  """Test in-hospital mortality linking."""
  in_hosp = compute_inhospital_mortality(synthetic_records_df, synthetic_admissions_df)

  # Subject 1 was admitted and died in hospital -> True
  subj1 = in_hosp.filter(pl.col("study_id") == 101).to_dicts()[0]
  assert subj1["in_hospital_mortality"] is True
  assert subj1["linked_hadm_id"] == 2001

  # Subject 2 was admitted and discharged alive -> False
  subj2 = in_hosp.filter(pl.col("study_id") == 102).to_dicts()[0]
  assert subj2["in_hospital_mortality"] is False

  # Subject 3 was not admitted -> None
  subj3 = in_hosp.filter(pl.col("study_id") == 103).to_dicts()[0]
  assert subj3["in_hospital_mortality"] is None
  assert subj3["linked_hadm_id"] is None


def test_follow_up_policy_activity_confirmed(
  synthetic_records_df: pl.DataFrame,
  synthetic_patients_df: pl.DataFrame,
  synthetic_admissions_df: pl.DataFrame,
) -> None:
  """Test activity-confirmed follow-up policy exclusion of short follow-up."""
  policy = FollowUpPolicy(
    require_minimum_followup=True,
    minimum_followup_days=30.0,
    censoring_strategy="activity_confirmed",
  )

  outcomes = compute_mortality_outcomes(
    synthetic_records_df,
    synthetic_patients_df,
    synthetic_admissions_df,
    follow_up_policy=policy,
  )

  # Subject 8 is alive with last discharge on 2150-01-10 (9 days follow-up < 30 days) -> Marked invalid under strict policy
  subj8 = outcomes.filter(pl.col("subject_id") == 8).to_dicts()[0]
  assert subj8["is_valid"] is False
  assert "Insufficient follow-up" in str(subj8["exclusion_reason"])


def test_filter_valid_outcomes(
  synthetic_records_df: pl.DataFrame,
  synthetic_patients_df: pl.DataFrame,
  synthetic_admissions_df: pl.DataFrame,
) -> None:
  """Test filtering of invalid outcomes."""
  outcomes = compute_mortality_outcomes(
    synthetic_records_df,
    synthetic_patients_df,
    synthetic_admissions_df,
  )
  assert len(outcomes) == 10
  valid = filter_valid_outcomes(outcomes)
  assert len(valid) == 9
  assert 9 not in valid["subject_id"].to_list()


def test_compute_outcome_statistics(
  synthetic_records_df: pl.DataFrame,
  synthetic_patients_df: pl.DataFrame,
  synthetic_admissions_df: pl.DataFrame,
) -> None:
  """Test outcome aggregation and summary dictionary."""
  outcomes = compute_mortality_outcomes(
    synthetic_records_df,
    synthetic_patients_df,
    synthetic_admissions_df,
  )
  stats = compute_outcome_statistics(outcomes)
  assert stats["total_records"] == 10
  assert stats["valid_records"] == 9
  assert stats["invalid_records"] == 1
  assert stats["30d_events"] == 3  # Subj 1, 2, 10
  assert stats["90d_events"] == 5  # Subj 1, 2, 3, 4, 10
  assert "in_hospital_events" in stats


def test_generate_outcome_report_markdown(
  synthetic_records_df: pl.DataFrame,
  synthetic_patients_df: pl.DataFrame,
  synthetic_admissions_df: pl.DataFrame,
) -> None:
  """Test Markdown report generation."""
  outcomes = compute_mortality_outcomes(
    synthetic_records_df,
    synthetic_patients_df,
    synthetic_admissions_df,
  )
  stats = compute_outcome_statistics(outcomes)
  md = generate_outcome_report_markdown(stats)
  assert "# MIMIC-IV Outcome Definition" in md
  assert "30-day All-Cause Mortality (Primary)" in md
  assert "Research Guardrail Compliance" in md
