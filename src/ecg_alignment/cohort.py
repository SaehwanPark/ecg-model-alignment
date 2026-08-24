"""Cohort construction, linkage, and index ECG selection logic."""

from dataclasses import dataclass
from typing import Any, Literal
import polars as pl


@dataclass(frozen=True)
class CohortStep:
  """Summary step in cohort derivation flow."""

  step_name: str
  description: str
  record_count: int
  subject_count: int
  records_excluded: int
  subjects_excluded: int


def compute_age_at_ecg(
  records_df: pl.DataFrame,
  patients_df: pl.DataFrame,
) -> pl.DataFrame:
  """Join ECG records with MIMIC-IV patients and compute patient age at ECG time.

  In MIMIC-IV, anchor_age is the patient's age in anchor_year.
  Age at ECG is calculated as: anchor_age + (ecg_year - anchor_year).

  Args:
    records_df: DataFrame with subject_id and ecg_year (or ecg_datetime).
    patients_df: DataFrame with subject_id, anchor_age, anchor_year, etc.

  Returns:
    Joined DataFrame including age_at_ecg column.
  """
  df = records_df
  if "ecg_year" not in df.columns and "ecg_datetime" in df.columns:
    df = df.with_columns(pl.col("ecg_datetime").dt.year().alias("ecg_year"))

  patient_cols = [
    c
    for c in [
      "subject_id",
      "gender",
      "anchor_age",
      "anchor_year",
      "anchor_year_group",
      "dod",
      "dod_date",
    ]
    if c in patients_df.columns
  ]
  patients_sub = patients_df.select(patient_cols)

  joined = df.join(patients_sub, on="subject_id", how="inner")
  return joined.with_columns(
    (pl.col("anchor_age") + (pl.col("ecg_year") - pl.col("anchor_year"))).alias("age_at_ecg")
  )


def filter_adult_records(
  df: pl.DataFrame,
  min_age: int = 18,
) -> pl.DataFrame:
  """Filter records to adult patients (age_at_ecg >= min_age).

  Args:
    df: DataFrame containing age_at_ecg column.
    min_age: Minimum age threshold (inclusive). Defaults to 18.

  Returns:
    Filtered DataFrame.
  """
  if "age_at_ecg" not in df.columns:
    raise ValueError("DataFrame must contain 'age_at_ecg' column to filter adult records.")
  return df.filter(pl.col("age_at_ecg") >= min_age)


def link_records_to_admissions(
  records_df: pl.DataFrame,
  admissions_df: pl.DataFrame,
  pre_admit_hours: int = 0,
) -> pl.DataFrame:
  """Link ECG records to hospital admissions based on temporal encounter windows.

  An ECG is linked to an admission if:
    ecg_datetime >= admit_dt - pre_admit_hours AND ecg_datetime <= disch_dt

  Args:
    records_df: DataFrame with subject_id and ecg_datetime.
    admissions_df: DataFrame with subject_id, hadm_id, admit_dt, disch_dt.
    pre_admit_hours: Number of hours before formal admit time to include (e.g. for ED presentation).

  Returns:
    DataFrame of matched ECG-admission pairs with admission metadata.
  """
  joined = records_df.join(admissions_df, on="subject_id", how="inner")

  start_col = (
    pl.col("admit_dt") - pl.duration(hours=pre_admit_hours)
    if pre_admit_hours > 0
    else pl.col("admit_dt")
  )

  return joined.filter(
    (pl.col("ecg_datetime") >= start_col) & (pl.col("ecg_datetime") <= pl.col("disch_dt"))
  )


def select_index_ecgs(
  records_df: pl.DataFrame,
  strategy: Literal["earliest", "latest"] = "earliest",
) -> pl.DataFrame:
  """Select a single index ECG per patient using a deterministic temporal rule.

  Ties are broken deterministically by study_id.

  Args:
    records_df: DataFrame containing subject_id, ecg_datetime, and study_id.
    strategy: "earliest" (default) or "latest".

  Returns:
    DataFrame with exactly one index ECG row per unique subject_id.
  """
  descending_order = strategy == "latest"

  sorted_df = records_df.sort(
    by=["subject_id", "ecg_datetime", "study_id"],
    descending=[False, descending_order, descending_order],
  )

  return sorted_df.group_by("subject_id", maintain_order=True).first()


def compute_linkage_statistics(
  records_df: pl.DataFrame,
  patients_df: pl.DataFrame,
  admissions_df: pl.DataFrame,
) -> dict[str, Any]:
  """Compute comprehensive aggregate linkage statistics across datasets.

  Args:
    records_df: ECG records DataFrame.
    patients_df: MIMIC-IV patients DataFrame.
    admissions_df: MIMIC-IV admissions DataFrame.

  Returns:
    Dictionary containing non-sensitive aggregate counts and linkage metrics.
  """
  total_ecgs = len(records_df)
  unique_ecg_subjs = records_df["subject_id"].n_unique()
  unique_study_ids = records_df["study_id"].n_unique()

  # Patients linkage
  ecg_subjs_set = set(records_df["subject_id"].unique().to_list())
  pat_subjs_set = set(patients_df["subject_id"].unique().to_list())
  linked_pat_subjs = ecg_subjs_set.intersection(pat_subjs_set)

  records_with_patients = compute_age_at_ecg(records_df, patients_df)
  linked_ecg_count = len(records_with_patients)

  # Adult ECGs
  adult_records = filter_adult_records(records_with_patients, min_age=18)
  adult_ecg_count = len(adult_records)
  adult_patient_count = adult_records["subject_id"].n_unique()

  # Index ECGs
  index_ecgs = select_index_ecgs(adult_records, strategy="earliest")
  index_ecg_count = len(index_ecgs)

  # Admissions linkage
  adm_subjs_set = set(admissions_df["subject_id"].unique().to_list())
  linked_adm_subjs = ecg_subjs_set.intersection(adm_subjs_set)

  in_adm_ecgs = link_records_to_admissions(records_df, admissions_df, pre_admit_hours=0)
  in_adm_24h_ecgs = link_records_to_admissions(records_df, admissions_df, pre_admit_hours=24)

  # ECGs per patient distribution
  counts_per_subj = records_df.group_by("subject_id").len()["len"]
  mean_val = counts_per_subj.mean()
  med_val = counts_per_subj.median()
  min_val = counts_per_subj.min()
  max_val = counts_per_subj.max()

  return {
    "total_ecgs": total_ecgs,
    "unique_ecg_subjects": unique_ecg_subjs,
    "unique_study_ids": unique_study_ids,
    "linked_patient_subjects": len(linked_pat_subjs),
    "linked_patient_subject_pct": len(linked_pat_subjs) / unique_ecg_subjs * 100.0,
    "linked_ecgs": linked_ecg_count,
    "linked_ecg_pct": linked_ecg_count / total_ecgs * 100.0,
    "adult_ecgs": adult_ecg_count,
    "adult_patients": adult_patient_count,
    "index_adult_ecgs": index_ecg_count,
    "subjects_with_admissions": len(linked_adm_subjs),
    "subjects_with_admissions_pct": len(linked_adm_subjs) / unique_ecg_subjs * 100.0,
    "ecgs_during_admission": in_adm_ecgs["study_id"].n_unique(),
    "ecgs_during_admission_or_24h_pre": in_adm_24h_ecgs["study_id"].n_unique(),
    "ecgs_per_patient_mean": float(mean_val) if isinstance(mean_val, (int, float)) else 0.0,
    "ecgs_per_patient_median": float(med_val) if isinstance(med_val, (int, float)) else 0.0,
    "ecgs_per_patient_min": int(min_val) if isinstance(min_val, (int, float)) else 0,
    "ecgs_per_patient_max": int(max_val) if isinstance(max_val, (int, float)) else 0,
  }


def generate_cohort_flow(
  records_df: pl.DataFrame,
  patients_df: pl.DataFrame,
) -> list[CohortStep]:
  """Generate step-by-step cohort flow attrition data.

  Args:
    records_df: Raw ECG records DataFrame.
    patients_df: MIMIC-IV patients DataFrame.

  Returns:
    List of CohortStep objects representing sequential inclusion/exclusion.
  """
  steps: list[CohortStep] = []

  # Step 1: All MIMIC-IV-ECG records
  total_records = len(records_df)
  total_subjs = records_df["subject_id"].n_unique()
  steps.append(
    CohortStep(
      step_name="All MIMIC-IV-ECG Records",
      description="All 12-lead ECG records present in MIMIC-IV-ECG v1.0",
      record_count=total_records,
      subject_count=total_subjs,
      records_excluded=0,
      subjects_excluded=0,
    )
  )

  # Step 2: Linked to MIMIC-IV Subject
  linked = compute_age_at_ecg(records_df, patients_df)
  linked_records = len(linked)
  linked_subjs = linked["subject_id"].n_unique()
  steps.append(
    CohortStep(
      step_name="Linked to MIMIC-IV Patients",
      description="ECG records linkable to MIMIC-IV v3.1 subject identifier",
      record_count=linked_records,
      subject_count=linked_subjs,
      records_excluded=total_records - linked_records,
      subjects_excluded=total_subjs - linked_subjs,
    )
  )

  # Step 3: Adult Patients (Age >= 18 at ECG)
  adults = filter_adult_records(linked, min_age=18)
  adult_records = len(adults)
  adult_subjs = adults["subject_id"].n_unique()
  steps.append(
    CohortStep(
      step_name="Adult Patients (Age >= 18)",
      description="ECG records for patients aged 18 years or older at time of ECG",
      record_count=adult_records,
      subject_count=adult_subjs,
      records_excluded=linked_records - adult_records,
      subjects_excluded=linked_subjs - adult_subjs,
    )
  )

  # Step 4: Earliest Index ECG per Unique Patient
  index_ecgs = select_index_ecgs(adults, strategy="earliest")
  index_records = len(index_ecgs)
  index_subjs = index_ecgs["subject_id"].n_unique()
  steps.append(
    CohortStep(
      step_name="Earliest Index ECG per Patient",
      description="Single earliest eligible ECG selected per unique patient",
      record_count=index_records,
      subject_count=index_subjs,
      records_excluded=adult_records - index_records,
      subjects_excluded=adult_subjs - index_subjs,
    )
  )

  return steps
