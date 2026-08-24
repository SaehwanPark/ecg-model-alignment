"""Outcome definition and mortality endpoint ascertainment from MIMIC-IV."""

from dataclasses import dataclass
from typing import Any, Literal
import polars as pl

DEFAULT_HORIZON_30D: float = 30.0
DEFAULT_HORIZON_90D: float = 90.0
DEFAULT_HORIZON_1YR: float = 365.25


@dataclass(frozen=True)
class OutcomeHorizon:
  """Configuration for a time-to-event mortality endpoint horizon."""

  name: str
  days: float
  description: str = ""


DEFAULT_HORIZONS: tuple[OutcomeHorizon, ...] = (
  OutcomeHorizon(name="30d", days=DEFAULT_HORIZON_30D, description="30-day all-cause mortality"),
  OutcomeHorizon(name="90d", days=DEFAULT_HORIZON_90D, description="90-day all-cause mortality"),
  OutcomeHorizon(name="1yr", days=DEFAULT_HORIZON_1YR, description="1-year all-cause mortality"),
)


@dataclass(frozen=True)
class FollowUpPolicy:
  """Policy for handling patient follow-up and censoring."""

  require_minimum_followup: bool = False
  minimum_followup_days: float = 30.0
  censoring_strategy: Literal["registry", "activity_confirmed"] = "registry"


def compute_patient_death_table(
  patients_df: pl.DataFrame,
  admissions_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
  """Build a unified patient death summary combining state registry and hospital records.

  Args:
    patients_df: MIMIC-IV patients DataFrame with subject_id, dod/dod_date.
    admissions_df: Optional MIMIC-IV admissions DataFrame with hadm_id, deathtime,
      hospital_expire_flag.

  Returns:
    DataFrame with subject_id, dod_date, first_hosp_death_dt, in_hosp_death_flag,
    last_disch_dt.
  """
  pat = patients_df
  if "dod_date" not in pat.columns and "dod" in pat.columns:
    pat = pat.with_columns(pl.col("dod").str.to_date("%Y-%m-%d", strict=False).alias("dod_date"))

  base_cols = ["subject_id"]
  if "dod_date" in pat.columns:
    base_cols.append("dod_date")

  summary = pat.select(base_cols)

  if admissions_df is not None and len(admissions_df) > 0:
    adm = admissions_df
    if "death_dt" not in adm.columns and "deathtime" in adm.columns:
      adm = adm.with_columns(
        pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("death_dt")
      )
    if "disch_dt" not in adm.columns and "dischtime" in adm.columns:
      adm = adm.with_columns(
        pl.col("dischtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("disch_dt")
      )

    hosp_deaths = (
      adm.filter(pl.col("hospital_expire_flag") == 1)
      .group_by("subject_id")
      .agg(
        pl.col("death_dt").min().alias("first_hosp_death_dt"),
        pl.lit(1).cast(pl.Int64).alias("in_hosp_death_flag"),
      )
    )

    last_discharges = adm.group_by("subject_id").agg(
      pl.col("disch_dt").max().alias("last_disch_dt")
    )

    summary = summary.join(hosp_deaths, on="subject_id", how="left").join(
      last_discharges, on="subject_id", how="left"
    )

    if "first_hosp_death_dt" in summary.columns:
      if summary["first_hosp_death_dt"].dtype == pl.Null:
        summary = summary.with_columns(pl.col("first_hosp_death_dt").cast(pl.Datetime("us")))
    else:
      summary = summary.with_columns(
        pl.lit(None).cast(pl.Datetime("us")).alias("first_hosp_death_dt")
      )

    if "last_disch_dt" in summary.columns:
      if summary["last_disch_dt"].dtype == pl.Null:
        summary = summary.with_columns(pl.col("last_disch_dt").cast(pl.Datetime("us")))
    else:
      summary = summary.with_columns(
        pl.lit(None).cast(pl.Datetime("us")).alias("last_disch_dt")
      )

    if "in_hosp_death_flag" not in summary.columns:
      summary = summary.with_columns(pl.lit(0).cast(pl.Int64).alias("in_hosp_death_flag"))
    else:
      summary = summary.with_columns(pl.col("in_hosp_death_flag").fill_null(0).cast(pl.Int64))
  else:
    summary = summary.with_columns(
      pl.lit(None).cast(pl.Datetime("us")).alias("first_hosp_death_dt"),
      pl.lit(0).cast(pl.Int64).alias("in_hosp_death_flag"),
      pl.lit(None).cast(pl.Datetime("us")).alias("last_disch_dt"),
    )


  return summary


def compute_mortality_outcomes(
  records_df: pl.DataFrame,
  patients_df: pl.DataFrame,
  admissions_df: pl.DataFrame | None = None,
  horizons: tuple[OutcomeHorizon, ...] = DEFAULT_HORIZONS,
  follow_up_policy: FollowUpPolicy | None = None,
) -> pl.DataFrame:
  """Compute time-to-event and binary mortality outcomes relative to index ECG timestamp.

  Each patient's index ECG timestamp is the temporal origin (T_0 = ecg_datetime).
  Time-to-death is computed in days.
  - If a patient dies before the ECG (time < 0), the record is marked invalid (is_valid = False).
  - If a patient dies on or after ECG (time >= 0), mortality within horizon H is True if
    time <= H, else False.
  - If a patient has no death record (dod is null), mortality is False under registry policy,
    or assessed against activity under activity_confirmed policy.

  Args:
    records_df: DataFrame containing subject_id, ecg_datetime, and study_id.
    patients_df: MIMIC-IV patients DataFrame containing subject_id and dod/dod_date.
    admissions_df: Optional MIMIC-IV admissions DataFrame.
    horizons: Tuple of OutcomeHorizon configurations (default: 30d, 90d, 1yr).
    follow_up_policy: Censoring and follow-up validation rules.

  Returns:
    DataFrame with comprehensive outcome annotations, validity flags, and horizon indicators.
  """
  policy = follow_up_policy or FollowUpPolicy()

  records = records_df
  if "ecg_datetime" not in records.columns and "ecg_time" in records.columns:
    records = records.with_columns(
      pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ecg_datetime")
    )

  death_summary = compute_patient_death_table(patients_df, admissions_df)
  joined = records.join(death_summary, on="subject_id", how="left")

  # Compute days to death
  joined = joined.with_columns(
    pl.when(pl.col("first_hosp_death_dt").is_not_null())
    .then(
      (pl.col("first_hosp_death_dt") - pl.col("ecg_datetime")).dt.total_seconds() / 86400.0
    )
    .when(pl.col("dod_date").is_not_null())
    .then(
      (pl.col("dod_date") - pl.col("ecg_datetime").dt.date()).dt.total_days().cast(pl.Float64)
    )
    .otherwise(None)
    .alias("days_to_death")
  )

  # Compute follow-up activity duration (from last discharge or ECG)
  joined = joined.with_columns(
    pl.when(pl.col("last_disch_dt").is_not_null())
    .then((pl.col("last_disch_dt") - pl.col("ecg_datetime")).dt.total_seconds() / 86400.0)
    .otherwise(0.0)
    .alias("activity_followup_days")
  )

  # Identify valid vs invalid records (e.g. ECG after recorded death)
  joined = joined.with_columns(
    pl.when(pl.col("days_to_death").is_not_null() & (pl.col("days_to_death") < 0.0))
    .then(pl.lit(False))
    .when(
      pl.lit(policy.require_minimum_followup)
      & pl.col("dod_date").is_null()
      & (pl.col("activity_followup_days") < policy.minimum_followup_days)
    )
    .then(pl.lit(False))
    .otherwise(pl.lit(True))
    .alias("is_valid"),
    pl.when(pl.col("days_to_death").is_not_null() & (pl.col("days_to_death") < 0.0))
    .then(pl.lit("ECG recorded after death timestamp"))
    .when(
      pl.lit(policy.require_minimum_followup)
      & pl.col("dod_date").is_null()
      & (pl.col("activity_followup_days") < policy.minimum_followup_days)
    )
    .then(pl.lit(f"Insufficient follow-up activity (< {policy.minimum_followup_days} days)"))
    .otherwise(None)
    .alias("exclusion_reason"),
  )

  # Add horizon columns
  exprs: list[pl.Expr] = []
  for horizon in horizons:
    col_name = f"mortality_{horizon.name}"
    time_col = f"time_to_{horizon.name}_days"

    exprs.append(
      pl.when(~pl.col("is_valid"))
      .then(None)
      .when(
        pl.col("days_to_death").is_not_null()
        & (pl.col("days_to_death") >= 0.0)
        & (pl.col("days_to_death") <= horizon.days)
      )
      .then(pl.lit(True))
      .otherwise(pl.lit(False))
      .alias(col_name)
    )

    exprs.append(
      pl.when(~pl.col("is_valid"))
      .then(None)
      .when(
        pl.col("days_to_death").is_not_null()
        & (pl.col("days_to_death") >= 0.0)
        & (pl.col("days_to_death") <= horizon.days)
      )
      .then(pl.col("days_to_death"))
      .otherwise(pl.lit(horizon.days))
      .alias(time_col)
    )

  joined = joined.with_columns(exprs)

  # In-hospital mortality calculation
  if admissions_df is not None and len(admissions_df) > 0:
    in_hosp = compute_inhospital_mortality(records, admissions_df)
    joined = joined.join(
      in_hosp.select(["study_id", "in_hospital_mortality", "linked_hadm_id"]),
      on="study_id",
      how="left",
    )
  else:
    joined = joined.with_columns(
      pl.lit(None).cast(pl.Boolean).alias("in_hospital_mortality"),
      pl.lit(None).cast(pl.Int64).alias("linked_hadm_id"),
    )

  return joined


def compute_inhospital_mortality(
  records_df: pl.DataFrame,
  admissions_df: pl.DataFrame,
  pre_admit_hours: int = 0,
) -> pl.DataFrame:
  """Compute in-hospital mortality for ECG records linked to hospital admissions.

  An ECG is linked to an admission if:
    ecg_datetime >= admit_dt - pre_admit_hours AND ecg_datetime <= disch_dt.

  In-hospital mortality is:
  - True if the linked admission ended in death (hospital_expire_flag == 1) and
    the ECG occurred on or before death.
  - False if the linked admission ended in alive discharge (hospital_expire_flag == 0).
  - None if the ECG was not linked to any admission.

  Args:
    records_df: ECG records DataFrame with subject_id, study_id, ecg_datetime.
    admissions_df: Admissions DataFrame with hadm_id, admit_dt, disch_dt,
      hospital_expire_flag, death_dt.
    pre_admit_hours: Window before formal admit time to include (default: 0).

  Returns:
    DataFrame with study_id, linked_hadm_id, in_hospital_mortality.
  """
  records = records_df
  if "ecg_datetime" not in records.columns and "ecg_time" in records.columns:
    records = records.with_columns(
      pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ecg_datetime")
    )

  adm = admissions_df
  if "admit_dt" not in adm.columns and "admittime" in adm.columns:
    adm = adm.with_columns(
      pl.col("admittime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("admit_dt")
    )
  if "disch_dt" not in adm.columns and "dischtime" in adm.columns:
    adm = adm.with_columns(
      pl.col("dischtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("disch_dt")
    )
  if "death_dt" not in adm.columns and "deathtime" in adm.columns:
    adm = adm.with_columns(
      pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("death_dt")
    )

  start_col = (
    pl.col("admit_dt") - pl.duration(hours=pre_admit_hours)
    if pre_admit_hours > 0
    else pl.col("admit_dt")
  )

  linked = records.join(adm, on="subject_id", how="inner").filter(
    (pl.col("ecg_datetime") >= start_col) & (pl.col("ecg_datetime") <= pl.col("disch_dt"))
  )

  linked = linked.with_columns(
    pl.when(
      (pl.col("hospital_expire_flag") == 1)
      & (
        pl.col("death_dt").is_null()
        | (pl.col("ecg_datetime") <= pl.col("death_dt"))
      )
    )
    .then(pl.lit(True))
    .when(pl.col("hospital_expire_flag") == 0)
    .then(pl.lit(False))
    .otherwise(pl.lit(False))
    .alias("in_hospital_mortality")
  )

  deduped = (
    linked.sort(by=["study_id", "admit_dt"])
    .group_by("study_id", maintain_order=True)
    .first()
    .select(
      pl.col("study_id"),
      pl.col("hadm_id").alias("linked_hadm_id"),
      pl.col("in_hospital_mortality"),
    )
  )

  result = records.select(["study_id"]).join(deduped, on="study_id", how="left")
  return result


def filter_valid_outcomes(outcomes_df: pl.DataFrame) -> pl.DataFrame:
  """Filter outcomes DataFrame to valid observations only.

  Args:
    outcomes_df: Outcomes DataFrame containing is_valid column.

  Returns:
    DataFrame with invalid observations removed.
  """
  return outcomes_df.filter(pl.col("is_valid"))


def compute_outcome_statistics(
  outcomes_df: pl.DataFrame,
  horizons: tuple[OutcomeHorizon, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
  """Compute comprehensive aggregate statistics for mortality outcomes.

  Args:
    outcomes_df: DataFrame output from compute_mortality_outcomes.
    horizons: Horizions evaluated.

  Returns:
    Dictionary containing non-sensitive aggregate counts and event rates.
  """
  total_n = len(outcomes_df)
  valid_df = filter_valid_outcomes(outcomes_df)
  valid_n = len(valid_df)
  invalid_n = total_n - valid_n

  stats: dict[str, Any] = {
    "total_records": total_n,
    "valid_records": valid_n,
    "invalid_records": invalid_n,
    "invalid_pct": (invalid_n / total_n * 100.0) if total_n > 0 else 0.0,
    "total_deaths_all_time": (
      valid_df.filter(pl.col("days_to_death").is_not_null()).shape[0] if valid_n > 0 else 0
    ),
  }

  for horizon in horizons:
    col_name = f"mortality_{horizon.name}"
    if col_name in valid_df.columns:
      events = valid_df.filter(pl.col(col_name) == True).shape[0] # noqa: E712
      rate = (events / valid_n * 100.0) if valid_n > 0 else 0.0
      stats[f"{horizon.name}_events"] = events
      stats[f"{horizon.name}_event_rate_pct"] = rate

  if "in_hospital_mortality" in valid_df.columns:
    in_hosp_evaluable = valid_df.filter(pl.col("in_hospital_mortality").is_not_null())
    evaluable_n = len(in_hosp_evaluable)
    in_hosp_events = in_hosp_evaluable.filter(pl.col("in_hospital_mortality") == True).shape[0] # noqa: E712
    stats["in_hospital_evaluable_n"] = evaluable_n
    stats["in_hospital_events"] = in_hosp_events
    stats["in_hospital_event_rate_pct"] = (
      (in_hosp_events / evaluable_n * 100.0) if evaluable_n > 0 else 0.0
    )

  return stats


def generate_outcome_report_markdown(
  stats: dict[str, Any],
  title: str = "MIMIC-IV Outcome Definition and Mortality Validation Report",
) -> str:
  """Generate a structured Markdown report summarizing outcome ascertainment.

  Args:
    stats: Statistics dictionary from compute_outcome_statistics.
    title: Report title.

  Returns:
    Formatted Markdown string with summary tables.
  """
  lines = [
    f"# {title}",
    "",
    "## 1. Overview and Primary Endpoint",
    "",
    "This report summarizes outcome ascertainment for the index-ECG cohort.",
    "The primary prognostic endpoint is **30-day all-cause mortality** after the index ECG timestamp.",
    "Secondary endpoints include **90-day mortality**, **1-year mortality**, and **in-hospital mortality**.",
    "",
    "## 2. Mortality Event Summary",
    "",
    "| Endpoint | Cohort N | Events (N) | Event Rate (%) |",
    "| :--- | :--- | :--- | :--- |",
    f"| **30-day All-Cause Mortality (Primary)** | {stats.get('valid_records', 0):,} | {stats.get('30d_events', 0):,} | {stats.get('30d_event_rate_pct', 0.0):.2f}% |",
    f"| **90-day All-Cause Mortality** | {stats.get('valid_records', 0):,} | {stats.get('90d_events', 0):,} | {stats.get('90d_event_rate_pct', 0.0):.2f}% |",
    f"| **1-year All-Cause Mortality** | {stats.get('valid_records', 0):,} | {stats.get('1yr_events', 0):,} | {stats.get('1yr_event_rate_pct', 0.0):.2f}% |",
    f"| **In-Hospital Mortality** | {stats.get('in_hospital_evaluable_n', 0):,} | {stats.get('in_hospital_events', 0):,} | {stats.get('in_hospital_event_rate_pct', 0.0):.2f}% |",
    "",
    "## 3. Data Integrity and Exclusion Summary",
    "",
    f"- **Total Candidate Records Evaluated:** {stats.get('total_records', 0):,}",
    f"- **Valid Analysis Records:** {stats.get('valid_records', 0):,}",
    f"- **Excluded Records (e.g. ECG recorded after death):** {stats.get('invalid_records', 0):,} ({stats.get('invalid_pct', 0.0):.4f}%)",
    f"- **All-Time Observed Deaths:** {stats.get('total_deaths_all_time', 0):,}",
    "",
    "## 4. Research Guardrail Compliance",
    "",
    "- All outcomes are derived solely from MIMIC-IV clinical and administrative timestamps.",
    "- No clinical outcome or demographic feature enters predictor models `A` (CIIS) or `B` (Transformer embeddings).",
    "- Exact temporal origin $T_0$ is defined as the index ECG acquisition timestamp.",
    "",
  ]
  return "\n".join(lines)
