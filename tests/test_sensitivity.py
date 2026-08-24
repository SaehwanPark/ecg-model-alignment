"""Tests for Stage 10 sensitivity and robustness analyses."""

import numpy as np
import polars as pl
import pytest

from ecg_alignment.scoring.traditional import (
  CIISMeasurements,
  compute_cornell_voltage,
  compute_sokolow_lyon_voltage,
  score_simplified_ischemic_ecg,
)
from ecg_alignment.sensitivity import (
  AlternativeTraditionalSensitivityResult,
  CohortAnchoringSensitivityResult,
  DemographicSubgroupSensitivityResult,
  FullSensitivityAnalysisResult,
  OutcomeHorizonSensitivityResult,
  ProbeArchitectureSensitivityResult,
  QualityFilterSensitivityResult,
  SecondaryTransformerSensitivityResult,
  evaluate_alternative_traditional_models,
  evaluate_cohort_index_sensitivity,
  evaluate_demographic_subgroups,
  evaluate_outcome_horizons,
  evaluate_probe_sensitivity,
  evaluate_quality_filter_sensitivity,
  evaluate_secondary_transformer,
  fit_custom_linear_probe,
  generate_sensitivity_report_markdown,
  run_sensitivity_analyses,
)


@pytest.fixture
def synthetic_sensitivity_data() -> dict[str, pl.DataFrame | np.ndarray]:
  """Create reproducible synthetic dataset for sensitivity testing."""
  rng = np.random.default_rng(42)
  n_test = 200
  n_dev = 400

  # Dev set
  dev_emb = rng.normal(size=(n_dev, 32))
  dev_y = (rng.uniform(size=n_dev) < 0.15).astype(np.int64)
  if np.sum(dev_y) == 0:
    dev_y[0] = 1

  # Test set
  test_emb = rng.normal(size=(n_test, 32))
  test_y_30d = (rng.uniform(size=n_test) < 0.10).astype(np.int64)
  if np.sum(test_y_30d) == 0:
    test_y_30d[0] = 1

  test_y_inhosp = (test_y_30d * (rng.uniform(size=n_test) < 0.7)).astype(np.int64)
  test_y_90d = np.clip(test_y_30d + (rng.uniform(size=n_test) < 0.05).astype(np.int64), 0, 1)
  test_y_1yr = np.clip(test_y_90d + (rng.uniform(size=n_test) < 0.05).astype(np.int64), 0, 1)

  # Scores
  a_scores = rng.normal(loc=12.0, scale=4.0, size=n_test)
  b_scores = 1.0 / (1.0 + np.exp(-rng.normal(loc=-2.0, scale=1.0, size=n_test)))

  subjs = [f"subj_{i:04d}" for i in range(n_test)]
  studies = [f"study_{i:04d}" for i in range(n_test)]

  earliest_df = pl.DataFrame({
    "subject_id": subjs,
    "study_id": studies,
    "split": ["test"] * n_test,
    "model_a_score": a_scores,
    "model_a_category": ["borderline"] * n_test,
    "model_a_valid": [True] * n_test,
    "model_b_score": b_scores,
    "model_b_log_odds": np.log(np.clip(b_scores / (1 - b_scores), 1e-10, 1e10)),
    "model_b_valid": [True] * n_test,
    "mortality_30d": test_y_30d,
    "inhospital_mortality": test_y_inhosp,
    "mortality_90d": test_y_90d,
    "mortality_1yr": test_y_1yr,
  })

  # Admission-anchored cohort (slightly smaller or shifted)
  adm_n = 150
  adm_y_30d = (rng.uniform(size=adm_n) < 0.12).astype(np.int64)
  if np.sum(adm_y_30d) == 0:
    adm_y_30d[0] = 1

  admission_df = pl.DataFrame({
    "subject_id": [f"adm_subj_{i:04d}" for i in range(adm_n)],
    "study_id": [f"adm_study_{i:04d}" for i in range(adm_n)],
    "split": ["test"] * adm_n,
    "model_a_score": rng.normal(loc=13.0, scale=4.0, size=adm_n),
    "model_a_category": ["borderline"] * adm_n,
    "model_a_valid": [True] * adm_n,
    "model_b_score": 1.0 / (1.0 + np.exp(-rng.normal(loc=-1.8, scale=1.0, size=adm_n))),
    "model_b_log_odds": rng.normal(loc=-1.8, scale=1.0, size=adm_n),
    "model_b_valid": [True] * adm_n,
    "mortality_30d": adm_y_30d,
  })

  # Demographic strata dataframe
  strata_df = pl.DataFrame({
    "subject_id": subjs,
    "age_group": ["<65" if i % 2 == 0 else ">=65" for i in range(n_test)],
    "gender": ["F" if i % 3 == 0 else "M" for i in range(n_test)],
  })

  return {
    "earliest_df": earliest_df,
    "admission_df": admission_df,
    "strata_df": strata_df,
    "dev_emb": dev_emb,
    "dev_y": dev_y,
    "test_emb": test_emb,
    "test_y": test_y_30d,
  }


def test_evaluate_cohort_index_sensitivity(
  synthetic_sensitivity_data: dict[str, pl.DataFrame | np.ndarray],
) -> None:
  earliest_df = synthetic_sensitivity_data["earliest_df"]
  admission_df = synthetic_sensitivity_data["admission_df"]
  assert isinstance(earliest_df, pl.DataFrame)
  assert isinstance(admission_df, pl.DataFrame)

  res = evaluate_cohort_index_sensitivity(
    earliest_test_df=earliest_df,
    admission_test_df=admission_df,
    n_bootstraps=50,
    seed=42,
  )

  assert isinstance(res, CohortAnchoringSensitivityResult)
  assert res.earliest_n == len(earliest_df)
  assert res.admission_n == len(admission_df)
  assert 0.0 <= res.earliest_model_a_auroc.point_estimate <= 1.0
  assert 0.0 <= res.admission_model_b_auroc.point_estimate <= 1.0
  assert -1.0 <= res.earliest_spearman_rho <= 1.0


def test_evaluate_outcome_horizons(
  synthetic_sensitivity_data: dict[str, pl.DataFrame | np.ndarray],
) -> None:
  earliest_df = synthetic_sensitivity_data["earliest_df"]
  assert isinstance(earliest_df, pl.DataFrame)

  horizons = evaluate_outcome_horizons(
    test_df=earliest_df,
    n_bootstraps=50,
    seed=42,
  )

  assert len(horizons) == 4
  for h in horizons:
    assert isinstance(h, OutcomeHorizonSensitivityResult)
    assert h.n_total == len(earliest_df)
    assert 0.0 <= h.model_a_auroc.point_estimate <= 1.0
    assert 0.0 <= h.model_b_auroc.point_estimate <= 1.0
    assert h.incremental_lrt_stat >= 0.0
    assert 0.0 <= h.incremental_pvalue <= 1.0


def test_fit_custom_linear_probe_and_probe_sensitivity(
  synthetic_sensitivity_data: dict[str, pl.DataFrame | np.ndarray],
) -> None:
  dev_emb = synthetic_sensitivity_data["dev_emb"]
  dev_y = synthetic_sensitivity_data["dev_y"]
  test_emb = synthetic_sensitivity_data["test_emb"]
  test_y = synthetic_sensitivity_data["test_y"]
  earliest_df = synthetic_sensitivity_data["earliest_df"]

  assert isinstance(dev_emb, np.ndarray)
  assert isinstance(dev_y, np.ndarray)
  assert isinstance(test_emb, np.ndarray)
  assert isinstance(test_y, np.ndarray)
  assert isinstance(earliest_df, pl.DataFrame)

  # Test Elastic-Net probe
  coef_en, int_en, mean_en, scale_en = fit_custom_linear_probe(
    dev_emb,
    dev_y,
    penalty="elasticnet",
    c_value=0.1,
    l1_ratio=0.5,
    solver="saga",
    standardize=True,
    random_state=42,
  )
  assert len(coef_en) == dev_emb.shape[1]
  assert mean_en is not None
  assert scale_en is not None

  # Test L1 probe
  coef_l1, int_l1, _, _ = fit_custom_linear_probe(
    dev_emb,
    dev_y,
    penalty="l1",
    c_value=0.1,
    solver="saga",
    standardize=True,
    random_state=42,
  )
  assert len(coef_l1) == dev_emb.shape[1]

  primary_scores = earliest_df["model_b_score"].to_numpy().astype(np.float64)
  probe_res = evaluate_probe_sensitivity(
    dev_embeddings=dev_emb,
    dev_labels=dev_y,
    test_embeddings=test_emb,
    test_labels=test_y,
    primary_test_scores=primary_scores,
    n_bootstraps=50,
    seed=42,
  )

  assert len(probe_res) == 6
  for p in probe_res:
    assert isinstance(p, ProbeArchitectureSensitivityResult)
    assert 0.0 <= p.test_auroc.point_estimate <= 1.0
    assert -1.0 <= p.rank_correlation_with_primary <= 1.0


def test_evaluate_quality_filter_sensitivity(
  synthetic_sensitivity_data: dict[str, pl.DataFrame | np.ndarray],
) -> None:
  earliest_df = synthetic_sensitivity_data["earliest_df"]
  assert isinstance(earliest_df, pl.DataFrame)

  mask = [i % 4 != 0 for i in range(len(earliest_df))]
  q_res = evaluate_quality_filter_sensitivity(
    test_df=earliest_df,
    high_quality_mask=mask,
    n_bootstraps=50,
    seed=42,
  )

  assert isinstance(q_res, QualityFilterSensitivityResult)
  assert q_res.n_total == sum(mask)
  assert 0.0 <= q_res.model_a_auroc.point_estimate <= 1.0
  assert 0.0 <= q_res.model_b_auroc.point_estimate <= 1.0


def test_evaluate_alternative_traditional_models(
  synthetic_sensitivity_data: dict[str, pl.DataFrame | np.ndarray],
) -> None:
  earliest_df = synthetic_sensitivity_data["earliest_df"]
  assert isinstance(earliest_df, pl.DataFrame)

  rng = np.random.default_rng(42)
  alt_scores = {
    "Cornell Voltage": rng.uniform(0.5, 3.5, size=len(earliest_df)),
    "Sokolow-Lyon Voltage": rng.uniform(1.0, 4.5, size=len(earliest_df)),
    "Simplified Ischemic Score": rng.uniform(0.0, 20.0, size=len(earliest_df)),
  }

  trad_res = evaluate_alternative_traditional_models(
    test_df=earliest_df,
    alternative_scores=alt_scores,
    n_bootstraps=50,
    seed=42,
  )

  assert len(trad_res) == 4  # Baseline CIIS + 3 alternatives
  for t in trad_res:
    assert isinstance(t, AlternativeTraditionalSensitivityResult)
    assert 0.0 <= t.traditional_auroc.point_estimate <= 1.0
    assert -1.0 <= t.spearman_with_model_b <= 1.0


def test_evaluate_secondary_transformer(
  synthetic_sensitivity_data: dict[str, pl.DataFrame | np.ndarray],
) -> None:
  dev_y = synthetic_sensitivity_data["dev_y"]
  test_y = synthetic_sensitivity_data["test_y"]
  earliest_df = synthetic_sensitivity_data["earliest_df"]

  assert isinstance(dev_y, np.ndarray)
  assert isinstance(test_y, np.ndarray)
  assert isinstance(earliest_df, pl.DataFrame)

  rng = np.random.default_rng(42)
  cards_dev = rng.normal(size=(len(dev_y), 64))
  cards_test = rng.normal(size=(len(test_y), 64))
  ciis_scores = earliest_df["model_a_score"].to_numpy().astype(np.float64)
  dbeta_scores = earliest_df["model_b_score"].to_numpy().astype(np.float64)

  sec_res = evaluate_secondary_transformer(
    cards_clip_dev_embeddings=cards_dev,
    dev_labels=dev_y,
    cards_clip_test_embeddings=cards_test,
    test_labels=test_y,
    ciis_test_scores=ciis_scores,
    dbeta_test_scores=dbeta_scores,
    n_bootstraps=50,
    seed=42,
  )

  assert isinstance(sec_res, SecondaryTransformerSensitivityResult)
  assert sec_res.embedding_dim == 64
  assert 0.0 <= sec_res.test_auroc.point_estimate <= 1.0
  assert -1.0 <= sec_res.spearman_with_dbeta <= 1.0


def test_evaluate_demographic_subgroups_and_firewall(
  synthetic_sensitivity_data: dict[str, pl.DataFrame | np.ndarray],
) -> None:
  earliest_df = synthetic_sensitivity_data["earliest_df"]
  strata_df = synthetic_sensitivity_data["strata_df"]

  assert isinstance(earliest_df, pl.DataFrame)
  assert isinstance(strata_df, pl.DataFrame)

  demo_res = evaluate_demographic_subgroups(
    test_predictions_df=earliest_df,
    evaluation_strata_df=strata_df,
    n_bootstraps=50,
    seed=42,
  )

  assert len(demo_res) == 4  # Age <65, Age >=65, Female, Male
  for d in demo_res:
    assert isinstance(d, DemographicSubgroupSensitivityResult)
    assert d.n_total > 0
    assert 0.0 <= d.model_a_auroc.point_estimate <= 1.0
    assert 0.0 <= d.model_b_auroc.point_estimate <= 1.0


def test_traditional_alternative_score_computation() -> None:
  # Synthetic 12-lead ECG waveform (500 Hz, 2.5s, 12 leads)
  fs = 500
  duration = 2.5
  n_samples = int(fs * duration)
  t = np.linspace(0, duration, n_samples)
  waveform = np.zeros((n_samples, 12), dtype=np.float64)

  # Add synthetic periodic QRS peaks
  for beat_time in [0.5, 1.2, 1.9]:
    idx = int(beat_time * fs)
    # R peak
    waveform[idx - 5 : idx + 5, :] = 1.0
    # S peak
    waveform[idx + 5 : idx + 15, :] = -0.5

  lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

  # Cornell voltage: R_aVL + S_V3
  cornell = compute_cornell_voltage(waveform, lead_names, fs=fs)
  assert isinstance(cornell, float)
  assert cornell >= 0.0

  # Sokolow-Lyon: S_V1 + max(R_V5, R_V6)
  sokolow = compute_sokolow_lyon_voltage(waveform, lead_names, fs=fs)
  assert isinstance(sokolow, float)
  assert sokolow >= 0.0

  # Simplified ischemic score
  meas = CIISMeasurements(
    avl_q_duration_ms=45.0,
    avl_t_pos_mv=0.2,
    avl_t_neg_mv=0.0,
    inv_avr_r_mv=0.5,
    inv_avr_t_pos_mv=0.1,
    lead2_qr_ratio=0.3,
    avf_qr_ratio=0.1,
    lead3_q_duration_ms=30.0,
    lead3_t_neg_mv=0.0,
    v1_t_pos_mv=0.1,
    v2_r_mv=0.8,
    v2_t_neg_mv=0.05,
    v3_qr_ratio=0.02,
    v5_s_mv=0.1,
  )
  simp_score = score_simplified_ischemic_ecg(meas)
  assert isinstance(simp_score, float)
  assert simp_score > 0.0


def test_full_sensitivity_analyses_orchestration_and_markdown(
  synthetic_sensitivity_data: dict[str, pl.DataFrame | np.ndarray],
) -> None:
  earliest_df = synthetic_sensitivity_data["earliest_df"]
  admission_df = synthetic_sensitivity_data["admission_df"]
  dev_emb = synthetic_sensitivity_data["dev_emb"]
  dev_y = synthetic_sensitivity_data["dev_y"]
  test_emb = synthetic_sensitivity_data["test_emb"]
  test_y = synthetic_sensitivity_data["test_y"]
  strata_df = synthetic_sensitivity_data["strata_df"]

  assert isinstance(earliest_df, pl.DataFrame)
  assert isinstance(admission_df, pl.DataFrame)
  assert isinstance(dev_emb, np.ndarray)
  assert isinstance(dev_y, np.ndarray)
  assert isinstance(test_emb, np.ndarray)
  assert isinstance(test_y, np.ndarray)
  assert isinstance(strata_df, pl.DataFrame)

  rng = np.random.default_rng(42)
  alt_scores = {
    "Cornell Voltage": rng.uniform(0.5, 3.5, size=len(earliest_df)),
    "Sokolow-Lyon Voltage": rng.uniform(1.0, 4.5, size=len(earliest_df)),
  }

  full_result = run_sensitivity_analyses(
    earliest_test_df=earliest_df,
    admission_test_df=admission_df,
    dev_embeddings=dev_emb,
    dev_labels=dev_y,
    test_embeddings=test_emb,
    test_labels=test_y,
    alternative_traditional_scores=alt_scores,
    evaluation_strata_df=strata_df,
    n_bootstraps=20,
    seed=42,
  )

  assert isinstance(full_result, FullSensitivityAnalysisResult)
  md = generate_sensitivity_report_markdown(full_result)
  assert isinstance(md, str)
  assert "# Stage 10 Validation Report: Sensitivity and Robustness Analyses" in md
  assert "Admission-Anchored Index ECG" in md
  assert "In-Hospital Mortality" in md
  assert "Elastic-Net" in md
  assert "CarDSLab ECG-CLIP" in md
  assert "Sex / Gender" in md
