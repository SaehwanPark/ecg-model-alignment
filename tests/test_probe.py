"""Tests for linear probe fitting, continuous prediction generation, and research guardrails."""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from ecg_alignment.probe import (
  DEFAULT_C_GRID,
  ProbeConfig,
  ProbeValidationStep,
  TrainedProbe,
  build_unified_prediction_table,
  compute_auroc,
  compute_binary_log_loss,
  compute_brier_score,
  compute_prediction_summary_statistics,
  extract_transformer_embeddings,
  fit_logistic_probe,
  generate_continuous_predictions_markdown,
  score_traditional_cohort,
  score_transformer_cohort,
  verify_predictor_firewall,
  verify_unified_prediction_table,
)
from ecg_alignment.scoring.base import (
  BaseEcgModelAdapter,
  ModelOutputResult,
  TransformerAdapterConfig,
)


# -----------------------------------------------------------------------------
# Metric Evaluation Tests
# -----------------------------------------------------------------------------


def test_compute_binary_log_loss() -> None:
  # Perfect predictions
  y_true = np.array([0, 1, 0, 1], dtype=np.int64)
  y_prob_perfect = np.array([0.0001, 0.9999, 0.0001, 0.9999], dtype=np.float64)
  loss_perfect = compute_binary_log_loss(y_true, y_prob_perfect)
  assert loss_perfect < 0.01

  # Neutral predictions
  y_prob_neutral = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
  loss_neutral = compute_binary_log_loss(y_true, y_prob_neutral)
  assert np.isclose(loss_neutral, np.log(2), atol=1e-3)

  # Edge case: empty input
  assert compute_binary_log_loss([], []) == 0.0

  # Edge case: single class
  y_single = np.array([1, 1, 1], dtype=np.int64)
  loss_single = compute_binary_log_loss(y_single, np.array([0.9, 0.9, 0.9]))
  assert loss_single > 0.0


def test_compute_auroc() -> None:
  y_true = np.array([0, 0, 1, 1], dtype=np.int64)
  y_score_perf = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
  assert compute_auroc(y_true, y_score_perf) == 1.0

  y_score_inv = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
  assert compute_auroc(y_true, y_score_inv) == 0.0

  # Single class fallback
  assert compute_auroc([1, 1], [0.5, 0.8]) == 0.5
  assert compute_auroc([], []) == 0.5


def test_compute_brier_score() -> None:
  y_true = np.array([0, 1], dtype=np.int64)
  y_prob_perf = np.array([0.0, 1.0], dtype=np.float64)
  assert compute_brier_score(y_true, y_prob_perf) == 0.0

  y_prob_worst = np.array([1.0, 0.0], dtype=np.float64)
  assert compute_brier_score(y_true, y_prob_worst) == 1.0

  assert compute_brier_score([], []) == 0.0


# -----------------------------------------------------------------------------
# ProbeConfig Tests
# -----------------------------------------------------------------------------


def test_probe_config_validation() -> None:
  cfg = ProbeConfig()
  assert cfg.model_name == "Manhph2211/D-BETA"
  assert cfg.regularization_c_grid == DEFAULT_C_GRID
  assert cfg.standardize is True
  assert cfg.tuning_metric == "log_loss"

  with pytest.raises(ValueError, match="must not be empty"):
    ProbeConfig(regularization_c_grid=())

  with pytest.raises(ValueError, match="strictly positive"):
    ProbeConfig(regularization_c_grid=(0.1, -1.0))


# -----------------------------------------------------------------------------
# Probe Fitting and Serialization Tests
# -----------------------------------------------------------------------------


def test_fit_logistic_probe_and_predict() -> None:
  rng = np.random.default_rng(42)
  n_dev = 100
  n_val = 40
  dim = 8

  # Synthetic linearly separable features with noise
  w_true = rng.normal(size=dim)
  x_dev = rng.normal(size=(n_dev, dim))
  logits_dev = x_dev @ w_true
  y_dev = (logits_dev > 0).astype(np.int64)

  x_val = rng.normal(size=(n_val, dim))
  logits_val = x_val @ w_true
  y_val = (logits_val > 0).astype(np.int64)

  config = ProbeConfig(
    regularization_c_grid=(0.01, 1.0, 100.0),
    standardize=True,
    tuning_metric="log_loss",
  )

  probe = fit_logistic_probe(x_dev, y_dev, x_val, y_val, config=config)

  assert isinstance(probe, TrainedProbe)
  assert probe.embedding_dim == dim
  assert probe.best_c in config.regularization_c_grid
  assert len(probe.tuning_history) == 3
  assert probe.scaler_mean is not None
  assert len(probe.scaler_mean) == dim

  # Test predictions
  probs_val = probe.predict_proba(x_val)
  logits_pred = probe.predict_log_odds(x_val)

  assert probs_val.shape == (n_val,)
  assert logits_pred.shape == (n_val,)
  assert np.all(probs_val >= 0.0) and np.all(probs_val <= 1.0)
  assert compute_auroc(y_val, probs_val) > 0.85


def test_fit_logistic_probe_error_handling() -> None:
  x_dev = np.zeros((10, 4))
  y_dev = np.ones(10, dtype=np.int64)  # Single class
  x_val = np.zeros((5, 4))
  y_val = np.array([0, 1, 0, 1, 0], dtype=np.int64)

  with pytest.raises(ValueError, match="both positive.*and negative"):
    fit_logistic_probe(x_dev, y_dev, x_val, y_val)

  # Shape mismatch
  with pytest.raises(ValueError, match="Dev samples"):
    fit_logistic_probe(x_dev[:8], np.array([0, 1] * 3), x_val, y_val)

  with pytest.raises(ValueError, match="dimension"):
    fit_logistic_probe(np.zeros((10, 4)), np.array([0, 1] * 5), np.zeros((5, 6)), y_val)


def test_trained_probe_json_serialization(tmp_path: Path) -> None:
  probe = TrainedProbe(
    coefficients=(0.5, -0.2, 0.1),
    intercept=-0.1,
    best_c=1.0,
    tuning_history=(
      ProbeValidationStep(c_value=0.1, val_log_loss=0.45, val_auroc=0.75, val_brier=0.12),
      ProbeValidationStep(c_value=1.0, val_log_loss=0.42, val_auroc=0.78, val_brier=0.11),
    ),
    scaler_mean=(0.0, 1.0, -0.5),
    scaler_scale=(1.0, 2.0, 0.5),
    config=ProbeConfig(),
  )

  save_file = tmp_path / "probe.json"
  probe.save_json(save_file)
  assert save_file.exists()

  loaded = TrainedProbe.load_json(save_file)
  assert loaded.coefficients == probe.coefficients
  assert loaded.intercept == probe.intercept
  assert loaded.best_c == probe.best_c
  assert len(loaded.tuning_history) == 2
  assert loaded.scaler_mean == probe.scaler_mean
  assert loaded.config.model_name == probe.config.model_name


# -----------------------------------------------------------------------------
# Predictor Firewall Tests
# -----------------------------------------------------------------------------


def test_verify_predictor_firewall() -> None:
  # Clean table with allowed non-predictor and score columns
  clean_df = pl.DataFrame({
    "subject_id": [1, 2],
    "study_id": [101, 102],
    "split": ["dev", "val"],
    "model_a_score": [5.0, 12.0],
    "model_a_category": ["normal", "borderline"],
    "model_a_valid": [True, True],
    "model_a_error": [None, None],
    "model_b_score": [0.03, 0.08],
    "model_b_log_odds": [-3.5, -2.4],
    "model_b_valid": [True, True],
    "model_b_error": [None, None],
    "mortality_30d": [False, False],
  })
  assert verify_predictor_firewall(clean_df) is True

  # Violating table with demographic / clinical features
  for bad_col in ["anchor_age", "gender", "race", "hadm_id", "vital_heart_rate", "lab_creatinine"]:
    dirty_df = clean_df.with_columns(pl.lit(1.0).alias(bad_col))
    with pytest.raises(ValueError, match="Predictor-information firewall violation"):
      verify_predictor_firewall(dirty_df)


# -----------------------------------------------------------------------------
# Unified Prediction Table Verification & Construction Tests
# -----------------------------------------------------------------------------


def test_build_and_verify_unified_prediction_table() -> None:
  cohort_df = pl.DataFrame({
    "subject_id": [1, 2, 3],
    "study_id": [101, 201, 301],
    "split": ["dev", "val", "test"],
    "mortality_30d": [False, True, False],
    "mortality_90d": [False, True, False],
  })

  model_a_df = pl.DataFrame({
    "subject_id": [1, 2, 3],
    "study_id": [101, 201, 301],
    "model_a_score": [4.0, 16.5, 8.0],
    "model_a_category": ["normal", "possible_injury", "normal"],
    "model_a_valid": [True, True, True],
    "model_a_error": [None, None, None],
  })

  model_b_df = pl.DataFrame({
    "subject_id": [1, 2, 3],
    "study_id": [101, 201, 301],
    "model_b_score": [0.02, 0.25, 0.05],
    "model_b_log_odds": [-3.8, -1.1, -2.9],
    "model_b_valid": [True, True, True],
    "model_b_error": [None, None, None],
  })

  unified = build_unified_prediction_table(cohort_df, model_a_df, model_b_df)
  assert len(unified) == 3
  assert verify_unified_prediction_table(unified) is True
  assert "model_a_score" in unified.columns
  assert "model_b_score" in unified.columns
  assert "mortality_30d" in unified.columns


def test_verify_unified_prediction_table_violations() -> None:
  # Duplicate patient ID (unit-of-analysis violation)
  duplicate_patient_df = pl.DataFrame({
    "subject_id": [1, 1],  # duplicate
    "study_id": [101, 102],
    "split": ["dev", "dev"],
    "model_a_score": [4.0, 5.0],
    "model_a_category": ["normal", "normal"],
    "model_a_valid": [True, True],
    "model_b_score": [0.02, 0.03],
    "model_b_log_odds": [-3.8, -3.5],
    "model_b_valid": [True, True],
    "mortality_30d": [False, False],
  })
  with pytest.raises(ValueError, match="Unit-of-analysis violation"):
    verify_unified_prediction_table(duplicate_patient_df)

  # Patient split overlap violation
  overlap_df = pl.DataFrame({
    "subject_id": [1, 1],
    "study_id": [101, 102],
    "split": ["dev", "test"],  # split overlap
    "model_a_score": [4.0, 5.0],
    "model_a_category": ["normal", "normal"],
    "model_a_valid": [True, True],
    "model_b_score": [0.02, 0.03],
    "model_b_log_odds": [-3.8, -3.5],
    "model_b_valid": [True, True],
    "mortality_30d": [False, False],
  })
  with pytest.raises(ValueError, match="Unit-of-analysis violation"):
    # Fails unit of analysis first
    verify_unified_prediction_table(overlap_df)

  # Invalid split labels
  invalid_split_df = pl.DataFrame({
    "subject_id": [1, 2],
    "study_id": [101, 102],
    "split": ["train", "test"],  # 'train' instead of 'dev'
    "model_a_score": [4.0, 5.0],
    "model_a_category": ["normal", "normal"],
    "model_a_valid": [True, True],
    "model_b_score": [0.02, 0.03],
    "model_b_log_odds": [-3.8, -3.5],
    "model_b_valid": [True, True],
    "mortality_30d": [False, False],
  })
  with pytest.raises(ValueError, match="Invalid split partition labels"):
    verify_unified_prediction_table(invalid_split_df)


# -----------------------------------------------------------------------------
# Batch Scoring & Probe Application Tests
# -----------------------------------------------------------------------------


def test_score_transformer_cohort() -> None:
  cohort_df = pl.DataFrame({
    "subject_id": [1, 2, 3],
    "study_id": [101, 201, 301],
  })

  embeddings = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
  ], dtype=np.float64)

  probe = TrainedProbe(
    coefficients=(2.0, -1.0),
    intercept=0.0,
    best_c=1.0,
    tuning_history=(),
    scaler_mean=None,
    scaler_scale=None,
    config=ProbeConfig(),
  )

  # All valid
  scored = score_transformer_cohort(cohort_df, embeddings, probe)
  assert len(scored) == 3
  assert scored["model_b_valid"].to_list() == [True, True, True]
  assert scored["model_b_log_odds"][0] == 2.0  # 2.0*1.0 + -1.0*0.0
  assert scored["model_b_log_odds"][1] == -1.0  # 2.0*0.0 + -1.0*1.0

  # With invalid mask
  scored_masked = score_transformer_cohort(
    cohort_df,
    embeddings,
    probe,
    valid_mask=[True, False, True],
    error_messages=[None, "Lead V1 missing", None],
  )
  assert scored_masked["model_b_valid"].to_list() == [True, False, True]
  assert scored_masked["model_b_score"][1] is None
  assert scored_masked["model_b_error"][1] == "Lead V1 missing"


# -----------------------------------------------------------------------------
# Summary Statistics and Report Generation Tests
# -----------------------------------------------------------------------------


def test_compute_prediction_summary_statistics_and_markdown() -> None:
  unified_df = pl.DataFrame({
    "subject_id": [1, 2, 3, 4, 5, 6],
    "study_id": [101, 201, 301, 401, 501, 601],
    "split": ["dev", "dev", "val", "val", "test", "test"],
    "model_a_score": [2.0, 12.0, 5.0, 22.0, 8.0, 18.0],
    "model_a_category": [
      "normal",
      "borderline",
      "normal",
      "probable_infarction",
      "normal",
      "possible_injury",
    ],
    "model_a_valid": [True, True, True, True, True, True],
    "model_a_error": [None] * 6,
    "model_b_score": [0.01, 0.20, 0.04, 0.40, 0.05, 0.35],
    "model_b_log_odds": [-4.5, -1.4, -3.2, -0.4, -2.9, -0.6],
    "model_b_valid": [True, True, True, True, True, True],
    "model_b_error": [None] * 6,
    "mortality_30d": [False, True, False, True, False, True],
  })

  stats = compute_prediction_summary_statistics(unified_df)
  assert stats["total_patients"] == 6
  assert stats["dev_n"] == 2
  assert stats["val_n"] == 2
  assert stats["test_n"] == 2
  assert stats["test_model_a_valid_n"] == 2
  assert stats["test_model_b_valid_n"] == 2
  assert "test_model_a_mean" in stats
  assert "test_model_b_prob_mean" in stats

  probe = TrainedProbe(
    coefficients=(1.0, 0.5),
    intercept=-1.0,
    best_c=1.0,
    tuning_history=(
      ProbeValidationStep(c_value=0.1, val_log_loss=0.50, val_auroc=0.80, val_brier=0.15),
      ProbeValidationStep(c_value=1.0, val_log_loss=0.45, val_auroc=0.85, val_brier=0.12),
    ),
    scaler_mean=(0.0, 0.0),
    scaler_scale=(1.0, 1.0),
    config=ProbeConfig(),
  )

  md = generate_continuous_predictions_markdown(stats, probe=probe)
  assert "# MIMIC-IV Continuous Prediction Generation" in md
  assert "Stage 8" in md
  assert "Score Generation Pipeline Overview" in md
  assert "Optimal Regularization Parameter" in md
  assert "Predictor-Information Firewall" in md


# -----------------------------------------------------------------------------
# Batch Cohort Extraction & Scoring Mock Tests
# -----------------------------------------------------------------------------


class MockEcgAdapter(BaseEcgModelAdapter):
  """Mock adapter for deterministic testing of extract_transformer_embeddings."""

  def __init__(self, embedding_dim: int = 4, fail_on_subject: int | None = None) -> None:
    self._config = TransformerAdapterConfig(
      model_name="mock-transformer",
      embedding_dim=embedding_dim,
    )
    self.fail_on_subject = fail_on_subject

  @property
  def config(self) -> TransformerAdapterConfig:
    return self._config

  def preprocess_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = 500,
  ) -> npt.NDArray[np.float64]:
    return signal_array

  def embed_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = 500,
  ) -> ModelOutputResult:
    if np.isnan(signal_array).any():
      return ModelOutputResult(is_valid=False, error_message="Signal contains NaNs")
    emb = np.ones(self._config.embedding_dim, dtype=np.float64) * 0.5
    return ModelOutputResult(is_valid=True, embedding=emb, output_dim=len(emb))

  def embed_batch(
    self,
    batch_records: Sequence[tuple[npt.NDArray[np.float64], Sequence[str], int]],
  ) -> Any:
    raise NotImplementedError


def test_extract_transformer_embeddings_and_traditional_cohort(monkeypatch: pytest.MonkeyPatch) -> None:
  cohort_df = pl.DataFrame({
    "subject_id": [1, 2],
    "study_id": [101, 102],
    "path": ["files/p10/p101", "files/p10/p102"],
  })

  def mock_read_waveform(record_path: Path | str) -> tuple[npt.NDArray[np.float64], list[str], int]:
    p = str(record_path)
    if "p102" in p:
      # Return NaN signal to test graceful failure handling
      sig = np.full((5000, 12), np.nan, dtype=np.float64)
    else:
      sig = np.zeros((5000, 12), dtype=np.float64)
    leads = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
    return sig, leads, 500

  monkeypatch.setattr("ecg_alignment.probe.read_ecg_waveform", mock_read_waveform)

  # 1. Test extract_transformer_embeddings
  adapter = MockEcgAdapter(embedding_dim=4)
  meta_df, embs = extract_transformer_embeddings(cohort_df, adapter)
  assert embs.shape == (2, 4)
  assert meta_df["is_valid"].to_list() == [True, False]
  assert meta_df["error_message"][1] == "Signal contains NaNs"

  # 2. Test score_traditional_cohort
  scored_a = score_traditional_cohort(cohort_df)
  assert len(scored_a) == 2
  assert scored_a["model_a_valid"][1] is False
  assert "NaN or Infinite" in (scored_a["model_a_error"][1] or "")

