"""Tests for Stage 9 primary statistical analysis module."""

from pathlib import Path
import numpy as np
import polars as pl
import pytest

from ecg_alignment.analysis import (
  BootstrapConfidenceInterval,
  DiscordanceResult,
  GlobalAlignmentResult,
  GlobalPerformanceResult,
  IncrementalInformationResult,
  PerformanceComparisonResult,
  PrimaryAnalysisResult,
  StratifiedAnalysisResult,
  bootstrap_confidence_interval,
  bootstrap_metric_difference,
  compare_models_performance,
  compute_auprc,
  compute_calibration_metrics,
  compute_discordance_analysis,
  compute_global_alignment,
  compute_global_performance,
  compute_incremental_information,
  compute_pearson_correlation,
  compute_spearman_correlation,
  compute_stratified_risk,
  generate_primary_figures,
  generate_primary_results_markdown,
  run_primary_analysis,
)
from ecg_alignment.probe import compute_auroc


# -----------------------------------------------------------------------------
# Metric & Pure Statistical Function Tests
# -----------------------------------------------------------------------------


def test_compute_spearman_and_pearson_correlation() -> None:
  # Perfectly aligned monotonic relationship
  x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

  rho, p_spearman = compute_spearman_correlation(x, y)
  r, p_pearson = compute_pearson_correlation(x, y)

  assert np.isclose(rho, 1.0)
  assert p_spearman < 0.01
  assert np.isclose(r, 1.0)
  assert p_pearson < 0.01

  # Inverse monotonic relationship
  y_inv = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
  rho_inv, _ = compute_spearman_correlation(x, y_inv)
  assert np.isclose(rho_inv, -1.0)

  # Degenerate / constant input
  x_const = np.array([1.0, 1.0, 1.0, 1.0])
  y_const = np.array([2.0, 2.0, 2.0, 2.0])
  rho_c, p_c = compute_spearman_correlation(x_const, y_const)
  assert rho_c == 0.0
  assert p_c == 1.0

  r_c, p_cr = compute_pearson_correlation(x_const, y_const)
  assert r_c == 0.0
  assert p_cr == 1.0

  # Short input (<3)
  assert compute_spearman_correlation([1.0], [2.0]) == (0.0, 1.0)
  assert compute_pearson_correlation([1.0], [2.0]) == (0.0, 1.0)


def test_compute_auprc() -> None:
  y_true = np.array([0, 0, 1, 1], dtype=np.int64)
  y_score_perf = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
  assert compute_auprc(y_true, y_score_perf) == 1.0

  # Single class fallback returns prevalence
  assert compute_auprc([1, 1], [0.5, 0.8]) == 1.0
  assert compute_auprc([0, 0], [0.5, 0.8]) == 0.0
  assert compute_auprc([], []) == 0.0


def test_compute_calibration_metrics() -> None:
  rng = np.random.default_rng(42)
  n = 200
  # Simulated calibrated probabilities
  p_true = rng.uniform(0.01, 0.99, size=n)
  y_true = (rng.uniform(0, 1, size=n) < p_true).astype(np.int64)

  slope, intercept = compute_calibration_metrics(y_true, p_true)
  assert 0.5 < slope < 1.5
  assert -1.0 < intercept < 1.0

  # Small input fallback
  assert compute_calibration_metrics([0, 1], [0.2, 0.8]) == (1.0, 0.0)


# -----------------------------------------------------------------------------
# Bootstrap Resampling Tests
# -----------------------------------------------------------------------------


def test_bootstrap_confidence_interval() -> None:
  rng = np.random.default_rng(42)
  n = 100
  y_true = np.array([0] * 50 + [1] * 50, dtype=np.int64)
  y_pred = y_true.astype(np.float64) * 2.0 + rng.normal(0, 0.2, size=n)

  ci = bootstrap_confidence_interval(
    compute_auroc,
    y_true,
    y_pred,
    n_bootstraps=100,
    confidence_level=0.95,
    random_seed=42,
  )

  assert isinstance(ci, BootstrapConfidenceInterval)
  assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper
  assert ci.ci_lower > 0.8
  assert "95% CI" in ci.formatted()

  # Small sample fallback
  ci_small = bootstrap_confidence_interval(compute_auroc, np.array([0, 1]), np.array([0.1, 0.9]))
  assert ci_small.ci_lower == ci_small.point_estimate == ci_small.ci_upper


def test_bootstrap_metric_difference() -> None:
  rng = np.random.default_rng(42)
  n = 150
  y_true = np.array([0] * 75 + [1] * 75, dtype=np.int64)
  y_pred_good = y_true.astype(np.float64) * 2.0 + rng.normal(0, 0.2, size=n)
  y_pred_poor = rng.normal(0, 1.0, size=n)

  diff_ci, p_val = bootstrap_metric_difference(
    compute_auroc,
    y_true,
    y_pred_good,
    y_pred_poor,
    n_bootstraps=100,
    random_seed=42,
  )

  assert diff_ci.point_estimate > 0.3
  assert diff_ci.ci_lower > 0.2
  assert p_val < 0.05

  # Small sample fallback
  diff_small, p_small = bootstrap_metric_difference(
    compute_auroc, np.array([0, 1]), np.array([0.9, 0.1]), np.array([0.1, 0.9])
  )
  assert p_small == 1.0


# -----------------------------------------------------------------------------
# Component Analyses Tests with Synthetic Cohorts
# -----------------------------------------------------------------------------


def _generate_synthetic_test_df(n_samples: int = 300, seed: int = 42) -> pl.DataFrame:
  """Helper to construct a realistic synthetic test DataFrame."""
  rng = np.random.default_rng(seed)

  # Traditional CIIS scores (points 0 to 40)
  a_scores = np.clip(rng.gamma(shape=3.0, scale=4.0, size=n_samples), 0, 45)

  # Categorize CIIS
  categories: list[str] = []
  for a in a_scores:
    if a < 10.0:
      categories.append("normal")
    elif a < 15.0:
      categories.append("borderline")
    elif a < 20.0:
      categories.append("possible_injury")
    else:
      categories.append("probable_infarction")

  # Model B scores correlated with A plus independent risk signal
  latent_risk = 0.08 * a_scores + rng.normal(0, 0.6, size=n_samples) - 1.5
  b_probs = 1.0 / (1.0 + np.exp(-latent_risk))
  b_logits = latent_risk

  # Outcomes generated from combined latent risk
  true_prob = 1.0 / (1.0 + np.exp(-(0.12 * a_scores + 2.5 * b_probs - 3.2)))
  mortality_30d = (rng.uniform(0, 1, size=n_samples) < true_prob).astype(np.int64)

  # Ensure at least some events in each category
  if np.sum(mortality_30d) < 10:
    mortality_30d[:10] = 1

  return pl.DataFrame({
    "subject_id": list(range(1, n_samples + 1)),
    "study_id": list(range(101, 101 + n_samples)),
    "split": ["test"] * n_samples,
    "model_a_score": a_scores.tolist(),
    "model_a_category": categories,
    "model_a_valid": [True] * n_samples,
    "model_a_error": [None] * n_samples,
    "model_b_score": b_probs.tolist(),
    "model_b_log_odds": b_logits.tolist(),
    "model_b_valid": [True] * n_samples,
    "model_b_error": [None] * n_samples,
    "mortality_30d": [bool(m) for m in mortality_30d],
    "mortality_90d": [bool(m) for m in mortality_30d],
    "mortality_1yr": [bool(m) for m in mortality_30d],
  })


def test_compute_global_alignment() -> None:
  df = _generate_synthetic_test_df(n_samples=200)
  a_scores = df["model_a_score"].to_numpy()
  b_scores = df["model_b_score"].to_numpy()
  y_true = df["mortality_30d"].cast(pl.Int64).to_numpy()

  align = compute_global_alignment(a_scores, b_scores, y_true=y_true)
  assert isinstance(align, GlobalAlignmentResult)
  assert align.spearman_rho > 0.0
  assert len(align.a_grid) == 25
  assert len(align.expected_b_smooth) == 25
  assert len(align.risk_surface_matrix) == 5
  assert len(align.risk_surface_counts) == 5
  assert sum(sum(row) for row in align.risk_surface_counts) == len(a_scores)


def test_compute_global_performance_and_comparison() -> None:
  df = _generate_synthetic_test_df(n_samples=200)
  y_true = df["mortality_30d"].cast(pl.Int64).to_numpy()
  a_scores = df["model_a_score"].to_numpy()
  b_scores = df["model_b_score"].to_numpy()

  perf_a = compute_global_performance(y_true, a_scores, model_name="Model A", is_probability=False, n_bootstraps=50)
  perf_b = compute_global_performance(y_true, b_scores, model_name="Model B", is_probability=True, n_bootstraps=50)

  assert isinstance(perf_a, GlobalPerformanceResult)
  assert isinstance(perf_b, GlobalPerformanceResult)
  assert perf_a.auroc.point_estimate > 0.5
  assert perf_b.auroc.point_estimate > 0.5
  assert perf_b.log_loss is not None

  comp = compare_models_performance(y_true, b_scores, a_scores, n_bootstraps=50)
  assert isinstance(comp, PerformanceComparisonResult)
  assert comp.delta_auroc.confidence_level == 0.95


def test_compute_stratified_risk() -> None:
  df = _generate_synthetic_test_df(n_samples=300)
  strat = compute_stratified_risk(df, n_quantiles=3)

  assert isinstance(strat, StratifiedAnalysisResult)
  assert len(strat.categories) == 4  # 4 CIIS categories
  for cat_res in strat.categories:
    if cat_res.n_total > 0:
      assert cat_res.model_b_mean >= 0.0
      assert len(cat_res.b_quantiles) == 3


def test_compute_discordance_analysis() -> None:
  df = _generate_synthetic_test_df(n_samples=300)
  disc = compute_discordance_analysis(df, a_cutoff=15.0, n_bootstraps=50)

  assert isinstance(disc, DiscordanceResult)
  assert len(disc.quadrants) == 4
  assert disc.a_threshold == 15.0
  assert disc.risk_diff_alow_bhigh_vs_alow_blow.point_estimate is not None
  assert disc.risk_ratio_alow_bhigh_vs_alow_blow.point_estimate is not None
  assert disc.risk_diff_ahigh_bhigh_vs_ahigh_blow.point_estimate is not None
  assert disc.risk_diff_ahigh_bhigh_vs_ahigh_blow.ci_lower <= disc.risk_diff_ahigh_bhigh_vs_ahigh_blow.ci_upper


def test_compute_incremental_information() -> None:
  dev_df = _generate_synthetic_test_df(n_samples=250, seed=1)
  dev_df = dev_df.with_columns(pl.lit("dev").alias("split"))

  test_df = _generate_synthetic_test_df(n_samples=150, seed=2)
  test_df = test_df.with_columns(pl.lit("test").alias("split"))

  inc = compute_incremental_information(dev_df, test_df, n_bootstraps=50)

  assert isinstance(inc, IncrementalInformationResult)
  assert inc.lrt_statistic >= 0.0
  assert inc.lrt_degrees_of_freedom == 1
  assert 0.0 <= inc.lrt_pvalue <= 1.0
  assert inc.auroc_improvement.point_estimate is not None


def test_run_primary_analysis_end_to_end(tmp_path: Path) -> None:
  dev_df = _generate_synthetic_test_df(n_samples=200, seed=10).with_columns(pl.lit("dev").alias("split"))
  val_df = _generate_synthetic_test_df(n_samples=100, seed=11).with_columns(pl.lit("val").alias("split"))
  test_df = _generate_synthetic_test_df(n_samples=150, seed=12).with_columns(pl.lit("test").alias("split"))

  unified_df = pl.concat([dev_df, val_df, test_df])

  result = run_primary_analysis(unified_df, n_bootstraps=50)
  assert isinstance(result, PrimaryAnalysisResult)
  assert result.n_patients == 150
  assert result.alignment.spearman_rho is not None
  assert result.performance_a.auroc.point_estimate > 0.5
  assert result.incremental.lrt_statistic >= 0.0

  # Markdown Report Generation
  md = generate_primary_results_markdown(result)
  assert "# Stage 9 Primary Analysis" in md
  assert "H1 (Partial Alignment)" in md
  assert "H2 (Residual Risk Gradients)" in md
  assert "H3 (Clinically Informative Discordance)" in md
  assert "H4 (Incremental Information)" in md
  assert "Likelihood Ratio Test" in md
  assert "Net Reclassification Improvement (NRI)" in md  # Documented exclusion

  # Figure Generation
  fig_dict = generate_primary_figures(result, output_dir=tmp_path / "figures")
  assert len(fig_dict) == 3
  assert (tmp_path / "figures" / "fig_alignment_risk_surface.png").exists()
  assert (tmp_path / "figures" / "fig_stratified_residual_risk.png").exists()
  assert (tmp_path / "figures" / "fig_discordance_quadrants.png").exists()


def test_run_primary_analysis_firewall_violation() -> None:
  test_df = _generate_synthetic_test_df(n_samples=50).with_columns(
    pl.lit(65.0).alias("anchor_age")  # Forbidden clinical variable
  )
  with pytest.raises(ValueError, match="Predictor-information firewall violation"):
    run_primary_analysis(test_df)
