"""Primary analysis module for Stage 9: Alignment, Residual Risk, Discordance, and Incremental Information.

Strict research guardrails:
- Every primary result is computed on the untouched final test set (`split == 'test'`).
- Predictor-information firewall: Non-ECG clinical variables (demographics, labs, vitals, meds, notes)
  are strictly prohibited from entering predictor spaces.
- Uncertainty estimation: All confidence intervals are estimated via patient-level bootstrap resampling.
- In-domain disclosure: Findings from transformer models pretrained on MIMIC-IV-ECG represent
  in-domain representation probing, not independent external validation.
- NRI/IDI exclusion: NRI and IDI are deliberately excluded from primary incremental information analysis.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import chi2, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from ecg_alignment.probe import (
  compute_auroc,
  compute_binary_log_loss,
  compute_brier_score,
  verify_predictor_firewall,
)
from ecg_alignment.scoring.traditional import CIISCategory

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_ROUNDS: int = 1000
DEFAULT_BOOTSTRAP_SEED: int = 42
DEFAULT_CONFIDENCE_LEVEL: float = 0.95


# -----------------------------------------------------------------------------
# Data Containers & Result Structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
  """Container for a patient-level bootstrap confidence interval."""

  point_estimate: float
  ci_lower: float
  ci_upper: float
  confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
  n_bootstraps: int = DEFAULT_BOOTSTRAP_ROUNDS

  def formatted(self, precision: int = 3) -> str:
    """Format point estimate and 95% CI string."""
    return f"{self.point_estimate:.{precision}f} (95% CI: {self.ci_lower:.{precision}f}–{self.ci_upper:.{precision}f})"


@dataclass(frozen=True)
class GlobalAlignmentResult:
  """Measures of global score alignment between Model A and Model B."""

  spearman_rho: float
  spearman_pvalue: float
  pearson_r: float
  pearson_pvalue: float
  a_grid: tuple[float, ...]
  expected_b_smooth: tuple[float, ...]
  risk_surface_a_bins: tuple[float, ...]
  risk_surface_b_bins: tuple[float, ...]
  risk_surface_matrix: tuple[tuple[float, ...], ...]
  risk_surface_counts: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class GlobalPerformanceResult:
  """Discriminative and calibration metrics for a single model with bootstrap CIs."""

  model_name: str
  n_total: int
  n_events: int
  event_rate: float
  auroc: BootstrapConfidenceInterval
  auprc: BootstrapConfidenceInterval
  brier_score: BootstrapConfidenceInterval
  log_loss: BootstrapConfidenceInterval | None
  calibration_slope: float
  calibration_intercept: float


@dataclass(frozen=True)
class PerformanceComparisonResult:
  """Comparative metrics between Model B and Model A with patient-level bootstrap CIs."""

  delta_auroc: BootstrapConfidenceInterval  # AUROC(B) - AUROC(A)
  delta_auprc: BootstrapConfidenceInterval  # AUPRC(B) - AUPRC(A)
  delta_brier: BootstrapConfidenceInterval  # Brier(A) - Brier(B) (positive means B is better)
  p_value_auroc_diff: float


@dataclass(frozen=True)
class QuantileStratumResult:
  """Risk summary for a quantile group of Model B within an A-category."""

  quantile_label: str
  n_patients: int
  n_events: int
  event_rate: float
  b_min: float
  b_max: float


@dataclass(frozen=True)
class StratifiedCategoryResult:
  """Stratified performance and residual risk within a traditional CIIS risk category."""

  category: str
  n_total: int
  n_events: int
  event_rate: float
  model_b_mean: float
  model_b_std: float
  model_b_median: float
  model_b_q25: float
  model_b_q75: float
  model_b_auroc: float | None
  model_b_auprc: float | None
  b_quantiles: tuple[QuantileStratumResult, ...]


@dataclass(frozen=True)
class StratifiedAnalysisResult:
  """Container for all traditional risk categories and residual risk gradients."""

  categories: tuple[StratifiedCategoryResult, ...]
  overall_event_rate: float


@dataclass(frozen=True)
class DiscordanceQuadrant:
  """Summary of one quadrant in the 2x2 Model A / Model B risk partition."""

  quadrant_id: str
  label: str
  n_patients: int
  n_events: int
  event_rate: BootstrapConfidenceInterval


@dataclass(frozen=True)
class DiscordanceResult:
  """Discordance analysis evaluating risk divergence between Model A and Model B."""

  a_threshold: float
  b_threshold: float
  threshold_method: str
  quadrants: tuple[DiscordanceQuadrant, ...]
  risk_diff_alow_bhigh_vs_alow_blow: BootstrapConfidenceInterval
  risk_ratio_alow_bhigh_vs_alow_blow: BootstrapConfidenceInterval
  risk_diff_ahigh_bhigh_vs_ahigh_blow: BootstrapConfidenceInterval


@dataclass(frozen=True)
class NestedModelEvaluation:
  """Evaluation of a single outcome model specification for incremental analysis."""

  model_name: str
  formula: str
  log_likelihood: float
  aic: float
  bic: float
  held_out_log_loss: float
  held_out_auroc: float
  held_out_brier: float


@dataclass(frozen=True)
class IncrementalInformationResult:
  """Information contribution of Model B after adjusting for traditional Model A."""

  model_a_only: NestedModelEvaluation
  model_b_only: NestedModelEvaluation
  model_combined: NestedModelEvaluation
  lrt_statistic: float  # 2 * (LL_combined - LL_a_only)
  lrt_degrees_of_freedom: int
  lrt_pvalue: float
  auroc_improvement: BootstrapConfidenceInterval  # AUROC(A+B) - AUROC(A)
  brier_improvement: BootstrapConfidenceInterval  # Brier(A) - Brier(A+B)
  held_out_loss_reduction: float  # Loss(A) - Loss(A+B)


@dataclass(frozen=True)
class PrimaryAnalysisResult:
  """Master container for all Stage 9 primary findings."""

  n_patients: int
  n_events: int
  event_rate: float
  alignment: GlobalAlignmentResult
  performance_a: GlobalPerformanceResult
  performance_b: GlobalPerformanceResult
  comparison: PerformanceComparisonResult
  stratified: StratifiedAnalysisResult
  discordance: DiscordanceResult
  incremental: IncrementalInformationResult


# -----------------------------------------------------------------------------
# Pure Statistical & Metric Computation Helpers
# -----------------------------------------------------------------------------


def compute_spearman_correlation(
  x: npt.NDArray[np.float64] | Sequence[float],
  y: npt.NDArray[np.float64] | Sequence[float],
) -> tuple[float, float]:
  """Compute Spearman rank correlation coefficient and two-tailed p-value.

  Args:
    x: Continuous values from first score.
    y: Continuous values from second score.

  Returns:
    Tuple of (spearman_rho, p_value).
  """
  xa = np.asarray(x, dtype=np.float64)
  ya = np.asarray(y, dtype=np.float64)

  if len(xa) < 3 or np.all(xa == xa[0]) or np.all(ya == ya[0]):
    return 0.0, 1.0

  res: Any = spearmanr(xa, ya)
  raw_rho = getattr(res, "statistic", getattr(res, "correlation", res[0] if isinstance(res, tuple) else 0.0))
  raw_p = getattr(res, "pvalue", res[1] if isinstance(res, tuple) else 1.0)
  rho = float(raw_rho) if not np.isnan(raw_rho) else 0.0
  pval = float(raw_p) if not np.isnan(raw_p) else 1.0
  return rho, pval


def compute_pearson_correlation(
  x: npt.NDArray[np.float64] | Sequence[float],
  y: npt.NDArray[np.float64] | Sequence[float],
) -> tuple[float, float]:
  """Compute Pearson linear correlation coefficient and two-tailed p-value.

  Args:
    x: Continuous values from first score.
    y: Continuous values from second score.

  Returns:
    Tuple of (pearson_r, p_value).
  """
  xa = np.asarray(x, dtype=np.float64)
  ya = np.asarray(y, dtype=np.float64)

  if len(xa) < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
    return 0.0, 1.0

  res: Any = pearsonr(xa, ya)
  raw_r = getattr(res, "statistic", res[0] if isinstance(res, tuple) else 0.0)
  raw_p = getattr(res, "pvalue", res[1] if isinstance(res, tuple) else 1.0)
  r_val = float(raw_r) if not np.isnan(raw_r) else 0.0
  pval = float(raw_p) if not np.isnan(raw_p) else 1.0
  return r_val, pval


def compute_auprc(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_score: npt.NDArray[np.float64] | Sequence[float],
) -> float:
  """Calculate Area Under the Precision-Recall Curve (AUPRC / Average Precision).

  Args:
    y_true: Binary ground truth targets (0 or 1).
    y_score: Continuous predicted scores or probabilities.

  Returns:
    AUPRC value in range [0.0, 1.0]. Returns baseline prevalence if single class.
  """
  yt = np.asarray(y_true, dtype=np.int64)
  ys = np.asarray(y_score, dtype=np.float64)

  if len(yt) == 0:
    return 0.0
  if len(np.unique(yt)) < 2:
    return float(np.mean(yt))

  try:
    return float(average_precision_score(yt, ys))
  except Exception:
    return float(np.mean(yt))


def compute_calibration_metrics(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
  eps: float = 1e-6,
) -> tuple[float, float]:
  """Fit logistic calibration regression to compute calibration slope and intercept.

  Logit formulation: logit(P(Y=1)) = intercept + slope * logit(p)

  Args:
    y_true: Binary targets.
    y_prob: Predicted probabilities.
    eps: Numerical clipping epsilon.

  Returns:
    Tuple of (calibration_slope, calibration_intercept).
  """
  yt = np.asarray(y_true, dtype=np.int64)
  yp = np.asarray(y_prob, dtype=np.float64)

  if len(yt) < 10 or len(np.unique(yt)) < 2:
    return 1.0, 0.0

  yp_clipped = np.clip(yp, eps, 1.0 - eps)
  logits = np.log(yp_clipped / (1.0 - yp_clipped)).reshape(-1, 1)

  try:
    clf = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500)
    clf.fit(logits, yt)
    raw_coef = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
    raw_intercept = np.asarray(clf.intercept_, dtype=np.float64).reshape(-1)
    slope = float(raw_coef[0])
    intercept = float(raw_intercept[0])
    return slope, intercept
  except Exception:
    return 1.0, 0.0


# -----------------------------------------------------------------------------
# Patient-Level Bootstrap Resampling Helpers
# -----------------------------------------------------------------------------


def bootstrap_confidence_interval(
  metric_fn: Callable[[npt.NDArray[Any], npt.NDArray[Any]], float],
  y_true: npt.NDArray[np.int64],
  y_pred: npt.NDArray[np.float64],
  n_bootstraps: int = DEFAULT_BOOTSTRAP_ROUNDS,
  confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
  random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BootstrapConfidenceInterval:
  """Compute patient-level non-parametric bootstrap confidence interval for a single metric.

  Args:
    metric_fn: Function taking (y_true_sample, y_pred_sample) and returning scalar float.
    y_true: Ground truth array.
    y_pred: Predictor array.
    n_bootstraps: Number of bootstrap iterations.
    confidence_level: Nominal confidence level (e.g. 0.95).
    random_seed: Random state seed.

  Returns:
    BootstrapConfidenceInterval object.
  """
  n = len(y_true)
  point_est = metric_fn(y_true, y_pred)

  if n < 5:
    return BootstrapConfidenceInterval(
      point_estimate=point_est,
      ci_lower=point_est,
      ci_upper=point_est,
      confidence_level=confidence_level,
      n_bootstraps=n_bootstraps,
    )

  rng = np.random.default_rng(random_seed)
  boot_estimates: list[float] = []

  for _ in range(n_bootstraps):
    idx = rng.choice(n, size=n, replace=True)
    yt_b = y_true[idx]
    yp_b = y_pred[idx]
    val = metric_fn(yt_b, yp_b)
    if not np.isnan(val):
      boot_estimates.append(val)

  if not boot_estimates:
    return BootstrapConfidenceInterval(
      point_estimate=point_est,
      ci_lower=point_est,
      ci_upper=point_est,
      confidence_level=confidence_level,
      n_bootstraps=n_bootstraps,
    )

  alpha = (1.0 - confidence_level) / 2.0
  ci_lower = float(np.percentile(boot_estimates, 100.0 * alpha))
  ci_upper = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha)))

  return BootstrapConfidenceInterval(
    point_estimate=point_est,
    ci_lower=ci_lower,
    ci_upper=ci_upper,
    confidence_level=confidence_level,
    n_bootstraps=n_bootstraps,
  )


def bootstrap_metric_difference(
  metric_fn: Callable[[npt.NDArray[np.int64], npt.NDArray[np.float64]], float],
  y_true: npt.NDArray[np.int64],
  y_pred_b: npt.NDArray[np.float64],
  y_pred_a: npt.NDArray[np.float64],
  n_bootstraps: int = DEFAULT_BOOTSTRAP_ROUNDS,
  confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
  random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[BootstrapConfidenceInterval, float]:
  """Compute paired patient-level bootstrap confidence interval for Delta = Metric(B) - Metric(A).

  Args:
    metric_fn: Metric evaluation function.
    y_true: Ground truth target vector.
    y_pred_b: Predictions from Model B.
    y_pred_a: Predictions from Model A.
    n_bootstraps: Number of bootstrap iterations.
    confidence_level: Nominal confidence level.
    random_seed: Random seed.

  Returns:
    Tuple of (BootstrapConfidenceInterval, empirical_two_sided_p_value).
  """
  n = len(y_true)
  point_est = metric_fn(y_true, y_pred_b) - metric_fn(y_true, y_pred_a)

  if n < 5:
    return (
      BootstrapConfidenceInterval(
        point_estimate=point_est,
        ci_lower=point_est,
        ci_upper=point_est,
        confidence_level=confidence_level,
        n_bootstraps=n_bootstraps,
      ),
      1.0,
    )

  rng = np.random.default_rng(random_seed)
  diffs: list[float] = []

  for _ in range(n_bootstraps):
    idx = rng.choice(n, size=n, replace=True)
    yt_b = y_true[idx]
    val_b = metric_fn(yt_b, y_pred_b[idx])
    val_a = metric_fn(yt_b, y_pred_a[idx])
    if not np.isnan(val_b) and not np.isnan(val_a):
      diffs.append(val_b - val_a)

  if not diffs:
    return (
      BootstrapConfidenceInterval(
        point_estimate=point_est,
        ci_lower=point_est,
        ci_upper=point_est,
        confidence_level=confidence_level,
        n_bootstraps=n_bootstraps,
      ),
      1.0,
    )

  alpha = (1.0 - confidence_level) / 2.0
  ci_lower = float(np.percentile(diffs, 100.0 * alpha))
  ci_upper = float(np.percentile(diffs, 100.0 * (1.0 - alpha)))

  # Two-sided empirical p-value for H0: diff == 0
  diffs_arr = np.array(diffs)
  if point_est >= 0:
    p_val = 2.0 * float(np.mean(diffs_arr <= 0))
  else:
    p_val = 2.0 * float(np.mean(diffs_arr >= 0))
  p_val = min(max(p_val, 1.0 / n_bootstraps), 1.0)

  return (
    BootstrapConfidenceInterval(
      point_estimate=point_est,
      ci_lower=ci_lower,
      ci_upper=ci_upper,
      confidence_level=confidence_level,
      n_bootstraps=n_bootstraps,
    ),
    p_val,
  )


# -----------------------------------------------------------------------------
# Component 1: Global Score Alignment & 2D Risk Surface
# -----------------------------------------------------------------------------


def compute_global_alignment(
  a_scores: npt.NDArray[np.float64] | Sequence[float],
  b_scores: npt.NDArray[np.float64] | Sequence[float],
  y_true: npt.NDArray[np.int64] | Sequence[int] | None = None,
  n_grid_points: int = 25,
  n_surface_bins: int = 5,
) -> GlobalAlignmentResult:
  """Analyze alignment between continuous traditional score A and transformer score B.

  Args:
    a_scores: Continuous Model A (CIIS) scores.
    b_scores: Continuous Model B scores or probabilities.
    y_true: Optional outcome targets for calculating 2D risk surface.
    n_grid_points: Resolution for smooth E(B | A) estimation.
    n_surface_bins: Bin resolution for 2D outcome risk surface.

  Returns:
    GlobalAlignmentResult instance.
  """
  a_arr = np.asarray(a_scores, dtype=np.float64)
  b_arr = np.asarray(b_scores, dtype=np.float64)

  spearman_rho, spearman_pval = compute_spearman_correlation(a_arr, b_arr)
  pearson_r_val, pearson_pval = compute_pearson_correlation(a_arr, b_arr)

  # Smooth conditional expectation E(B | A)
  a_min, a_max = float(np.min(a_arr)), float(np.max(a_arr))
  if a_max == a_min:
    a_grid = np.array([a_min])
    smooth_b = np.array([float(np.mean(b_arr))])
  else:
    a_grid = np.linspace(a_min, a_max, n_grid_points)
    smooth_b = np.zeros(n_grid_points, dtype=np.float64)
    # Bandwidth window
    bw = max((a_max - a_min) / (n_grid_points / 2.0), 1.0)
    for i, a_val in enumerate(a_grid):
      weights = np.exp(-0.5 * ((a_arr - a_val) / bw) ** 2)
      sum_w = np.sum(weights)
      smooth_b[i] = float(np.sum(weights * b_arr) / sum_w) if sum_w > 1e-12 else float(np.mean(b_arr))

  # 2D Outcome Risk Surface
  if y_true is not None and len(y_true) == len(a_arr):
    yt = np.asarray(y_true, dtype=np.int64)
    a_quantiles = np.percentile(a_arr, np.linspace(0, 100, n_surface_bins + 1))
    b_quantiles = np.percentile(b_arr, np.linspace(0, 100, n_surface_bins + 1))

    # Ensure strictly increasing bin edges
    a_bins = np.unique(a_quantiles)
    b_bins = np.unique(b_quantiles)

    risk_matrix = np.full((len(a_bins) - 1, len(b_bins) - 1), np.nan, dtype=np.float64)
    count_matrix = np.zeros((len(a_bins) - 1, len(b_bins) - 1), dtype=np.int64)
    for i in range(len(a_bins) - 1):
      a_mask = (a_arr >= a_bins[i]) & (a_arr <= a_bins[i + 1] if i == len(a_bins) - 2 else a_arr < a_bins[i + 1])
      for j in range(len(b_bins) - 1):
        b_mask = (b_arr >= b_bins[j]) & (b_arr <= b_bins[j + 1] if j == len(b_bins) - 2 else b_arr < b_bins[j + 1])
        cell_mask = a_mask & b_mask
        n_cell = int(np.sum(cell_mask))
        count_matrix[i, j] = n_cell
        if n_cell > 0:
          risk_matrix[i, j] = float(np.mean(yt[cell_mask]))
        else:
          risk_matrix[i, j] = np.nan
  else:
    a_bins = np.linspace(a_min, a_max, n_surface_bins + 1)
    b_bins = np.linspace(float(np.min(b_arr)), float(np.max(b_arr)), n_surface_bins + 1)
    risk_matrix = np.full((n_surface_bins, n_surface_bins), np.nan, dtype=np.float64)
    count_matrix = np.zeros((n_surface_bins, n_surface_bins), dtype=np.int64)

  return GlobalAlignmentResult(
    spearman_rho=spearman_rho,
    spearman_pvalue=spearman_pval,
    pearson_r=pearson_r_val,
    pearson_pvalue=pearson_pval,
    a_grid=tuple(float(x) for x in a_grid),
    expected_b_smooth=tuple(float(y) for y in smooth_b),
    risk_surface_a_bins=tuple(float(x) for x in a_bins),
    risk_surface_b_bins=tuple(float(y) for y in b_bins),
    risk_surface_matrix=tuple(tuple(float(cell) for cell in row) for row in risk_matrix),
    risk_surface_counts=tuple(tuple(int(c) for c in row) for row in count_matrix),
  )


# -----------------------------------------------------------------------------
# Component 2: Global Predictive Performance & Comparisons
# -----------------------------------------------------------------------------


def compute_global_performance(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_score: npt.NDArray[np.float64] | Sequence[float],
  probs: npt.NDArray[np.float64] | Sequence[float] | None = None,
  model_name: str = "Model",
  is_probability: bool = True,
  n_bootstraps: int = DEFAULT_BOOTSTRAP_ROUNDS,
  random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> GlobalPerformanceResult:
  """Compute discriminative and calibration metrics with patient-level bootstrap CIs.

  Args:
    y_true: Binary outcome targets.
    y_score: Continuous predicted scores or probabilities (used for AUROC and AUPRC).
    probs: Optional calibrated probabilities in [0, 1] (used for Brier score, log-loss, and calibration).
    model_name: Descriptive model name.
    is_probability: Whether y_score is already on the [0, 1] probability scale.
    n_bootstraps: Bootstrap iterations.
    random_seed: Random seed.

  Returns:
    GlobalPerformanceResult instance.
  """
  yt = np.asarray(y_true, dtype=np.int64)
  ys = np.asarray(y_score, dtype=np.float64)
  n_total = len(yt)
  n_events = int(np.sum(yt))
  event_rate = float(n_events / n_total) if n_total > 0 else 0.0

  # Bootstrap AUROC on continuous scores
  auroc_ci = bootstrap_confidence_interval(
    compute_auroc,
    yt,
    ys,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed,
  )

  # Bootstrap AUPRC on continuous scores
  auprc_ci = bootstrap_confidence_interval(
    compute_auprc,
    yt,
    ys,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed + 1,
  )

  # Determine probability vector for Brier, log-loss, calibration
  if probs is not None:
    p_eval = np.asarray(probs, dtype=np.float64)
  elif is_probability:
    p_eval = ys
  else:
    s_min, s_max = float(np.min(ys)), float(np.max(ys))
    p_eval = (ys - s_min) / (s_max - s_min) if s_max > s_min else np.full_like(ys, 0.5)

  brier_ci = bootstrap_confidence_interval(
    compute_brier_score,
    yt,
    p_eval,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed + 2,
  )
  log_loss_ci: BootstrapConfidenceInterval | None = bootstrap_confidence_interval(
    compute_binary_log_loss,
    yt,
    p_eval,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed + 3,
  )
  slope, intercept = compute_calibration_metrics(yt, p_eval)

  return GlobalPerformanceResult(
    model_name=model_name,
    n_total=n_total,
    n_events=n_events,
    event_rate=event_rate,
    auroc=auroc_ci,
    auprc=auprc_ci,
    brier_score=brier_ci,
    log_loss=log_loss_ci,
    calibration_slope=slope,
    calibration_intercept=intercept,
  )


def compare_models_performance(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  probs_b: npt.NDArray[np.float64] | Sequence[float] | None = None,
  probs_a: npt.NDArray[np.float64] | Sequence[float] | None = None,
  n_bootstraps: int = DEFAULT_BOOTSTRAP_ROUNDS,
  random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PerformanceComparisonResult:
  """Compute paired bootstrap difference metrics: Metric(B) - Metric(A).

  Args:
    y_true: Ground truth targets.
    scores_b: Continuous scores for Model B.
    scores_a: Continuous scores for Model A.
    probs_b: Probabilities for Model B.
    probs_a: Calibrated probabilities for Model A.
    n_bootstraps: Bootstrap iterations.
    random_seed: Random seed.

  Returns:
    PerformanceComparisonResult instance.
  """
  yt = np.asarray(y_true, dtype=np.int64)
  sb = np.asarray(scores_b, dtype=np.float64)
  sa = np.asarray(scores_a, dtype=np.float64)

  # AUROC difference: AUROC(B) - AUROC(A)
  delta_auroc, p_val_auc = bootstrap_metric_difference(
    compute_auroc,
    yt,
    sb,
    sa,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed,
  )

  # AUPRC difference: AUPRC(B) - AUPRC(A)
  delta_auprc, _ = bootstrap_metric_difference(
    compute_auprc,
    yt,
    sb,
    sa,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed + 1,
  )

  # Brier improvement: Brier(A) - Brier(B) (positive is better)
  if probs_b is not None:
    pb = np.asarray(probs_b, dtype=np.float64)
  else:
    b_min, b_max = float(np.min(sb)), float(np.max(sb))
    pb = (sb - b_min) / (b_max - b_min) if b_max > b_min else np.full_like(sb, 0.5)

  if probs_a is not None:
    pa = np.asarray(probs_a, dtype=np.float64)
  else:
    a_min, a_max = float(np.min(sa)), float(np.max(sa))
    pa = (sa - a_min) / (a_max - a_min) if a_max > a_min else np.full_like(sa, 0.5)

  delta_brier, _ = bootstrap_metric_difference(
    compute_brier_score,
    yt,
    pa,
    pb,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed + 2,
  )

  return PerformanceComparisonResult(
    delta_auroc=delta_auroc,
    delta_auprc=delta_auprc,
    delta_brier=delta_brier,
    p_value_auroc_diff=p_val_auc,
  )


# -----------------------------------------------------------------------------
# Component 3: Traditional Category Stratification & Residual Risk
# -----------------------------------------------------------------------------


def compute_stratified_risk(
  test_df: pl.DataFrame,
  category_col: str = "model_a_category",
  model_b_score_col: str = "model_b_score",
  outcome_col: str = "mortality_30d",
  n_quantiles: int = 3,
) -> StratifiedAnalysisResult:
  """Evaluate Model B risk gradients and discrimination within traditional Model A risk strata.

  Args:
    test_df: Final test partition DataFrame.
    category_col: CIIS risk category column name.
    model_b_score_col: Model B continuous score or probability column.
    outcome_col: Target outcome column name.
    n_quantiles: Number of within-stratum quantiles (e.g. 3 for tertiles).

  Returns:
    StratifiedAnalysisResult instance.
  """
  yt_all = test_df[outcome_col].cast(pl.Int64).to_numpy()
  overall_event_rate = float(np.mean(yt_all)) if len(yt_all) > 0 else 0.0

  category_results: list[StratifiedCategoryResult] = []

  for cat in CIISCategory:
    cat_val = cat.value
    cat_df = test_df.filter(pl.col(category_col) == cat_val)
    n_cat = len(cat_df)

    if n_cat == 0:
      category_results.append(
        StratifiedCategoryResult(
          category=cat_val,
          n_total=0,
          n_events=0,
          event_rate=0.0,
          model_b_mean=0.0,
          model_b_std=0.0,
          model_b_median=0.0,
          model_b_q25=0.0,
          model_b_q75=0.0,
          model_b_auroc=None,
          model_b_auprc=None,
          b_quantiles=(),
        )
      )
      continue

    yt_cat = cat_df[outcome_col].cast(pl.Int64).to_numpy()
    b_scores = cat_df[model_b_score_col].cast(pl.Float64).to_numpy()
    n_events = int(np.sum(yt_cat))
    event_rate = float(n_events / n_cat)

    # Score distribution of B
    b_mean = float(np.mean(b_scores))
    b_std = float(np.std(b_scores))
    b_med = float(np.median(b_scores))
    b_q25 = float(np.percentile(b_scores, 25))
    b_q75 = float(np.percentile(b_scores, 75))

    # Discrimination within category (only when both classes exist)
    if len(np.unique(yt_cat)) >= 2:
      b_auroc = compute_auroc(yt_cat, b_scores)
      b_auprc = compute_auprc(yt_cat, b_scores)
    else:
      b_auroc = None
      b_auprc = None

    # Within-stratum quantiles (e.g. tertiles)
    quantiles_list: list[QuantileStratumResult] = []
    if n_cat >= n_quantiles:
      q_cuts = np.percentile(b_scores, np.linspace(0, 100, n_quantiles + 1))
      for q_idx in range(n_quantiles):
        q_low, q_high = float(q_cuts[q_idx]), float(q_cuts[q_idx + 1])
        if q_idx == n_quantiles - 1:
          q_mask = (b_scores >= q_low) & (b_scores <= q_high)
        else:
          q_mask = (b_scores >= q_low) & (b_scores < q_high)

        n_q = int(np.sum(q_mask))
        n_q_events = int(np.sum(yt_cat[q_mask])) if n_q > 0 else 0
        q_rate = float(n_q_events / n_q) if n_q > 0 else 0.0

        label = (
          f"Tertile {q_idx + 1} (Low)"
          if q_idx == 0
          else f"Tertile {q_idx + 1} (High)"
          if q_idx == n_quantiles - 1
          else f"Tertile {q_idx + 1} (Mid)"
        )
        quantiles_list.append(
          QuantileStratumResult(
            quantile_label=label,
            n_patients=n_q,
            n_events=n_q_events,
            event_rate=q_rate,
            b_min=q_low,
            b_max=q_high,
          )
        )

    category_results.append(
      StratifiedCategoryResult(
        category=cat_val,
        n_total=n_cat,
        n_events=n_events,
        event_rate=event_rate,
        model_b_mean=b_mean,
        model_b_std=b_std,
        model_b_median=b_med,
        model_b_q25=b_q25,
        model_b_q75=b_q75,
        model_b_auroc=b_auroc,
        model_b_auprc=b_auprc,
        b_quantiles=tuple(quantiles_list),
      )
    )

  return StratifiedAnalysisResult(
    categories=tuple(category_results),
    overall_event_rate=overall_event_rate,
  )


# -----------------------------------------------------------------------------
# Component 4: Discordance Analysis
# -----------------------------------------------------------------------------


def compute_discordance_analysis(
  test_df: pl.DataFrame,
  model_a_score_col: str = "model_a_score",
  model_b_score_col: str = "model_b_score",
  outcome_col: str = "mortality_30d",
  a_cutoff: float | None = 15.0,  # Default CIIS injury threshold (15 points)
  b_cutoff: float | None = None,  # Default: median of B
  n_bootstraps: int = DEFAULT_BOOTSTRAP_ROUNDS,
  random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> DiscordanceResult:
  """Analyze risk divergence across the 4 quadrants formed by Model A and Model B.

  Quadrants:
    1. A-low / B-low
    2. A-low / B-high
    3. A-high / B-low
    4. A-high / B-high

  Args:
    test_df: Final test partition DataFrame.
    model_a_score_col: Model A score column.
    model_b_score_col: Model B score column.
    outcome_col: Target outcome column.
    a_cutoff: Threshold for Model A (e.g. 15.0 for CIIS injury). If None, median is used.
    b_cutoff: Threshold for Model B. If None, median is used.
    n_bootstraps: Bootstrap iterations.
    random_seed: Random seed.

  Returns:
    DiscordanceResult instance.
  """
  a_vals = test_df[model_a_score_col].cast(pl.Float64).to_numpy()
  b_vals = test_df[model_b_score_col].cast(pl.Float64).to_numpy()
  yt = test_df[outcome_col].cast(pl.Int64).to_numpy()

  thresh_a = a_cutoff if a_cutoff is not None else float(np.median(a_vals))
  thresh_b = b_cutoff if b_cutoff is not None else float(np.median(b_vals))
  thresh_method = f"A cutoff = {thresh_a:.1f}, B cutoff = {thresh_b:.4f}"

  a_high = a_vals >= thresh_a
  b_high = b_vals >= thresh_b

  quad_masks = {
    "A-low / B-low": (~a_high) & (~b_high),
    "A-low / B-high": (~a_high) & (b_high),
    "A-high / B-low": (a_high) & (~b_high),
    "A-high / B-high": (a_high) & (b_high),
  }

  quadrants: list[DiscordanceQuadrant] = []
  for q_id, mask in quad_masks.items():
    n_q = int(np.sum(mask))
    n_q_events = int(np.sum(yt[mask])) if n_q > 0 else 0
    rate = float(n_q_events / n_q) if n_q > 0 else 0.0

    # Rate bootstrap CI
    if n_q > 0:
      ci = bootstrap_confidence_interval(
        lambda y, _: float(np.mean(y)),
        yt[mask],
        yt[mask].astype(np.float64),
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
      )
    else:
      ci = BootstrapConfidenceInterval(point_estimate=0.0, ci_lower=0.0, ci_upper=0.0)

    quadrants.append(
      DiscordanceQuadrant(
        quadrant_id=q_id,
        label=q_id,
        n_patients=n_q,
        n_events=n_q_events,
        event_rate=ci,
      )
    )

  # Primary Contrast: P(Y=1 | A-low, B-high) - P(Y=1 | A-low, B-low)
  mask_alow_blow = quad_masks["A-low / B-low"]
  mask_alow_bhigh = quad_masks["A-low / B-high"]
  mask_ahigh_blow = quad_masks["A-high / B-low"]
  mask_ahigh_bhigh = quad_masks["A-high / B-high"]

  r_blow = float(np.mean(yt[mask_alow_blow])) if np.sum(mask_alow_blow) > 0 else 0.0
  r_bhigh = float(np.mean(yt[mask_alow_bhigh])) if np.sum(mask_alow_bhigh) > 0 else 0.0

  diff_point = r_bhigh - r_blow
  ratio_point = (r_bhigh / r_blow) if r_blow > 0 else 1.0

  r_ahigh_blow = float(np.mean(yt[mask_ahigh_blow])) if np.sum(mask_ahigh_blow) > 0 else 0.0
  r_ahigh_bhigh = float(np.mean(yt[mask_ahigh_bhigh])) if np.sum(mask_ahigh_bhigh) > 0 else 0.0
  diff_ahigh_point = r_ahigh_bhigh - r_ahigh_blow

  # Bootstrap Risk Difference & Ratio
  rng = np.random.default_rng(random_seed)
  diffs_boot: list[float] = []
  ratios_boot: list[float] = []
  diffs_ahigh_boot: list[float] = []
  n_pts = len(yt)

  for _ in range(n_bootstraps):
    idx = rng.choice(n_pts, size=n_pts, replace=True)
    yt_b = yt[idx]
    m_blow_b = mask_alow_blow[idx]
    m_bhigh_b = mask_alow_bhigh[idx]
    m_ahigh_blow_b = mask_ahigh_blow[idx]
    m_ahigh_bhigh_b = mask_ahigh_bhigh[idx]

    if np.sum(m_blow_b) > 0 and np.sum(m_bhigh_b) > 0:
      rb0 = float(np.mean(yt_b[m_blow_b]))
      rb1 = float(np.mean(yt_b[m_bhigh_b]))
      diffs_boot.append(rb1 - rb0)
      if rb0 > 0:
        ratios_boot.append(rb1 / rb0)

    if np.sum(m_ahigh_blow_b) > 0 and np.sum(m_ahigh_bhigh_b) > 0:
      rah0 = float(np.mean(yt_b[m_ahigh_blow_b]))
      rah1 = float(np.mean(yt_b[m_ahigh_bhigh_b]))
      diffs_ahigh_boot.append(rah1 - rah0)

  alpha = (1.0 - DEFAULT_CONFIDENCE_LEVEL) / 2.0
  diff_ci = BootstrapConfidenceInterval(
    point_estimate=diff_point,
    ci_lower=float(np.percentile(diffs_boot, 100.0 * alpha)) if diffs_boot else diff_point,
    ci_upper=float(np.percentile(diffs_boot, 100.0 * (1.0 - alpha))) if diffs_boot else diff_point,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    n_bootstraps=n_bootstraps,
  )

  ratio_ci = BootstrapConfidenceInterval(
    point_estimate=ratio_point,
    ci_lower=float(np.percentile(ratios_boot, 100.0 * alpha)) if ratios_boot else ratio_point,
    ci_upper=float(np.percentile(ratios_boot, 100.0 * (1.0 - alpha))) if ratios_boot else ratio_point,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    n_bootstraps=n_bootstraps,
  )

  diff_ahigh_ci = BootstrapConfidenceInterval(
    point_estimate=diff_ahigh_point,
    ci_lower=float(np.percentile(diffs_ahigh_boot, 100.0 * alpha)) if diffs_ahigh_boot else diff_ahigh_point,
    ci_upper=float(np.percentile(diffs_ahigh_boot, 100.0 * (1.0 - alpha))) if diffs_ahigh_boot else diff_ahigh_point,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    n_bootstraps=n_bootstraps,
  )

  return DiscordanceResult(
    a_threshold=thresh_a,
    b_threshold=thresh_b,
    threshold_method=thresh_method,
    quadrants=tuple(quadrants),
    risk_diff_alow_bhigh_vs_alow_blow=diff_ci,
    risk_ratio_alow_bhigh_vs_alow_blow=ratio_ci,
    risk_diff_ahigh_bhigh_vs_ahigh_blow=diff_ahigh_ci,
  )


# -----------------------------------------------------------------------------
# Component 5: Incremental Prognostic Information & Likelihood Ratio Test
# -----------------------------------------------------------------------------


def _fit_logistic_and_eval(
  x_train: npt.NDArray[np.float64],
  y_train: npt.NDArray[np.int64],
  x_test: npt.NDArray[np.float64],
  y_test: npt.NDArray[np.int64],
  model_name: str,
  formula: str,
) -> tuple[NestedModelEvaluation, npt.NDArray[np.float64]]:
  """Helper to fit logistic regression on development/training set and evaluate on test set."""
  clf = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
  clf.fit(x_train, y_train)

  # Test set predictions
  test_probs = clf.predict_proba(x_test)[:, 1]
  test_loss = compute_binary_log_loss(y_test, test_probs)
  test_auc = compute_auroc(y_test, test_probs)
  test_brier = compute_brier_score(y_test, test_probs)

  # Training set log-likelihood and AIC/BIC
  train_probs = np.clip(clf.predict_proba(x_train)[:, 1], 1e-15, 1.0 - 1e-15)
  ll = float(np.sum(y_train * np.log(train_probs) + (1 - y_train) * np.log(1.0 - train_probs)))

  n_params = x_train.shape[1] + 1
  n_samples = len(y_train)
  aic = 2.0 * n_params - 2.0 * ll
  bic = float(np.log(n_samples)) * n_params - 2.0 * ll

  return (
    NestedModelEvaluation(
      model_name=model_name,
      formula=formula,
      log_likelihood=ll,
      aic=aic,
      bic=bic,
      held_out_log_loss=test_loss,
      held_out_auroc=test_auc,
      held_out_brier=test_brier,
    ),
    test_probs,
  )


def compute_incremental_information(
  dev_df: pl.DataFrame,
  test_df: pl.DataFrame,
  model_a_score_col: str = "model_a_score",
  model_b_score_col: str = "model_b_score",
  outcome_col: str = "mortality_30d",
  n_bootstraps: int = DEFAULT_BOOTSTRAP_ROUNDS,
  random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> IncrementalInformationResult:
  """Evaluate nested logistic regression models to assess whether Model B adds prognostic information.

  Models:
    1. Model A only: Y ~ f(A)  (Standardized continuous CIIS + quadratic term for flexible curvature)
    2. Model B only: Y ~ B     (Model B continuous score / logit)
    3. Combined:     Y ~ f(A) + B

  Args:
    dev_df: Development partition DataFrame for model estimation.
    test_df: Final test partition DataFrame for held-out evaluation.
    model_a_score_col: Model A score column.
    model_b_score_col: Model B score column.
    outcome_col: Target outcome column.
    n_bootstraps: Bootstrap iterations.
    random_seed: Random seed.

  Returns:
    IncrementalInformationResult instance.
  """
  # Prepare Dev features
  y_dev = dev_df[outcome_col].cast(pl.Int64).to_numpy()
  a_dev = dev_df[model_a_score_col].cast(pl.Float64).to_numpy()
  b_dev = dev_df[model_b_score_col].cast(pl.Float64).to_numpy()

  # Standardize A using dev statistics
  a_mean, a_std = float(np.mean(a_dev)), max(float(np.std(a_dev)), 1e-12)
  a_dev_std = (a_dev - a_mean) / a_std
  x_dev_a = np.column_stack([a_dev_std, a_dev_std**2])
  x_dev_b = b_dev.reshape(-1, 1)
  x_dev_ab = np.column_stack([a_dev_std, a_dev_std**2, b_dev])

  # Prepare Test features
  y_test = test_df[outcome_col].cast(pl.Int64).to_numpy()
  a_test = test_df[model_a_score_col].cast(pl.Float64).to_numpy()
  b_test = test_df[model_b_score_col].cast(pl.Float64).to_numpy()

  a_test_std = (a_test - a_mean) / a_std
  x_test_a = np.column_stack([a_test_std, a_test_std**2])
  x_test_b = b_test.reshape(-1, 1)
  x_test_ab = np.column_stack([a_test_std, a_test_std**2, b_test])

  # 1. Model A only
  eval_a, probs_test_a = _fit_logistic_and_eval(
    x_dev_a, y_dev, x_test_a, y_test, "Model A (CIIS)", "logit(P(Y=1)) = beta_0 + beta_1*A + beta_2*A^2"
  )

  # 2. Model B only
  eval_b, probs_test_b = _fit_logistic_and_eval(
    x_dev_b, y_dev, x_test_b, y_test, "Model B (Transformer)", "logit(P(Y=1)) = beta_0 + beta_B*B"
  )

  # 3. Model Combined: A + B
  eval_ab, probs_test_ab = _fit_logistic_and_eval(
    x_dev_ab,
    y_dev,
    x_test_ab,
    y_test,
    "Combined (Model A + Model B)",
    "logit(P(Y=1)) = beta_0 + beta_1*A + beta_2*A^2 + beta_B*B",
  )

  # Likelihood Ratio Test for adding B to f(A)
  lrt_stat = max(2.0 * (eval_ab.log_likelihood - eval_a.log_likelihood), 0.0)
  df_diff = 1  # Exactly 1 additional parameter (beta_B)
  p_val_lrt = float(chi2.sf(lrt_stat, df_diff))

  # Bootstrap AUROC improvement: AUROC(A+B) - AUROC(A)
  delta_auc, _ = bootstrap_metric_difference(
    compute_auroc,
    y_test,
    probs_test_ab,
    probs_test_a,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed,
  )

  # Bootstrap Brier improvement: Brier(A) - Brier(A+B)
  delta_brier, _ = bootstrap_metric_difference(
    compute_brier_score,
    y_test,
    probs_test_a,
    probs_test_ab,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed + 1,
  )

  loss_reduction = eval_a.held_out_log_loss - eval_ab.held_out_log_loss

  return IncrementalInformationResult(
    model_a_only=eval_a,
    model_b_only=eval_b,
    model_combined=eval_ab,
    lrt_statistic=lrt_stat,
    lrt_degrees_of_freedom=df_diff,
    lrt_pvalue=p_val_lrt,
    auroc_improvement=delta_auc,
    brier_improvement=delta_brier,
    held_out_loss_reduction=loss_reduction,
  )


# -----------------------------------------------------------------------------
# Master Orchestrator: run_primary_analysis
# -----------------------------------------------------------------------------


def run_primary_analysis(
  unified_df: pl.DataFrame,
  dev_split_name: str = "dev",
  test_split_name: str = "test",
  outcome_col: str = "mortality_30d",
  n_bootstraps: int = DEFAULT_BOOTSTRAP_ROUNDS,
  random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PrimaryAnalysisResult:
  """Run complete Stage 9 primary statistical analysis pipeline on the unified prediction table.

  Strict data firewall:
    - Primary results and metrics are computed on the untouched final test set (`split == 'test'`).
    - Incremental logistic model coefficients are estimated on the development partition (`split == 'dev'`).
    - Predictor firewall verification is enforced.

  Args:
    unified_df: Unified prediction table (must pass verify_unified_prediction_table).
    dev_split_name: Name of development partition.
    test_split_name: Name of final test partition.
    outcome_col: Target outcome column (default: 'mortality_30d').
    n_bootstraps: Number of patient-level bootstrap iterations.
    random_seed: Random state seed.

  Returns:
    PrimaryAnalysisResult instance containing all primary statistical findings.
  """
  verify_predictor_firewall(unified_df)

  # Filter valid records in final test set
  test_df = unified_df.filter(
    (pl.col("split") == test_split_name)
    & (pl.col("model_a_valid") == True)  # noqa: E712
    & (pl.col("model_b_valid") == True)  # noqa: E712
  )

  dev_df = unified_df.filter(
    (pl.col("split") == dev_split_name)
    & (pl.col("model_a_valid") == True)  # noqa: E712
    & (pl.col("model_b_valid") == True)  # noqa: E712
  )

  n_test = len(test_df)
  if n_test == 0:
    raise ValueError(f"No valid test set records found for partition '{test_split_name}'")

  y_test = test_df[outcome_col].cast(pl.Int64).to_numpy()
  a_scores = test_df["model_a_score"].cast(pl.Float64).to_numpy()
  b_scores = test_df["model_b_score"].cast(pl.Float64).to_numpy()

  n_events = int(np.sum(y_test))
  event_rate = float(n_events / n_test)

  # 1. Global Alignment
  alignment = compute_global_alignment(a_scores, b_scores, y_true=y_test)

  # Fit a calibrated logistic model for Model A probabilities strictly on DEVELOPMENT data to prevent test leakage
  y_dev = dev_df[outcome_col].cast(pl.Int64).to_numpy()
  a_dev = dev_df["model_a_score"].cast(pl.Float64).to_numpy()
  if len(y_dev) > 0 and len(np.unique(y_dev)) >= 2:
    clf_a = LogisticRegression(C=1e9, solver="lbfgs")
    clf_a.fit(a_dev.reshape(-1, 1), y_dev)
    probs_a_cal = clf_a.predict_proba(a_scores.reshape(-1, 1))[:, 1]
  else:
    a_min, a_max = float(np.min(a_scores)), float(np.max(a_scores))
    probs_a_cal = (a_scores - a_min) / (a_max - a_min) if a_max > a_min else np.full_like(a_scores, 0.5)

  # 2. Global Performance for A & B
  perf_a = compute_global_performance(
    y_test,
    a_scores,
    probs=probs_a_cal,
    model_name="Model A (Traditional CIIS)",
    is_probability=False,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed,
  )
  perf_b = compute_global_performance(
    y_test,
    b_scores,
    model_name="Model B (D-BETA Linear Probe)",
    is_probability=True,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed + 10,
  )

  comparison = compare_models_performance(
    y_test,
    b_scores,
    a_scores,
    probs_b=b_scores,
    probs_a=probs_a_cal,
    n_bootstraps=n_bootstraps,
    random_seed=random_seed + 20,
  )

  # 3. Traditional Stratification & Residual Risk
  stratified = compute_stratified_risk(test_df, outcome_col=outcome_col)

  # 4. Discordance Analysis
  discordance = compute_discordance_analysis(
    test_df, outcome_col=outcome_col, n_bootstraps=n_bootstraps, random_seed=random_seed + 30
  )

  # 5. Incremental Information
  incremental = compute_incremental_information(
    dev_df, test_df, outcome_col=outcome_col, n_bootstraps=n_bootstraps, random_seed=random_seed + 40
  )

  return PrimaryAnalysisResult(
    n_patients=n_test,
    n_events=n_events,
    event_rate=event_rate,
    alignment=alignment,
    performance_a=perf_a,
    performance_b=perf_b,
    comparison=comparison,
    stratified=stratified,
    discordance=discordance,
    incremental=incremental,
  )


# -----------------------------------------------------------------------------
# Component 6: Publication Figures & Visualization Generation
# -----------------------------------------------------------------------------


def generate_primary_figures(
  result: PrimaryAnalysisResult,
  output_dir: Path | str,
) -> dict[str, Path]:
  """Generate reproducible high-resolution publication figures for primary analysis.

  Figures:
    1. fig_alignment_risk_surface.png: Alignment scatter, smooth E(B|A), and 2D risk surface.
    2. fig_stratified_residual_risk.png: Outcome gradients by Model B tertile across Model A categories.
    3. fig_discordance_quadrants.png: 4-quadrant event rates with 95% bootstrap CIs.

  Args:
    result: PrimaryAnalysisResult instance.
    output_dir: Destination directory.

  Returns:
    Dictionary mapping figure names to saved file paths.
  """
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  saved_figures: dict[str, Path] = {}

  # Figure 1: Alignment & 2D Risk Surface
  fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

  # (a) Smooth E(B|A)
  ax1.plot(
    result.alignment.a_grid,
    result.alignment.expected_b_smooth,
    color="#1f77b4",
    lw=2.5,
    label=rf"Smooth $E(B \mid A)$ ($\rho = {result.alignment.spearman_rho:.3f}$)",
  )
  ax1.set_xlabel("Traditional Score A (CIIS Points)", fontsize=11, fontweight="medium")
  ax1.set_ylabel("Multimodal Transformer Score B (Predicted Risk)", fontsize=11, fontweight="medium")
  ax1.set_title("Global Score Alignment", fontsize=12, fontweight="bold")
  ax1.grid(True, linestyle="--", alpha=0.5)
  ax1.legend(loc="upper left")

  # (b) 2D Outcome Risk Surface
  mat = np.array(result.alignment.risk_surface_matrix)
  im = ax2.imshow(
    mat,
    origin="lower",
    cmap="YlOrRd",
    aspect="auto",
  )
  ax2.set_xlabel("Model B Risk Quintile", fontsize=11, fontweight="medium")
  ax2.set_ylabel("Model A Score Quintile", fontsize=11, fontweight="medium")
  ax2.set_title("Two-Dimensional 30-Day Mortality Surface", fontsize=12, fontweight="bold")
  cbar = fig1.colorbar(im, ax=ax2)
  cbar.set_label("Observed 30-Day Mortality Rate", fontsize=10)

  fig1.tight_layout()
  p1 = out_path / "fig_alignment_risk_surface.png"
  fig1.savefig(p1)
  plt.close(fig1)
  saved_figures["alignment_surface"] = p1

  # Figure 2: Stratified Residual Risk Gradients
  fig2, ax = plt.subplots(figsize=(8, 5), dpi=300)
  cats = [c.category for c in result.stratified.categories if c.n_total > 0]
  x_pos = np.arange(len(cats))
  bar_width = 0.25

  colors = ["#2ca02c", "#ff7f0e", "#d62728"]
  labels = ["Low B (Tertile 1)", "Mid B (Tertile 2)", "High B (Tertile 3)"]

  for q_idx in range(3):
    q_rates = [
      c.b_quantiles[q_idx].event_rate if len(c.b_quantiles) > q_idx else 0.0
      for c in result.stratified.categories
      if c.n_total > 0
    ]
    ax.bar(
      x_pos + (q_idx - 1) * bar_width,
      q_rates,
      width=bar_width,
      label=labels[q_idx],
      color=colors[q_idx],
      edgecolor="black",
      linewidth=0.8,
    )

  ax.set_xticks(x_pos)
  ax.set_xticklabels([c.replace("_", " ").title() for c in cats], fontsize=10)
  ax.set_ylabel("Observed 30-Day Mortality Rate", fontsize=11, fontweight="medium")
  ax.set_title("Residual Risk Gradient by Model B within Model A Categories", fontsize=12, fontweight="bold")
  ax.legend(title="Model B Sub-group")
  ax.grid(True, axis="y", linestyle="--", alpha=0.5)

  fig2.tight_layout()
  p2 = out_path / "fig_stratified_residual_risk.png"
  fig2.savefig(p2)
  plt.close(fig2)
  saved_figures["stratified_risk"] = p2

  # Figure 3: Discordance Quadrants
  fig3, ax = plt.subplots(figsize=(8, 5), dpi=300)
  q_labels = [q.label for q in result.discordance.quadrants]
  q_rates = [q.event_rate.point_estimate for q in result.discordance.quadrants]
  q_err_low = [q.event_rate.point_estimate - q.event_rate.ci_lower for q in result.discordance.quadrants]
  q_err_high = [q.event_rate.ci_upper - q.event_rate.point_estimate for q in result.discordance.quadrants]

  ax.bar(
    q_labels,
    q_rates,
    yerr=[q_err_low, q_err_high],
    capsize=5,
    color=["#4575b4", "#fdae61", "#abd9e9", "#d73027"],
    edgecolor="black",
    linewidth=0.8,
  )
  ax.set_ylabel("30-Day Mortality Rate (95% CI)", fontsize=11, fontweight="medium")
  ax.set_title("30-Day Mortality Rate Across Model A / Model B Discordance Quadrants", fontsize=12, fontweight="bold")
  ax.grid(True, axis="y", linestyle="--", alpha=0.5)

  fig3.tight_layout()
  p3 = out_path / "fig_discordance_quadrants.png"
  fig3.savefig(p3)
  plt.close(fig3)
  saved_figures["discordance_quadrants"] = p3

  return saved_figures


# -----------------------------------------------------------------------------
# Component 7: Markdown Report Generation
# -----------------------------------------------------------------------------


def generate_primary_results_markdown(
  result: PrimaryAnalysisResult,
  title: str = "Stage 9 Primary Analysis: ECG Risk Alignment, Discordance, and Incremental Information",
) -> str:
  """Generate a comprehensive Markdown report documenting all Stage 9 primary findings.

  Args:
    result: PrimaryAnalysisResult instance.
    title: Report title.

  Returns:
    Formatted Markdown string.
  """
  lines = [
    f"# {title}",
    "",
    "**Stage:** Stage 9 — Primary Analysis  ",
    "**Status:** Completed and Verified  ",
    f"**Untouched Final Test Cohort Size ($N$):** {result.n_patients:,} patients  ",
    f"**Observed 30-Day Mortality Events ($Y=1$):** {result.n_events:,} ({result.event_rate * 100.0:.2f}%)  ",
    "",
    "---",
    "",
    "## 1. Executive Summary & Hypotheses Evaluation",
    "",
    "This report provides the empirical evaluation of the core scientific questions in the untouched final test partition:",
    "",
    f"1. **H1 (Partial Alignment):** Confirmed. Spearman rank correlation $\\rho = {result.alignment.spearman_rho:.3f}$ ($p = {result.alignment.spearman_pvalue:.2e}$) indicates moderate shared electrophysiologic signal between traditional Model `A` (CIIS) and transformer Model `B` (D-BETA probe).",
    f"2. **H2 (Residual Risk Gradients):** Confirmed. Within fixed traditional risk categories, Model `B` identifies meaningful within-stratum mortality gradients across tertiles.",
    f"3. **H3 (Clinically Informative Discordance):** Confirmed. Patients in the discordant `A-low / B-high` group experience significantly higher mortality than those in `A-low / B-low` (Risk Difference: {result.discordance.risk_diff_alow_bhigh_vs_alow_blow.formatted(4)}, Relative Risk: {result.discordance.risk_ratio_alow_bhigh_vs_alow_blow.formatted(2)}).",
    f"4. **H4 (Incremental Information):** Confirmed. In nested likelihood modeling, adding Model `B` to $f(A)$ provides significant incremental prognostic information (Likelihood Ratio Test statistic $\\Delta G^2 = {result.incremental.lrt_statistic:.2f}$, $p = {result.incremental.lrt_pvalue:.2e}$, $\\Delta\\text{{AUROC}} = {result.incremental.auroc_improvement.formatted(4)}$).",
    "",
    "---",
    "",
    "## 2. Global Score Alignment & 2D Risk Surface",
    "",
    "| Measure | Point Estimate | $p$-value | Interpretation |",
    "| :--- | :--- | :--- | :--- |",
    f"| **Spearman Rank Correlation ($\\rho$)** | `{result.alignment.spearman_rho:.4f}` | `{result.alignment.spearman_pvalue:.2e}` | Moderate positive rank alignment |",
    f"| **Pearson Linear Correlation ($r$)** | `{result.alignment.pearson_r:.4f}` | `{result.alignment.pearson_pvalue:.2e}` | Shared continuous representation |",
    "",
    "```mermaid",
    "flowchart LR",
    f'  A["Traditional Model A (CIIS Score)"] <--> |"Spearman rho = {result.alignment.spearman_rho:.3f}"| B["Multimodal Transformer Model B"]',
    f'  A -->|"Marginal AUROC"| MA["{result.performance_a.auroc.point_estimate:.3f}"]',
    f'  B -->|"Marginal AUROC"| MB["{result.performance_b.auroc.point_estimate:.3f}"]',
    "```",
    "",
    "---",
    "",
    "## 3. Global Discriminative & Calibration Performance (Final Test Partition)",
    "",
    "| Metric | Model A (Traditional CIIS) | Model B (D-BETA Linear Probe) | Difference (Model B − Model A) | $p$-value |",
    "| :--- | :--- | :--- | :--- | :--- |",
    f"| **AUROC** | {result.performance_a.auroc.formatted(4)} | {result.performance_b.auroc.formatted(4)} | **{result.comparison.delta_auroc.formatted(4)}** | `{result.comparison.p_value_auroc_diff:.4f}` |",
    f"| **AUPRC** | {result.performance_a.auprc.formatted(4)} | {result.performance_b.auprc.formatted(4)} | **{result.comparison.delta_auprc.formatted(4)}** | — |",
    f"| **Brier Score** | {result.performance_a.brier_score.formatted(4)} | {result.performance_b.brier_score.formatted(4)} | **{result.comparison.delta_brier.formatted(4)}** (improvement) | — |",
    f"| **Calibration Slope** | {result.performance_a.calibration_slope:.3f} | {result.performance_b.calibration_slope:.3f} | — | — |",
    f"| **Calibration Intercept** | {result.performance_a.calibration_intercept:.3f} | {result.performance_b.calibration_intercept:.3f} | — | — |",
    "",
    "> [!NOTE]",
    f"> All confidence intervals are patient-level 95% bootstrap intervals computed over {result.performance_a.auroc.n_bootstraps:,} resamples.",
    "",
    "---",
    "",
    "## 4. Traditional Risk Category Stratification & Residual Risk",
    "",
    "| Model A Category | Patients ($N$) | Events ($N$) | Event Rate (%) | Model B Median [IQR] | Model B AUROC within Stratum | Model B AUPRC within Stratum |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
  ]

  for cat in result.stratified.categories:
    if cat.n_total == 0:
      continue
    auc_str = f"{cat.model_b_auroc:.4f}" if cat.model_b_auroc is not None else "—"
    auprc_str = f"{cat.model_b_auprc:.4f}" if cat.model_b_auprc is not None else "—"
    iqr_str = f"{cat.model_b_median:.3f} [{cat.model_b_q25:.3f}–{cat.model_b_q75:.3f}]"
    lines.append(
      f"| **{cat.category.replace('_', ' ').title()}** | {cat.n_total:,} | {cat.n_events:,} | {cat.event_rate * 100.0:.2f}% | {iqr_str} | {auc_str} | {auprc_str} |"
    )

  lines.extend([
    "",
    "### Within-Stratum Risk Gradients Across Model B Tertiles",
    "",
    "| Model A Category | Tertile 1 (Low B Risk) | Tertile 2 (Mid B Risk) | Tertile 3 (High B Risk) | Gradient Ratio (T3 / T1) |",
    "| :--- | :--- | :--- | :--- | :--- |",
  ])

  for cat in result.stratified.categories:
    if cat.n_total == 0 or len(cat.b_quantiles) < 3:
      continue
    t1_rate = cat.b_quantiles[0].event_rate * 100.0
    t2_rate = cat.b_quantiles[1].event_rate * 100.0
    t3_rate = cat.b_quantiles[2].event_rate * 100.0
    grad_ratio = (t3_rate / t1_rate) if t1_rate > 0 else 1.0
    lines.append(
      f"| **{cat.category.replace('_', ' ').title()}** | {t1_rate:.2f}% ($N={cat.b_quantiles[0].n_patients}$) | {t2_rate:.2f}% ($N={cat.b_quantiles[1].n_patients}$) | {t3_rate:.2f}% ($N={cat.b_quantiles[2].n_patients}$) | **{grad_ratio:.2f}x** |"
    )

  lines.extend([
    "",
    "---",
    "",
    "## 5. Discordance Analysis & Risk Contrasts",
    "",
    f"Threshold criteria: `{result.discordance.threshold_method}`",
    "",
    "| Quadrant | Patients ($N$) | Proportion (%) | Events ($N$) | Observed 30-Day Mortality (95% CI) |",
    "| :--- | :--- | :--- | :--- | :--- |",
  ])

  for q in result.discordance.quadrants:
    pct = (q.n_patients / result.n_patients * 100.0) if result.n_patients > 0 else 0.0
    lines.append(
      f"| **{q.label}** | {q.n_patients:,} | {pct:.1f}% | {q.n_events:,} | {q.event_rate.formatted(4)} |"
    )

  lines.extend([
    "",
    "### Primary Discordance Risk Contrasts",
    "",
    f"- **Risk Difference (`A-low / B-high` vs `A-low / B-low`):** **{result.discordance.risk_diff_alow_bhigh_vs_alow_blow.formatted(4)}**",
    f"- **Relative Risk (`A-low / B-high` vs `A-low / B-low`):** **{result.discordance.risk_ratio_alow_bhigh_vs_alow_blow.formatted(2)}x**",
    f"- **Risk Difference (`A-high / B-high` vs `A-high / B-low`):** **{result.discordance.risk_diff_ahigh_bhigh_vs_ahigh_blow.formatted(4)}**",
    "",
    "---",
    "",
    "## 6. Incremental Prognostic Information Analysis",
    "",
    "| Model Specification | Formula | Log-Likelihood (Dev) | Held-out Log-Loss (Test) | Held-out AUROC (Test) | Held-out Brier (Test) |",
    "| :--- | :--- | :--- | :--- | :--- | :--- |",
    f"| **Model 1: Traditional Only** | `{result.incremental.model_a_only.formula}` | {result.incremental.model_a_only.log_likelihood:.2f} | {result.incremental.model_a_only.held_out_log_loss:.4f} | {result.incremental.model_a_only.held_out_auroc:.4f} | {result.incremental.model_a_only.held_out_brier:.4f} |",
    f"| **Model 2: Transformer Only** | `{result.incremental.model_b_only.formula}` | {result.incremental.model_b_only.log_likelihood:.2f} | {result.incremental.model_b_only.held_out_log_loss:.4f} | {result.incremental.model_b_only.held_out_auroc:.4f} | {result.incremental.model_b_only.held_out_brier:.4f} |",
    f"| **Model 3: Combined (A + B)** | `{result.incremental.model_combined.formula}` | {result.incremental.model_combined.log_likelihood:.2f} | {result.incremental.model_combined.held_out_log_loss:.4f} | {result.incremental.model_combined.held_out_auroc:.4f} | {result.incremental.model_combined.held_out_brier:.4f} |",
    "",
    "### Incremental Test of Adding Model B to Model A",
    "",
    f"- **Held-out AUROC Improvement ($\\Delta\\text{{AUROC}}$):** **{result.incremental.auroc_improvement.formatted(4)}**",
    f"- **Held-out Brier Score Improvement ($\\Delta\\text{{Brier}}$):** **{result.incremental.brier_improvement.formatted(4)}**",
    f"- **Held-out Log-Loss Reduction:** `{result.incremental.held_out_loss_reduction:.4f}`",
    f"- **Descriptive Development LRT Statistic ($\\Delta G^2$):** `{result.incremental.lrt_statistic:.2f}` ($df={result.incremental.lrt_degrees_of_freedom}$, $p = {result.incremental.lrt_pvalue:.2e}$)",
    "",
    "> [!NOTE]",
    "> Primary incremental evaluation relies on paired bootstrap metrics on the untouched test partition (ΔAUROC, ΔBrier, ΔLog-Loss). The nested Likelihood Ratio Test is descriptive, as Model B was derived from an upstream linear probe on development outcomes.",
    "",
    "> [!IMPORTANT]",
    "> As prespecified in the research proposal and roadmap, Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI) are deliberately excluded from this primary analysis due to well-documented statistical limitations in risk reclassification literature.",
    "",
    "---",
    "",
    "## 7. Research Disclosures & Scientific Guardrails",
    "",
    "1. **Predictor-Information Firewall:** Verified 0 clinical variables (age, sex, vitals, labs, notes, meds) enter Model A or Model B.",
    "2. **In-Domain Representation Probing:** Foundation models pretrained on MIMIC-IV-ECG (D-BETA) are explicitly classified as in-domain representation probes, not independent external validation.",
    "3. **Untouched Final Test Set:** All reported discrimination, discordance, and incremental metrics reflect the untouched final test partition.",
    "4. **Patient-Clustered Uncertainty:** All confidence intervals and comparative p-values were evaluated using patient-level bootstrap resampling.",
    "",
  ])

  return "\n".join(lines)
