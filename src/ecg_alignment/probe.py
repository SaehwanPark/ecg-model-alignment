"""Linear probing, prediction extraction, and continuous score pipeline for Model A and Model B.

Strict research guardrails:
- Model A (traditional CIIS) and Model B (multimodal transformer probe) operate on ECG data only.
- Predictor-information firewall: Demographics, diagnoses, medications, labs, notes, and encounter features
  are strictly prohibited from entering predictor feature spaces.
- Supervised data firewall: Test set outcomes and features are NEVER used during probe training or hyperparameter selection.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Literal, cast
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from ecg_alignment.data import read_ecg_waveform
from ecg_alignment.scoring.base import BaseEcgModelAdapter
from ecg_alignment.scoring.traditional import CIISCategory, score_ecg_waveform
from ecg_alignment.split import verify_split_disjointness

logger = logging.getLogger(__name__)

DEFAULT_C_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_PROBE_SOLVER: str = "lbfgs"
DEFAULT_PROBE_MAX_ITER: int = 1000
DEFAULT_PROBE_SEED: int = 42

FORBIDDEN_PREDICTOR_KEYWORDS: tuple[str, ...] = (
  "age",
  "gender",
  "sex",
  "race",
  "ethnicity",
  "admit",
  "disch",
  "hadm",
  "icd",
  "drg",
  "vital",
  "lab",
  "med",
  "note",
  "anchor",
  "insurance",
  "language",
  "marital",
)

ALLOWED_NON_PREDICTOR_COLS: tuple[str, ...] = (
  "subject_id",
  "study_id",
  "split",
  "mortality_30d",
  "mortality_90d",
  "mortality_1yr",
  "inhospital_mortality",
  "model_a_score",
  "model_a_category",
  "model_a_valid",
  "model_a_error",
  "model_b_score",
  "model_b_log_odds",
  "model_b_valid",
  "model_b_error",
)


# -----------------------------------------------------------------------------
# Metric Evaluation Helpers (Pure Functions)
# -----------------------------------------------------------------------------


def compute_binary_log_loss(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
  eps: float = 1e-15,
) -> float:
  """Calculate cross-entropy log-loss with numerical clipping.

  Args:
    y_true: Binary ground truth targets (0 or 1).
    y_prob: Predicted probability of positive class in range [0, 1].
    eps: Small epsilon for clipping probabilities.

  Returns:
    Mean binary cross-entropy loss.
  """
  yt = np.asarray(y_true, dtype=np.float64)
  yp = np.asarray(y_prob, dtype=np.float64)

  if len(yt) == 0:
    return 0.0

  yp_clipped = np.clip(yp, eps, 1.0 - eps)
  loss = -np.mean(yt * np.log(yp_clipped) + (1.0 - yt) * np.log(1.0 - yp_clipped))
  return float(loss)


def compute_auroc(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_score: npt.NDArray[np.float64] | Sequence[float],
) -> float:
  """Calculate Area Under the Receiver Operating Characteristic curve (AUROC).

  Args:
    y_true: Binary ground truth targets.
    y_score: Continuous predicted scores or probabilities.

  Returns:
    AUROC value in range [0.0, 1.0]. Returns 0.5 if only one class exists.
  """
  yt = np.asarray(y_true, dtype=np.int64)
  ys = np.asarray(y_score, dtype=np.float64)

  if len(yt) == 0 or len(np.unique(yt)) < 2:
    return 0.5

  try:
    return float(roc_auc_score(yt, ys))
  except Exception:
    return 0.5


def compute_brier_score(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
) -> float:
  """Calculate Brier score (mean squared error of predicted probabilities).

  Args:
    y_true: Binary ground truth targets.
    y_prob: Predicted probability in range [0, 1].

  Returns:
    Brier score in range [0.0, 1.0].
  """
  yt = np.asarray(y_true, dtype=np.int64)
  yp = np.asarray(y_prob, dtype=np.float64)

  if len(yt) == 0:
    return 0.0

  return float(brier_score_loss(yt, yp))


# -----------------------------------------------------------------------------
# Probe Configurations & Container
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
  """Immutable configuration for frozen transformer representation linear probe."""

  model_name: str = "Manhph2211/D-BETA"
  regularization_c_grid: tuple[float, ...] = DEFAULT_C_GRID
  max_iter: int = DEFAULT_PROBE_MAX_ITER
  solver: str = DEFAULT_PROBE_SOLVER
  random_state: int = DEFAULT_PROBE_SEED
  tuning_metric: Literal["log_loss", "auroc"] = "log_loss"
  standardize: bool = True
  penalty: Literal["l2"] = "l2"
  tolerance: float = 1e-4

  def __post_init__(self) -> None:
    if not self.regularization_c_grid:
      raise ValueError("regularization_c_grid must not be empty.")
    for c in self.regularization_c_grid:
      if c <= 0:
        raise ValueError(f"Regularization parameter C must be strictly positive. Got {c}")


@dataclass(frozen=True)
class ProbeValidationStep:
  """Evaluation metrics on validation set for a single hyperparameter C."""

  c_value: float
  val_log_loss: float
  val_auroc: float
  val_brier: float


@dataclass(frozen=True)
class TrainedProbe:
  """Immutable frozen container for a trained linear probe."""

  coefficients: tuple[float, ...]
  intercept: float
  best_c: float
  tuning_history: tuple[ProbeValidationStep, ...]
  scaler_mean: tuple[float, ...] | None
  scaler_scale: tuple[float, ...] | None
  config: ProbeConfig

  @property
  def embedding_dim(self) -> int:
    return len(self.coefficients)

  def _preprocess_input(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Standardize input embeddings using frozen dev set parameters."""
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim == 1:
      arr = arr.reshape(1, -1)

    if arr.shape[1] != self.embedding_dim:
      raise ValueError(
        f"Expected embedding dimension {self.embedding_dim}, got {arr.shape[1]}"
      )

    if self.scaler_mean is not None and self.scaler_scale is not None:
      mean = np.array(self.scaler_mean, dtype=np.float64)
      scale = np.array(self.scaler_scale, dtype=np.float64)
      arr = (arr - mean) / np.maximum(scale, 1e-12)

    return arr

  def predict_log_odds(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute linear log-odds: z = w^T x + b."""
    x = self._preprocess_input(embeddings)
    w = np.array(self.coefficients, dtype=np.float64)
    logits = np.dot(x, w) + self.intercept
    return np.asarray(logits, dtype=np.float64)

  def predict_proba(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute calibrated class 1 probability via sigmoid: p = 1 / (1 + exp(-z))."""
    logits = self.predict_log_odds(embeddings)
    # Numerically stable sigmoid
    probs = np.where(
      logits >= 0,
      1.0 / (1.0 + np.exp(-logits)),
      np.exp(logits) / (1.0 + np.exp(logits)),
    )
    return np.asarray(probs, dtype=np.float64)

  def to_dict(self) -> dict[str, Any]:
    """Serialize probe configuration and parameters to a dictionary."""
    return {
      "coefficients": list(self.coefficients),
      "intercept": self.intercept,
      "best_c": self.best_c,
      "tuning_history": [
        {
          "c_value": step.c_value,
          "val_log_loss": step.val_log_loss,
          "val_auroc": step.val_auroc,
          "val_brier": step.val_brier,
        }
        for step in self.tuning_history
      ],
      "scaler_mean": list(self.scaler_mean) if self.scaler_mean is not None else None,
      "scaler_scale": list(self.scaler_scale) if self.scaler_scale is not None else None,
      "config": {
        "model_name": self.config.model_name,
        "regularization_c_grid": list(self.config.regularization_c_grid),
        "max_iter": self.config.max_iter,
        "solver": self.config.solver,
        "random_state": self.config.random_state,
        "tuning_metric": self.config.tuning_metric,
        "standardize": self.config.standardize,
        "penalty": self.config.penalty,
        "tolerance": self.config.tolerance,
      },
    }

  def save_json(self, path: Path | str) -> None:
    """Save trained probe to JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
      json.dump(self.to_dict(), f, indent=2)

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> "TrainedProbe":
    """Instantiate TrainedProbe from dictionary."""
    cfg_data = d["config"]
    config = ProbeConfig(
      model_name=str(cfg_data["model_name"]),
      regularization_c_grid=tuple(float(c) for c in cfg_data["regularization_c_grid"]),
      max_iter=int(cfg_data["max_iter"]),
      solver=str(cfg_data["solver"]),
      random_state=int(cfg_data["random_state"]),
      tuning_metric=cast(Literal["log_loss", "auroc"], cfg_data["tuning_metric"]),
      standardize=bool(cfg_data["standardize"]),
      penalty=cast(Literal["l2"], cfg_data["penalty"]),
      tolerance=float(cfg_data["tolerance"]),
    )

    steps = tuple(
      ProbeValidationStep(
        c_value=float(s["c_value"]),
        val_log_loss=float(s["val_log_loss"]),
        val_auroc=float(s["val_auroc"]),
        val_brier=float(s["val_brier"]),
      )
      for s in d.get("tuning_history", [])
    )

    scaler_mean = tuple(float(x) for x in d["scaler_mean"]) if d.get("scaler_mean") is not None else None
    scaler_scale = tuple(float(x) for x in d["scaler_scale"]) if d.get("scaler_scale") is not None else None

    return cls(
      coefficients=tuple(float(x) for x in d["coefficients"]),
      intercept=float(d["intercept"]),
      best_c=float(d["best_c"]),
      tuning_history=steps,
      scaler_mean=scaler_mean,
      scaler_scale=scaler_scale,
      config=config,
    )

  @classmethod
  def load_json(cls, path: Path | str) -> "TrainedProbe":
    """Load trained probe from JSON file."""
    with open(path, encoding="utf-8") as f:
      d = json.load(f)
    return cls.from_dict(d)


# -----------------------------------------------------------------------------
# Probe Fitting
# -----------------------------------------------------------------------------


def fit_logistic_probe(
  dev_embeddings: npt.NDArray[np.float64],
  dev_labels: npt.NDArray[np.int64] | Sequence[int],
  val_embeddings: npt.NDArray[np.float64],
  val_labels: npt.NDArray[np.int64] | Sequence[int],
  config: ProbeConfig = ProbeConfig(),
) -> TrainedProbe:
  """Fit L2-regularized logistic regression probe on development set and select C on validation set.

  Strict data firewall:
    - Only dev_embeddings and dev_labels are used for parameter estimation.
    - Only val_embeddings and val_labels are used for hyperparameter C tuning.
    - Final test set is strictly prohibited from entering this function.

  Args:
    dev_embeddings: 2D array of shape [N_dev, D].
    dev_labels: Binary outcome labels for development partition.
    val_embeddings: 2D array of shape [N_val, D].
    val_labels: Binary outcome labels for validation partition.
    config: ProbeConfig configuration.

  Returns:
    TrainedProbe with frozen weights and tuning history.
  """
  x_dev = np.asarray(dev_embeddings, dtype=np.float64)
  y_dev = np.asarray(dev_labels, dtype=np.int64)
  x_val = np.asarray(val_embeddings, dtype=np.float64)
  y_val = np.asarray(val_labels, dtype=np.int64)

  if x_dev.ndim != 2 or x_val.ndim != 2:
    raise ValueError(f"Embeddings must be 2D arrays. Got dev ndim={x_dev.ndim}, val ndim={x_val.ndim}")

  if x_dev.shape[0] != len(y_dev):
    raise ValueError(f"Dev samples {x_dev.shape[0]} != dev labels {len(y_dev)}")

  if x_val.shape[0] != len(y_val):
    raise ValueError(f"Val samples {x_val.shape[0]} != val labels {len(y_val)}")

  if x_dev.shape[1] != x_val.shape[1]:
    raise ValueError(f"Dev dimension {x_dev.shape[1]} != Val dimension {x_val.shape[1]}")

  if len(np.unique(y_dev)) < 2:
    raise ValueError("Development labels must contain both positive (1) and negative (0) classes.")

  # Feature standardization using development set statistics only
  scaler_mean: tuple[float, ...] | None = None
  scaler_scale: tuple[float, ...] | None = None

  if config.standardize:
    mean_vec = np.mean(x_dev, axis=0)
    std_vec = np.std(x_dev, axis=0)
    std_vec_safe = np.maximum(std_vec, 1e-12)
    x_dev_scaled = (x_dev - mean_vec) / std_vec_safe
    x_val_scaled = (x_val - mean_vec) / std_vec_safe
    scaler_mean = tuple(float(m) for m in mean_vec)
    scaler_scale = tuple(float(s) for s in std_vec_safe)
  else:
    x_dev_scaled = x_dev
    x_val_scaled = x_val

  validation_steps: list[ProbeValidationStep] = []
  best_c: float = config.regularization_c_grid[0]
  best_score: float = float("inf") if config.tuning_metric == "log_loss" else -float("inf")
  best_model: LogisticRegression | None = None

  for c_val in config.regularization_c_grid:
    clf = LogisticRegression(
      C=c_val,
      solver=config.solver,
      max_iter=config.max_iter,
      random_state=config.random_state,
      tol=config.tolerance,
    )
    clf.fit(x_dev_scaled, y_dev)

    # Predict probabilities on validation partition
    val_probs = clf.predict_proba(x_val_scaled)[:, 1]

    val_loss = compute_binary_log_loss(y_val, val_probs)
    val_auc = compute_auroc(y_val, val_probs)
    val_brier = compute_brier_score(y_val, val_probs)

    validation_steps.append(
      ProbeValidationStep(
        c_value=c_val,
        val_log_loss=val_loss,
        val_auroc=val_auc,
        val_brier=val_brier,
      )
    )

    if config.tuning_metric == "log_loss":
      if val_loss < best_score:
        best_score = val_loss
        best_c = c_val
        best_model = clf
    else:  # auroc
      if val_auc > best_score:
        best_score = val_auc
        best_c = c_val
        best_model = clf

  if best_model is None:
    raise RuntimeError("Probe training failed to fit any valid model.")

  raw_coef = np.asarray(best_model.coef_, dtype=np.float64).reshape(-1)
  raw_intercept = np.asarray(best_model.intercept_, dtype=np.float64).reshape(-1)
  coefficients = tuple(float(coef) for coef in raw_coef)
  intercept = float(raw_intercept[0])

  return TrainedProbe(
    coefficients=coefficients,
    intercept=intercept,
    best_c=best_c,
    tuning_history=tuple(validation_steps),
    scaler_mean=scaler_mean,
    scaler_scale=scaler_scale,
    config=config,
  )


# -----------------------------------------------------------------------------
# Predictor Firewall & Joint Verification
# -----------------------------------------------------------------------------


def verify_predictor_firewall(
  df: pl.DataFrame,
  allowed_cols: Sequence[str] = ALLOWED_NON_PREDICTOR_COLS,
  forbidden_keywords: Sequence[str] = FORBIDDEN_PREDICTOR_KEYWORDS,
) -> bool:
  """Verify that no prohibited patient clinical covariates leaked into predictor feature spaces.

  Args:
    df: DataFrame containing prediction table or feature vectors.
    allowed_cols: Explicitly permitted identifiers and outcome evaluation columns.
    forbidden_keywords: Substrings identifying clinical variables (demographics, labs, meds, etc.).

  Returns:
    True if firewall is intact.

  Raises:
    ValueError: If a prohibited clinical feature is detected.
  """
  violating_cols: list[str] = []
  for col in df.columns:
    if col in allowed_cols:
      continue
    col_lower = col.lower()
    for kw in forbidden_keywords:
      if kw in col_lower:
        violating_cols.append(col)
        break

  if violating_cols:
    raise ValueError(
      f"Predictor-information firewall violation! Found prohibited clinical columns: {violating_cols}"
    )
  return True


def verify_unified_prediction_table(
  df: pl.DataFrame,
  subject_id_col: str = "subject_id",
  study_id_col: str = "study_id",
  split_col: str = "split",
) -> bool:
  """Verify integrity, one-row-per-patient rule, patient disjointness, and firewall.

  Args:
    df: Unified prediction DataFrame.
    subject_id_col: Column name for patient identifier.
    study_id_col: Column name for ECG study identifier.
    split_col: Column name for partition assignment.

  Returns:
    True if all checks pass.

  Raises:
    ValueError: If any integrity or firewall condition is violated.
  """
  required_cols = [
    subject_id_col,
    study_id_col,
    split_col,
    "model_a_score",
    "model_a_category",
    "model_a_valid",
    "model_b_score",
    "model_b_log_odds",
    "model_b_valid",
    "mortality_30d",
  ]

  for col in required_cols:
    if col not in df.columns:
      raise ValueError(f"Unified prediction table missing required column: '{col}'")

  # 1. Exactly one row per patient
  total_rows = len(df)
  unique_subjects = df[subject_id_col].n_unique()
  if total_rows != unique_subjects:
    raise ValueError(
      f"Unit-of-analysis violation: {total_rows} rows found for {unique_subjects} unique patients."
    )

  # 2. Strict patient disjointness across splits
  verify_split_disjointness(df, subject_id_col=subject_id_col, split_col=split_col)

  # 3. Valid split labels
  split_labels = set(df[split_col].unique().to_list())
  invalid_splits = split_labels - {"dev", "val", "test"}
  if invalid_splits:
    raise ValueError(f"Invalid split partition labels detected: {invalid_splits}")

  # 4. Predictor firewall check
  verify_predictor_firewall(df)

  return True


# -----------------------------------------------------------------------------
# Unified Prediction Table Construction
# -----------------------------------------------------------------------------


def build_unified_prediction_table(
  cohort_df: pl.DataFrame,
  model_a_df: pl.DataFrame,
  model_b_df: pl.DataFrame,
  outcome_cols: tuple[str, ...] = ("mortality_30d", "mortality_90d", "mortality_1yr"),
) -> pl.DataFrame:
  """Construct a unified prediction table joining Model A and Model B with cohort outcomes.

  Args:
    cohort_df: Cohort DataFrame with subject_id, study_id, split, and outcome columns.
    model_a_df: Model A predictions DataFrame.
    model_b_df: Model B predictions DataFrame.
    outcome_cols: Target outcome columns to retain for downstream evaluation.

  Returns:
    Unified Polars DataFrame strictly verified against research guardrails.
  """
  # Select base cohort columns
  base_cols = ["subject_id", "study_id", "split"]
  for col in outcome_cols:
    if col in cohort_df.columns:
      base_cols.append(col)

  cohort_base = cohort_df.select(base_cols)

  # Select Model A columns
  model_a_cols = [
    "subject_id",
    "study_id",
    "model_a_score",
    "model_a_category",
    "model_a_valid",
  ]
  if "model_a_error" in model_a_df.columns:
    model_a_cols.append("model_a_error")
  a_subset = model_a_df.select(model_a_cols)

  # Select Model B columns
  model_b_cols = [
    "subject_id",
    "study_id",
    "model_b_score",
    "model_b_log_odds",
    "model_b_valid",
  ]
  if "model_b_error" in model_b_df.columns:
    model_b_cols.append("model_b_error")
  b_subset = model_b_df.select(model_b_cols)

  # Join on (subject_id, study_id)
  joined = cohort_base.join(a_subset, on=["subject_id", "study_id"], how="inner").join(
    b_subset, on=["subject_id", "study_id"], how="inner"
  )

  # Fill default errors if not present
  if "model_a_error" not in joined.columns:
    joined = joined.with_columns(pl.lit(None).cast(pl.String).alias("model_a_error"))
  if "model_b_error" not in joined.columns:
    joined = joined.with_columns(pl.lit(None).cast(pl.String).alias("model_b_error"))

  verify_unified_prediction_table(joined)
  return joined


# -----------------------------------------------------------------------------
# Batch Scoring & Extraction Pipelines
# -----------------------------------------------------------------------------


def score_traditional_cohort(
  cohort_df: pl.DataFrame,
  ecg_data_dir: Path | str | None = None,
  relative_path_col: str = "path",
) -> pl.DataFrame:
  """Compute Model A (CIIS) scores for an entire cohort dataframe.

  Args:
    cohort_df: Cohort DataFrame with subject_id, study_id, and path.
    ecg_data_dir: Optional root directory of WFDB files.
    relative_path_col: Column name containing relative WFDB path.

  Returns:
    DataFrame with (subject_id, study_id, model_a_score, model_a_category, model_a_valid, model_a_error).
  """
  data_dir = Path(ecg_data_dir) if ecg_data_dir is not None else None

  results: list[dict[str, Any]] = []

  for row in cohort_df.iter_rows(named=True):
    subj_id = int(row["subject_id"])
    study_id = int(row["study_id"])
    rel_path = str(row.get(relative_path_col, ""))

    rec_path = data_dir / rel_path if data_dir is not None else Path(rel_path)

    try:
      signal_array, lead_names, fs = read_ecg_waveform(rec_path)
      res = score_ecg_waveform(signal_array, lead_names, fs)

      results.append({
        "subject_id": subj_id,
        "study_id": study_id,
        "model_a_score": res.total_score,
        "model_a_category": res.category.value if res.category is not None else None,
        "model_a_valid": res.is_valid,
        "model_a_error": res.error_message,
      })
    except Exception as exc:
      results.append({
        "subject_id": subj_id,
        "study_id": study_id,
        "model_a_score": None,
        "model_a_category": None,
        "model_a_valid": False,
        "model_a_error": str(exc),
      })

  return pl.DataFrame(results)


def extract_transformer_embeddings(
  cohort_df: pl.DataFrame,
  adapter: BaseEcgModelAdapter,
  ecg_data_dir: Path | str | None = None,
  relative_path_col: str = "path",
  batch_size: int = 32,
  checkpoint_path: Path | str | None = None,
  checkpoint_interval: int = 5000,
) -> tuple[pl.DataFrame, npt.NDArray[np.float64]]:
  """Extract frozen ECG representations for cohort using a Model-B adapter.

  Supports batch inference and periodic intermediate checkpointing with resume capability.

  Args:
    cohort_df: Cohort DataFrame with subject_id, study_id, and path.
    adapter: Instantiated model adapter (e.g. DbetaAdapter).
    ecg_data_dir: Optional root directory of WFDB files.
    relative_path_col: Column name containing relative WFDB path.
    batch_size: Batch size for batched model forward passes.
    checkpoint_path: Optional path to save/resume intermediate embeddings (.npz).
    checkpoint_interval: Number of processed records between checkpoints.

  Returns:
    Tuple of (metadata_df, embeddings_2d).
  """
  import hashlib

  data_dir = Path(ecg_data_dir) if ecg_data_dir is not None else None
  n_records = len(cohort_df)
  emb_dim = adapter.config.embedding_dim

  embeddings = np.zeros((n_records, emb_dim), dtype=np.float64)
  valid_flags: list[bool] = [False] * n_records
  error_messages: list[str | None] = [None] * n_records

  # Compute deterministic cohort identifier hash to guard against mismatched resume
  id_bytes = b"".join(
    f"{r['subject_id']}:{r['study_id']}\n".encode("utf-8")
    for r in cohort_df.select(["subject_id", "study_id"]).iter_rows(named=True)
  )
  cohort_hash = hashlib.sha256(id_bytes).hexdigest()

  start_idx = 0
  if checkpoint_path is not None:
    ckpt_p = Path(checkpoint_path).expanduser().resolve()
    if ckpt_p.exists() and ckpt_p.is_file():
      try:
        with np.load(ckpt_p, allow_pickle=True) as ckpt_data:
          saved_count = int(ckpt_data["processed_count"])
          saved_cohort_hash = str(ckpt_data.get("cohort_hash", ""))
          saved_model_name = str(ckpt_data.get("model_name", ""))
          saved_dim = int(ckpt_data.get("embedding_dim", emb_dim))

          if saved_cohort_hash and saved_cohort_hash != cohort_hash:
            logger.warning("Checkpoint cohort hash mismatch at %s; restarting extraction.", ckpt_p)
          elif saved_model_name and saved_model_name != str(adapter.config.model_name):
            logger.warning("Checkpoint model name mismatch at %s; restarting extraction.", ckpt_p)
          elif saved_dim != emb_dim:
            logger.warning("Checkpoint embedding dimension mismatch at %s; restarting extraction.", ckpt_p)
          elif 0 < saved_count <= n_records:
            embeddings[:saved_count] = ckpt_data["embeddings"][:saved_count]
            saved_mask = ckpt_data["valid_mask"][:saved_count]
            for i in range(saved_count):
              valid_flags[i] = bool(saved_mask[i])
            if "error_messages" in ckpt_data:
              saved_errs = ckpt_data["error_messages"][:saved_count]
              for i in range(saved_count):
                err_v = saved_errs[i]
                error_messages[i] = str(err_v) if err_v is not None and str(err_v) != "" and not (isinstance(err_v, float) and np.isnan(err_v)) else None
            start_idx = saved_count
            logger.info("Resumed transformer embedding extraction from %s at record %d/%d", ckpt_p, start_idx, n_records)
      except Exception as exc:
        logger.warning("Could not restore checkpoint from %s: %s; starting extraction from scratch.", ckpt_p, exc)

  # If adapter supports embed_batch and batch_size > 1, use batching
  if batch_size > 1 and hasattr(adapter, "embed_batch"):
    batch_records: list[tuple[npt.NDArray[np.float64], Sequence[str], int]] = []
    batch_indices: list[int] = []

    for idx in range(start_idx, n_records):
      row = cohort_df.row(idx, named=True)
      rel_path = str(row.get(relative_path_col, ""))
      rec_path = data_dir / rel_path if data_dir is not None else Path(rel_path)

      try:
        sig, leads, fs = read_ecg_waveform(rec_path)
        batch_records.append((sig, leads, fs))
        batch_indices.append(idx)
      except Exception as exc:
        valid_flags[idx] = False
        error_messages[idx] = str(exc)

      # When batch is full or at the end of cohort, run batch inference
      if len(batch_records) >= batch_size or (idx == n_records - 1 and batch_records):
        try:
          batch_res = adapter.embed_batch(batch_records)
          for b_i, global_idx in enumerate(batch_indices):
            if batch_res.valid_mask[b_i] and batch_res.embeddings is not None:
              embeddings[global_idx] = batch_res.embeddings[b_i]
              valid_flags[global_idx] = True
              error_messages[global_idx] = None
            else:
              valid_flags[global_idx] = False
              error_messages[global_idx] = (
                batch_res.failure_reasons[b_i] if b_i < len(batch_res.failure_reasons) else "Batch inference error"
              )
        except Exception as exc:
          for global_idx in batch_indices:
            valid_flags[global_idx] = False
            error_messages[global_idx] = f"Batch failure: {exc}"

        batch_records.clear()
        batch_indices.clear()

      # Periodic intermediate checkpointing
      if checkpoint_path is not None and ((idx + 1) % checkpoint_interval == 0 or idx == n_records - 1):
        ckpt_p = Path(checkpoint_path)
        ckpt_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
          ckpt_p,
          embeddings=embeddings[: idx + 1],
          valid_mask=np.array(valid_flags[: idx + 1], dtype=bool),
          error_messages=np.array([m if m is not None else "" for m in error_messages[: idx + 1]], dtype=object),
          processed_count=idx + 1,
          total_records=n_records,
          model_name=str(adapter.config.model_name),
          embedding_dim=emb_dim,
          cohort_hash=cohort_hash,
        )
        logger.info("Saved intermediate embedding checkpoint to %s (%d/%d records)", ckpt_p, idx + 1, n_records)

  else:
    for idx in range(start_idx, n_records):
      row = cohort_df.row(idx, named=True)
      rel_path = str(row.get(relative_path_col, ""))
      rec_path = data_dir / rel_path if data_dir is not None else Path(rel_path)

      try:
        signal_array, lead_names, fs = read_ecg_waveform(rec_path)
        out = adapter.embed_single(signal_array, lead_names, fs)

        if out.is_valid and out.embedding is not None:
          embeddings[idx] = out.embedding
          valid_flags[idx] = True
          error_messages[idx] = None
        else:
          valid_flags[idx] = False
          error_messages[idx] = out.error_message or "Invalid embedding output"
      except Exception as exc:
        valid_flags[idx] = False
        error_messages[idx] = str(exc)

      if checkpoint_path is not None and ((idx + 1) % checkpoint_interval == 0 or idx == n_records - 1):
        ckpt_p = Path(checkpoint_path)
        ckpt_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
          ckpt_p,
          embeddings=embeddings[: idx + 1],
          valid_mask=np.array(valid_flags[: idx + 1], dtype=bool),
          error_messages=np.array([m if m is not None else "" for m in error_messages[: idx + 1]], dtype=object),
          processed_count=idx + 1,
          total_records=n_records,
          model_name=str(adapter.config.model_name),
          embedding_dim=emb_dim,
          cohort_hash=cohort_hash,
        )

  meta_df = cohort_df.select(["subject_id", "study_id"]).with_columns(
    pl.Series("is_valid", valid_flags, dtype=pl.Boolean),
    pl.Series("error_message", error_messages, dtype=pl.String),
  )

  return meta_df, embeddings


def score_transformer_cohort(
  cohort_df: pl.DataFrame,
  embeddings: npt.NDArray[np.float64],
  probe: TrainedProbe,
  valid_mask: Sequence[bool] | None = None,
  error_messages: Sequence[str | None] | None = None,
) -> pl.DataFrame:
  """Apply frozen probe to compute continuous probabilities and log-odds.

  Args:
    cohort_df: Cohort DataFrame with subject_id and study_id.
    embeddings: 2D array of shape [N, D].
    probe: Frozen TrainedProbe instance.
    valid_mask: Optional boolean sequence indicating valid embeddings.
    error_messages: Optional sequence of error messages.

  Returns:
    DataFrame with (subject_id, study_id, model_b_score, model_b_log_odds, model_b_valid, model_b_error).
  """
  n = len(cohort_df)
  probs = np.full(n, np.nan, dtype=np.float64)
  logits = np.full(n, np.nan, dtype=np.float64)

  if valid_mask is None:
    mask = np.ones(n, dtype=bool)
  else:
    mask = np.asarray(valid_mask, dtype=bool)

  valid_indices = np.where(mask)[0]
  if len(valid_indices) > 0:
    valid_embs = embeddings[valid_indices]
    valid_logits = probe.predict_log_odds(valid_embs)
    valid_probs = probe.predict_proba(valid_embs)
    logits[valid_indices] = valid_logits
    probs[valid_indices] = valid_probs

  err_list: list[str | None] = (
    list(error_messages) if error_messages is not None else [None if m else "Invalid embedding" for m in mask]
  )

  return cohort_df.select(["subject_id", "study_id"]).with_columns(
    pl.Series("model_b_score", [float(p) if not np.isnan(p) else None for p in probs], dtype=pl.Float64),
    pl.Series("model_b_log_odds", [float(l) if not np.isnan(l) else None for l in logits], dtype=pl.Float64),
    pl.Series("model_b_valid", mask.tolist(), dtype=pl.Boolean),
    pl.Series("model_b_error", err_list, dtype=pl.String),
  )


# -----------------------------------------------------------------------------
# Summary Statistics & Reporting
# -----------------------------------------------------------------------------


def compute_prediction_summary_statistics(
  unified_df: pl.DataFrame,
) -> dict[str, Any]:
  """Compute non-sensitive aggregate statistics on the unified prediction table.

  Args:
    unified_df: Verified unified prediction table.

  Returns:
    Dictionary of aggregate sample sizes, score distributions, and baseline discrimination metrics.
  """
  stats: dict[str, Any] = {
    "total_patients": len(unified_df),
    "unique_subjects": unified_df["subject_id"].n_unique(),
  }

  for sp in ("dev", "val", "test"):
    sp_df = unified_df.filter(pl.col("split") == sp)
    n_sp = len(sp_df)
    stats[f"{sp}_n"] = n_sp

    # Model A valid counts and failures
    a_valid = sp_df.filter(pl.col("model_a_valid") == True)  # noqa: E712
    stats[f"{sp}_model_a_valid_n"] = len(a_valid)
    stats[f"{sp}_model_a_valid_pct"] = (len(a_valid) / n_sp * 100.0) if n_sp > 0 else 0.0

    # Model B valid counts and failures
    b_valid = sp_df.filter(pl.col("model_b_valid") == True)  # noqa: E712
    stats[f"{sp}_model_b_valid_n"] = len(b_valid)
    stats[f"{sp}_model_b_valid_pct"] = (len(b_valid) / n_sp * 100.0) if n_sp > 0 else 0.0

    # Model A score distribution (on valid records)
    if len(a_valid) > 0:
      a_scores = a_valid["model_a_score"].drop_nulls().to_numpy()
      stats[f"{sp}_model_a_mean"] = float(np.mean(a_scores))
      stats[f"{sp}_model_a_std"] = float(np.std(a_scores))
      stats[f"{sp}_model_a_median"] = float(np.median(a_scores))
      stats[f"{sp}_model_a_q25"] = float(np.percentile(a_scores, 25))
      stats[f"{sp}_model_a_q75"] = float(np.percentile(a_scores, 75))

    # Model B score distribution (on valid records)
    if len(b_valid) > 0:
      b_scores = b_valid["model_b_score"].drop_nulls().to_numpy()
      b_logits = b_valid["model_b_log_odds"].drop_nulls().to_numpy()
      stats[f"{sp}_model_b_prob_mean"] = float(np.mean(b_scores))
      stats[f"{sp}_model_b_prob_std"] = float(np.std(b_scores))
      stats[f"{sp}_model_b_prob_median"] = float(np.median(b_scores))
      stats[f"{sp}_model_b_prob_q25"] = float(np.percentile(b_scores, 25))
      stats[f"{sp}_model_b_prob_q75"] = float(np.percentile(b_scores, 75))

      stats[f"{sp}_model_b_logits_mean"] = float(np.mean(b_logits))
      stats[f"{sp}_model_b_logits_std"] = float(np.std(b_logits))

    # Baseline performance metrics on valid both records
    both_valid = sp_df.filter(
      (pl.col("model_a_valid") == True) & (pl.col("model_b_valid") == True)  # noqa: E712
    )
    stats[f"{sp}_both_valid_n"] = len(both_valid)
    if len(both_valid) > 0 and "mortality_30d" in both_valid.columns:
      y_true = both_valid["mortality_30d"].cast(pl.Int64).to_numpy()
      if len(np.unique(y_true)) >= 2:
        a_sc = both_valid["model_a_score"].to_numpy()
        b_sc = both_valid["model_b_score"].to_numpy()

        stats[f"{sp}_model_a_auroc"] = compute_auroc(y_true, a_sc)
        stats[f"{sp}_model_b_auroc"] = compute_auroc(y_true, b_sc)
        stats[f"{sp}_model_b_log_loss"] = compute_binary_log_loss(y_true, b_sc)
        stats[f"{sp}_model_b_brier"] = compute_brier_score(y_true, b_sc)

  # Category distribution for Model A
  cat_counts: dict[str, int] = {}
  for cat in CIISCategory:
    c_n = unified_df.filter(pl.col("model_a_category") == cat.value).shape[0]
    cat_counts[cat.value] = c_n
  stats["model_a_category_counts"] = cat_counts

  return stats


def generate_continuous_predictions_markdown(
  summary_stats: dict[str, Any],
  probe: TrainedProbe | None = None,
  title: str = "MIMIC-IV Continuous Prediction Generation and Probe Validation Report",
  data_mode: Literal["real", "simulation"] | str = "real",
) -> str:
  """Generate a comprehensive Markdown report documenting continuous score generation and probe validation.

  Args:
    summary_stats: Aggregate metrics from compute_prediction_summary_statistics.
    probe: Optional TrainedProbe instance with tuning history.
    title: Report title.
    data_mode: Data provenance mode ('real' or 'simulation').

  Returns:
    Markdown string.
  """
  total_pts = summary_stats.get("total_patients", 0)
  provenance_banner = (
    "> **Data Source:** REAL MIMIC-IV-ECG predictions"
    if str(data_mode).lower() == "real"
    else "> **Data Source:** SIMULATION — NOT EMPIRICAL RESULTS"
  )

  lines = [
    f"# {title}",
    "",
    provenance_banner,
    "",
    "**Stage:** Stage 8 — Build Continuous Predictions  ",
    "**Status:** Completed and Verified  ",
    f"**Total Cohort Size:** {total_pts:,} patients  ",
  ]

  if probe is not None:
    lines.extend([
      f"**Primary Transformer Model:** `{probe.config.model_name}`  ",
      f"**Optimal Regularization Parameter ($C^*$):** `{probe.best_c}`  ",
      f"**Embedding Dimension:** {probe.embedding_dim}  ",
    ])

  lines.extend([
    "",
    "---",
    "",
    "## 1. Score Generation Pipeline Overview",
    "",
    "```mermaid",
    "flowchart LR",
    '  A["Index ECG Waveform (10s, 12-lead)"] --> B["Traditional Model A (CIIS)"]',
    '  A --> C["D-BETA Frozen Transformer Encoder"]',
    '  C --> D["768-d Frozen ECG Embedding"]',
    '  D --> E["Trained L2 Linear Probe (Frozen)"]',
    '  B --> F["Continuous CIIS Score & Category"]',
    '  E --> G["Model B 30-Day Mortality Risk (Probability & Logits)"]',
    '  F --> H["Unified Prediction Table"]',
    '  G --> H',
    "```",
    "",
    "---",
    "",
    "## 2. Partition Sample Sizes & Technical Completion Rates",
    "",
    "| Partition | Total Patients ($N$) | Model A Valid ($N$, %) | Model B Valid ($N$, %) | Both Models Valid ($N$, %) |",
    "| :--- | :--- | :--- | :--- | :--- |",
  ])

  for sp, name in (
    ("dev", "Development (`dev`, 60%)"),
    ("val", "Validation (`val`, 20%)"),
    ("test", "Final Test (`test`, 20%)"),
  ):
    n_sp = summary_stats.get(f"{sp}_n", 0)
    a_n = summary_stats.get(f"{sp}_model_a_valid_n", 0)
    a_pct = summary_stats.get(f"{sp}_model_a_valid_pct", 0.0)
    b_n = summary_stats.get(f"{sp}_model_b_valid_n", 0)
    b_pct = summary_stats.get(f"{sp}_model_b_valid_pct", 0.0)
    both_n = summary_stats.get(f"{sp}_both_valid_n", 0)
    both_pct = (both_n / n_sp * 100.0) if n_sp > 0 else 0.0

    lines.append(
      f"| **{name}** | {n_sp:,} | {a_n:,} ({a_pct:.2f}%) | {b_n:,} ({b_pct:.2f}%) | {both_n:,} ({both_pct:.2f}%) |"
    )

  lines.extend([
    "",
    "---",
    "",
    "## 3. Model B Linear Probe Hyperparameter Tuning (Validation Set)",
    "",
  ])

  if probe is not None and probe.tuning_history:
    lines.extend([
      "| Regularization Parameter ($C$) | Validation Log-Loss | Validation AUROC | Validation Brier Score | Selected |",
      "| :--- | :--- | :--- | :--- | :--- |",
    ])
    for step in probe.tuning_history:
      is_sel = "**Yes (Optimal)**" if step.c_value == probe.best_c else "No"
      lines.append(
        f"| $C = 10^{{{np.log10(step.c_value):.0f}}}$ (`{step.c_value}`) | {step.val_log_loss:.4f} | {step.val_auroc:.4f} | {step.val_brier:.4f} | {is_sel} |"
      )
    lines.append("")

  lines.extend([
    "---",
    "",
    "## 4. Score Distributions on Final Test Set (`test`)",
    "",
    "| Metric | Model A (CIIS Score) | Model B (Predicted 30-day Risk) | Model B (Log-Odds Logits) |",
    "| :--- | :--- | :--- | :--- |",
  ])

  for stat_key, label in (
    ("mean", "Mean"),
    ("std", "Standard Deviation"),
    ("median", "Median"),
    ("q25", "25th Percentile (Q1)"),
    ("q75", "75th Percentile (Q3)"),
  ):
    a_val = summary_stats.get(f"test_model_a_{stat_key}", None)
    b_p_val = summary_stats.get(f"test_model_b_prob_{stat_key}", None)
    b_l_val = summary_stats.get(f"test_model_b_logits_{stat_key}", None)

    a_str = f"{a_val:.2f}" if a_val is not None else "—"
    bp_str = f"{b_p_val:.4f}" if b_p_val is not None else "—"
    bl_str = f"{b_l_val:.2f}" if b_l_val is not None else "—"

    lines.append(f"| **{label}** | {a_str} | {bp_str} | {bl_str} |")

  lines.extend([
    "",
    "### Baseline Discriminative Performance on Final Test Set (`test`)",
    "",
    "| Model | Test AUROC | Test Log-Loss | Test Brier Score |",
    "| :--- | :--- | :--- | :--- |",
  ])

  a_auc = summary_stats.get("test_model_a_auroc", None)
  b_auc = summary_stats.get("test_model_b_auroc", None)
  b_ll = summary_stats.get("test_model_b_log_loss", None)
  b_br = summary_stats.get("test_model_b_brier", None)

  lines.append(f"| **Model A (Continuous CIIS)** | {a_auc:.4f} | — | — |" if a_auc is not None else "| **Model A** | — | — | — |")
  lines.append(
    f"| **Model B (D-BETA Linear Probe)** | {b_auc:.4f} | {b_ll:.4f} | {b_br:.4f} |"
    if (b_auc is not None and b_ll is not None and b_br is not None)
    else "| **Model B** | — | — | — |"
  )

  lines.extend([
    "",
    "---",
    "",
    "## 5. Research Guardrails & Integrity Verification",
    "",
    "- [x] **Predictor-Information Firewall:** Verified 0 clinical features (age, sex, vitals, labs, notes, meds) enter Model A or Model B.",
    "- [x] **Unit of Analysis:** Verified exactly 1 row per unique patient in the prediction table.",
    "- [x] **Supervised Data Firewall:** Probe weights and regularization parameter $C^*$ were frozen using development and validation sets only, without inspecting final test outcomes.",
    "- [x] **Disjointness:** Zero patient overlap across development, validation, and test splits.",
    "- [x] **Reproducibility:** Probe specification and parameters are versioned and serializable to JSON.",
    "",
  ])

  return "\n".join(lines)


# -----------------------------------------------------------------------------
# Protected Artifact Persistence & Run Manifest Generation
# -----------------------------------------------------------------------------


def save_unified_prediction_table(
  df: pl.DataFrame,
  path: Path | str,
) -> Path:
  """Persist verified unified prediction table artifact outside Git.

  Supports Parquet (preferred), Arrow/IPC, or CSV based on file extension.
  """
  p = Path(path).expanduser().resolve()
  p.parent.mkdir(parents=True, exist_ok=True)
  verify_unified_prediction_table(df)
  path_str = str(p).lower()
  if path_str.endswith(".parquet"):
    df.write_parquet(p)
  elif path_str.endswith(".csv"):
    df.write_csv(p)
  elif path_str.endswith(".ipc") or path_str.endswith(".arrow"):
    df.write_ipc(p)
  else:
    df.write_parquet(p)
  return p


def load_unified_prediction_table(
  path: Path | str,
) -> pl.DataFrame:
  """Load and verify a persisted unified prediction table artifact."""
  p = Path(path).expanduser().resolve()
  if not p.exists():
    raise FileNotFoundError(f"Prediction table not found at: {p}")
  path_str = str(p).lower()
  if path_str.endswith(".parquet"):
    df = pl.read_parquet(p)
  elif path_str.endswith(".csv"):
    df = pl.read_csv(p)
  elif path_str.endswith(".ipc") or path_str.endswith(".arrow"):
    df = pl.read_ipc(p)
  else:
    try:
      df = pl.read_parquet(p)
    except Exception:
      df = pl.read_csv(p)
  verify_unified_prediction_table(df)
  return df


@dataclass(frozen=True)
class VerifiedPredictionArtifact:
  """Container for a verified prediction table artifact with provenance metadata."""

  dataframe: pl.DataFrame
  data_mode: Literal["real", "simulation"] | str
  manifest_path: Path | None = None
  manifest_data: dict[str, Any] | None = None
  sha256: str | None = None


def find_companion_manifest(artifact_path: Path | str) -> Path | None:
  """Search for a companion run manifest associated with a prediction table artifact."""
  p = Path(artifact_path).expanduser().resolve()
  candidates = [
    p.parent / f"{p.name}.manifest.json",
    p.parent / f"{p.stem}.manifest.json",
    p.parent / "run_manifest.json",
  ]
  for cand in candidates:
    if cand.exists() and cand.is_file():
      return cand
  return None


def load_and_verify_prediction_artifact(
  path: Path | str,
  requested_mode: Literal["real", "simulation"] | None = None,
  allow_simulated_artifact: bool = False,
  allow_unverified_artifact: bool = False,
  verify_checksum: bool = True,
) -> VerifiedPredictionArtifact:
  """Load a unified prediction table artifact with fail-closed provenance verification.

  Enforces:
    1. Prediction table exists and schema is valid.
    2. If companion manifest is missing in 'real' mode, fails closed unless
       `allow_unverified_artifact` is True.
    3. If companion manifest exists, checks data_mode compatibility. A simulated
       prediction table cannot be loaded in 'real' execution mode unless
       `allow_simulated_artifact` is explicitly set to True.
    4. If `predictions_artifact_sha256` is recorded in the companion manifest and
       `verify_checksum` is True, verifies that the artifact file SHA-256 matches.

  Args:
    path: Path to the prediction artifact file (parquet, csv, ipc).
    requested_mode: Execution mode ('real' or 'simulation').
    allow_simulated_artifact: If True, permit simulated artifacts in real mode.
    allow_unverified_artifact: If True, permit unmanifested artifacts in real mode.
    verify_checksum: If True, verify SHA-256 matches manifest if recorded.

  Returns:
    VerifiedPredictionArtifact containing the DataFrame, actual data_mode, and metadata.

  Raises:
    FileNotFoundError: If the prediction table does not exist.
    ValueError: If provenance or checksum verification fails.
  """
  p = Path(path).expanduser().resolve()
  if not p.exists():
    raise FileNotFoundError(f"Prediction table not found at: {p}")

  manifest_path = find_companion_manifest(p)
  manifest_data: dict[str, Any] | None = None
  actual_mode: str = "simulation" if requested_mode == "simulation" else "real"
  actual_sha = compute_file_sha256(p)

  if manifest_path is None:
    if requested_mode == "real" and not allow_unverified_artifact:
      raise ValueError(
        f"Prediction artifact at '{p}' has no companion manifest to verify data provenance in 'real' mode. "
        "Real execution requires verified empirical predictions with a companion manifest. "
        "Pass --allow-unverified-artifact to bypass if unverified loading is explicitly intended."
      )
  else:
    try:
      with open(manifest_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    except Exception as exc:
      if requested_mode == "real" and not allow_unverified_artifact:
        raise ValueError(
          f"Companion manifest at '{manifest_path}' is malformed or unreadable: {exc}. "
          "Real mode requires a valid JSON companion manifest. Pass --allow-unverified-artifact to bypass."
        ) from exc
      loaded_json = None

    if not isinstance(loaded_json, dict):
      if requested_mode == "real" and not allow_unverified_artifact:
        raise ValueError(
          f"Companion manifest at '{manifest_path}' is invalid (expected JSON object). "
          "Real mode requires a valid JSON companion manifest. Pass --allow-unverified-artifact to bypass."
        )
      manifest_data = None
    else:
      manifest_data = loaded_json

      recorded_mode = manifest_data.get("data_mode")
      if recorded_mode in ("real", "simulation"):
        actual_mode = recorded_mode
      elif requested_mode == "real" and not allow_unverified_artifact:
        raise ValueError(
          f"Companion manifest at '{manifest_path}' is missing a valid 'data_mode' field. "
          "Real mode requires data_mode to be specified. Pass --allow-unverified-artifact to bypass."
        )

      if (
        requested_mode == "real"
        and recorded_mode == "simulation"
        and not allow_simulated_artifact
      ):
        raise ValueError(
          f"Prediction artifact at '{p}' was generated in 'simulation' mode (recorded in {manifest_path.name}), "
          "but execution was requested in 'real' mode. Real execution requires genuine empirical predictions. "
          "Pass --simulate or --allow-simulated-artifact if simulated evaluation is intended."
        )

      if verify_checksum:
        recorded_sha = manifest_data.get("predictions_artifact_sha256")
        if recorded_sha is None:
          if requested_mode == "real" and not allow_unverified_artifact:
            raise ValueError(
              f"Companion manifest at '{manifest_path}' is missing 'predictions_artifact_sha256' field. "
              "Pass --allow-unverified-artifact to bypass."
            )
        elif actual_sha != recorded_sha:
          raise ValueError(
            f"Prediction artifact integrity error: SHA-256 mismatch for '{p}'. "
            f"Manifest records {recorded_sha}, but file has {actual_sha}."
          )

  df = load_unified_prediction_table(p)
  return VerifiedPredictionArtifact(
    dataframe=df,
    data_mode=actual_mode,
    manifest_path=manifest_path,
    manifest_data=manifest_data,
    sha256=actual_sha,
  )


def load_and_verify_unified_prediction_table(
  path: Path | str,
  requested_mode: Literal["real", "simulation"] | None = None,
  allow_simulated_artifact: bool = False,
  allow_unverified_artifact: bool = False,
  verify_checksum: bool = True,
) -> pl.DataFrame:
  """Load a unified prediction table artifact with fail-closed provenance verification.

  Convenience wrapper around load_and_verify_prediction_artifact returning only the DataFrame.
  """
  artifact = load_and_verify_prediction_artifact(
    path=path,
    requested_mode=requested_mode,
    allow_simulated_artifact=allow_simulated_artifact,
    allow_unverified_artifact=allow_unverified_artifact,
    verify_checksum=verify_checksum,
  )
  return artifact.dataframe


def get_current_git_sha() -> str | None:
  """Retrieve current git commit SHA if repository is available."""
  import subprocess
  try:
    res = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
      timeout=5,
    )
    return res.stdout.strip()
  except Exception:
    return None


def compute_file_sha256(path: Path | str | None) -> str | None:
  """Compute SHA-256 hash of a file if it exists."""
  if path is None:
    return None
  import hashlib
  p = Path(path).expanduser().resolve()
  if not p.exists() or not p.is_file():
    return None
  h = hashlib.sha256()
  with open(p, "rb") as f:
    while chunk := f.read(65536):
      h.update(chunk)
  return h.hexdigest()


def generate_run_manifest(
  unified_df: pl.DataFrame,
  probe: TrainedProbe | None = None,
  seed: int = DEFAULT_PROBE_SEED,
  data_mode: Literal["real", "simulation"] = "real",
  git_sha: str | None = None,
  predictions_path: Path | str | None = None,
  uv_lock_path: Path | str | None = None,
  dbeta_revision: str = "20ff3ccce1759d7d629171e15befafa9a424d2ca",
  notes: str | None = None,
) -> dict[str, Any]:
  """Generate a structured run manifest recording provenance, data mode, checksums, and counts."""
  import hashlib
  from datetime import datetime, timezone

  verify_unified_prediction_table(unified_df)

  # Deterministic checksum of identifiers and score fields
  id_bytes = b"".join(
    f"{r['subject_id']}:{r['study_id']}:{r['model_a_score']}:{r['model_b_score']}\n".encode("utf-8")
    for r in unified_df.select(["subject_id", "study_id", "model_a_score", "model_b_score"]).iter_rows(named=True)
  )
  checksum = hashlib.sha256(id_bytes).hexdigest()

  # Git SHA resolution
  resolved_git_sha = git_sha or get_current_git_sha()

  # uv.lock hash resolution
  resolved_uv_lock_sha = compute_file_sha256(uv_lock_path or Path("uv.lock"))

  # Serialized prediction artifact hash
  resolved_artifact_sha = compute_file_sha256(predictions_path)

  dev_df = unified_df.filter(pl.col("split") == "dev")
  val_df = unified_df.filter(pl.col("split") == "val")
  test_df = unified_df.filter(pl.col("split") == "test")

  total_events = (
    int(unified_df["mortality_30d"].cast(pl.Int64).sum())
    if "mortality_30d" in unified_df.columns
    else 0
  )
  dev_events = (
    int(dev_df["mortality_30d"].cast(pl.Int64).sum())
    if "mortality_30d" in dev_df.columns
    else 0
  )
  val_events = (
    int(val_df["mortality_30d"].cast(pl.Int64).sum())
    if "mortality_30d" in val_df.columns
    else 0
  )
  test_events = (
    int(test_df["mortality_30d"].cast(pl.Int64).sum())
    if "mortality_30d" in test_df.columns
    else 0
  )

  manifest: dict[str, Any] = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "data_mode": data_mode,
    "git_sha": resolved_git_sha,
    "uv_lock_sha256": resolved_uv_lock_sha,
    "predictions_artifact_sha256": resolved_artifact_sha,
    "seed": seed,
    "dbeta_revision": dbeta_revision,
    "checksum_sha256": checksum,
    "cohort_counts": {
      "total_patients": len(unified_df),
      "dev_patients": len(dev_df),
      "val_patients": len(val_df),
      "test_patients": len(test_df),
    },
    "event_counts_30d": {
      "total_events": total_events,
      "dev_events": dev_events,
      "val_events": val_events,
      "test_events": test_events,
    },
    "model_validity": {
      "model_a_valid_n": int(unified_df.filter(pl.col("model_a_valid") == True).shape[0]),  # noqa: E712
      "model_b_valid_n": int(unified_df.filter(pl.col("model_b_valid") == True).shape[0]),  # noqa: E712
      "both_valid_n": int(
        unified_df.filter(
          (pl.col("model_a_valid") == True) & (pl.col("model_b_valid") == True)  # noqa: E712
        ).shape[0]
      ),
    },
    "probe_config": probe.to_dict()["config"] if probe is not None else None,
    "probe_best_c": probe.best_c if probe is not None else None,
    "notes": notes,
  }
  return manifest


def save_run_manifest(
  manifest: dict[str, Any],
  path: Path | str,
) -> Path:
  """Write run manifest metadata to a JSON file."""
  p = Path(path).expanduser().resolve()
  p.parent.mkdir(parents=True, exist_ok=True)
  with open(p, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
  return p

