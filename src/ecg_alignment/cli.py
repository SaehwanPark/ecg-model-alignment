"""Command-line interface and workflow orchestration for ECG Model Alignment.

Provides reproducible commands for data inventory, scoring, probe training,
primary statistical analysis, sensitivity evaluations, and research interpretation.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
import numpy as np
import polars as pl

from ecg_alignment.analysis import (
  PrimaryAnalysisResult,
  generate_primary_figures,
  generate_primary_results_markdown,
  run_primary_analysis,
)
from ecg_alignment.cohort import (
  compute_linkage_statistics,
  generate_cohort_flow,
  link_records_to_admissions,
  select_index_ecgs,
)
from ecg_alignment.data import (
  DataPaths,
  load_admissions,
  load_patients,
  load_record_list,
)
from ecg_alignment.interpretation import (
  ResearchInterpretationSynthesis,
  generate_research_interpretation_markdown,
  synthesize_research_interpretation,
)
from ecg_alignment.outcomes import (
  compute_mortality_outcomes,
  compute_patient_death_table,
)
from ecg_alignment.probe import (
  DEFAULT_PROBE_SEED,
  ProbeConfig,
  TrainedProbe,
  build_unified_prediction_table,
  compute_prediction_summary_statistics,
  extract_transformer_embeddings,
  fit_logistic_probe,
  generate_continuous_predictions_markdown,
  generate_run_manifest,
  load_unified_prediction_table,
  save_run_manifest,
  save_unified_prediction_table,
  score_traditional_cohort,
  score_transformer_cohort,
)
from ecg_alignment.scoring.dbeta import DbetaAdapter, DbetaConfig
from ecg_alignment.sensitivity import (
  FullSensitivityAnalysisResult,
  generate_sensitivity_report_markdown,
  run_sensitivity_analyses,
)
from ecg_alignment.split import (
  DEFAULT_SPLIT_SEED,
  SplitRatios,
  assign_patient_disjoint_split,
  build_primary_cohort_and_split,
  derive_primary_cohort,
  generate_cohort_flow_markdown,
)

DEFAULT_MIMIC_ROOT = "~/data/mimiciv/3.1"
DEFAULT_ECG_ROOT = "~/data/mimic-iv-ecg/1.0"
ENV_MIMIC_ROOT = "MIMIC_ROOT"
ENV_ECG_ROOT = "MIMIC_ECG_ROOT"
ENV_ECG_ROOT_ALT = "ECG_ROOT"


@dataclass(frozen=True)
class CliContext:
  """Resolved execution context for CLI commands."""

  mimic_root: Path
  ecg_root: Path
  output_dir: Path
  seed: int
  verbose: bool
  simulate: bool = False
  predictions_path: Path | None = None
  max_records: int | None = None
  n_bootstraps: int = 50
  device: str = "cpu"
  batch_size: int = 32


def resolve_data_paths(
  mimic_root: str | Path | None = None,
  ecg_root: str | Path | None = None,
) -> DataPaths:
  """Resolve MIMIC and ECG data paths with environment variable fallbacks.

  Precedence:
    1. Explicit function arguments / CLI flags.
    2. Environment variables (MIMIC_ROOT, MIMIC_ECG_ROOT / ECG_ROOT).
    3. Standard local defaults (~/data/mimiciv/3.1, ~/data/mimic-iv-ecg/1.0).
  """
  resolved_mimic = (
    mimic_root
    if mimic_root is not None
    else os.environ.get(ENV_MIMIC_ROOT, DEFAULT_MIMIC_ROOT)
  )
  resolved_ecg = (
    ecg_root
    if ecg_root is not None
    else os.environ.get(
      ENV_ECG_ROOT, os.environ.get(ENV_ECG_ROOT_ALT, DEFAULT_ECG_ROOT)
    )
  )

  mimic_path = Path(str(resolved_mimic)).expanduser().resolve()
  ecg_path = Path(str(resolved_ecg)).expanduser().resolve()

  return DataPaths(
    mimic_iv_dir=mimic_path,
    mimic_iv_ecg_dir=ecg_path,
  )


def _find_data_file(base_dir: Path, relative_candidates: list[str]) -> Path:
  """Find the first existing file among relative candidates under base_dir."""
  for rel in relative_candidates:
    p = base_dir / rel
    if p.exists():
      return p
  return base_dir / relative_candidates[0]


def load_dataset_tables(paths: DataPaths) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
  """Load record list, patients, and admissions dataframes resolving paths."""
  record_path = _find_data_file(paths.mimic_iv_ecg_dir, ["record_list.csv", "files/record_list.csv"])
  patients_path = _find_data_file(paths.mimic_iv_dir, ["hosp/patients.csv.gz", "hosp/patients.csv", "patients.csv.gz", "patients.csv"])
  admissions_path = _find_data_file(paths.mimic_iv_dir, ["hosp/admissions.csv.gz", "hosp/admissions.csv", "admissions.csv.gz", "admissions.csv"])

  if not record_path.exists():
    raise FileNotFoundError(f"Record list not found at: {record_path}")
  if not patients_path.exists():
    raise FileNotFoundError(f"Patients table not found at: {patients_path}")
  if not admissions_path.exists():
    raise FileNotFoundError(f"Admissions table not found at: {admissions_path}")

  record_df = load_record_list(record_path)
  patients_df = load_patients(patients_path)
  admissions_df = load_admissions(admissions_path)

  return record_df, patients_df, admissions_df


def build_parser() -> argparse.ArgumentParser:
  """Construct the top-level argument parser and subcommands."""
  common_flags = argparse.ArgumentParser(add_help=False)
  common_flags.add_argument(
    "--mimic-root",
    type=str,
    default=None,
    help=f"Path to MIMIC-IV root directory (env: {ENV_MIMIC_ROOT})",
  )
  common_flags.add_argument(
    "--ecg-root",
    type=str,
    default=None,
    help=f"Path to MIMIC-IV-ECG root directory (env: {ENV_ECG_ROOT})",
  )
  common_flags.add_argument(
    "--output-dir",
    type=str,
    default="./reports",
    help="Directory to save generated reports and figure artifacts",
  )
  common_flags.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_SPLIT_SEED,
    help="Deterministic random seed for partitioning and bootstrapping",
  )
  common_flags.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    default=False,
    help="Enable verbose progress output",
  )
  common_flags.add_argument(
    "--simulate",
    action="store_true",
    default=False,
    help="Execute in simulation mode using synthetic scores and embeddings",
  )
  common_flags.add_argument(
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "mps", "cuda"],
    help="Computation device for transformer model inference ('cpu', 'mps', 'cuda')",
  )
  common_flags.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Inference batch size for transformer representation extraction",
  )
  common_flags.add_argument(
    "--predictions-path",
    type=str,
    default=None,
    help="Path to load or save authoritative unified prediction table artifact",
  )
  common_flags.add_argument(
    "--max-records",
    type=int,
    default=None,
    help="Optional maximum number of records to process for quick testing",
  )
  common_flags.add_argument(
    "--n-bootstraps",
    type=int,
    default=50,
    help="Number of patient-level bootstrap iterations",
  )

  parser = argparse.ArgumentParser(
    prog="ecg-alignment",
    description="ECG Model Alignment: Traditional ECG Risk vs Multimodal Transformer Representations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=[common_flags],
  )
  parser.add_argument(
    "--version",
    action="version",
    version="%(prog)s 0.1.0",
  )

  subparsers = parser.add_subparsers(
    dest="command",
    title="subcommands",
    description="Valid subcommands for ECG alignment stages",
  )

  # 1. inventory subcommand
  inventory_parser = subparsers.add_parser(
    "inventory",
    help="Run Stage 1 data inventory and linkage statistics",
    parents=[common_flags],
  )
  inventory_parser.add_argument(
    "--report-out",
    type=str,
    default=None,
    help="Optional output file path for data inventory markdown report",
  )

  # 2. cohort subcommand
  cohort_parser = subparsers.add_parser(
    "cohort",
    help="Derive Stage 7 primary patient cohort and patient-disjoint split",
    parents=[common_flags],
  )
  cohort_parser.add_argument(
    "--report-out",
    type=str,
    default=None,
    help="Optional output file path for cohort flow markdown report",
  )

  # 3. probe subcommand
  probe_parser = subparsers.add_parser(
    "probe",
    help="Train Stage 8 frozen-embedding linear probe on development cohort",
    parents=[common_flags],
  )
  probe_parser.add_argument(
    "--report-out",
    type=str,
    default=None,
    help="Optional output file path for continuous predictions markdown report",
  )

  # 4. analyze subcommand
  analyze_parser = subparsers.add_parser(
    "analyze",
    help="Execute Stage 9 primary statistical analysis pipeline",
    parents=[common_flags],
  )
  analyze_parser.add_argument(
    "--generate-figures",
    action="store_true",
    default=True,
    help="Generate matplotlib figure artifacts",
  )
  analyze_parser.add_argument(
    "--report-out",
    type=str,
    default=None,
    help="Optional output file path for primary results markdown report",
  )

  # 5. sensitivity subcommand
  sens_parser = subparsers.add_parser(
    "sensitivity",
    help="Execute Stage 10 comprehensive sensitivity and robustness battery",
    parents=[common_flags],
  )
  sens_parser.add_argument(
    "--report-out",
    type=str,
    default=None,
    help="Optional output file path for sensitivity analyses markdown report",
  )

  # 6. interpret subcommand
  interp_parser = subparsers.add_parser(
    "interpret",
    help="Synthesize Stage 11 research interpretation and validation roadmap",
    parents=[common_flags],
  )
  interp_parser.add_argument(
    "--report-out",
    type=str,
    default=None,
    help="Optional output file path for research interpretation markdown report",
  )

  # 7. pipeline subcommand
  pipeline_parser = subparsers.add_parser(
    "pipeline",
    help="Execute end-to-end reproducible research pipeline (Stages 1-11)",
    parents=[common_flags],
  )
  pipeline_parser.add_argument(
    "--skip-figures",
    action="store_true",
    default=False,
    help="Skip figure generation during pipeline execution",
  )

  return parser


def create_cli_context(args: argparse.Namespace) -> CliContext:
  """Build a frozen CliContext from parsed arguments."""
  data_paths = resolve_data_paths(
    mimic_root=getattr(args, "mimic_root", None),
    ecg_root=getattr(args, "ecg_root", None),
  )
  output_dir = Path(getattr(args, "output_dir", "./reports") or "./reports").expanduser().resolve()
  seed = int(getattr(args, "seed", DEFAULT_SPLIT_SEED))
  verbose = bool(getattr(args, "verbose", False))
  simulate = bool(getattr(args, "simulate", False))
  pred_path_arg = getattr(args, "predictions_path", None)
  predictions_path = Path(pred_path_arg).expanduser().resolve() if pred_path_arg is not None else None
  max_records = int(getattr(args, "max_records", 0)) if getattr(args, "max_records", None) is not None else None
  n_bootstraps = int(getattr(args, "n_bootstraps", 50))
  device = str(getattr(args, "device", "cpu") or "cpu")
  batch_size = int(getattr(args, "batch_size", 32) or 32)

  return CliContext(
    mimic_root=data_paths.mimic_iv_dir,
    ecg_root=data_paths.mimic_iv_ecg_dir,
    output_dir=output_dir,
    seed=seed,
    verbose=verbose,
    simulate=simulate,
    predictions_path=predictions_path,
    max_records=max_records,
    n_bootstraps=n_bootstraps,
    device=device,
    batch_size=batch_size,
  )


def run_inventory(ctx: CliContext, report_out: str | None = None) -> int:
  """Execute data inventory stage."""
  if ctx.verbose:
    print(f"Loading MIMIC-IV and ECG data from:\n  MIMIC: {ctx.mimic_root}\n  ECG:   {ctx.ecg_root}")

  paths = DataPaths(mimic_iv_dir=ctx.mimic_root, mimic_iv_ecg_dir=ctx.ecg_root)
  try:
    record_df, patients_df, admissions_df = load_dataset_tables(paths)
  except FileNotFoundError as exc:
    print(f"Data file error: {exc}", file=sys.stderr)
    return 1

  stats = compute_linkage_statistics(record_df, patients_df, admissions_df)

  print(f"[Inventory] Total ECGs: {stats.get('total_ecgs', 0):,}")
  print(f"[Inventory] Unique patients: {stats.get('unique_ecg_subjects', 0):,}")
  print(f"[Inventory] Linked to admission: {stats.get('ecgs_during_admission', 0):,}")
  print(f"[Inventory] Index adult ECGs: {stats.get('index_adult_ecgs', 0):,}")

  if report_out:
    out_path = Path(report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flow_steps = generate_cohort_flow(record_df, patients_df)
    report_lines = [
      "# Stage 1 Data Inventory and Linkage Report",
      "",
      f"- **Total ECGs:** {stats.get('total_ecgs', 0):,}",
      f"- **Unique Patients:** {stats.get('unique_ecg_subjects', 0):,}",
      f"- **Linked to Admission:** {stats.get('ecgs_during_admission', 0):,}",
      f"- **Index Adult ECGs:** {stats.get('index_adult_ecgs', 0):,}",
      "",
      "## Cohort Flow Summary",
      "",
      "| Step | Description | Records | Subjects |",
      "| :--- | :--- | :--- | :--- |",
    ]
    for s in flow_steps:
      report_lines.append(f"| {s.step_name} | {s.description} | {s.record_count:,} | {s.subject_count:,} |")
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[Inventory] Report written to: {out_path}")

  return 0


def run_cohort(ctx: CliContext, report_out: str | None = None) -> int:
  """Execute cohort definition and split stage."""
  if ctx.verbose:
    print(f"Deriving primary cohort and patient-disjoint split (seed={ctx.seed})...")

  paths = DataPaths(mimic_iv_dir=ctx.mimic_root, mimic_iv_ecg_dir=ctx.ecg_root)
  try:
    record_df, patients_df, admissions_df = load_dataset_tables(paths)
  except FileNotFoundError as exc:
    print(f"Data file error: {exc}", file=sys.stderr)
    return 1

  split_res = build_primary_cohort_and_split(
    records_df=record_df,
    patients_df=patients_df,
    admissions_df=admissions_df,
    seed=ctx.seed,
  )

  print(f"[Cohort] Primary cohort size: {len(split_res.cohort_df):,} patients")
  print(f"[Cohort] Split sizes: Dev={split_res.split_record_counts.get('dev', 0):,}, Val={split_res.split_record_counts.get('val', 0):,}, Test={split_res.split_record_counts.get('test', 0):,}")

  out_path = Path(report_out) if report_out else ctx.output_dir / "cohort-flow.md"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  md = generate_cohort_flow_markdown(split_res)
  out_path.write_text(md, encoding="utf-8")
  print(f"[Cohort] Flow report written to: {out_path}")

  return 0


def check_waveforms_available(cohort_df: pl.DataFrame, ecg_root: Path) -> bool:
  """Check if WFDB signal files exist for cohort records under ecg_root."""
  if len(cohort_df) == 0:
    return False
  checked = 0
  for row in cohort_df.iter_rows(named=True):
    rel_p = str(row.get("path", ""))
    if not rel_p:
      continue
    full_p = ecg_root / rel_p
    if (
      full_p.with_suffix(".hea").exists()
      or Path(f"{full_p}.hea").exists()
      or full_p.exists()
    ):
      return True
    checked += 1
    if checked >= 5:
      break
  return False


def build_real_predictions_for_cohort(
  split_res: Any,
  ecg_root: Path,
  seed: int = DEFAULT_PROBE_SEED,
  device: str = "cpu",
  batch_size: int = 32,
  verbose: bool = False,
) -> tuple[pl.DataFrame, TrainedProbe]:
  """Score Model A (CIIS) and Model B (D-BETA + linear probe) on real waveforms."""
  cohort_df = split_res.cohort_df

  if verbose:
    print(f"[Scoring] Computing traditional Model A (CIIS) for {len(cohort_df):,} waveforms...")
  model_a_df = score_traditional_cohort(cohort_df, ecg_data_dir=ecg_root)

  if verbose:
    print(f"[Scoring] Extracting Model B (D-BETA) representations on {device} (batch size {batch_size}) for {len(cohort_df):,} waveforms...")
  adapter = DbetaAdapter(DbetaConfig(device=device, batch_size=batch_size))
  meta_df, embeddings = extract_transformer_embeddings(cohort_df, adapter=adapter, ecg_data_dir=ecg_root)

  dev_mask = (cohort_df["split"] == "dev").to_numpy()
  val_mask = (cohort_df["split"] == "val").to_numpy()
  valid_mask = meta_df["is_valid"].to_numpy()

  dev_train_mask = dev_mask & valid_mask
  val_tune_mask = val_mask & valid_mask

  dev_embs = embeddings[dev_train_mask]
  dev_y = cohort_df.filter(pl.Series(dev_train_mask))["mortality_30d"].cast(pl.Int64).to_numpy()

  val_embs = embeddings[val_tune_mask]
  val_y = cohort_df.filter(pl.Series(val_tune_mask))["mortality_30d"].cast(pl.Int64).to_numpy()

  if len(np.unique(dev_y)) < 2:
    raise ValueError("Development set must contain at least 1 positive and 1 negative outcome.")

  if verbose:
    print(f"[Probe] Fitting linear probe on {len(dev_y):,} dev waveforms (tuning on {len(val_y):,} val waveforms)...")
  probe = fit_logistic_probe(dev_embs, dev_y, val_embs, val_y, config=ProbeConfig(random_state=seed))

  model_b_df = score_transformer_cohort(
    cohort_df,
    embeddings=embeddings,
    probe=probe,
    valid_mask=meta_df["is_valid"].to_list(),
    error_messages=meta_df["error_message"].to_list(),
  )

  unified = build_unified_prediction_table(cohort_df, model_a_df, model_b_df)
  return unified, probe


def simulate_cohort_predictions(
  cohort_df: pl.DataFrame,
  seed: int = 42,
) -> tuple[pl.DataFrame, TrainedProbe]:
  """Generate reproducible Model A and Model B prediction tables for cohort in simulation mode."""
  rng = np.random.default_rng(seed)
  n = len(cohort_df)

  # Traditional CIIS scores
  a_scores = np.clip(rng.gamma(shape=3.0, scale=4.0, size=n), 0, 45)
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

  model_a_df = pl.DataFrame({
    "subject_id": cohort_df["subject_id"].to_list(),
    "study_id": cohort_df["study_id"].to_list(),
    "model_a_score": a_scores.tolist(),
    "model_a_category": categories,
    "model_a_valid": [True] * n,
    "model_a_error": [None] * n,
  })

  # Transformer Model B embeddings & probe
  dim = 16
  embs = rng.normal(size=(n, dim))
  w_true = rng.normal(size=dim)
  latent_risk = 0.08 * a_scores + (embs @ w_true) * 0.2 - 1.5
  b_probs = 1.0 / (1.0 + np.exp(-latent_risk))

  model_b_df = pl.DataFrame({
    "subject_id": cohort_df["subject_id"].to_list(),
    "study_id": cohort_df["study_id"].to_list(),
    "model_b_score": b_probs.tolist(),
    "model_b_log_odds": latent_risk.tolist(),
    "model_b_valid": [True] * n,
    "model_b_error": [None] * n,
  })

  # Ensure mortality_30d exists
  if "mortality_30d" not in cohort_df.columns:
    true_prob = 1.0 / (1.0 + np.exp(-(0.12 * a_scores + 2.5 * b_probs - 3.2)))
    m30 = (rng.uniform(0, 1, size=n) < true_prob).astype(np.int64)
    if np.sum(m30) < 5:
      m30[:5] = 1
    cohort_df = cohort_df.with_columns(
      pl.Series("mortality_30d", [bool(m) for m in m30], dtype=pl.Boolean),
      pl.Series("mortality_90d", [bool(m) for m in m30], dtype=pl.Boolean),
      pl.Series("mortality_1yr", [bool(m) for m in m30], dtype=pl.Boolean),
    )

  unified = build_unified_prediction_table(cohort_df, model_a_df, model_b_df)

  dev_mask = (unified["split"] == "dev").to_numpy()
  val_mask = (unified["split"] == "val").to_numpy()

  dev_embs = embs[dev_mask]
  dev_y = unified.filter(pl.col("split") == "dev")["mortality_30d"].cast(pl.Int64).to_numpy().copy()
  val_embs = embs[val_mask]
  val_y = unified.filter(pl.col("split") == "val")["mortality_30d"].cast(pl.Int64).to_numpy().copy()

  # Ensure both classes exist in dev/val
  if len(np.unique(dev_y)) < 2 and len(dev_y) >= 2:
    dev_y[0] = 1
    dev_y[0] = 0
  if len(np.unique(val_y)) < 2 and len(val_y) >= 2:
    val_y[0] = 1
    val_y[1] = 0
  elif len(val_y) == 1:
    val_embs = np.vstack([val_embs, val_embs])
    val_y = np.array([1, 0], dtype=np.int64)

  probe = fit_logistic_probe(dev_embs, dev_y, val_embs, val_y, config=ProbeConfig(random_state=seed))
  return unified, probe


# Alias for backward compatibility
_build_synthetic_predictions_for_cohort = simulate_cohort_predictions


def run_probe(
  ctx: CliContext,
  report_out: str | None = None,
) -> int:
  """Execute continuous predictions and linear probe stage."""
  if ctx.verbose:
    print(f"Executing probe stage (seed={ctx.seed}, simulate={ctx.simulate}, device={ctx.device})...")

  # If precomputed predictions artifact exists, load it
  if ctx.predictions_path is not None and ctx.predictions_path.exists():
    if ctx.verbose:
      print(f"Loading cached predictions from {ctx.predictions_path}...")
    try:
      unified_table = load_unified_prediction_table(ctx.predictions_path)
      stats = compute_prediction_summary_statistics(unified_table)
      out_path = Path(report_out) if report_out else ctx.output_dir / "continuous-predictions.md"
      out_path.parent.mkdir(parents=True, exist_ok=True)
      md = generate_continuous_predictions_markdown(stats)
      out_path.write_text(md, encoding="utf-8")
      print(f"[Probe] Report written to: {out_path}")
      return 0
    except Exception as exc:
      print(f"[Probe] Error loading predictions artifact: {exc}", file=sys.stderr)
      return 1

  paths = DataPaths(mimic_iv_dir=ctx.mimic_root, mimic_iv_ecg_dir=ctx.ecg_root)
  try:
    record_df, patients_df, admissions_df = load_dataset_tables(paths)
  except FileNotFoundError as exc:
    print(f"Data file error: {exc}", file=sys.stderr)
    return 1

  split_res = build_primary_cohort_and_split(record_df, patients_df, admissions_df, seed=ctx.seed)

  if ctx.max_records is not None and ctx.max_records > 0:
    cohort_sub = split_res.cohort_df.head(ctx.max_records)
    split_res = build_primary_cohort_and_split(
      record_df.filter(pl.col("subject_id").is_in(cohort_sub["subject_id"])),
      patients_df,
      admissions_df,
      seed=ctx.seed,
    )

  has_waveforms = check_waveforms_available(split_res.cohort_df, ctx.ecg_root)

  if ctx.simulate:
    unified_table, probe = simulate_cohort_predictions(split_res.cohort_df, seed=ctx.seed)
    data_mode = "simulation"
  else:
    if not has_waveforms:
      print(
        f"[Probe] Error: WFDB waveform files not found in ECG root directory ({ctx.ecg_root}). "
        "Real scoring mode requires waveform files. Run with --simulate to run synthetic mode.",
        file=sys.stderr,
      )
      return 1
    try:
      unified_table, probe = build_real_predictions_for_cohort(
        split_res,
        ecg_root=ctx.ecg_root,
        seed=ctx.seed,
        device=ctx.device,
        batch_size=ctx.batch_size,
        verbose=ctx.verbose,
      )
      data_mode = "real"
    except Exception as exc:
      print(
        f"[Probe] Error: Real scoring execution failed: {exc}. "
        "Terminating run (fail-closed). Run with --simulate if synthetic evaluation is desired.",
        file=sys.stderr,
      )
      return 1

  # Save prediction artifact
  save_target = ctx.predictions_path or (ctx.output_dir / "predictions.parquet")
  try:
    save_unified_prediction_table(unified_table, save_target)
    if ctx.verbose:
      print(f"[Probe] Saved prediction table to {save_target}")
  except Exception as e:
    if ctx.verbose:
      print(f"[Probe] Notice: unable to write parquet ({e}); skipping artifact cache.")

  # Save run manifest
  manifest = generate_run_manifest(
    unified_table,
    probe=probe,
    seed=ctx.seed,
    data_mode=data_mode,
    predictions_path=save_target if save_target.exists() else None,
  )
  save_run_manifest(manifest, ctx.output_dir / "run_manifest.json")

  stats = compute_prediction_summary_statistics(unified_table)
  print(f"[Probe] Best regularization C: {probe.best_c}")
  print(f"[Probe] Unified prediction table rows: {len(unified_table):,}")

  out_path = Path(report_out) if report_out else ctx.output_dir / "continuous-predictions.md"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  md = generate_continuous_predictions_markdown(stats, probe=probe)
  out_path.write_text(md, encoding="utf-8")
  print(f"[Probe] Report written to: {out_path}")

  return 0


def _get_unified_table(ctx: CliContext) -> pl.DataFrame:
  """Retrieve unified prediction table from cache or derive it."""
  if ctx.predictions_path is not None and ctx.predictions_path.exists():
    return load_unified_prediction_table(ctx.predictions_path)
  default_cache = ctx.output_dir / "predictions.parquet"
  if default_cache.exists():
    return load_unified_prediction_table(default_cache)

  paths = DataPaths(mimic_iv_dir=ctx.mimic_root, mimic_iv_ecg_dir=ctx.ecg_root)
  record_df, patients_df, admissions_df = load_dataset_tables(paths)
  split_res = build_primary_cohort_and_split(record_df, patients_df, admissions_df, seed=ctx.seed)
  if ctx.max_records is not None and ctx.max_records > 0:
    cohort_sub = split_res.cohort_df.head(ctx.max_records)
    split_res = build_primary_cohort_and_split(
      record_df.filter(pl.col("subject_id").is_in(cohort_sub["subject_id"])),
      patients_df,
      admissions_df,
      seed=ctx.seed,
    )

  has_waveforms = check_waveforms_available(split_res.cohort_df, ctx.ecg_root)
  if ctx.simulate:
    unified_table, _ = simulate_cohort_predictions(split_res.cohort_df, seed=ctx.seed)
  else:
    if not has_waveforms:
      raise FileNotFoundError(
        f"Real mode requested but WFDB waveforms not found in {ctx.ecg_root}. "
        "Provide precomputed predictions or run with --simulate."
      )
    unified_table, _ = build_real_predictions_for_cohort(
      split_res,
      ecg_root=ctx.ecg_root,
      seed=ctx.seed,
      device=ctx.device,
      batch_size=ctx.batch_size,
      verbose=ctx.verbose,
    )
  return unified_table


def run_analyze(
  ctx: CliContext,
  generate_figs: bool = True,
  report_out: str | None = None,
) -> int:
  """Execute primary statistical analysis stage."""
  if ctx.verbose:
    print("Running primary statistical analysis on test cohort...")

  try:
    unified_table = _get_unified_table(ctx)
  except FileNotFoundError as exc:
    print(f"Data file error: {exc}", file=sys.stderr)
    return 1
  except Exception as exc:
    print(f"Analysis preparation error: {exc}", file=sys.stderr)
    return 1

  primary_result = run_primary_analysis(
    unified_table,
    n_bootstraps=ctx.n_bootstraps,
    random_seed=ctx.seed,
  )

  print(f"[Analysis] Test N: {primary_result.n_patients:,}")
  print(f"[Analysis] Spearman rho: {primary_result.alignment.spearman_rho:.4f}")
  print(f"[Analysis] AUROC Model A: {primary_result.performance_a.auroc.point_estimate:.4f}")
  print(f"[Analysis] AUROC Model B: {primary_result.performance_b.auroc.point_estimate:.4f}")
  print(f"[Analysis] Incremental LRT p-value: {primary_result.incremental.lrt_pvalue:.4e}")

  ctx.output_dir.mkdir(parents=True, exist_ok=True)
  if generate_figs:
    fig_dir = ctx.output_dir / "figures"
    generate_primary_figures(primary_result, output_dir=fig_dir)
    print(f"[Analysis] Figures generated under: {fig_dir}")

  out_path = Path(report_out) if report_out else ctx.output_dir / "primary-results.md"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  md = generate_primary_results_markdown(primary_result)
  out_path.write_text(md, encoding="utf-8")
  print(f"[Analysis] Report written to: {out_path}")

  return 0


def run_sensitivity(ctx: CliContext, report_out: str | None = None) -> int:
  """Execute sensitivity analyses battery."""
  if ctx.verbose:
    print("Running comprehensive sensitivity and robustness analyses...")

  try:
    unified_table = _get_unified_table(ctx)
  except FileNotFoundError as exc:
    print(f"Data file error: {exc}", file=sys.stderr)
    return 1
  except Exception as exc:
    print(f"Sensitivity preparation error: {exc}", file=sys.stderr)
    return 1

  test_df = unified_table.filter(pl.col("split") == "test")
  dev_df = unified_table.filter(pl.col("split") == "dev")

  strata_df: pl.DataFrame | None = None
  try:
    paths = DataPaths(mimic_iv_dir=ctx.mimic_root, mimic_iv_ecg_dir=ctx.ecg_root)
    _, patients_df, _ = load_dataset_tables(paths)
    joined = test_df.select(["subject_id"]).join(patients_df, on="subject_id", how="left")
    if "anchor_age" in joined.columns and "gender" in joined.columns:
      strata_df = joined.with_columns(
        pl.when(pl.col("anchor_age") < 65).then(pl.lit("<65")).otherwise(pl.lit(">=65")).alias("age_group")
      ).select(["subject_id", "age_group", "gender"])
  except Exception:
    strata_df = None

  sens_result = run_sensitivity_analyses(
    earliest_test_df=test_df,
    dev_df=dev_df,
    evaluation_strata_df=strata_df,
    n_bootstraps=min(ctx.n_bootstraps, 20),
    seed=ctx.seed,
  )

  print(f"[Sensitivity] Evaluated {len(sens_result.outcome_horizons)} outcome horizons")
  print(f"[Sensitivity] Evaluated {len(sens_result.probe_architectures)} probe configurations")
  print(f"[Sensitivity] Evaluated {len(sens_result.alternative_traditional)} traditional scoring criteria")
  print(f"[Sensitivity] Evaluated {len(sens_result.demographic_subgroups)} demographic strata")

  out_path = Path(report_out) if report_out else ctx.output_dir / "sensitivity-analyses.md"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  md = generate_sensitivity_report_markdown(sens_result)
  out_path.write_text(md, encoding="utf-8")
  print(f"[Sensitivity] Report written to: {out_path}")

  return 0


def run_interpret(
  ctx: CliContext,
  primary_result: PrimaryAnalysisResult | None = None,
  sensitivity_result: FullSensitivityAnalysisResult | None = None,
  report_out: str | None = None,
) -> int:
  """Execute research interpretation and validation roadmap synthesis."""
  if ctx.verbose:
    print("Synthesizing research interpretation and scientific translation roadmap...")

  if primary_result is None:
    try:
      unified_table = _get_unified_table(ctx)
      primary_result = run_primary_analysis(
        unified_table,
        n_bootstraps=ctx.n_bootstraps,
        random_seed=ctx.seed,
      )
    except Exception as exc:
      print(f"[Interpretation] Error deriving primary analysis: {exc}", file=sys.stderr)
      return 1

  if sensitivity_result is None:
    try:
      unified_table = _get_unified_table(ctx)
      test_df = unified_table.filter(pl.col("split") == "test")
      dev_df = unified_table.filter(pl.col("split") == "dev")
      sensitivity_result = run_sensitivity_analyses(
        earliest_test_df=test_df,
        dev_df=dev_df,
        n_bootstraps=min(ctx.n_bootstraps, 20),
        seed=ctx.seed,
      )
    except Exception:
      sensitivity_result = None

  synthesis = synthesize_research_interpretation(
    primary_result=primary_result,
    sensitivity_result=sensitivity_result,
  )

  print(f"[Interpretation] Alignment Strength: {synthesis.alignment.strength.value}")
  print(f"[Interpretation] Within-A Gradients: {synthesis.within_a_gradients.summary_narrative}")
  print(f"[Interpretation] Contamination Risk: {synthesis.contamination_audit.has_any_contamination}")
  print(f"[Interpretation] Recommendation Status: {synthesis.external_validation_recommendation.status.value}")

  out_path = Path(report_out) if report_out else ctx.output_dir / "research-interpretation.md"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  md = generate_research_interpretation_markdown(synthesis)
  out_path.write_text(md, encoding="utf-8")
  print(f"[Interpretation] Report written to: {out_path}")

  return 0


def run_pipeline(ctx: CliContext, skip_figures: bool = False) -> int:
  """Execute the complete end-to-end research workflow."""
  print("=== Executing ECG Model Alignment Pipeline ===")
  print(f"Config: MIMIC={ctx.mimic_root}, ECG={ctx.ecg_root}, Output={ctx.output_dir}, Seed={ctx.seed}")

  rc = run_inventory(ctx)
  if rc != 0:
    return rc

  rc = run_cohort(ctx)
  if rc != 0:
    return rc

  rc = run_probe(ctx)
  if rc != 0:
    return rc

  try:
    unified_table = _get_unified_table(ctx)
  except Exception as exc:
    print(f"Pipeline error retrieving prediction table: {exc}", file=sys.stderr)
    return 1

  primary_result = run_primary_analysis(
    unified_table,
    n_bootstraps=ctx.n_bootstraps,
    random_seed=ctx.seed,
  )

  ctx.output_dir.mkdir(parents=True, exist_ok=True)
  if not skip_figures:
    fig_dir = ctx.output_dir / "figures"
    generate_primary_figures(primary_result, output_dir=fig_dir)

  prim_path = ctx.output_dir / "primary-results.md"
  prim_path.write_text(generate_primary_results_markdown(primary_result), encoding="utf-8")

  test_df = unified_table.filter(pl.col("split") == "test")
  dev_df = unified_table.filter(pl.col("split") == "dev")

  strata_df: pl.DataFrame | None = None
  try:
    paths = DataPaths(mimic_iv_dir=ctx.mimic_root, mimic_iv_ecg_dir=ctx.ecg_root)
    _, patients_df, _ = load_dataset_tables(paths)
    joined = test_df.select(["subject_id"]).join(patients_df, on="subject_id", how="left")
    if "anchor_age" in joined.columns and "gender" in joined.columns:
      strata_df = joined.with_columns(
        pl.when(pl.col("anchor_age") < 65).then(pl.lit("<65")).otherwise(pl.lit(">=65")).alias("age_group")
      ).select(["subject_id", "age_group", "gender"])
  except Exception:
    strata_df = None

  sens_result = run_sensitivity_analyses(
    earliest_test_df=test_df,
    dev_df=dev_df,
    evaluation_strata_df=strata_df,
    n_bootstraps=min(ctx.n_bootstraps, 20),
    seed=ctx.seed,
  )
  sens_path = ctx.output_dir / "sensitivity-analyses.md"
  sens_path.write_text(generate_sensitivity_report_markdown(sens_result), encoding="utf-8")

  rc = run_interpret(ctx, primary_result=primary_result, sensitivity_result=sens_result)
  if rc != 0:
    return rc

  print("=== Pipeline Complete: All Stages Executed Successfully ===")
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """Main CLI entrypoint."""
  parser = build_parser()
  args = parser.parse_args(argv if argv is not None else sys.argv[1:])

  if not hasattr(args, "command") or args.command is None:
    parser.print_help()
    return 0

  ctx = create_cli_context(args)

  if args.command == "inventory":
    return run_inventory(ctx, report_out=getattr(args, "report_out", None))
  elif args.command == "cohort":
    return run_cohort(ctx, report_out=getattr(args, "report_out", None))
  elif args.command == "probe":
    return run_probe(ctx, report_out=getattr(args, "report_out", None))
  elif args.command == "analyze":
    return run_analyze(
      ctx,
      generate_figs=getattr(args, "generate_figures", True),
      report_out=getattr(args, "report_out", None),
    )
  elif args.command == "sensitivity":
    return run_sensitivity(ctx, report_out=getattr(args, "report_out", None))
  elif args.command == "interpret":
    return run_interpret(ctx, report_out=getattr(args, "report_out", None))
  elif args.command == "pipeline":
    return run_pipeline(ctx, skip_figures=getattr(args, "skip_figures", False))
  else:
    parser.print_help()
    return 1


if __name__ == "__main__":
  sys.exit(main())

