"""Tests for the ecg_alignment CLI and pipeline orchestration."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from ecg_alignment.cli import (
  DEFAULT_ECG_ROOT,
  DEFAULT_MIMIC_ROOT,
  ENV_ECG_ROOT,
  ENV_ECG_ROOT_ALT,
  ENV_MIMIC_ROOT,
  build_parser,
  create_cli_context,
  main,
  resolve_data_paths,
  run_analyze,
  run_cohort,
  run_interpret,
  run_inventory,
  run_pipeline,
  run_probe,
  run_sensitivity,
)


def test_resolve_data_paths_explicit() -> None:
  """Explicit paths override defaults and environment variables."""
  paths = resolve_data_paths(
    mimic_root="/tmp/custom_mimic",
    ecg_root="/tmp/custom_ecg",
  )
  assert str(paths.mimic_iv_dir).endswith("custom_mimic")
  assert str(paths.mimic_iv_ecg_dir).endswith("custom_ecg")


def test_resolve_data_paths_env(monkeypatch: pytest.MonkeyPatch) -> None:
  """Environment variables are used when explicit arguments are omitted."""
  monkeypatch.setenv(ENV_MIMIC_ROOT, "/tmp/env_mimic")
  monkeypatch.setenv(ENV_ECG_ROOT, "/tmp/env_ecg")

  paths = resolve_data_paths()
  assert str(paths.mimic_iv_dir).endswith("env_mimic")
  assert str(paths.mimic_iv_ecg_dir).endswith("env_ecg")


def test_resolve_data_paths_alt_env(monkeypatch: pytest.MonkeyPatch) -> None:
  """Alternative ECG_ROOT environment variable is supported."""
  monkeypatch.delenv(ENV_ECG_ROOT, raising=False)
  monkeypatch.setenv(ENV_ECG_ROOT_ALT, "/tmp/alt_ecg")

  paths = resolve_data_paths()
  assert str(paths.mimic_iv_ecg_dir).endswith("alt_ecg")


def test_resolve_data_paths_default(monkeypatch: pytest.MonkeyPatch) -> None:
  """Default paths are resolved when no args or env vars are set."""
  monkeypatch.delenv(ENV_MIMIC_ROOT, raising=False)
  monkeypatch.delenv(ENV_ECG_ROOT, raising=False)
  monkeypatch.delenv(ENV_ECG_ROOT_ALT, raising=False)

  paths = resolve_data_paths()
  assert "mimiciv" in str(paths.mimic_iv_dir)
  assert "mimic-iv-ecg" in str(paths.mimic_iv_ecg_dir)


def test_build_parser_subcommands() -> None:
  """Parser registers all staged subcommands and standard flags."""
  parser = build_parser()

  # Test valid subcommand parsing
  args_inv = parser.parse_args(["inventory", "--report-out", "/tmp/inv.md"])
  assert args_inv.command == "inventory"
  assert args_inv.report_out == "/tmp/inv.md"

  args_cohort = parser.parse_args(["cohort", "--report-out", "/tmp/flow.md"])
  assert args_cohort.command == "cohort"
  assert args_cohort.report_out == "/tmp/flow.md"

  args_probe = parser.parse_args(["probe", "--seed", "123"])
  assert args_probe.command == "probe"
  assert args_probe.seed == 123

  args_analyze = parser.parse_args(["analyze", "--generate-figures"])
  assert args_analyze.command == "analyze"
  assert args_analyze.generate_figures is True

  args_sens = parser.parse_args(["sensitivity"])
  assert args_sens.command == "sensitivity"

  args_interp = parser.parse_args(["interpret"])
  assert args_interp.command == "interpret"

  args_pipe = parser.parse_args(["pipeline", "--skip-figures"])
  assert args_pipe.command == "pipeline"
  assert args_pipe.skip_figures is True


def test_cli_context_creation() -> None:
  """create_cli_context builds typed CliContext with expanded paths."""
  parser = build_parser()
  args = parser.parse_args([
    "inventory",
    "--mimic-root", "/tmp/test_mimic",
    "--ecg-root", "/tmp/test_ecg",
    "--output-dir", "/tmp/test_out",
    "--seed", "999",
    "--verbose",
  ])
  ctx = create_cli_context(args)
  assert ctx.seed == 999
  assert ctx.verbose is True
  assert str(ctx.mimic_root).endswith("test_mimic")
  assert str(ctx.ecg_root).endswith("test_ecg")
  assert str(ctx.output_dir).endswith("test_out")


def test_main_no_args(capsys: pytest.CaptureFixture[str]) -> None:
  """Invoking main with no arguments prints help and returns 0."""
  rc = main([])
  captured = capsys.readouterr()
  assert rc == 0
  assert "usage: ecg-alignment" in captured.out


def test_cli_missing_files_error_handling(tmp_path: Path) -> None:
  """CLI subcommands return error code 1 when input data files are missing."""
  parser = build_parser()
  args = parser.parse_args([
    "inventory",
    "--mimic-root", str(tmp_path / "nonexistent_mimic"),
    "--ecg-root", str(tmp_path / "nonexistent_ecg"),
  ])
  ctx = create_cli_context(args)

  assert run_inventory(ctx) == 1
  assert run_cohort(ctx) == 1
  assert run_probe(ctx) == 1
  assert run_analyze(ctx) == 1
  assert run_sensitivity(ctx) == 1


@pytest.fixture
def synthetic_mimic_env(tmp_path: Path) -> tuple[Path, Path]:
  """Create a synthetic MIMIC and MIMIC-ECG directory structure with test CSVs."""
  mimic_dir = tmp_path / "mimiciv" / "3.1"
  hosp_dir = mimic_dir / "hosp"
  hosp_dir.mkdir(parents=True, exist_ok=True)

  ecg_dir = tmp_path / "mimic-iv-ecg" / "1.0"
  files_dir = ecg_dir / "files" / "p1000" / "p10000001"
  files_dir.mkdir(parents=True, exist_ok=True)

  # Generate 20 synthetic patients
  pat_lines = ["subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod"]
  adm_lines = ["subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,insurance,language,marital_status,race,edregtime,edouttime,hospital_expire_flag"]
  rec_lines = ["subject_id,study_id,file_name,ecg_time,path"]

  for i in range(1, 25):
    s_id = 10000000 + i
    h_id = 20000000 + i
    st_id = 40000000 + i
    gender = "M" if i % 2 == 0 else "F"
    age = 40 + (i * 2)
    dod = f"2150-01-20" if i % 3 == 0 else ""
    pat_lines.append(f"{s_id},{gender},{age},2150,2014 - 2016,{dod}")
    adm_lines.append(f"{s_id},{h_id},2150-01-01 08:00:00,2150-01-10 12:00:00,,URGENT,EMERGENCY ROOM,HOME,Medicare,English,MARRIED,WHITE,2150-01-01 06:00:00,2150-01-01 08:00:00,0")
    rec_lines.append(f"{s_id},{st_id},{st_id},2150-01-01 09:00:00,files/p1000/p10000001/{st_id}")

  patients_csv = hosp_dir / "patients.csv"
  patients_csv.write_text("\n".join(pat_lines) + "\n", encoding="utf-8")

  admissions_csv = hosp_dir / "admissions.csv"
  admissions_csv.write_text("\n".join(adm_lines) + "\n", encoding="utf-8")

  record_list_csv = ecg_dir / "record_list.csv"
  record_list_csv.write_text("\n".join(rec_lines) + "\n", encoding="utf-8")

  return mimic_dir, ecg_dir


def test_cli_subcommands_execution(
  synthetic_mimic_env: tuple[Path, Path],
  tmp_path: Path,
) -> None:
  """Execute each CLI subcommand against the synthetic environment."""
  mimic_dir, ecg_dir = synthetic_mimic_env
  out_dir = tmp_path / "reports_out"

  # 1. Inventory
  inv_rc = main([
    "inventory",
    "--mimic-root", str(mimic_dir),
    "--ecg-root", str(ecg_dir),
    "--output-dir", str(out_dir),
    "--report-out", str(out_dir / "custom_inventory.md"),
  ])
  assert inv_rc == 0
  assert (out_dir / "custom_inventory.md").exists()

  # 2. Cohort
  cohort_rc = main([
    "cohort",
    "--mimic-root", str(mimic_dir),
    "--ecg-root", str(ecg_dir),
    "--output-dir", str(out_dir),
  ])
  assert cohort_rc == 0
  assert (out_dir / "cohort-flow.md").exists()

  # 3. Probe
  probe_rc = main([
    "probe",
    "--mimic-root", str(mimic_dir),
    "--ecg-root", str(ecg_dir),
    "--output-dir", str(out_dir),
  ])
  assert probe_rc == 0
  assert (out_dir / "continuous-predictions.md").exists()

  # 4. Analyze
  analyze_rc = main([
    "analyze",
    "--mimic-root", str(mimic_dir),
    "--ecg-root", str(ecg_dir),
    "--output-dir", str(out_dir),
  ])
  assert analyze_rc == 0
  assert (out_dir / "primary-results.md").exists()

  # 5. Sensitivity
  sens_rc = main([
    "sensitivity",
    "--mimic-root", str(mimic_dir),
    "--ecg-root", str(ecg_dir),
    "--output-dir", str(out_dir),
  ])
  assert sens_rc == 0
  assert (out_dir / "sensitivity-analyses.md").exists()

  # 6. Interpret
  interp_rc = main([
    "interpret",
    "--mimic-root", str(mimic_dir),
    "--ecg-root", str(ecg_dir),
    "--output-dir", str(out_dir),
  ])
  assert interp_rc == 0
  assert (out_dir / "research-interpretation.md").exists()

  # 7. Full pipeline
  pipe_rc = main([
    "pipeline",
    "--mimic-root", str(mimic_dir),
    "--ecg-root", str(ecg_dir),
    "--output-dir", str(out_dir),
    "--skip-figures",
  ])
  assert pipe_rc == 0
