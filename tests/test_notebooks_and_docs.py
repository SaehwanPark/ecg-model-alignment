"""Tests for notebook integrity and documentation consistency."""

import json
from pathlib import Path
import pytest


def test_research_notebook_json_structure() -> None:
  repo_root = Path(__file__).resolve().parent.parent
  nb_path = repo_root / "notebooks" / "01_research_flow_and_findings.ipynb"

  assert nb_path.exists(), f"Notebook not found at: {nb_path}"

  with open(nb_path, "r", encoding="utf-8") as f:
    nb_data = json.load(f)

  assert nb_data.get("nbformat") == 4
  assert "cells" in nb_data
  assert len(nb_data["cells"]) >= 10

  code_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "code"]
  md_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "markdown"]

  assert len(code_cells) >= 5
  assert len(md_cells) >= 5

  # Check that each code cell has non-empty source
  for cell in code_cells:
    source_lines = cell.get("source", [])
    assert len(source_lines) > 0, "Encountered empty code cell in notebook"


def test_research_notebook_execution_smoke() -> None:
  repo_root = Path(__file__).resolve().parent.parent
  nb_path = repo_root / "notebooks" / "01_research_flow_and_findings.ipynb"

  with open(nb_path, "r", encoding="utf-8") as f:
    nb_data = json.load(f)

  # Execute cells sequentially in a shared namespace
  global_env: dict[str, object] = {}
  for idx, cell in enumerate(nb_data["cells"]):
    if cell.get("cell_type") == "code":
      code_str = "".join(cell.get("source", []))
      try:
        exec(code_str, global_env)
      except Exception as exc:
        pytest.fail(f"Notebook execution failed at cell {idx}: {exc}")


def test_development_guide_structure() -> None:
  repo_root = Path(__file__).resolve().parent.parent
  dev_path = repo_root / "docs" / "development.md"

  assert dev_path.exists(), f"Development guide not found at: {dev_path}"
  content = dev_path.read_text(encoding="utf-8")

  assert "Environment & Dependency Management" in content
  assert "Python Conventions & Code Architecture" in content
  assert "Quality Assurance: Testing & Type Checking" in content
  assert "Predictor-Information Firewall & Data Safety" in content
  assert "Pull Request Workflow" in content


def test_readme_integrity_and_findings() -> None:
  repo_root = Path(__file__).resolve().parent.parent
  readme_path = repo_root / "README.md"

  assert readme_path.exists(), f"README.md not found at: {readme_path}"
  content = readme_path.read_text(encoding="utf-8")

  assert "ECG Risk Alignment" in content
  assert "Empirical Findings" in content
  assert "Interactive Walkthrough & Notebooks" in content
  assert "Predictor-Information Firewall" in content
  assert "Quickstart & CLI Orchestration" in content
  assert "Repository Structure" in content
  assert "01_research_flow_and_findings.ipynb" in content
  assert "reports/primary-results.md" in content
  assert "docs/development.md" in content


def test_notebooks_readme_exists() -> None:
  repo_root = Path(__file__).resolve().parent.parent
  readme_path = repo_root / "notebooks" / "README.md"

  assert readme_path.exists(), f"notebooks/README.md not found at: {readme_path}"
  content = readme_path.read_text(encoding="utf-8")
  assert "01_research_flow_and_findings.ipynb" in content
