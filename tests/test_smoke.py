"""Smoke tests for ecg_alignment package initialization."""

import ecg_alignment


def test_package_version() -> None:
  assert ecg_alignment.__version__ == "0.1.0"


def test_package_exports() -> None:
  assert hasattr(ecg_alignment, "__version__")
  assert ecg_alignment.__all__ == ["__version__"]
