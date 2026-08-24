"""Tests for D-BETA primary multimodal transformer adapter and waveform pipeline (Stage 5)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch
import numpy as np
import numpy.typing as npt
import pytest
import torch

from ecg_alignment.data import (
  CANONICAL_12_LEADS,
  DEFAULT_SAMPLING_RATE_HZ,
  DEFAULT_SIGNAL_LENGTH_SAMPLES,
  DataPaths,
  load_record_list,
  read_ecg_waveform,
)
from ecg_alignment.scoring.base import (
  STANDARD_CLINICAL_12_LEADS,
  BatchInferenceResult,
  InputModality,
  ModelOutputResult,
)
from ecg_alignment.scoring.dbeta import (
  DEFAULT_DBETA_EMBEDDING_DIM,
  DEFAULT_DBETA_MODEL_NAME,
  DEFAULT_DBETA_REVISION,
  DEFAULT_DBETA_SAMPLING_RATE_HZ,
  DEFAULT_DBETA_SIGNAL_LENGTH,
  DbetaAdapter,
  DbetaConfig,
  load_dbeta_model,
)


# -----------------------------------------------------------------------------
# Fixtures and Deterministic Mock D-BETA Modules
# -----------------------------------------------------------------------------


@pytest.fixture
def synthetic_12lead_signal() -> tuple[npt.NDArray[np.float64], list[str], int]:
  """Generate a synthetic 10-second 12-lead ECG signal at 500 Hz."""
  fs = 500
  n_samples = 5000
  n_leads = len(CANONICAL_12_LEADS)
  t = np.linspace(0, 10, n_samples, endpoint=False)

  sig = np.zeros((n_samples, n_leads), dtype=np.float64)
  for i in range(n_leads):
    sig[:, i] = np.sin(2 * np.pi * (1.0 + 0.2 * i) * t) * 0.5

  return sig, list(CANONICAL_12_LEADS), fs


class MockDbetaOutput:
  """Mock output container for D-BETA AutoModel forward passes."""

  def __init__(self, tensor: torch.Tensor) -> None:
    self.pooler_output = tensor
    self.last_hidden_state = tensor.unsqueeze(1)  # [B, 1, 768]


class MockDbetaModel(torch.nn.Module):
  """Deterministic mock D-BETA waveform transformer for offline unit testing."""

  def __init__(self, embedding_dim: int = DEFAULT_DBETA_EMBEDDING_DIM) -> None:
    super().__init__()
    self.embedding_dim = embedding_dim
    # Linear projection from 12 leads mean signal to embedding_dim
    self.proj = torch.nn.Linear(12, embedding_dim, bias=False)
    with torch.no_grad():
      torch.nn.init.eye_(self.proj.weight[:12, :12])

  def forward(self, ecg_tensors: torch.Tensor) -> MockDbetaOutput:
    # ecg_tensors shape: [batch, 12, 5000]
    lead_means = ecg_tensors.mean(dim=2)  # [batch, 12]
    embed = self.proj(lead_means)  # [batch, 768]
    return MockDbetaOutput(embed)

  def eval(self) -> "MockDbetaModel":
    return self


@pytest.fixture
def mock_dbeta_adapter() -> DbetaAdapter:
  """Instantiate a DbetaAdapter with a deterministic mock model."""
  mock_model = MockDbetaModel(embedding_dim=DEFAULT_DBETA_EMBEDDING_DIM)
  config = DbetaConfig(batch_size=4, device="cpu", normalize_embeddings=False)
  return DbetaAdapter(config=config, model=mock_model)


# -----------------------------------------------------------------------------
# Unit Tests: Config and Preprocessing
# -----------------------------------------------------------------------------


def test_dbeta_config_defaults():
  """Verify default configuration parameters for D-BETA adapter."""
  cfg = DbetaConfig()
  assert cfg.model_name == DEFAULT_DBETA_MODEL_NAME
  assert cfg.model_version == DEFAULT_DBETA_REVISION
  assert cfg.input_modality == InputModality.WAVEFORM
  assert cfg.embedding_dim == 768
  assert cfg.normalize_embeddings is False
  assert cfg.preprocess_config.target_fs == DEFAULT_DBETA_SAMPLING_RATE_HZ
  assert cfg.preprocess_config.target_sig_len == DEFAULT_DBETA_SIGNAL_LENGTH
  assert cfg.preprocess_config.target_leads == STANDARD_CLINICAL_12_LEADS


def test_dbeta_preprocess_single_shape_and_leads(
  mock_dbeta_adapter: DbetaAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify waveform is standardized to shape [12, 5000] in standard clinical lead order."""
  sig, leads, fs = synthetic_12lead_signal
  prep = mock_dbeta_adapter.preprocess_single(sig, leads, fs)

  assert isinstance(prep, np.ndarray)
  assert prep.shape == (12, 5000)
  assert prep.dtype == np.float64


# -----------------------------------------------------------------------------
# Unit Tests: Inference (Single and Batch)
# -----------------------------------------------------------------------------


def test_embed_single_success(
  mock_dbeta_adapter: DbetaAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify embed_single produces a valid 768-d embedding."""
  sig, leads, fs = synthetic_12lead_signal
  res = mock_dbeta_adapter.embed_single(sig, leads, fs)

  assert isinstance(res, ModelOutputResult)
  assert res.is_valid is True
  assert res.embedding is not None
  assert res.embedding.shape == (768,)
  assert res.output_dim == 768
  assert res.error_message is None
  assert res.metadata["input_shape"] == (12, 5000)


def test_embed_single_normalized(
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify normalize_embeddings=True produces unit L2 norm."""
  sig, leads, fs = synthetic_12lead_signal
  mock_model = MockDbetaModel()
  cfg = DbetaConfig(normalize_embeddings=True)
  adapter = DbetaAdapter(config=cfg, model=mock_model)

  res = adapter.embed_single(sig, leads, fs)
  assert res.is_valid is True
  assert res.embedding is not None
  norm = float(np.linalg.norm(res.embedding))
  assert pytest.approx(norm, rel=1e-4) == 1.0


def test_embed_single_failure_nan(
  mock_dbeta_adapter: DbetaAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify corrupted/NaN signals return is_valid=False gracefully."""
  sig, leads, fs = synthetic_12lead_signal
  corrupted = sig.copy()
  corrupted[100, 3] = np.nan

  res = mock_dbeta_adapter.embed_single(corrupted, leads, fs)
  assert res.is_valid is False
  assert res.embedding is None
  assert res.error_message is not None
  assert "NaN" in res.error_message


def test_embed_single_failure_inf(
  mock_dbeta_adapter: DbetaAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify infinite signals return is_valid=False gracefully."""
  sig, leads, fs = synthetic_12lead_signal
  corrupted = sig.copy()
  corrupted[200, 1] = np.inf

  res = mock_dbeta_adapter.embed_single(corrupted, leads, fs)
  assert res.is_valid is False
  assert res.embedding is None


def test_embed_single_missing_lead(
  mock_dbeta_adapter: DbetaAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify incomplete leads return is_valid=False."""
  sig, leads, fs = synthetic_12lead_signal
  incomplete_leads = leads[:-1]  # drop last lead
  res = mock_dbeta_adapter.embed_single(sig[:, :-1], incomplete_leads, fs)

  assert res.is_valid is False
  assert res.embedding is None
  assert res.error_message is not None


def test_embed_batch_mixed_records(
  mock_dbeta_adapter: DbetaAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify embed_batch correctly identifies valid and invalid records in batch."""
  sig, leads, fs = synthetic_12lead_signal

  valid_rec = (sig, leads, fs)
  nan_rec = (np.full_like(sig, np.nan), leads, fs)
  empty_rec = (np.empty((0, len(leads))), leads, fs)

  batch = [valid_rec, nan_rec, valid_rec, empty_rec, valid_rec]
  res = mock_dbeta_adapter.embed_batch(batch)

  assert isinstance(res, BatchInferenceResult)
  assert res.count_total == 5
  assert res.count_valid == 3
  assert res.valid_mask == (True, False, True, False, True)
  assert res.embeddings is not None
  assert res.embeddings.shape == (5, 768)

  # Check non-zero embeddings for valid rows and zero for invalid rows
  assert not np.all(res.embeddings[0] == 0.0)
  assert np.all(res.embeddings[1] == 0.0)
  assert not np.all(res.embeddings[2] == 0.0)
  assert np.all(res.embeddings[3] == 0.0)
  assert not np.all(res.embeddings[4] == 0.0)


def test_embed_batch_empty(mock_dbeta_adapter: DbetaAdapter):
  """Verify embed_batch on empty list."""
  res = mock_dbeta_adapter.embed_batch([])
  assert res.count_total == 0
  assert res.count_valid == 0
  assert res.embeddings is not None
  assert res.embeddings.shape == (0, 768)


def test_determinism(
  mock_dbeta_adapter: DbetaAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify identical input signals yield bitwise identical embeddings."""
  sig, leads, fs = synthetic_12lead_signal

  res1 = mock_dbeta_adapter.embed_single(sig, leads, fs)
  res2 = mock_dbeta_adapter.embed_single(sig, leads, fs)

  assert res1.is_valid and res2.is_valid
  assert res1.embedding is not None and res2.embedding is not None
  np.testing.assert_allclose(res1.embedding, res2.embedding, rtol=1e-10, atol=1e-10)


def test_batch_chunking(
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify batch inference works accurately across multiple chunks."""
  sig, leads, fs = synthetic_12lead_signal
  mock_model = MockDbetaModel()
  cfg = DbetaConfig(batch_size=2)
  adapter = DbetaAdapter(config=cfg, model=mock_model)

  batch = [(sig * (i + 1), leads, fs) for i in range(5)]
  res = adapter.embed_batch(batch)

  assert res.count_total == 5
  assert res.count_valid == 5
  assert res.embeddings is not None
  assert res.embeddings.shape == (5, 768)

  # Check individual vs batched consistency
  for i in range(5):
    single_res = adapter.embed_single(sig * (i + 1), leads, fs)
    assert single_res.embedding is not None
    np.testing.assert_allclose(res.embeddings[i], single_res.embedding, rtol=1e-6, atol=1e-6)


# -----------------------------------------------------------------------------
# Unit Tests: Model Loading & Error Diagnostics
# -----------------------------------------------------------------------------


def test_load_dbeta_model_gated_error():
  """Verify load_dbeta_model raises clear PermissionError when HF access is gated."""
  with patch("transformers.AutoModel.from_pretrained") as mock_from_pretrained:
    mock_from_pretrained.side_effect = Exception("403 Client Error: Cannot access gated repo")

    with pytest.raises(PermissionError) as exc_info:
      load_dbeta_model("Manhph2211/D-BETA")

    assert "CC BY-NC 4.0" in str(exc_info.value)
    assert "local_checkpoint_path" in str(exc_info.value)


def test_load_dbeta_model_local_path():
  """Verify load_dbeta_model with local_checkpoint_path invokes AutoModel properly."""
  mock_model = MockDbetaModel()
  with patch("transformers.AutoModel.from_pretrained", return_value=mock_model) as mock_from_pretrained:
    loaded = load_dbeta_model(local_checkpoint_path="/mock/path/to/dbeta")
    mock_from_pretrained.assert_called_once_with("/mock/path/to/dbeta", trust_remote_code=True)
    assert loaded is not None


# -----------------------------------------------------------------------------
# Integration Test: Real Local MIMIC-IV-ECG Data
# -----------------------------------------------------------------------------


def test_real_mimic_waveform_inference_with_mock_dbeta():
  """Smoke test loading real MIMIC-IV-ECG waveforms and generating representations."""
  paths = DataPaths()
  if not paths.record_list_path.exists():
    pytest.skip("Local MIMIC-IV-ECG record_list.csv not present.")

  records = load_record_list(paths.record_list_path)
  sample_rows = records.head(5).to_dicts()

  batch_records: list[tuple[npt.NDArray[np.float64], list[str], int]] = []
  for row in sample_rows:
    wfdb_path = paths.mimic_iv_ecg_dir / row["path"]
    sig, leads, fs = read_ecg_waveform(wfdb_path)
    batch_records.append((sig, leads, fs))

  mock_model = MockDbetaModel()
  adapter = DbetaAdapter(config=DbetaConfig(batch_size=2), model=mock_model)
  res = adapter.embed_batch(batch_records)

  assert res.count_total == 5
  assert res.count_valid == 5
  assert res.embeddings is not None
  assert res.embeddings.shape == (5, 768)
  assert np.isfinite(res.embeddings).all()
