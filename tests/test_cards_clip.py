"""Tests for CarDSLab ECG-CLIP adapter, preprocessing, and inference pipeline (Stage 4)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from PIL import Image
import pytest
import torch

from ecg_alignment.data import CANONICAL_12_LEADS, DEFAULT_SAMPLING_RATE_HZ, DataPaths, load_record_list, read_ecg_waveform
from ecg_alignment.scoring.base import (
  BatchInferenceResult,
  ImageRenderConfig,
  InputModality,
  ModelOutputResult,
)
from ecg_alignment.scoring.cards_clip import (
  DEFAULT_CARDS_CLIP_EMBEDDING_DIM,
  DEFAULT_CARDS_CLIP_IMAGE_SIZE,
  DEFAULT_CARDS_CLIP_MODEL_NAME,
  DEFAULT_CARDS_CLIP_REVISION,
  CardsClipAdapter,
  CardsClipConfig,
  compute_centroid_similarity_score,
  cosine_similarity_1d,
)


# -----------------------------------------------------------------------------
# Fixtures and Mock Models
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
    sig[:, i] = np.sin(2 * np.pi * (1.0 + 0.3 * i) * t) * 0.5

  return sig, list(CANONICAL_12_LEADS), fs


class MockClipOutput:
  """Mock output for CLIPModel.get_image_features."""

  def __init__(self, tensor: torch.Tensor) -> None:
    self.pooler_output = tensor
    self.image_embeds = tensor


class MockClipModel(torch.nn.Module):
  """Deterministic mock CLIP model for fast, offline unit testing."""

  def __init__(self, embedding_dim: int = DEFAULT_CARDS_CLIP_EMBEDDING_DIM) -> None:
    super().__init__()
    self.embedding_dim = embedding_dim
    # Linear projection to generate reproducible embeddings from pixel mean
    self.proj = torch.nn.Linear(3, embedding_dim, bias=False)
    with torch.no_grad():
      torch.nn.init.eye_(self.proj.weight[:3, :3])

  def get_image_features(self, pixel_values: torch.Tensor) -> MockClipOutput:
    # pixel_values shape: [batch, 3, 384, 384]
    batch_size = pixel_values.shape[0]
    mean_rgb = pixel_values.mean(dim=(2, 3))  # [batch, 3]
    embed = self.proj(mean_rgb)  # [batch, 512]
    return MockClipOutput(embed)

  def eval(self) -> "MockClipModel":
    return self


class MockClipProcessor:
  """Deterministic mock processor mapping PIL Images to PyTorch tensors."""

  def __call__(
    self,
    images: Image.Image | list[Image.Image],
    return_tensors: str = "pt",
  ) -> dict[str, torch.Tensor]:
    if isinstance(images, Image.Image):
      img_list = [images]
    else:
      img_list = images

    tensors: list[torch.Tensor] = []
    for img in img_list:
      arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3]
      arr_ch = np.transpose(arr, (2, 0, 1))  # [3, H, W]
      tensors.append(torch.from_numpy(arr_ch))

    batch_tensor = torch.stack(tensors, dim=0)
    return {"pixel_values": batch_tensor}


@pytest.fixture
def mock_adapter() -> CardsClipAdapter:
  """Instantiate a CardsClipAdapter with deterministic mock modules."""
  mock_model = MockClipModel(embedding_dim=DEFAULT_CARDS_CLIP_EMBEDDING_DIM)
  mock_proc = MockClipProcessor()
  config = CardsClipConfig(batch_size=4, device="cpu", normalize_embeddings=True)
  return CardsClipAdapter(config=config, model=mock_model, processor=mock_proc)


# -----------------------------------------------------------------------------
# Unit Tests: Config and Mathematical Helpers
# -----------------------------------------------------------------------------


def test_cards_clip_config_defaults():
  """Verify default configuration parameters for CarDSLab ECG-CLIP."""
  cfg = CardsClipConfig()
  assert cfg.model_name == DEFAULT_CARDS_CLIP_MODEL_NAME
  assert cfg.model_version == DEFAULT_CARDS_CLIP_REVISION
  assert cfg.input_modality == InputModality.IMAGE
  assert cfg.embedding_dim == 512
  assert cfg.normalize_embeddings is True
  assert cfg.image_config.target_pixel_size == (384, 384)
  assert cfg.image_config.layout == "standard_3x4"


def test_cosine_similarity_1d():
  """Verify cosine similarity calculation across canonical geometric angles."""
  v1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  v2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  v3 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
  v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
  zero = np.array([0.0, 0.0, 0.0], dtype=np.float64)

  assert pytest.approx(cosine_similarity_1d(v1, v2), rel=1e-5) == 1.0
  assert pytest.approx(cosine_similarity_1d(v1, v3), abs=1e-7) == 0.0
  assert pytest.approx(cosine_similarity_1d(v1, v4), rel=1e-5) == -1.0
  assert cosine_similarity_1d(v1, zero) == 0.0


def test_compute_centroid_similarity_score():
  """Verify binary softmax scoring from case and control reference centroids."""
  case_cent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  ctrl_cent = np.array([0.0, 1.0, 0.0], dtype=np.float64)

  # Exact match with case
  score_case = compute_centroid_similarity_score(case_cent, case_cent, ctrl_cent, temperature=1.0)
  assert score_case > 0.7

  # Exact match with control
  score_ctrl = compute_centroid_similarity_score(ctrl_cent, case_cent, ctrl_cent, temperature=1.0)
  assert score_ctrl < 0.3

  # Equidistant vector
  equi = np.array([1.0, 1.0, 0.0], dtype=np.float64)
  score_equi = compute_centroid_similarity_score(equi, case_cent, ctrl_cent, temperature=1.0)
  assert pytest.approx(score_equi, abs=1e-5) == 0.5


# -----------------------------------------------------------------------------
# Unit Tests: Preprocessing and Image Rendering
# -----------------------------------------------------------------------------


def test_preprocess_single_renders_image(
  mock_adapter: CardsClipAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify single ECG preprocessing renders a standardized 384x384 image."""
  sig, leads, fs = synthetic_12lead_signal
  img = mock_adapter.preprocess_single(sig, leads, fs)

  assert isinstance(img, Image.Image)
  assert img.size == (384, 384)
  assert img.mode == "RGB"


# -----------------------------------------------------------------------------
# Unit Tests: Adapter Inference (Single & Batch)
# -----------------------------------------------------------------------------


def test_embed_single_success(
  mock_adapter: CardsClipAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify embed_single produces a valid 512-d normalized embedding."""
  sig, leads, fs = synthetic_12lead_signal
  res = mock_adapter.embed_single(sig, leads, fs)

  assert isinstance(res, ModelOutputResult)
  assert res.is_valid is True
  assert res.embedding is not None
  assert res.embedding.shape == (512,)
  assert res.output_dim == 512
  assert res.error_message is None

  # Verify L2 normalization
  norm = float(np.linalg.norm(res.embedding))
  assert pytest.approx(norm, rel=1e-4) == 1.0


def test_embed_single_failure_nan(
  mock_adapter: CardsClipAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify embed_single gracefully returns invalid result on corrupted/NaN signals."""
  sig, leads, fs = synthetic_12lead_signal
  corrupted = sig.copy()
  corrupted[100, 2] = np.nan

  res = mock_adapter.embed_single(corrupted, leads, fs)
  assert res.is_valid is False
  assert res.embedding is None
  assert res.error_message is not None
  assert "NaN" in res.error_message


def test_embed_single_missing_lead(
  mock_adapter: CardsClipAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify embed_single handles missing required leads cleanly."""
  sig, leads, fs = synthetic_12lead_signal
  incomplete_leads = leads[:-1]  # missing V6
  res = mock_adapter.embed_single(sig[:, :-1], incomplete_leads, fs)

  assert res.is_valid is False
  assert res.embedding is None
  assert res.error_message is not None


def test_embed_batch_mixed_records(
  mock_adapter: CardsClipAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify embed_batch handles combinations of valid and corrupted records."""
  sig, leads, fs = synthetic_12lead_signal

  valid_rec = (sig, leads, fs)
  nan_rec = (np.full_like(sig, np.nan), leads, fs)
  empty_rec = (np.empty((0, len(leads))), leads, fs)

  batch = [valid_rec, nan_rec, valid_rec, empty_rec]
  res = mock_adapter.embed_batch(batch)

  assert isinstance(res, BatchInferenceResult)
  assert res.count_total == 4
  assert res.count_valid == 2
  assert res.valid_mask == (True, False, True, False)
  assert res.embeddings is not None
  assert res.embeddings.shape == (4, 512)

  # Check valid entries have unit norm
  assert pytest.approx(float(np.linalg.norm(res.embeddings[0])), rel=1e-4) == 1.0
  assert pytest.approx(float(np.linalg.norm(res.embeddings[2])), rel=1e-4) == 1.0

  # Check invalid entries are zeroed out
  assert np.all(res.embeddings[1] == 0.0)
  assert np.all(res.embeddings[3] == 0.0)


def test_embed_batch_empty(mock_adapter: CardsClipAdapter):
  """Verify embed_batch on empty input sequence."""
  res = mock_adapter.embed_batch([])
  assert res.count_total == 0
  assert res.count_valid == 0
  assert res.embeddings is not None
  assert res.embeddings.shape == (0, 512)


def test_determinism(
  mock_adapter: CardsClipAdapter,
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Verify identical input signals produce identical embeddings."""
  sig, leads, fs = synthetic_12lead_signal

  res1 = mock_adapter.embed_single(sig, leads, fs)
  res2 = mock_adapter.embed_single(sig, leads, fs)

  assert res1.is_valid and res2.is_valid
  assert res1.embedding is not None and res2.embedding is not None
  np.testing.assert_allclose(res1.embedding, res2.embedding, rtol=1e-7, atol=1e-7)


# -----------------------------------------------------------------------------
# Integration Test with Real Pretrained Weights
# -----------------------------------------------------------------------------


def test_real_cards_clip_pretrained_inference(
  synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int],
):
  """Integration test validating inference with real CarDSLab pretrained model when accessible."""
  sig, leads, fs = synthetic_12lead_signal

  adapter = CardsClipAdapter()
  assert adapter.config.model_name == DEFAULT_CARDS_CLIP_MODEL_NAME

  try:
    res = adapter.embed_single(sig, leads, fs)
  except Exception as exc:
    err_str = str(exc).lower()
    if any(k in err_str for k in ("401", "403", "gated", "restricted", "access", "unauthorized")):
      pytest.skip(f"CarDSLab pretrained model not authenticated in test environment: {exc}")
    raise

  if not res.is_valid and res.error_message is not None:
    err_str = res.error_message.lower()
    if any(k in err_str for k in ("401", "403", "gated", "restricted", "access", "unauthorized")):
      pytest.skip(f"CarDSLab pretrained model not authenticated in test environment: {res.error_message}")

  assert res.is_valid is True
  assert res.embedding is not None
  assert res.embedding.shape == (512,)
  assert np.isfinite(res.embedding).all()

  # Check unit norm
  norm = float(np.linalg.norm(res.embedding))
  assert pytest.approx(norm, rel=1e-3) == 1.0
