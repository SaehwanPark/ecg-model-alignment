"""Tests for ECG preprocessing, canonical lead validation, resampling, and image rendering."""

from collections.abc import Sequence
from pathlib import Path
import numpy as np
import numpy.typing as npt
from PIL import Image
import pytest

from ecg_alignment.data import CANONICAL_12_LEADS, DEFAULT_SAMPLING_RATE_HZ, DataPaths, load_record_list, read_ecg_waveform
from ecg_alignment.scoring.base import (
  CABRERA_12_LEADS,
  STANDARD_CLINICAL_12_LEADS,
  BaseEcgModelAdapter,
  BatchInferenceResult,
  CropAlign,
  ImageRenderConfig,
  InputModality,
  ModelOutputResult,
  TransformerAdapterConfig,
  WaveformPadMode,
  WaveformPreprocessConfig,
)
from ecg_alignment.scoring.preprocess import (
  pad_or_crop_waveform,
  preprocess_waveform,
  render_12lead_ecg_image,
  resample_waveform,
  scale_waveform_units,
  validate_and_reorder_leads,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def synthetic_12lead_signal() -> tuple[npt.NDArray[np.float64], list[str], int]:
  """Generate a synthetic 10-second 12-lead signal at 500 Hz."""
  fs = 500
  n_samples = 5000
  n_leads = len(CANONICAL_12_LEADS)
  t = np.linspace(0, 10, n_samples, endpoint=False)

  # Generate distinct signals for each lead to test channel preservation
  sig = np.zeros((n_samples, n_leads), dtype=np.float64)
  for i in range(n_leads):
    sig[:, i] = np.sin(2 * np.pi * (1.0 + 0.5 * i) * t) * (1.0 + 0.1 * i)

  return sig, list(CANONICAL_12_LEADS), fs


# -----------------------------------------------------------------------------
# Unit Tests: Lead Ordering and Validation
# -----------------------------------------------------------------------------


def test_validate_and_reorder_leads_identity(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, leads, _ = synthetic_12lead_signal
  out = validate_and_reorder_leads(sig, leads, CANONICAL_12_LEADS)
  assert out.shape == sig.shape
  assert np.array_equal(out, sig)


def test_validate_and_reorder_leads_permuted(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, leads, _ = synthetic_12lead_signal
  # Permute leads
  perm = [5, 2, 0, 1, 3, 4, 11, 10, 9, 8, 7, 6]
  perm_leads = [leads[i] for i in perm]
  perm_sig = sig[:, perm]

  reordered = validate_and_reorder_leads(perm_sig, perm_leads, CANONICAL_12_LEADS)
  assert reordered.shape == sig.shape
  assert np.allclose(reordered, sig)


def test_validate_and_reorder_leads_cabrera(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, leads, _ = synthetic_12lead_signal
  cabrera = validate_and_reorder_leads(sig, leads, CABRERA_12_LEADS)
  assert cabrera.shape == (5000, 12)

  # Check that lead -aVR is inverted aVR
  avr_idx = leads.index("aVR")
  minus_avr_idx = CABRERA_12_LEADS.index("-aVR")
  assert np.allclose(cabrera[:, minus_avr_idx], -1.0 * sig[:, avr_idx])


def test_validate_and_reorder_leads_missing_lead():
  sig = np.zeros((1000, 3))
  leads = ["I", "II", "III"]
  with pytest.raises(ValueError, match="Missing required lead"):
    validate_and_reorder_leads(sig, leads, CANONICAL_12_LEADS)


def test_validate_and_reorder_leads_duplicate_leads():
  sig = np.zeros((1000, 4))
  leads = ["I", "II", "II", "V1"]
  with pytest.raises(ValueError, match="Duplicate lead names"):
    validate_and_reorder_leads(sig, leads, ("I", "II"))


def test_validate_and_reorder_leads_dimension_mismatch():
  sig = np.zeros((1000, 3))
  leads = ["I", "II"]
  with pytest.raises(ValueError, match="Number of columns"):
    validate_and_reorder_leads(sig, leads, ("I", "II"))


# -----------------------------------------------------------------------------
# Unit Tests: Waveform Resampling
# -----------------------------------------------------------------------------


def test_resample_waveform_same_frequency(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, _, fs = synthetic_12lead_signal
  res = resample_waveform(sig, original_fs=fs, target_fs=fs)
  assert res.shape == sig.shape
  assert np.array_equal(res, sig)


def test_resample_waveform_downsampling(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, _, fs = synthetic_12lead_signal
  # 500 Hz (5000 samples) -> 250 Hz (2500 samples)
  res = resample_waveform(sig, original_fs=fs, target_fs=250)
  assert res.shape == (2500, 12)

  # 500 Hz -> 100 Hz (1000 samples)
  res100 = resample_waveform(sig, original_fs=fs, target_fs=100)
  assert res100.shape == (1000, 12)


def test_resample_waveform_upsampling(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, _, _ = synthetic_12lead_signal
  sub_sig = sig[:1000, :]  # 1000 samples at 100 Hz (10 seconds)
  res = resample_waveform(sub_sig, original_fs=100, target_fs=500)
  assert res.shape == (5000, 12)


def test_resample_waveform_invalid_rates():
  sig = np.zeros((100, 2))
  with pytest.raises(ValueError, match="Sampling rates must be positive"):
    resample_waveform(sig, original_fs=0, target_fs=500)
  with pytest.raises(ValueError, match="Sampling rates must be positive"):
    resample_waveform(sig, original_fs=500, target_fs=-10)


# -----------------------------------------------------------------------------
# Unit Tests: Padding and Cropping
# -----------------------------------------------------------------------------


def test_pad_or_crop_waveform_exact_length():
  sig = np.ones((5000, 12))
  out = pad_or_crop_waveform(sig, target_length=5000)
  assert out.shape == (5000, 12)
  assert np.array_equal(out, sig)


def test_pad_or_crop_waveform_padding_constant():
  sig = np.ones((3000, 12))
  out = pad_or_crop_waveform(sig, target_length=5000, pad_mode=WaveformPadMode.CONSTANT, fill_value=0.0)
  assert out.shape == (5000, 12)
  assert np.all(out[:3000, :] == 1.0)
  assert np.all(out[3000:, :] == 0.0)


def test_pad_or_crop_waveform_padding_edge():
  sig = np.arange(10, dtype=np.float64).reshape(10, 1)
  out = pad_or_crop_waveform(sig, target_length=15, pad_mode=WaveformPadMode.EDGE)
  assert out.shape == (15, 1)
  assert out[9, 0] == 9.0
  assert np.all(out[9:, 0] == 9.0)


def test_pad_or_crop_waveform_cropping():
  sig = np.arange(100, dtype=np.float64).reshape(100, 1)

  # Crop START (keep first 60)
  out_start = pad_or_crop_waveform(sig, target_length=60, crop_align=CropAlign.START)
  assert out_start.shape == (60, 1)
  assert out_start[0, 0] == 0.0
  assert out_start[-1, 0] == 59.0

  # Crop END (keep last 60)
  out_end = pad_or_crop_waveform(sig, target_length=60, crop_align=CropAlign.END)
  assert out_end.shape == (60, 1)
  assert out_end[0, 0] == 40.0
  assert out_end[-1, 0] == 99.0

  # Crop CENTER (center 60: 20 to 80)
  out_center = pad_or_crop_waveform(sig, target_length=60, crop_align=CropAlign.CENTER)
  assert out_center.shape == (60, 1)
  assert out_center[0, 0] == 20.0
  assert out_center[-1, 0] == 79.0


# -----------------------------------------------------------------------------
# Unit Tests: Unit Conversion
# -----------------------------------------------------------------------------


def test_scale_waveform_units():
  sig_mv = np.array([[1.5, -0.5]], dtype=np.float64)

  # mV -> uV (1 mV = 1000 uV)
  sig_uv = scale_waveform_units(sig_mv, source_unit="mV", target_unit="uV")
  assert np.allclose(sig_uv, [[1500.0, -500.0]])

  # uV -> mV
  sig_mv_back = scale_waveform_units(sig_uv, source_unit="uV", target_unit="mV")
  assert np.allclose(sig_mv_back, sig_mv)

  # mV -> V (1 mV = 0.001 V)
  sig_v = scale_waveform_units(sig_mv, source_unit="mV", target_unit="V")
  assert np.allclose(sig_v, [[0.0015, -0.0005]])

  # Invalid unit raises
  with pytest.raises(ValueError, match="Unknown source unit"):
    scale_waveform_units(sig_mv, source_unit="invalid")


# -----------------------------------------------------------------------------
# Integration Test: Full Preprocess Pipeline & Determinism
# -----------------------------------------------------------------------------


def test_preprocess_waveform_pipeline(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, leads, fs = synthetic_12lead_signal

  config = WaveformPreprocessConfig(
    target_fs=250,
    target_sig_len=2500,
    target_leads=STANDARD_CLINICAL_12_LEADS,
    target_unit="uV",
    filter_baseline=True,
  )

  proc1 = preprocess_waveform(sig, leads, fs=fs, config=config, source_unit="mV")
  assert proc1.shape == (2500, 12)

  # Test strict determinism: identical inputs must yield bitwise identical output
  proc2 = preprocess_waveform(sig, leads, fs=fs, config=config, source_unit="mV")
  assert np.array_equal(proc1, proc2)


# -----------------------------------------------------------------------------
# Unit & Determinism Tests: ECG Image Rendering
# -----------------------------------------------------------------------------


def test_render_12lead_ecg_image_standard_3x4(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, leads, fs = synthetic_12lead_signal

  config = ImageRenderConfig(
    layout="standard_3x4",
    target_pixel_size=(384, 384),
    dpi=100,
  )

  img1 = render_12lead_ecg_image(sig, leads, fs=fs, config=config)
  assert isinstance(img1, Image.Image)
  assert img1.size == (384, 384)
  assert img1.mode == "RGB"

  # Strict image determinism check
  img2 = render_12lead_ecg_image(sig, leads, fs=fs, config=config)
  arr1 = np.array(img1)
  arr2 = np.array(img2)
  assert np.array_equal(arr1, arr2)


def test_render_12lead_ecg_image_stacked_12(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, leads, fs = synthetic_12lead_signal

  config = ImageRenderConfig(
    layout="stacked_12",
    target_pixel_size=(224, 224),
    dpi=100,
  )

  img = render_12lead_ecg_image(sig, leads, fs=fs, config=config)
  assert img.size == (224, 224)
  assert img.mode == "RGB"


def test_render_12lead_ecg_image_invalid_layout(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, leads, fs = synthetic_12lead_signal
  config = ImageRenderConfig(layout="invalid_layout")
  with pytest.raises(ValueError, match="Unsupported image layout"):
    render_12lead_ecg_image(sig, leads, fs=fs, config=config)


# -----------------------------------------------------------------------------
# Base Adapter Protocol and Mock Adapter Test
# -----------------------------------------------------------------------------


class MockEcgModelAdapter(BaseEcgModelAdapter):
  """Deterministic mock adapter for testing the Model-B interface contract."""

  def __init__(self, embedding_dim: int = 768):
    self._config = TransformerAdapterConfig(
      model_name="mock-transformer-base",
      model_version="v1.0",
      input_modality=InputModality.WAVEFORM,
      embedding_dim=embedding_dim,
      preprocess_config=WaveformPreprocessConfig(target_fs=500, target_sig_len=5000),
    )

  @property
  def config(self) -> TransformerAdapterConfig:
    return self._config

  def preprocess_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = DEFAULT_SAMPLING_RATE_HZ,
  ) -> npt.NDArray[np.float64]:
    return preprocess_waveform(signal_array, lead_names, fs=fs, config=self.config.preprocess_config)

  def embed_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = DEFAULT_SAMPLING_RATE_HZ,
  ) -> ModelOutputResult:
    try:
      proc = self.preprocess_single(signal_array, lead_names, fs)
      # Deterministic mock projection: compute channel-wise summary vector
      mean_lead = np.mean(proc, axis=0)  # 12-d
      # Tile or project to embedding_dim
      reps = int(np.ceil(self.config.embedding_dim / len(mean_lead)))
      tiled = np.tile(mean_lead, reps)[: self.config.embedding_dim]
      embedding = tiled / (np.linalg.norm(tiled) + 1e-8)

      return ModelOutputResult(
        is_valid=True,
        embedding=embedding,
        score=float(np.mean(embedding)),
        model_name=self.config.model_name,
        model_version=self.config.model_version,
        output_dim=self.config.embedding_dim,
        metadata={"target_sig_len": self.config.preprocess_config.target_sig_len},
      )
    except Exception as exc:
      return ModelOutputResult(
        is_valid=False,
        model_name=self.config.model_name,
        model_version=self.config.model_version,
        error_message=str(exc),
      )

  def embed_batch(
    self,
    batch_records: Sequence[tuple[npt.NDArray[np.float64], Sequence[str], int]],
  ) -> BatchInferenceResult:
    results: list[ModelOutputResult] = [
      self.embed_single(sig, leads, fs) for sig, leads, fs in batch_records
    ]

    valid_mask = tuple(r.is_valid for r in results)
    failure_reasons = tuple(r.error_message for r in results)
    count_total = len(results)
    count_valid = sum(1 for v in valid_mask if v)

    if count_valid > 0:
      emb_dim = self.config.embedding_dim
      emb_matrix = np.zeros((count_total, emb_dim), dtype=np.float64)
      scores = np.zeros(count_total, dtype=np.float64)
      for i, r in enumerate(results):
        if r.is_valid and r.embedding is not None:
          emb_matrix[i, :] = r.embedding
          scores[i] = r.score if r.score is not None else 0.0
    else:
      emb_matrix = None
      scores = None

    return BatchInferenceResult(
      embeddings=emb_matrix,
      scores=scores,
      valid_mask=valid_mask,
      model_name=self.config.model_name,
      failure_reasons=failure_reasons,
      count_total=count_total,
      count_valid=count_valid,
    )


def test_mock_adapter_inference(synthetic_12lead_signal: tuple[npt.NDArray[np.float64], list[str], int]):
  sig, leads, fs = synthetic_12lead_signal
  adapter = MockEcgModelAdapter(embedding_dim=768)

  # Single inference
  result = adapter.embed_single(sig, leads, fs)
  assert result.is_valid is True
  assert result.embedding is not None
  assert result.embedding.shape == (768,)
  assert result.output_dim == 768
  assert result.score is not None

  # Batch inference
  bad_sig = np.zeros((100, 2))  # Missing leads
  batch = [(sig, leads, fs), (bad_sig, ["I", "II"], fs)]
  batch_res = adapter.embed_batch(batch)

  assert batch_res.count_total == 2
  assert batch_res.count_valid == 1
  assert batch_res.valid_mask == (True, False)
  assert batch_res.embeddings is not None
  assert batch_res.embeddings.shape == (2, 768)
  assert batch_res.failure_reasons[1] is not None
  assert "Missing required lead" in str(batch_res.failure_reasons[1])


# -----------------------------------------------------------------------------
# Local MIMIC Data Smoke Test (if data available)
# -----------------------------------------------------------------------------


def test_mimic_local_sample_preprocessing():
  paths = DataPaths()
  if not paths.record_list_path.exists():
    pytest.skip("Local MIMIC-IV-ECG record_list.csv not present.")

  records_df = load_record_list(paths.record_list_path)
  sample_rows = records_df.head(5).to_dicts()

  for row in sample_rows:
    wfdb_path = paths.mimic_iv_ecg_dir / str(row["path"])
    if not wfdb_path.with_suffix(".hea").exists():
      continue

    signal_array, lead_names, fs = read_ecg_waveform(wfdb_path)

    # 1. Waveform preprocessing
    config = WaveformPreprocessConfig(target_fs=500, target_sig_len=5000)
    proc_sig = preprocess_waveform(signal_array, lead_names, fs, config)
    assert proc_sig.shape == (5000, 12)

    # 2. Image rendering
    img_config = ImageRenderConfig(layout="standard_3x4", target_pixel_size=(384, 384))
    img = render_12lead_ecg_image(signal_array, lead_names, fs, img_config)
    assert img.size == (384, 384)
    assert img.mode == "RGB"
