"""Common adapter interfaces and immutable configurations for multimodal transformer ECG models (Model B).

Authoritative references:
- Pham Hung M, Saeed A, Ma D. Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners.
  ICML 2025; PMLR 267:49277-49291.
- Oikonomou EK, et al. TARGET-AI: a foundational approach for the targeted deployment of artificial
  intelligence electrocardiography in the electronic health record. medRxiv. 2025.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np
import numpy.typing as npt

from ecg_alignment.data import CANONICAL_12_LEADS, DEFAULT_SAMPLING_RATE_HZ, DEFAULT_SIGNAL_LENGTH_SAMPLES

# Standard clinical 12-lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
STANDARD_CLINICAL_12_LEADS: tuple[str, ...] = (
  "I",
  "II",
  "III",
  "aVR",
  "aVL",
  "aVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
)

# Cabrera 12-lead order (anatomically sequential frontal leads followed by precordial leads):
# aVL, I, -aVR, II, aVF, III, V1-V6
CABRERA_12_LEADS: tuple[str, ...] = (
  "aVL",
  "I",
  "-aVR",
  "II",
  "aVF",
  "III",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
)


class InputModality(str, Enum):
  """Supported input modality for ECG transformer models."""

  WAVEFORM = "waveform"
  IMAGE = "image"


class WaveformPadMode(str, Enum):
  """Padding mode when signal length is shorter than target length."""

  CONSTANT = "constant"
  EDGE = "edge"


class CropAlign(str, Enum):
  """Alignment for cropping when signal is longer than target length."""

  START = "start"
  CENTER = "center"
  END = "end"


@dataclass(frozen=True)
class WaveformPreprocessConfig:
  """Immutable configuration for 12-lead ECG waveform preprocessing."""

  target_fs: int = DEFAULT_SAMPLING_RATE_HZ
  target_sig_len: int = DEFAULT_SIGNAL_LENGTH_SAMPLES
  target_leads: tuple[str, ...] = CANONICAL_12_LEADS
  target_unit: str = "mV"  # Standard units: "mV", "uV", "V"
  filter_baseline: bool = False
  pad_mode: WaveformPadMode = WaveformPadMode.CONSTANT
  crop_align: CropAlign = CropAlign.START
  fill_value: float = 0.0


@dataclass(frozen=True)
class ImageRenderConfig:
  """Immutable configuration for rendering 12-lead ECG waveforms into standardized images."""

  layout: str = "standard_3x4"  # "standard_3x4", "stacked_12", "grid_4x3"
  dpi: int = 100
  figsize_inches: tuple[float, float] = (10.0, 6.0)
  target_pixel_size: tuple[int, int] | None = (384, 384)  # (width, height) in pixels
  line_width: float = 1.0
  line_color: str = "#000000"
  background_color: str = "#FFFFFF"
  grid_color_major: str = "#FFB6C1"  # Light pink standard ECG grid
  grid_color_minor: str = "#FFE4E1"
  show_grid: bool = True
  show_labels: bool = True
  sample_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ
  gain_mm_per_mv: float = 10.0
  speed_mm_per_sec: float = 25.0


@dataclass(frozen=True)
class TransformerAdapterConfig:
  """Immutable configuration for an ECG transformer model adapter."""

  model_name: str
  model_version: str | None = None
  input_modality: InputModality = InputModality.WAVEFORM
  embedding_dim: int = 768
  preprocess_config: WaveformPreprocessConfig = field(default_factory=WaveformPreprocessConfig)
  image_config: ImageRenderConfig | None = None
  batch_size: int = 32
  device: str = "cpu"


@dataclass(frozen=True)
class ModelOutputResult:
  """Standardized output result for a single ECG model inference call."""

  is_valid: bool
  embedding: npt.NDArray[np.float64] | None = None
  score: float | None = None
  model_name: str = ""
  model_version: str | None = None
  output_dim: int | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  error_message: str | None = None


@dataclass(frozen=True)
class BatchInferenceResult:
  """Standardized batch inference result for multiple ECGs."""

  embeddings: npt.NDArray[np.float64] | None = None  # Shape [N, D]
  scores: npt.NDArray[np.float64] | None = None  # Shape [N]
  valid_mask: tuple[bool, ...] = ()
  model_name: str = ""
  failure_reasons: tuple[str | None, ...] = ()
  count_total: int = 0
  count_valid: int = 0


class BaseEcgModelAdapter(ABC):
  """Abstract base class for model adapters producing standardized Model-B embeddings/scores."""

  @property
  @abstractmethod
  def config(self) -> TransformerAdapterConfig:
    """Return the immutable adapter configuration."""
    ...

  @abstractmethod
  def preprocess_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = DEFAULT_SAMPLING_RATE_HZ,
  ) -> Any:
    """Preprocess a single raw ECG record into the model's expected input format."""
    ...

  @abstractmethod
  def embed_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = DEFAULT_SAMPLING_RATE_HZ,
  ) -> ModelOutputResult:
    """Compute frozen embedding for a single raw ECG recording."""
    ...

  @abstractmethod
  def embed_batch(
    self,
    batch_records: Sequence[tuple[npt.NDArray[np.float64], Sequence[str], int]],
  ) -> BatchInferenceResult:
    """Compute frozen embeddings for a batch of raw ECG recordings."""
    ...
