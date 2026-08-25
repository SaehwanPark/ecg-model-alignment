"""CarDSLab ECG-CLIP engineering prototype adapter for multimodal image-based ECG representation.

Model reference:
- Model repository: CarDSLab/ecg-clip-beit-base-384
- Base architecture: BEiT vision transformer (microsoft/beit-base-patch16-384) + custom text projection
- Target dimension: 512-dimensional normalized embedding
- Manuscript: TARGET-AI (Oikonomou EK, et al. medRxiv 2025)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from typing import Any, cast
import numpy as np
import numpy.typing as npt
from PIL import Image

from ecg_alignment.data import CANONICAL_12_LEADS, DEFAULT_SAMPLING_RATE_HZ
from ecg_alignment.scoring.base import (
  BaseEcgModelAdapter,
  BatchInferenceResult,
  ImageRenderConfig,
  InputModality,
  ModelOutputResult,
  TransformerAdapterConfig,
  WaveformPreprocessConfig,
)
from ecg_alignment.scoring.preprocess import render_12lead_ecg_image

logger = logging.getLogger(__name__)

DEFAULT_CARDS_CLIP_MODEL_NAME: str = "CarDSLab/ecg-clip-beit-base-384"
DEFAULT_CARDS_CLIP_REVISION: str = "80131ef06310dd1c8f2efe9082709c82433dc66e"
DEFAULT_CARDS_CLIP_EMBEDDING_DIM: int = 512
DEFAULT_CARDS_CLIP_IMAGE_SIZE: tuple[int, int] = (384, 384)


@dataclass(frozen=True)
class CardsClipConfig(TransformerAdapterConfig):
  """Immutable configuration for CarDSLab ECG-CLIP image adapter."""

  model_name: str = DEFAULT_CARDS_CLIP_MODEL_NAME
  model_version: str | None = DEFAULT_CARDS_CLIP_REVISION
  input_modality: InputModality = InputModality.IMAGE
  embedding_dim: int = DEFAULT_CARDS_CLIP_EMBEDDING_DIM
  preprocess_config: WaveformPreprocessConfig = field(
    default_factory=lambda: WaveformPreprocessConfig(
      target_fs=DEFAULT_SAMPLING_RATE_HZ,
      target_leads=CANONICAL_12_LEADS,
    )
  )
  image_config: ImageRenderConfig = field(
    default_factory=lambda: ImageRenderConfig(
      layout="standard_3x4",
      target_pixel_size=DEFAULT_CARDS_CLIP_IMAGE_SIZE,
      show_labels=True,
      show_grid=True,
    )
  )
  normalize_embeddings: bool = True
  batch_size: int = 32
  device: str = "cpu"


def cosine_similarity_1d(
  vec_a: npt.NDArray[np.float64],
  vec_b: npt.NDArray[np.float64],
  eps: float = 1e-12,
) -> float:
  """Compute cosine similarity between two 1D vectors."""
  norm_a = float(np.linalg.norm(vec_a))
  norm_b = float(np.linalg.norm(vec_b))
  if norm_a < eps or norm_b < eps:
    return 0.0
  dot_val = float(np.dot(vec_a, vec_b))
  return dot_val / (norm_a * norm_b)


def compute_centroid_similarity_score(
  embedding: npt.NDArray[np.float64],
  case_centroid: npt.NDArray[np.float64],
  control_centroid: npt.NDArray[np.float64],
  temperature: float = 1.0,
) -> float:
  """Compute zero-shot phenotype probability from case and control reference centroids.

  Uses softmax over cosine similarities scaled by temperature:
    P(case) = exp(sim(emb, case) / T) / (exp(sim(emb, case) / T) + exp(sim(emb, ctrl) / T))

  Args:
    embedding: 1D normalized embedding vector [D].
    case_centroid: 1D case reference centroid [D].
    control_centroid: 1D control reference centroid [D].
    temperature: Softmax temperature parameter.

  Returns:
    Float probability in [0.0, 1.0].
  """
  sim_case = cosine_similarity_1d(embedding, case_centroid)
  sim_ctrl = cosine_similarity_1d(embedding, control_centroid)

  # Numerically stable binary softmax
  diff = (sim_case - sim_ctrl) / max(temperature, 1e-6)
  prob_case = 1.0 / (1.0 + float(np.exp(-diff)))
  return prob_case


def load_cards_clip_model(
  model_name: str = DEFAULT_CARDS_CLIP_MODEL_NAME,
  revision: str = DEFAULT_CARDS_CLIP_REVISION,
  device: str = "cpu",
) -> tuple[Any, Any]:
  """Load CarDSLab ECG-CLIP PyTorch model and CLIP image processor.

  Args:
    model_name: Hugging Face model repository ID or local path.
    revision: Git commit hash or branch/tag revision.
    device: Target PyTorch execution device ("cpu", "cuda", "mps").

  Returns:
    Tuple of (model, processor) in evaluation mode.
  """
  import torch
  from transformers import CLIPImageProcessor, CLIPModel

  try:
    loaded_model = cast(Any, CLIPModel.from_pretrained(model_name, revision=revision))
    loaded_processor = cast(Any, CLIPImageProcessor.from_pretrained(model_name, revision=revision))
  except Exception as err:
    err_str = str(err)
    if "401" in err_str or "403" in err_str or "gated" in err_str.lower() or "restricted" in err_str.lower():
      msg = (
        f"Access to Hugging Face repository '{model_name}' requires authentication under research terms "
        f"(https://huggingface.co/{model_name}). Ensure your HF_TOKEN has granted repository access."
      )
      logger.error(msg)
      raise PermissionError(msg) from err
    logger.error("Failed to load CarDSLab ECG-CLIP model from %s: %s", model_name, err)
    raise RuntimeError(f"Failed to load CarDSLab ECG-CLIP model from {model_name}: {err}") from err

  model = loaded_model.to(torch.device(device))
  model.eval()
  return model, loaded_processor


class CardsClipAdapter(BaseEcgModelAdapter):
  """Adapter for extracting 512-d representations and zero-shot scores with CarDSLab ECG-CLIP."""

  def __init__(
    self,
    config: CardsClipConfig | None = None,
    model: Any | None = None,
    processor: Any | None = None,
  ) -> None:
    """Initialize adapter with optional pre-instantiated model/processor for test isolation."""
    self._config = config or CardsClipConfig()
    self._model = model
    self._processor = processor

  @property
  def config(self) -> CardsClipConfig:
    return self._config

  def _ensure_model_loaded(self) -> tuple[Any, Any]:
    """Lazy load model and processor if not already instantiated."""
    if self._model is None or self._processor is None:
      model, processor = load_cards_clip_model(
        model_name=self._config.model_name,
        revision=self._config.model_version or DEFAULT_CARDS_CLIP_REVISION,
        device=self._config.device,
      )
      self._model = model
      self._processor = processor
    return self._model, self._processor

  def preprocess_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = DEFAULT_SAMPLING_RATE_HZ,
  ) -> Image.Image:
    """Render 12-lead ECG into canonical 3x4 layout image."""
    return render_12lead_ecg_image(
      signal_array=signal_array,
      lead_names=lead_names,
      fs=fs,
      config=self._config.image_config,
    )

  def _extract_embedding_from_tensors(
    self,
    pixel_values: Any,
  ) -> npt.NDArray[np.float64]:
    """Run model vision forward pass and return extracted normalized embeddings as NumPy array."""
    import torch

    model, _ = self._ensure_model_loaded()

    with torch.no_grad():
      # Support both transformers 4.x and 5.x return formats
      feat = model.get_image_features(pixel_values=pixel_values)
      if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        raw_embeds = feat.pooler_output
      elif hasattr(feat, "image_embeds") and feat.image_embeds is not None:
        raw_embeds = feat.image_embeds
      elif isinstance(feat, torch.Tensor):
        raw_embeds = feat
      else:
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        raw_embeds = model.visual_projection(vision_outputs[1])

      if self._config.normalize_embeddings:
        norm = torch.norm(raw_embeds, p=2, dim=-1, keepdim=True)
        norm_embeds = raw_embeds / torch.clamp(norm, min=1e-12)
        out_tensor = norm_embeds
      else:
        out_tensor = raw_embeds

      result_np = out_tensor.detach().cpu().numpy().astype(np.float64)
      return cast(npt.NDArray[np.float64], result_np)

  def embed_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = DEFAULT_SAMPLING_RATE_HZ,
  ) -> ModelOutputResult:
    """Compute frozen 512-d ECG embedding for a single raw ECG recording."""
    # Check signal validity
    if signal_array.size == 0 or np.isnan(signal_array).any() or np.isinf(signal_array).any():
      return ModelOutputResult(
        is_valid=False,
        model_name=self._config.model_name,
        model_version=self._config.model_version,
        error_message="Input signal is empty or contains NaN/infinite values",
      )

    try:
      img = self.preprocess_single(signal_array, lead_names, fs)
      _, processor = self._ensure_model_loaded()
      inputs = processor(images=img, return_tensors="pt")
      pixel_values = inputs["pixel_values"].to(self._config.device)

      embed_2d = self._extract_embedding_from_tensors(pixel_values)
      embed_1d = embed_2d[0]

      return ModelOutputResult(
        is_valid=True,
        embedding=embed_1d,
        model_name=self._config.model_name,
        model_version=self._config.model_version,
        output_dim=len(embed_1d),
        metadata={
          "modality": self._config.input_modality.value,
          "image_size": self._config.image_config.target_pixel_size,
          "normalized": self._config.normalize_embeddings,
        },
      )
    except Exception as err:
      logger.exception("Error generating embedding for ECG record: %s", err)
      return ModelOutputResult(
        is_valid=False,
        model_name=self._config.model_name,
        model_version=self._config.model_version,
        error_message=str(err),
      )

  def embed_batch(
    self,
    batch_records: Sequence[tuple[npt.NDArray[np.float64], Sequence[str], int]],
  ) -> BatchInferenceResult:
    """Compute frozen embeddings for a batch of raw ECG recordings."""
    if not batch_records:
      return BatchInferenceResult(
        embeddings=np.empty((0, self._config.embedding_dim), dtype=np.float64),
        valid_mask=(),
        model_name=self._config.model_name,
        failure_reasons=(),
        count_total=0,
        count_valid=0,
      )

    valid_mask: list[bool] = []
    failure_reasons: list[str | None] = []
    valid_images: list[Image.Image] = []
    valid_indices: list[int] = []

    for idx, (sig, leads, fs) in enumerate(batch_records):
      if sig.size == 0 or np.isnan(sig).any() or np.isinf(sig).any():
        valid_mask.append(False)
        failure_reasons.append("Waveform contains NaN, Inf, or empty values")
        continue

      try:
        img = self.preprocess_single(sig, leads, fs)
        valid_images.append(img)
        valid_indices.append(idx)
        valid_mask.append(True)
        failure_reasons.append(None)
      except Exception as exc:
        valid_mask.append(False)
        failure_reasons.append(f"Preprocessing error: {exc}")

    n_total = len(batch_records)
    n_valid = len(valid_images)

    all_embeddings = np.zeros((n_total, self._config.embedding_dim), dtype=np.float64)

    if n_valid > 0:
      _, processor = self._ensure_model_loaded()
      batch_size = max(1, self._config.batch_size)

      for start_i in range(0, n_valid, batch_size):
        chunk_images = valid_images[start_i : start_i + batch_size]
        chunk_indices = valid_indices[start_i : start_i + batch_size]

        inputs = processor(images=chunk_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._config.device)
        chunk_embeds = self._extract_embedding_from_tensors(pixel_values)

        for local_i, target_i in enumerate(chunk_indices):
          all_embeddings[target_i] = chunk_embeds[local_i]

    return BatchInferenceResult(
      embeddings=all_embeddings,
      valid_mask=tuple(valid_mask),
      model_name=self._config.model_name,
      failure_reasons=tuple(failure_reasons),
      count_total=n_total,
      count_valid=n_valid,
    )
