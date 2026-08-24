"""D-BETA primary multimodal transformer adapter for 12-lead ECG representation extraction.

Authoritative reference:
- Pham Hung M, Saeed A, Ma D. Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners.
  Proceedings of the 42nd International Conference on Machine Learning. PMLR 267:49277-49291; 2025.
- Hugging Face repository: Manhph2211/D-BETA (Commit SHA: 20ff3ccce1759d7d629171e15befafa9a424d2ca)
- License: CC BY-NC 4.0 (Research use only, non-commercial, attribution required)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from typing import Any, cast
import numpy as np
import numpy.typing as npt

from ecg_alignment.data import DEFAULT_SAMPLING_RATE_HZ, DEFAULT_SIGNAL_LENGTH_SAMPLES
from ecg_alignment.scoring.base import (
  STANDARD_CLINICAL_12_LEADS,
  BaseEcgModelAdapter,
  BatchInferenceResult,
  InputModality,
  ModelOutputResult,
  TransformerAdapterConfig,
  WaveformPreprocessConfig,
)
from ecg_alignment.scoring.preprocess import preprocess_waveform

logger = logging.getLogger(__name__)

DEFAULT_DBETA_MODEL_NAME: str = "Manhph2211/D-BETA"
DEFAULT_DBETA_REVISION: str = "20ff3ccce1759d7d629171e15befafa9a424d2ca"
DEFAULT_DBETA_EMBEDDING_DIM: int = 768
DEFAULT_DBETA_SIGNAL_LENGTH: int = DEFAULT_SIGNAL_LENGTH_SAMPLES
DEFAULT_DBETA_SAMPLING_RATE_HZ: int = DEFAULT_SAMPLING_RATE_HZ


@dataclass(frozen=True)
class DbetaConfig(TransformerAdapterConfig):
  """Immutable configuration for D-BETA 12-lead waveform transformer adapter."""

  model_name: str = DEFAULT_DBETA_MODEL_NAME
  model_version: str | None = DEFAULT_DBETA_REVISION
  input_modality: InputModality = InputModality.WAVEFORM
  embedding_dim: int = DEFAULT_DBETA_EMBEDDING_DIM
  preprocess_config: WaveformPreprocessConfig = field(
    default_factory=lambda: WaveformPreprocessConfig(
      target_fs=DEFAULT_DBETA_SAMPLING_RATE_HZ,
      target_sig_len=DEFAULT_DBETA_SIGNAL_LENGTH,
      target_leads=STANDARD_CLINICAL_12_LEADS,
      target_unit="mV",
    )
  )
  normalize_embeddings: bool = False
  batch_size: int = 32
  device: str = "cpu"
  trust_remote_code: bool = True
  local_checkpoint_path: str | None = None


def load_dbeta_model(
  model_name: str = DEFAULT_DBETA_MODEL_NAME,
  revision: str = DEFAULT_DBETA_REVISION,
  device: str = "cpu",
  trust_remote_code: bool = True,
  local_checkpoint_path: str | None = None,
) -> Any:
  """Load D-BETA PyTorch model in evaluation mode.

  Args:
    model_name: Hugging Face model repository ID.
    revision: Git commit hash or revision tag.
    device: PyTorch target device ('cpu', 'cuda', 'mps').
    trust_remote_code: Whether to allow custom modeling code from repository.
    local_checkpoint_path: Optional local directory or checkpoint path to load from.

  Returns:
    Pretrained PyTorch model in eval mode on target device.

  Raises:
    PermissionError: If remote access is restricted or gated terms have not been accepted.
    RuntimeError: If model instantiation or weight loading fails.
  """
  import torch
  from transformers import AutoModel

  target_source = local_checkpoint_path if local_checkpoint_path is not None else model_name
  logger.info("Loading D-BETA model from %s (revision %s) on %s", target_source, revision, device)

  try:
    if local_checkpoint_path is not None:
      loaded_model = cast(
        Any,
        AutoModel.from_pretrained(
          local_checkpoint_path,
          trust_remote_code=trust_remote_code,
        ),
      )
    else:
      loaded_model = cast(
        Any,
        AutoModel.from_pretrained(
          model_name,
          revision=revision,
          trust_remote_code=trust_remote_code,
        ),
      )
  except Exception as err:
    err_str = str(err)
    if "403" in err_str or "gated" in err_str.lower() or "forbidden" in err_str.lower():
      msg = (
        f"Access to Hugging Face repository '{model_name}' requires approval under CC BY-NC 4.0 "
        f"research terms (https://huggingface.co/{model_name}). To load locally, provide weights via "
        f"'local_checkpoint_path' or ensure your HF_TOKEN has granted repository access."
      )
      logger.error(msg)
      raise PermissionError(msg) from err
    logger.error("Failed to load D-BETA model from %s: %s", target_source, err)
    raise RuntimeError(f"Failed to load D-BETA model from {target_source}: {err}") from err

  model = loaded_model.to(torch.device(device))
  model.eval()
  return model


class DbetaAdapter(BaseEcgModelAdapter):
  """Adapter for extracting 768-d representations from raw 12-lead ECGs using D-BETA."""

  def __init__(
    self,
    config: DbetaConfig | None = None,
    model: Any | None = None,
  ) -> None:
    """Initialize adapter with optional pre-instantiated model for test isolation."""
    self._config = config or DbetaConfig()
    self._model = model

  @property
  def config(self) -> DbetaConfig:
    return self._config

  def _ensure_model_loaded(self) -> Any:
    """Lazy load model if not already instantiated."""
    if self._model is None:
      self._model = load_dbeta_model(
        model_name=self._config.model_name,
        revision=self._config.model_version or DEFAULT_DBETA_REVISION,
        device=self._config.device,
        trust_remote_code=self._config.trust_remote_code,
        local_checkpoint_path=self._config.local_checkpoint_path,
      )
    return self._model

  def preprocess_single(
    self,
    signal_array: npt.NDArray[np.float64],
    lead_names: Sequence[str],
    fs: int = DEFAULT_SAMPLING_RATE_HZ,
  ) -> npt.NDArray[np.float64]:
    """Standardize 12-lead ECG signal into shape [12, 5000] in canonical clinical lead order."""
    standardized_2d = preprocess_waveform(
      signal_array=signal_array,
      lead_names=lead_names,
      fs=fs,
      config=self._config.preprocess_config,
    )
    # Transpose from [5000, 12] to [12, 5000] expected by D-BETA waveform transformer
    transposed = standardized_2d.T
    return np.asarray(transposed, dtype=np.float64)

  def _extract_embeddings_from_tensors(
    self,
    ecg_tensors: Any,
  ) -> npt.NDArray[np.float64]:
    """Run model forward pass on [B, 12, 5000] tensor and return [B, 768] embeddings."""
    import torch

    model = self._ensure_model_loaded()

    with torch.no_grad():
      output = model(ecg_tensors)

      if hasattr(output, "pooler_output") and output.pooler_output is not None:
        raw_embeds = output.pooler_output
      elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        # Use first token [CLS] embedding from sequence
        raw_embeds = output.last_hidden_state[:, 0, :]
      elif isinstance(output, torch.Tensor):
        raw_embeds = output
      elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        raw_embeds = output[0]
      else:
        raise ValueError(f"Unrecognized D-BETA model output structure: {type(output)}")

      if self._config.normalize_embeddings:
        norm = torch.norm(raw_embeds, p=2, dim=-1, keepdim=True)
        out_tensor = raw_embeds / torch.clamp(norm, min=1e-12)
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
    """Compute frozen 768-d ECG embedding for a single raw ECG recording."""
    import torch

    if signal_array.size == 0 or np.isnan(signal_array).any() or np.isinf(signal_array).any():
      return ModelOutputResult(
        is_valid=False,
        model_name=self._config.model_name,
        model_version=self._config.model_version,
        error_message="Input signal is empty or contains NaN/infinite values",
      )

    try:
      prep_sig = self.preprocess_single(signal_array, lead_names, fs)
      # Shape: [1, 12, 5000]
      tensor = torch.from_numpy(prep_sig).unsqueeze(0).float().to(self._config.device)

      embed_2d = self._extract_embeddings_from_tensors(tensor)
      embed_1d = embed_2d[0]

      return ModelOutputResult(
        is_valid=True,
        embedding=embed_1d,
        model_name=self._config.model_name,
        model_version=self._config.model_version,
        output_dim=len(embed_1d),
        metadata={
          "modality": self._config.input_modality.value,
          "input_shape": prep_sig.shape,
          "leads": self._config.preprocess_config.target_leads,
          "normalized": self._config.normalize_embeddings,
        },
      )
    except Exception as err:
      logger.exception("Error generating D-BETA embedding: %s", err)
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
    import torch

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
    valid_tensors: list[torch.Tensor] = []
    valid_indices: list[int] = []

    for idx, (sig, leads, fs) in enumerate(batch_records):
      if sig.size == 0 or np.isnan(sig).any() or np.isinf(sig).any():
        valid_mask.append(False)
        failure_reasons.append("Waveform contains NaN, Inf, or empty values")
        continue

      try:
        prep_sig = self.preprocess_single(sig, leads, fs)
        tensor_item = torch.from_numpy(prep_sig).float()
        valid_tensors.append(tensor_item)
        valid_indices.append(idx)
        valid_mask.append(True)
        failure_reasons.append(None)
      except Exception as exc:
        valid_mask.append(False)
        failure_reasons.append(f"Preprocessing error: {exc}")

    n_total = len(batch_records)
    n_valid = len(valid_tensors)

    all_embeddings = np.zeros((n_total, self._config.embedding_dim), dtype=np.float64)

    if n_valid > 0:
      batch_size = max(1, self._config.batch_size)

      for start_i in range(0, n_valid, batch_size):
        chunk_tensors = valid_tensors[start_i : start_i + batch_size]
        chunk_indices = valid_indices[start_i : start_i + batch_size]

        stacked_batch = torch.stack(chunk_tensors, dim=0).to(self._config.device)
        chunk_embeds = self._extract_embeddings_from_tensors(stacked_batch)

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
