"""Pure deterministic preprocessing and image rendering for 12-lead ECG signals.

Converts raw WFDB signals into canonical, standardized waveforms and images suitable
for Model-B transformer encoders (such as D-BETA and CarDSLab ECG-CLIP).
"""

from collections import Counter
from collections.abc import Sequence
import io
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
from PIL import Image
import scipy.signal as signal

from ecg_alignment.data import CANONICAL_12_LEADS, DEFAULT_SAMPLING_RATE_HZ
from ecg_alignment.scoring.base import (
  CropAlign,
  ImageRenderConfig,
  WaveformPadMode,
  WaveformPreprocessConfig,
)
from ecg_alignment.scoring.traditional import filter_ecg_signal

# Standard voltage and time grid specifications for clinical ECG paper
# Standard speed: 25 mm/s; Standard gain: 10 mm/mV (0.1 mV = 1 mm; 1 mV = 10 mm)
GRID_MINOR_TIME_SEC: float = 0.04  # 1 mm = 0.04 s (40 ms)
GRID_MAJOR_TIME_SEC: float = 0.20  # 5 mm = 0.20 s (200 ms)
GRID_MINOR_VOLT_MV: float = 0.10   # 1 mm = 0.10 mV
GRID_MAJOR_VOLT_MV: float = 0.50   # 5 mm = 0.50 mV

UNIT_CONVERSION_FACTORS: dict[str, float] = {
  "mv": 1.0,
  "millivolt": 1.0,
  "millivolts": 1.0,
  "uv": 1e-3,
  "microvolt": 1e-3,
  "microvolts": 1e-3,
  "v": 1e3,
  "volt": 1e3,
  "volts": 1e3,
}


def validate_and_reorder_leads(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  target_leads: Sequence[str] = CANONICAL_12_LEADS,
) -> npt.NDArray[np.float64]:
  """Validate lead presence, check for duplicates, and reorder channels to target order.

  Args:
    signal_array: 2D numpy array [sig_len, n_channels].
    lead_names: Sequence of lead names matching columns of signal_array.
    target_leads: Desired sequence and ordering of leads.

  Returns:
    2D numpy array [sig_len, len(target_leads)] with columns ordered by target_leads.

  Raises:
    ValueError: If shapes mismatch, duplicates exist, or required target leads are missing.
  """
  if signal_array.ndim != 2:
    raise ValueError(f"Expected 2D signal array [sig_len, n_channels], got shape {signal_array.shape}")

  if signal_array.shape[1] != len(lead_names):
    raise ValueError(
      f"Number of columns in signal_array ({signal_array.shape[1]}) does not match "
      f"lead_names length ({len(lead_names)})"
    )

  # Check for duplicates in input lead_names
  counts = Counter(lead_names)
  duplicates = [name for name, cnt in counts.items() if cnt > 1]
  if duplicates:
    raise ValueError(f"Duplicate lead names detected in input: {duplicates}")

  # Build lookup map
  lead_to_col = {name: idx for idx, name in enumerate(lead_names)}

  # Verify all target leads are resolvable
  missing_leads: list[str] = []
  col_indices: list[int] = []
  invert_flags: list[bool] = []

  for target in target_leads:
    if target in lead_to_col:
      col_indices.append(lead_to_col[target])
      invert_flags.append(False)
    elif target == "-aVR" and "aVR" in lead_to_col:
      # Inverted aVR synthesized from aVR
      col_indices.append(lead_to_col["aVR"])
      invert_flags.append(True)
    else:
      missing_leads.append(target)

  if missing_leads:
    raise ValueError(f"Missing required lead(s): {missing_leads}. Available leads: {list(lead_names)}")

  reordered = np.zeros((signal_array.shape[0], len(target_leads)), dtype=np.float64)
  for out_col, (in_col, invert) in enumerate(zip(col_indices, invert_flags, strict=True)):
    col_sig = signal_array[:, in_col]
    reordered[:, out_col] = -1.0 * col_sig if invert else col_sig

  return reordered


def resample_waveform(
  signal_array: npt.NDArray[np.float64],
  original_fs: int,
  target_fs: int,
) -> npt.NDArray[np.float64]:
  """Resample multi-channel ECG waveform to target sampling frequency.

  Args:
    signal_array: 2D numpy array [sig_len, n_channels].
    original_fs: Current sampling rate in Hz.
    target_fs: Desired sampling rate in Hz.

  Returns:
    Resampled 2D numpy array [new_sig_len, n_channels].

  Raises:
    ValueError: If sampling rates are non-positive or signal is invalid.
  """
  if original_fs <= 0 or target_fs <= 0:
    raise ValueError(f"Sampling rates must be positive. Got original_fs={original_fs}, target_fs={target_fs}")

  if signal_array.shape[0] == 0:
    return np.empty((0, signal_array.shape[1]), dtype=np.float64)

  if original_fs == target_fs:
    return signal_array.copy()

  num_target_samples = int(round(signal_array.shape[0] * (target_fs / original_fs)))
  if num_target_samples <= 0:
    raise ValueError(f"Target sample count must be positive. Computed {num_target_samples} samples.")

  # Use Fourier method along axis 0
  resampled = signal.resample(signal_array, num_target_samples, axis=0)
  return np.asarray(resampled, dtype=np.float64)


def pad_or_crop_waveform(
  signal_array: npt.NDArray[np.float64],
  target_length: int,
  pad_mode: WaveformPadMode = WaveformPadMode.CONSTANT,
  crop_align: CropAlign = CropAlign.START,
  fill_value: float = 0.0,
) -> npt.NDArray[np.float64]:
  """Adjust 2D signal length along time dimension to exactly match target_length.

  Args:
    signal_array: 2D numpy array [sig_len, n_channels].
    target_length: Desired number of time samples.
    pad_mode: Padding strategy ('constant' or 'edge').
    crop_align: Alignment when cropping ('start', 'center', or 'end').
    fill_value: Value to pad when pad_mode is constant.

  Returns:
    2D numpy array [target_length, n_channels].
  """
  if target_length <= 0:
    raise ValueError(f"target_length must be positive. Got {target_length}")

  cur_len, n_channels = signal_array.shape

  if cur_len == target_length:
    return signal_array.copy()

  if cur_len > target_length:
    # Crop
    if crop_align == CropAlign.START:
      return signal_array[:target_length, :].copy()
    if crop_align == CropAlign.END:
      return signal_array[cur_len - target_length :, :].copy()
    if crop_align == CropAlign.CENTER:
      start = (cur_len - target_length) // 2
      return signal_array[start : start + target_length, :].copy()
    raise ValueError(f"Unsupported crop alignment: {crop_align}")

  # Pad
  pad_needed = target_length - cur_len
  if pad_mode == WaveformPadMode.CONSTANT:
    pad_width = ((0, pad_needed), (0, 0))
    return np.pad(signal_array, pad_width, mode="constant", constant_values=fill_value)
  if pad_mode == WaveformPadMode.EDGE:
    pad_width = ((0, pad_needed), (0, 0))
    return np.pad(signal_array, pad_width, mode="edge")

  raise ValueError(f"Unsupported pad mode: {pad_mode}")


def scale_waveform_units(
  signal_array: npt.NDArray[np.float64],
  source_unit: str = "mV",
  target_unit: str = "mV",
) -> npt.NDArray[np.float64]:
  """Convert waveform amplitude units between mV, uV, and V.

  Args:
    signal_array: 2D numpy array.
    source_unit: Unit of input signal ("mV", "uV", or "V").
    target_unit: Desired output unit ("mV", "uV", or "V").

  Returns:
    Scaled 2D numpy array.
  """
  src_key = source_unit.strip().lower()
  tgt_key = target_unit.strip().lower()

  if src_key not in UNIT_CONVERSION_FACTORS:
    raise ValueError(f"Unknown source unit '{source_unit}'. Supported: {list(UNIT_CONVERSION_FACTORS.keys())}")
  if tgt_key not in UNIT_CONVERSION_FACTORS:
    raise ValueError(f"Unknown target unit '{target_unit}'. Supported: {list(UNIT_CONVERSION_FACTORS.keys())}")

  factor = UNIT_CONVERSION_FACTORS[src_key] / UNIT_CONVERSION_FACTORS[tgt_key]
  if factor == 1.0:
    return signal_array.copy()
  return signal_array * factor


def preprocess_waveform(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
  config: WaveformPreprocessConfig = WaveformPreprocessConfig(),
  source_unit: str = "mV",
) -> npt.NDArray[np.float64]:
  """Complete pure pipeline for 12-lead ECG waveform normalization and standardization.

  Pipeline steps:
  1. Validate leads and reorder channels to config.target_leads.
  2. Scale amplitude to config.target_unit.
  3. Optional zero-phase highpass baseline drift filtering.
  4. Resample to config.target_fs.
  5. Pad or crop to config.target_sig_len.

  Args:
    signal_array: Raw 2D signal array [sig_len, n_leads].
    lead_names: Lead channel names.
    fs: Original sampling frequency in Hz.
    config: WaveformPreprocessConfig.
    source_unit: Input amplitude unit.

  Returns:
    Standardized 2D array of shape [target_sig_len, len(target_leads)] in target units.
  """
  # 1. Lead validation and reordering
  reordered = validate_and_reorder_leads(signal_array, lead_names, config.target_leads)

  # 2. Unit scaling
  scaled = scale_waveform_units(reordered, source_unit=source_unit, target_unit=config.target_unit)

  # 3. Optional baseline filtering
  if config.filter_baseline:
    filtered = np.zeros_like(scaled)
    for c in range(scaled.shape[1]):
      filtered[:, c] = filter_ecg_signal(scaled[:, c], fs)
    work_sig = filtered
  else:
    work_sig = scaled

  # 4. Resampling
  resampled = resample_waveform(work_sig, original_fs=fs, target_fs=config.target_fs)

  # 5. Length standardization
  standardized = pad_or_crop_waveform(
    resampled,
    target_length=config.target_sig_len,
    pad_mode=config.pad_mode,
    crop_align=config.crop_align,
    fill_value=config.fill_value,
  )

  return standardized


# -----------------------------------------------------------------------------
# Deterministic ECG Image Rendering (for Image-based Transformer Adapters)
# -----------------------------------------------------------------------------


def render_12lead_ecg_image(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
  config: ImageRenderConfig = ImageRenderConfig(),
) -> Image.Image:
  """Render 12-lead ECG waveform deterministically into a standard clinical ECG layout image.

  Layout "standard_3x4":
  - 4 columns x 3 rows of 2.5-second lead segments:
      Col 0: I, II, III
      Col 1: aVR, aVL, aVF
      Col 2: V1, V2, V3
      Col 3: V4, V5, V6
  - Bottom strip: full 10-second continuous Lead II rhythm strip.

  Layout "stacked_12":
  - 12 full-width horizontal rows for all 12 leads.

  Args:
    signal_array: 2D numpy array [sig_len, n_leads] in mV.
    lead_names: Channel lead names.
    fs: Sampling rate in Hz.
    config: ImageRenderConfig.

  Returns:
    PIL.Image object with exact target dimensions and RGB color mode.
  """
  # Standardize leads to canonical clinical order
  std_leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
  ordered = validate_and_reorder_leads(signal_array, lead_names, std_leads)

  # Create headless Figure and Canvas
  fig = Figure(figsize=config.figsize_inches, dpi=config.dpi, facecolor=config.background_color)
  canvas = FigureCanvasAgg(fig)

  n_samples = ordered.shape[0]
  time_total_sec = n_samples / fs

  if config.layout == "standard_3x4":
    # 4 columns, 3 rows + 1 rhythm strip row at bottom
    # Lead mapping by grid position (row, col):
    # row 0 (top): I (col 0), aVR (col 1), V1 (col 2), V4 (col 3)
    # row 1 (mid): II (col 0), aVL (col 1), V2 (col 2), V5 (col 3)
    # row 2 (low): III (col 0), aVF (col 1), V3 (col 2), V6 (col 3)
    # row 3 (btm): Lead II (full width 10s)
    grid_layout: list[list[tuple[str, int]]] = [
      [("I", 0), ("aVR", 3), ("V1", 6), ("V4", 9)],
      [("II", 1), ("aVL", 4), ("V2", 7), ("V5", 10)],
      [("III", 2), ("aVF", 5), ("V3", 8), ("V6", 11)],
    ]

    # Subplot grid with 4 rows: top 3 equal height, bottom rhythm strip
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.15, wspace=0.08)

    col_duration_sec = time_total_sec / 4.0
    col_samples = int(round(col_duration_sec * fs))

    for r_idx in range(3):
      for c_idx in range(4):
        lead_name, lead_col = grid_layout[r_idx][c_idx]
        ax = fig.add_subplot(gs[r_idx, c_idx])

        start_samp = min(c_idx * col_samples, n_samples)
        end_samp = min((c_idx + 1) * col_samples, n_samples)
        seg_sig = ordered[start_samp:end_samp, lead_col] if end_samp > start_samp else np.zeros(col_samples)

        t_vec = np.linspace(0.0, col_duration_sec, len(seg_sig))
        ax.plot(t_vec, seg_sig, color=config.line_color, linewidth=config.line_width)

        if config.show_labels:
          ax.text(
            0.04,
            0.82,
            lead_name,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color=config.line_color,
          )

        ax.set_ylim(-2.0, 2.5)
        ax.set_xlim(0, col_duration_sec)
        ax.axis("off")

    # Bottom rhythm strip (Lead II)
    ax_rhythm = fig.add_subplot(gs[3, :])
    t_full = np.linspace(0.0, time_total_sec, n_samples)
    ax_rhythm.plot(t_full, ordered[:, 1], color=config.line_color, linewidth=config.line_width)
    if config.show_labels:
      ax_rhythm.text(
        0.01,
        0.82,
        "II (Rhythm)",
        transform=ax_rhythm.transAxes,
        fontsize=9,
        fontweight="bold",
        color=config.line_color,
      )
    ax_rhythm.set_ylim(-2.0, 2.5)
    ax_rhythm.set_xlim(0, time_total_sec)
    ax_rhythm.axis("off")

  elif config.layout == "stacked_12":
    # 12 horizontal subplots
    gs = fig.add_gridspec(12, 1, hspace=0.1)
    t_full = np.linspace(0.0, time_total_sec, n_samples)
    for idx, name in enumerate(std_leads):
      ax = fig.add_subplot(gs[idx, 0])
      ax.plot(t_full, ordered[:, idx], color=config.line_color, linewidth=config.line_width)
      if config.show_labels:
        ax.text(
          0.01,
          0.70,
          name,
          transform=ax.transAxes,
          fontsize=8,
          fontweight="bold",
          color=config.line_color,
        )
      ax.set_ylim(-2.0, 2.5)
      ax.set_xlim(0, time_total_sec)
      ax.axis("off")

  else:
    raise ValueError(f"Unsupported image layout '{config.layout}'. Supported: 'standard_3x4', 'stacked_12'")

  fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
  canvas.draw()

  # Extract RGBA buffer from canvas
  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=config.dpi, facecolor=config.background_color, bbox_inches=None)
  buf.seek(0)
  img = Image.open(buf).convert("RGB")

  # Optional resize to target pixel size
  if config.target_pixel_size is not None:
    img = img.resize(config.target_pixel_size, resample=Image.Resampling.BILINEAR)

  return img
