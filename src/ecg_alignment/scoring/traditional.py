"""Traditional ECG risk scoring models, focusing on the Cardiac Infarction/Injury Score (CIIS).

Authoritative references:
- Rautaharju PM, Warren JW, Jain U, Wolf HK, Nielsen CL. Cardiac infarction injury score:
  an electrocardiographic coding scheme for ischemic heart disease. Circulation. 1981;64(2):249-256.
- Dekker JM, Schouten EG, Pool J, Kok FJ. Cardiac Infarction Injury Score predicts
  cardiovascular mortality in apparently healthy men and women. Br Heart J. 1994;72(1):39-44.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast
import numpy as np
import numpy.typing as npt
import scipy.signal as signal

from ecg_alignment.data import CANONICAL_12_LEADS, DEFAULT_SAMPLING_RATE_HZ


class CIISCategory(str, Enum):
  """Published clinical risk categories for CIIS."""

  NORMAL = "normal"  # CIIS < 10
  BORDERLINE = "borderline"  # 10 <= CIIS < 15
  POSSIBLE_INJURY = "possible_injury"  # 15 <= CIIS < 20
  PROBABLE_INFARCTION = "probable_infarction"  # CIIS >= 20


@dataclass(frozen=True)
class LeadFeatures:
  """Morphologic measurements extracted for a single ECG lead."""

  q_duration_ms: float
  q_amplitude_mv: float
  r_amplitude_mv: float
  s_amplitude_mv: float
  t_pos_amplitude_mv: float
  t_neg_amplitude_mv: float
  qr_ratio: float


@dataclass(frozen=True)
class CIISMeasurements:
  """Complete set of measurements required to compute CIIS."""

  avl_q_duration_ms: float
  avl_t_pos_mv: float
  avl_t_neg_mv: float
  inv_avr_r_mv: float  # R amplitude in -aVR lead
  inv_avr_t_pos_mv: float  # Positive T amplitude in -aVR lead
  lead2_qr_ratio: float
  avf_qr_ratio: float
  lead3_q_duration_ms: float
  lead3_t_neg_mv: float
  v1_t_pos_mv: float
  v2_r_mv: float
  v2_t_neg_mv: float
  v3_qr_ratio: float
  v5_s_mv: float


@dataclass(frozen=True)
class CIISItemScores:
  """Itemized points contributing to the total CIIS."""

  item1_avl_q_duration: float
  item2_avl_t_amplitude: float
  item3_avr_r_amplitude: float
  item4_avr_t_amplitude: float
  item5_lead2_avf_qr_ratio: float
  item6_lead3_avl_q_duration: float
  item7_lead3_t_amplitude: float
  item8_v1_t_amplitude: float
  item9_v2_r_amplitude: float
  item10_v2_t_amplitude: float
  item11_v3_qr_ratio: float
  item12_v5_s_amplitude: float


@dataclass(frozen=True)
class CIISScoreResult:
  """Complete score output including continuous score, category, and diagnostic metadata."""

  is_valid: bool
  total_score: float | None = None
  category: CIISCategory | None = None
  item_scores: CIISItemScores | None = None
  measurements: CIISMeasurements | None = None
  error_message: str | None = None


# -----------------------------------------------------------------------------
# Pure Item Scoring Functions
# Calibration standard: 1 mV = 10 mm (0.1 mV = 1 mm; 0.025 mV = 0.25 mm)
# -----------------------------------------------------------------------------


def score_item1_avl_q_duration(q_dur_ms: float) -> float:
  """Score Item 1: Lead aVL Q-wave duration."""
  if q_dur_ms <= 0.0:
    return 5.0
  if q_dur_ms <= 15.0:
    return 1.0
  if q_dur_ms <= 25.0:
    return 3.0
  if q_dur_ms <= 35.0:
    return 9.0
  if q_dur_ms <= 45.0:
    return 10.0
  return 12.0


def score_item2_avl_t_amplitude(t_pos_mv: float, t_neg_mv: float) -> float:
  """Score Item 2: Lead aVL T-wave amplitude."""
  if t_neg_mv > 0.025:
    # Inverted T wave: 3 points + 2.3 points per mm (23 points per mV)
    t_neg_mm = t_neg_mv * 10.0
    return 3.0 + 2.3 * t_neg_mm
  if t_pos_mv <= 0.05:  # <= 0.5 mm flat or low positive
    return 3.0
  if t_pos_mv > 0.30:  # > 3.0 mm tall positive
    return 3.0
  return 0.0


def score_item3_avr_r_amplitude(inv_avr_r_mv: float) -> float:
  """Score Item 3: Lead -aVR R-wave amplitude."""
  r_mm = inv_avr_r_mv * 10.0
  if r_mm < 5.0:
    return -1.0 * r_mm
  return 0.0


def score_item4_avr_t_amplitude(inv_avr_t_pos_mv: float) -> float:
  """Score Item 4: Lead -aVR positive T-wave amplitude."""
  t_mm = inv_avr_t_pos_mv * 10.0
  if t_mm <= 0.5:
    return 6.0
  if t_mm <= 1.5:
    return 3.0
  if t_mm <= 2.5:
    return 0.0
  if t_mm <= 3.5:
    return -2.0
  if t_mm <= 4.5:
    return -5.0
  return -5.0 - 2.0 * (t_mm - 4.0)


def score_item5_lead2_avf_qr_ratio(lead2_qr: float, avf_qr: float) -> float:
  """Score Item 5: Largest Q/R ratio in Lead II or aVF."""
  if max(lead2_qr, avf_qr) >= 0.25:
    return 12.0
  return 0.0


def score_item6_lead3_avl_q_duration(lead3_q_dur_ms: float, avl_q_dur_ms: float) -> float:
  """Score Item 6: Duration of Q in Lead III or aVL >= 40 ms."""
  if max(lead3_q_dur_ms, avl_q_dur_ms) >= 40.0:
    return 5.0
  return 0.0


def score_item7_lead3_t_amplitude(lead3_t_neg_mv: float) -> float:
  """Score Item 7: Amplitude of inverted T in Lead III > 1.0 mm (0.1 mV)."""
  if lead3_t_neg_mv > 0.10:
    return 7.0
  return 0.0


def score_item8_v1_t_amplitude(v1_t_pos_mv: float) -> float:
  """Score Item 8: Amplitude of positive T in Lead V1 > 2.0 mm (0.2 mV)."""
  if v1_t_pos_mv > 0.20:
    return 5.0
  return 0.0


def score_item9_v2_r_amplitude(v2_r_mv: float) -> float:
  """Score Item 9: Amplitude of R in Lead V2 < 3.0 mm (0.3 mV) or > 14.0 mm (1.4 mV)."""
  if v2_r_mv < 0.30 or v2_r_mv > 1.40:
    return 5.0
  return 0.0


def score_item10_v2_t_amplitude(v2_t_neg_mv: float) -> float:
  """Score Item 10: Amplitude of inverted T in Lead V2 >= 0.25 mm (0.025 mV)."""
  if v2_t_neg_mv >= 0.025:
    return 5.0
  return 0.0


def score_item11_v3_qr_ratio(v3_qr: float) -> float:
  """Score Item 11: Largest Q/R ratio in Lead V3 > 1/20 (0.05)."""
  if v3_qr > 0.05:
    return 9.0
  return 0.0


def score_item12_v5_s_amplitude(v5_s_mv: float) -> float:
  """Score Item 12: Amplitude of S in Lead V5 < 2.0 mm (0.2 mV)."""
  if v5_s_mv < 0.20:
    return 5.0
  return 0.0


def classify_ciis_category(ciis_score: float) -> CIISCategory:
  """Classify continuous CIIS into published clinical risk categories.

  Args:
    ciis_score: Continuous CIIS total score.

  Returns:
    CIISCategory enum member.
  """
  if ciis_score < 10.0:
    return CIISCategory.NORMAL
  if ciis_score < 15.0:
    return CIISCategory.BORDERLINE
  if ciis_score < 20.0:
    return CIISCategory.POSSIBLE_INJURY
  return CIISCategory.PROBABLE_INFARCTION


def compute_ciis_from_measurements(measurements: CIISMeasurements) -> CIISScoreResult:
  """Compute CIIS score and itemized breakdown from explicit ECG measurements.

  Args:
    measurements: CIISMeasurements dataclass.

  Returns:
    CIISScoreResult with continuous score, category, and item breakdown.
  """
  item1 = score_item1_avl_q_duration(measurements.avl_q_duration_ms)
  item2 = score_item2_avl_t_amplitude(measurements.avl_t_pos_mv, measurements.avl_t_neg_mv)
  item3 = score_item3_avr_r_amplitude(measurements.inv_avr_r_mv)
  item4 = score_item4_avr_t_amplitude(measurements.inv_avr_t_pos_mv)
  item5 = score_item5_lead2_avf_qr_ratio(measurements.lead2_qr_ratio, measurements.avf_qr_ratio)
  item6 = score_item6_lead3_avl_q_duration(
    measurements.lead3_q_duration_ms, measurements.avl_q_duration_ms
  )
  item7 = score_item7_lead3_t_amplitude(measurements.lead3_t_neg_mv)
  item8 = score_item8_v1_t_amplitude(measurements.v1_t_pos_mv)
  item9 = score_item9_v2_r_amplitude(measurements.v2_r_mv)
  item10 = score_item10_v2_t_amplitude(measurements.v2_t_neg_mv)
  item11 = score_item11_v3_qr_ratio(measurements.v3_qr_ratio)
  item12 = score_item12_v5_s_amplitude(measurements.v5_s_mv)

  item_scores = CIISItemScores(
    item1_avl_q_duration=item1,
    item2_avl_t_amplitude=item2,
    item3_avr_r_amplitude=item3,
    item4_avr_t_amplitude=item4,
    item5_lead2_avf_qr_ratio=item5,
    item6_lead3_avl_q_duration=item6,
    item7_lead3_t_amplitude=item7,
    item8_v1_t_amplitude=item8,
    item9_v2_r_amplitude=item9,
    item10_v2_t_amplitude=item10,
    item11_v3_qr_ratio=item11,
    item12_v5_s_amplitude=item12,
  )

  total = (
    item1
    + item2
    + item3
    + item4
    + item5
    + item6
    + item7
    + item8
    + item9
    + item10
    + item11
    + item12
  )
  category = classify_ciis_category(total)

  return CIISScoreResult(
    is_valid=True,
    total_score=float(total),
    category=category,
    item_scores=item_scores,
    measurements=measurements,
    error_message=None,
  )


# -----------------------------------------------------------------------------
# Deterministic Waveform Feature Extraction & Delineation
# -----------------------------------------------------------------------------


def filter_ecg_signal(
  signal_1d: npt.NDArray[np.float64],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> npt.NDArray[np.float64]:
  """Apply zero-phase high-pass filter to remove baseline drift without phase distortion.

  Args:
    signal_1d: 1D NumPy array representing single lead waveform in mV.
    fs: Sampling frequency in Hz.

  Returns:
    Filtered 1D array.
  """
  cutoff_hz = 0.67
  nyq = fs / 2.0
  b, a = cast(
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    signal.butter(2, cutoff_hz / nyq, btype="highpass", output="ba"),
  )
  filtered: npt.NDArray[np.float64] = signal.filtfilt(b, a, signal_1d)
  return filtered


def detect_qrs_peaks(
  lead_signal: npt.NDArray[np.float64],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> npt.NDArray[np.int64]:
  """Detect QRS R-peaks using an energy-derivative envelope.

  Args:
    lead_signal: 1D signal (preferably Lead II or vector magnitude).
    fs: Sampling rate in Hz.

  Returns:
    Array of integer sample indices corresponding to detected R-peaks.
  """
  if len(lead_signal) < fs:
    return np.empty(0, dtype=np.int64)

  # Bandpass 5-18 Hz for QRS energy
  nyq = fs / 2.0
  b, a = cast(
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    signal.butter(2, [5.0 / nyq, 18.0 / nyq], btype="bandpass", output="ba"),
  )
  bp_sig: npt.NDArray[np.float64] = signal.filtfilt(b, a, lead_signal)

  diff_sig = np.diff(bp_sig, prepend=bp_sig[0])
  squared = diff_sig**2
  win_len = max(1, int(0.10 * fs))  # 100 ms moving average
  integrated: npt.NDArray[np.float64] = np.convolve(
    squared, np.ones(win_len) / win_len, mode="same"
  )

  mean_val = float(np.mean(integrated))
  std_val = float(np.std(integrated))
  threshold = mean_val + 0.4 * std_val
  min_distance = int(0.28 * fs)  # 280 ms refractory window

  peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)
  if len(peaks) == 0:
    return np.empty(0, dtype=np.int64)

  # Refine peak to local extreme in raw signal
  refined: list[int] = []
  search_win = int(0.08 * fs)
  for p in peaks:
    start = max(0, int(p) - search_win)
    end = min(len(lead_signal), int(p) + search_win)
    if end > start:
      sub = lead_signal[start:end]
      local_idx = int(np.argmax(np.abs(sub)))
      refined.append(start + local_idx)

  return np.asarray(refined, dtype=np.int64)


def extract_median_beat(
  signals_12lead: npt.NDArray[np.float64],
  r_peaks: npt.NDArray[np.int64],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> tuple[npt.NDArray[np.float64], int]:
  """Extract a time-aligned representative median beat across all beats in the recording.

  Args:
    signals_12lead: 2D array [samples, 12].
    r_peaks: Array of R-peak sample indices.
    fs: Sampling rate in Hz.

  Returns:
    Tuple of (median_beat [beat_samples, 12], r_peak_index_in_beat).
  """
  pre_samples = int(0.25 * fs)  # 250 ms before R
  post_samples = int(0.45 * fs)  # 450 ms after R
  total_samples = pre_samples + post_samples

  valid_beats: list[npt.NDArray[np.float64]] = []
  n_samples = signals_12lead.shape[0]

  for r in r_peaks:
    start = int(r) - pre_samples
    end = int(r) + post_samples
    if start >= 0 and end <= n_samples:
      beat = signals_12lead[start:end, :]
      valid_beats.append(beat)

  if not valid_beats:
    raise ValueError("No valid complete beats could be extracted within recording boundaries.")

  stacked = np.stack(valid_beats, axis=0)  # [n_beats, total_samples, 12]
  median_beat: npt.NDArray[np.float64] = np.median(stacked, axis=0)
  return median_beat, pre_samples


def extract_lead_features(
  lead_signal: npt.NDArray[np.float64],
  r_idx: int,
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> LeadFeatures:
  """Extract fiducials and amplitudes from a single-lead median beat.

  Args:
    lead_signal: 1D array representing the median beat for one lead in mV.
    r_idx: Sample index of the R-peak in lead_signal.
    fs: Sampling rate in Hz.

  Returns:
    LeadFeatures dataclass.
  """
  # Baseline is measured in late PR segment (60 to 30 ms before R peak)
  base_start = max(0, r_idx - int(0.08 * fs))
  base_end = max(base_start + 1, r_idx - int(0.03 * fs))
  baseline = float(np.median(lead_signal[base_start:base_end]))

  zeroed = lead_signal - baseline

  # QRS onset: search backwards from R peak for deflection onset
  qrs_search_start = max(0, r_idx - int(0.08 * fs))
  qrs_onset = qrs_search_start
  for i in range(r_idx, qrs_search_start, -1):
    if np.abs(zeroed[i]) < 0.015:  # Below 15 uV
      qrs_onset = i
      break

  # Q-wave detection (between qrs_onset and R peak)
  q_dur_ms = 0.0
  q_amp_mv = 0.0
  if r_idx > qrs_onset:
    q_window = zeroed[qrs_onset:r_idx]
    min_q = float(np.min(q_window))
    if min_q < -0.02:  # Significant negative deflection (> 20 uV)
      q_amp_mv = float(np.abs(min_q))
      q_dur_samples = r_idx - qrs_onset
      q_dur_ms = float(q_dur_samples / fs * 1000.0)

  # R-wave detection (positive deflection at/around r_idx)
  r_search_start = max(0, r_idx - int(0.02 * fs))
  r_search_end = min(len(zeroed), r_idx + int(0.04 * fs))
  max_r = float(np.max(zeroed[r_search_start:r_search_end]))
  r_amp_mv = float(max(0.0, max_r))

  # S-wave detection (negative deflection following R peak within 80 ms)
  s_search_start = r_idx
  s_search_end = min(len(zeroed), r_idx + int(0.08 * fs))
  s_amp_mv = 0.0
  if s_search_end > s_search_start:
    min_s = float(np.min(zeroed[s_search_start:s_search_end]))
    if min_s < 0.0:
      s_amp_mv = float(np.abs(min_s))

  # Q/R ratio
  qr_ratio = q_amp_mv / r_amp_mv if r_amp_mv > 0.02 else (10.0 if q_amp_mv > 0.02 else 0.0)

  # ST-T wave window (from 60 ms after R peak to 400 ms after R peak)
  t_start = min(len(zeroed) - 1, r_idx + int(0.06 * fs))
  t_end = min(len(zeroed), r_idx + int(0.40 * fs))

  t_pos_amp_mv = 0.0
  t_neg_amp_mv = 0.0
  if t_end > t_start:
    t_window = zeroed[t_start:t_end]
    max_t = float(np.max(t_window))
    min_t = float(np.min(t_window))
    if max_t > 0.0:
      t_pos_amp_mv = float(max_t)
    if min_t < 0.0:
      t_neg_amp_mv = float(np.abs(min_t))

  return LeadFeatures(
    q_duration_ms=q_dur_ms,
    q_amplitude_mv=q_amp_mv,
    r_amplitude_mv=r_amp_mv,
    s_amplitude_mv=s_amp_mv,
    t_pos_amplitude_mv=t_pos_amp_mv,
    t_neg_amplitude_mv=t_neg_amp_mv,
    qr_ratio=qr_ratio,
  )


def extract_ciis_measurements_from_waveform(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> CIISMeasurements:
  """Extract all 12 CIIS measurements from a 12-lead ECG waveform array.

  Args:
    signal_array: 2D array of shape [sig_len, n_sig].
    lead_names: List of lead names matching columns of signal_array.
    fs: Sampling rate in Hz.

  Returns:
    CIISMeasurements dataclass.
  """
  lead_map = {name: i for i, name in enumerate(lead_names)}
  for req in CANONICAL_12_LEADS:
    if req not in lead_map:
      raise ValueError(f"Required lead {req} not found in lead names: {lead_names}")

  # Filter each lead
  filtered_signals = np.zeros_like(signal_array)
  for i in range(signal_array.shape[1]):
    filtered_signals[:, i] = filter_ecg_signal(signal_array[:, i], fs)

  # Detect R-peaks using Lead II
  lead2_idx = lead_map["II"]
  r_peaks = detect_qrs_peaks(filtered_signals[:, lead2_idx], fs)
  if len(r_peaks) < 2:
    # Fallback to V4/V5 if Lead II has low amplitude
    v5_idx = lead_map["V5"]
    r_peaks = detect_qrs_peaks(filtered_signals[:, v5_idx], fs)
    if len(r_peaks) < 2:
      raise ValueError(f"Insufficient R-peaks detected for median beat extraction: {len(r_peaks)}")

  # Extract median beat
  median_beat, r_idx = extract_median_beat(filtered_signals, r_peaks, fs)

  # Extract features for all leads
  feats: dict[str, LeadFeatures] = {}
  for lead_name, col_idx in lead_map.items():
    feats[lead_name] = extract_lead_features(median_beat[:, col_idx], r_idx, fs)

  # Extract inverted aVR features (lead -aVR)
  inv_avr_signal = -1.0 * median_beat[:, lead_map["aVR"]]
  inv_avr_feats = extract_lead_features(inv_avr_signal, r_idx, fs)

  avl = feats["aVL"]
  lead2 = feats["II"]
  lead3 = feats["III"]
  avf = feats["aVF"]
  v1 = feats["V1"]
  v2 = feats["V2"]
  v3 = feats["V3"]
  v5 = feats["V5"]

  return CIISMeasurements(
    avl_q_duration_ms=avl.q_duration_ms,
    avl_t_pos_mv=avl.t_pos_amplitude_mv,
    avl_t_neg_mv=avl.t_neg_amplitude_mv,
    inv_avr_r_mv=inv_avr_feats.r_amplitude_mv,
    inv_avr_t_pos_mv=inv_avr_feats.t_pos_amplitude_mv,
    lead2_qr_ratio=lead2.qr_ratio,
    avf_qr_ratio=avf.qr_ratio,
    lead3_q_duration_ms=lead3.q_duration_ms,
    lead3_t_neg_mv=lead3.t_neg_amplitude_mv,
    v1_t_pos_mv=v1.t_pos_amplitude_mv,
    v2_r_mv=v2.r_amplitude_mv,
    v2_t_neg_mv=v2.t_neg_amplitude_mv,
    v3_qr_ratio=v3.qr_ratio,
    v5_s_mv=v5.s_amplitude_mv,
  )


def score_ecg_waveform(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> CIISScoreResult:
  """Calculate CIIS continuous score and risk category directly from a 12-lead ECG waveform.

  Args:
    signal_array: 2D array of shape [sig_len, 12] in mV.
    lead_names: Sequence of lead names matching array channels.
    fs: Sampling frequency in Hz.

  Returns:
    CIISScoreResult with total_score, category, item_scores, measurements, and status.
  """
  if np.isnan(signal_array).any() or np.isinf(signal_array).any():
    return CIISScoreResult(
      is_valid=False,
      error_message="Waveform contains NaN or Infinite values.",
    )

  if signal_array.shape[0] < fs:
    return CIISScoreResult(
      is_valid=False,
      error_message=f"Waveform duration too short: {signal_array.shape[0]} samples at {fs} Hz.",
    )

  try:
    measurements = extract_ciis_measurements_from_waveform(signal_array, lead_names, fs)
    return compute_ciis_from_measurements(measurements)
  except Exception as exc:
    return CIISScoreResult(
      is_valid=False,
      error_message=f"Waveform scoring failure: {exc}",
    )
