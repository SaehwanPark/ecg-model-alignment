"""Tests for traditional ECG risk scoring (CIIS) and waveform delineation."""

import numpy as np
import pytest

from ecg_alignment.data import CANONICAL_12_LEADS, DEFAULT_SAMPLING_RATE_HZ, DataPaths, load_record_list, read_ecg_waveform
from ecg_alignment.scoring.traditional import (
  CIISCategory,
  CIISMeasurements,
  classify_ciis_category,
  compute_ciis_from_measurements,
  extract_ciis_measurements_from_waveform,
  extract_lead_features,
  extract_median_beat,
  score_ecg_waveform,
  score_item1_avl_q_duration,
  score_item2_avl_t_amplitude,
  score_item3_avr_r_amplitude,
  score_item4_avr_t_amplitude,
  score_item5_lead2_avf_qr_ratio,
  score_item6_lead3_avl_q_duration,
  score_item7_lead3_t_amplitude,
  score_item8_v1_t_amplitude,
  score_item9_v2_r_amplitude,
  score_item10_v2_t_amplitude,
  score_item11_v3_qr_ratio,
  score_item12_v5_s_amplitude,
)


# -----------------------------------------------------------------------------
# Unit Tests: Individual Item Scoring Functions
# -----------------------------------------------------------------------------


def test_score_item1_avl_q_duration():
  assert score_item1_avl_q_duration(0.0) == 5.0
  assert score_item1_avl_q_duration(10.0) == 1.0
  assert score_item1_avl_q_duration(15.0) == 1.0
  assert score_item1_avl_q_duration(20.0) == 3.0
  assert score_item1_avl_q_duration(25.0) == 3.0
  assert score_item1_avl_q_duration(30.0) == 9.0
  assert score_item1_avl_q_duration(35.0) == 9.0
  assert score_item1_avl_q_duration(40.0) == 10.0
  assert score_item1_avl_q_duration(45.0) == 10.0
  assert score_item1_avl_q_duration(50.0) == 12.0
  assert score_item1_avl_q_duration(60.0) == 12.0


def test_score_item2_avl_t_amplitude():
  # Normal upright T (e.g. 0.15 mV / 1.5 mm): 0 points
  assert score_item2_avl_t_amplitude(0.15, 0.0) == 0.0
  # Flat or low T (<= 0.05 mV / 0.5 mm): +3 points
  assert score_item2_avl_t_amplitude(0.04, 0.0) == 3.0
  # Tall T (> 0.30 mV / 3.0 mm): +3 points
  assert score_item2_avl_t_amplitude(0.35, 0.0) == 3.0
  # Inverted T (0.10 mV / 1.0 mm negative): 3 + 2.3 * 1.0 = 5.3
  assert pytest.approx(score_item2_avl_t_amplitude(0.0, 0.10), rel=1e-3) == 5.3
  # Inverted T (0.20 mV / 2.0 mm negative): 3 + 2.3 * 2.0 = 7.6
  assert pytest.approx(score_item2_avl_t_amplitude(0.0, 0.20), rel=1e-3) == 7.6


def test_score_item3_avr_r_amplitude():
  # In -aVR, R < 5 mm (0.50 mV): score is -1 * R_mm
  assert score_item3_avr_r_amplitude(0.20) == -2.0  # 2 mm -> -2.0
  assert score_item3_avr_r_amplitude(0.40) == -4.0  # 4 mm -> -4.0
  assert score_item3_avr_r_amplitude(0.50) == 0.0  # 5 mm -> 0.0
  assert score_item3_avr_r_amplitude(0.80) == 0.0  # 8 mm -> 0.0


def test_score_item4_avr_t_amplitude():
  # In -aVR:
  assert score_item4_avr_t_amplitude(0.0) == 6.0  # 0 mm -> 6.0
  assert score_item4_avr_t_amplitude(0.10) == 3.0  # 1 mm -> 3.0
  assert score_item4_avr_t_amplitude(0.20) == 0.0  # 2 mm -> 0.0
  assert score_item4_avr_t_amplitude(0.30) == -2.0  # 3 mm -> -2.0
  assert score_item4_avr_t_amplitude(0.40) == -5.0  # 4 mm -> -5.0
  assert score_item4_avr_t_amplitude(0.50) == -7.0  # 5 mm -> -5 - 2*1 = -7.0
  assert score_item4_avr_t_amplitude(0.60) == -9.0  # 6 mm -> -5 - 2*2 = -9.0


def test_score_item5_lead2_avf_qr_ratio():
  assert score_item5_lead2_avf_qr_ratio(0.10, 0.15) == 0.0
  assert score_item5_lead2_avf_qr_ratio(0.25, 0.10) == 12.0
  assert score_item5_lead2_avf_qr_ratio(0.10, 0.30) == 12.0


def test_score_item6_lead3_avl_q_duration():
  assert score_item6_lead3_avl_q_duration(20.0, 30.0) == 0.0
  assert score_item6_lead3_avl_q_duration(40.0, 20.0) == 5.0
  assert score_item6_lead3_avl_q_duration(20.0, 42.0) == 5.0


def test_score_item7_lead3_t_amplitude():
  assert score_item7_lead3_t_amplitude(0.05) == 0.0
  assert score_item7_lead3_t_amplitude(0.10) == 0.0
  assert score_item7_lead3_t_amplitude(0.15) == 7.0


def test_score_item8_v1_t_amplitude():
  assert score_item8_v1_t_amplitude(0.15) == 0.0
  assert score_item8_v1_t_amplitude(0.20) == 0.0
  assert score_item8_v1_t_amplitude(0.25) == 5.0


def test_score_item9_v2_r_amplitude():
  assert score_item9_v2_r_amplitude(0.25) == 5.0  # < 0.30 mV
  assert score_item9_v2_r_amplitude(0.60) == 0.0  # in [0.30, 1.40]
  assert score_item9_v2_r_amplitude(1.50) == 5.0  # > 1.40 mV


def test_score_item10_v2_t_amplitude():
  assert score_item10_v2_t_amplitude(0.01) == 0.0
  assert score_item10_v2_t_amplitude(0.025) == 5.0
  assert score_item10_v2_t_amplitude(0.10) == 5.0


def test_score_item11_v3_qr_ratio():
  assert score_item11_v3_qr_ratio(0.02) == 0.0
  assert score_item11_v3_qr_ratio(0.05) == 0.0
  assert score_item11_v3_qr_ratio(0.08) == 9.0


def test_score_item12_v5_s_amplitude():
  assert score_item12_v5_s_amplitude(0.15) == 5.0  # < 0.20 mV
  assert score_item12_v5_s_amplitude(0.20) == 0.0
  assert score_item12_v5_s_amplitude(0.50) == 0.0


# -----------------------------------------------------------------------------
# Unit Tests: Risk Category Classification & Boundaries
# -----------------------------------------------------------------------------


def test_classify_ciis_category():
  assert classify_ciis_category(-5.0) == CIISCategory.NORMAL
  assert classify_ciis_category(0.0) == CIISCategory.NORMAL
  assert classify_ciis_category(9.9) == CIISCategory.NORMAL
  assert classify_ciis_category(10.0) == CIISCategory.BORDERLINE
  assert classify_ciis_category(14.9) == CIISCategory.BORDERLINE
  assert classify_ciis_category(15.0) == CIISCategory.POSSIBLE_INJURY
  assert classify_ciis_category(19.9) == CIISCategory.POSSIBLE_INJURY
  assert classify_ciis_category(20.0) == CIISCategory.PROBABLE_INFARCTION
  assert classify_ciis_category(35.0) == CIISCategory.PROBABLE_INFARCTION


def test_compute_ciis_from_measurements_normal():
  # Normal baseline ECG measurements
  meas = CIISMeasurements(
    avl_q_duration_ms=0.0,  # absent -> +5
    avl_t_pos_mv=0.15,  # normal -> 0
    avl_t_neg_mv=0.0,
    inv_avr_r_mv=0.50,  # 5 mm -> 0
    inv_avr_t_pos_mv=0.20,  # 2 mm -> 0
    lead2_qr_ratio=0.0,  # 0
    avf_qr_ratio=0.0,  # 0
    lead3_q_duration_ms=0.0,  # 0
    lead3_t_neg_mv=0.0,  # 0
    v1_t_pos_mv=0.10,  # 0
    v2_r_mv=0.60,  # 0
    v2_t_neg_mv=0.0,  # 0
    v3_qr_ratio=0.0,  # 0
    v5_s_mv=0.40,  # 0
  )
  res = compute_ciis_from_measurements(meas)
  assert res.is_valid is True
  assert res.total_score == 5.0
  assert res.category == CIISCategory.NORMAL
  assert res.item_scores is not None
  assert res.item_scores.item1_avl_q_duration == 5.0


def test_compute_ciis_from_measurements_infarction():
  # Severe infarction ECG measurements
  meas = CIISMeasurements(
    avl_q_duration_ms=50.0,  # >=50ms -> +12
    avl_t_pos_mv=0.0,
    avl_t_neg_mv=0.20,  # inverted 2mm -> 3 + 2.3*2 = 7.6
    inv_avr_r_mv=0.20,  # 2mm -> -2.0
    inv_avr_t_pos_mv=0.0,  # 0mm -> +6.0
    lead2_qr_ratio=0.35,  # >=0.25 -> +12.0
    avf_qr_ratio=0.30,
    lead3_q_duration_ms=45.0,  # >=40ms -> +5.0
    lead3_t_neg_mv=0.25,  # inverted >1mm -> +7.0
    v1_t_pos_mv=0.30,  # upright >2mm -> +5.0
    v2_r_mv=0.15,  # <3mm -> +5.0
    v2_t_neg_mv=0.05,  # inverted >=0.25mm -> +5.0
    v3_qr_ratio=0.15,  # >0.05 -> +9.0
    v5_s_mv=0.10,  # <2mm -> +5.0
  )
  res = compute_ciis_from_measurements(meas)
  assert res.is_valid is True
  expected_total = 12.0 + 7.6 - 2.0 + 6.0 + 12.0 + 5.0 + 7.0 + 5.0 + 5.0 + 5.0 + 9.0 + 5.0
  assert pytest.approx(res.total_score or 0.0, rel=1e-3) == expected_total
  assert res.category == CIISCategory.PROBABLE_INFARCTION


# -----------------------------------------------------------------------------
# Unit Tests: Waveform Feature Extraction & Delineation
# -----------------------------------------------------------------------------


def _generate_synthetic_ecg(
  duration_sec: float = 10.0,
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
  hr_bpm: float = 60.0,
) -> tuple[np.ndarray, list[str]]:
  """Create a synthetic 12-lead ECG signal with standard P-QRS-T complexes."""
  n_samples = int(duration_sec * fs)
  t = np.arange(n_samples) / fs
  signals = np.zeros((n_samples, 12), dtype=np.float64)

  beat_period = 60.0 / hr_bpm
  beat_times = np.arange(0.5, duration_sec - 0.5, beat_period)

  for bt in beat_times:
    b_idx = int(bt * fs)
    # Add QRS complex to Lead II (index 1) and V5 (index 10)
    for lead_idx in range(12):
      # R wave
      r_width = int(0.04 * fs)
      r_start = max(0, b_idx - r_width)
      r_end = min(n_samples, b_idx + r_width)
      signals[r_start:r_end, lead_idx] += 1.0 * np.exp(
        -0.5 * ((np.arange(r_start, r_end) - b_idx) / (0.015 * fs)) ** 2
      )

      # S wave
      s_idx = b_idx + int(0.03 * fs)
      s_start = max(0, s_idx - r_width)
      s_end = min(n_samples, s_idx + r_width)
      signals[s_start:s_end, lead_idx] -= 0.3 * np.exp(
        -0.5 * ((np.arange(s_start, s_end) - s_idx) / (0.015 * fs)) ** 2
      )

      # T wave
      t_idx = b_idx + int(0.20 * fs)
      t_width = int(0.08 * fs)
      t_start = max(0, t_idx - t_width)
      t_end = min(n_samples, t_idx + t_width)
      signals[t_start:t_end, lead_idx] += 0.25 * np.exp(
        -0.5 * ((np.arange(t_start, t_end) - t_idx) / (0.04 * fs)) ** 2
      )

  return signals, list(CANONICAL_12_LEADS)


def test_score_ecg_waveform_synthetic():
  signals, leads = _generate_synthetic_ecg()
  result = score_ecg_waveform(signals, leads, fs=DEFAULT_SAMPLING_RATE_HZ)
  assert result.is_valid is True
  assert result.total_score is not None
  assert isinstance(result.total_score, float)
  assert result.category is not None


def test_score_ecg_waveform_invalid_inputs():
  # NaN signal
  signals, leads = _generate_synthetic_ecg()
  signals[100, 0] = np.nan
  res_nan = score_ecg_waveform(signals, leads)
  assert res_nan.is_valid is False
  assert res_nan.error_message is not None
  assert "NaN" in res_nan.error_message

  # Infinite signal
  signals, leads = _generate_synthetic_ecg()
  signals[100, 0] = np.inf
  res_inf = score_ecg_waveform(signals, leads)
  assert res_inf.is_valid is False
  assert res_inf.error_message is not None
  assert "Infinite" in res_inf.error_message

  # Too short waveform
  short_sig = signals[:100, :]
  res_short = score_ecg_waveform(short_sig, leads)
  assert res_short.is_valid is False
  assert res_short.error_message is not None
  assert "too short" in res_short.error_message

  # Missing leads
  res_missing = score_ecg_waveform(signals, ["I", "II"])
  assert res_missing.is_valid is False
  assert res_missing.error_message is not None


# -----------------------------------------------------------------------------
# Integration Test: Real MIMIC-IV-ECG Data
# -----------------------------------------------------------------------------


def test_score_real_mimic_ecg_sample():
  paths = DataPaths()
  if not paths.record_list_path.exists():
    pytest.skip("Local MIMIC-IV-ECG dataset not found.")

  records = load_record_list(paths.record_list_path).head(5).to_dicts()
  for r in records:
    wf_path = paths.mimic_iv_ecg_dir / r["path"]
    sig, leads, fs = read_ecg_waveform(wf_path)
    result = score_ecg_waveform(sig, leads, fs)
    assert result.is_valid is True
    assert result.total_score is not None
    assert result.category in [
      CIISCategory.NORMAL,
      CIISCategory.BORDERLINE,
      CIISCategory.POSSIBLE_INJURY,
      CIISCategory.PROBABLE_INFARCTION,
    ]
    assert result.item_scores is not None
