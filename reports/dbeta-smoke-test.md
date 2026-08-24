# Stage 5 Validation Report: D-BETA Primary Transformer Path

## 1. Executive Summary

This report documents the implementation, unit testing, and empirical smoke test validation of **Model `B` Primary Transformer Path**: **D-BETA** (*Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners*, ICML 2025), developed by Hung Manh Pham, Aaqib Saeed, and Dong Ma.

D-BETA serves as the primary peer-reviewed multimodal foundation model for 12-lead ECG representation learning. It learns joint electrophysiologic and clinical report representations through cross-modal masked and contrastive pretraining, producing a 768-dimensional representation directly from raw multi-lead ECG waveforms.

### Key Validation Outcomes:
- **Predictor Firewall Compliant**: Embeddings are computed strictly from patient-specific ECG waveforms (`[12, 5000]`). No non-ECG clinical features, demographics, diagnoses, medications, laboratory values, or notes are permitted into the encoder.
- **Pure & Deterministic**: Waveform preprocessing, channel reordering to standard clinical 12-lead sequence, and inference in evaluation mode produce bitwise deterministic 768-dimensional feature vectors.
- **High Technical Success Rate**: **99.00%** (99 / 100) valid embedding generations on a sample of consecutive MIMIC-IV-ECG records.
- **Explicit Failure Diagnostics**: **1.00%** (1 / 100) technical failure gracefully identified and isolated due to raw waveform corruption (NaNs) without pipeline disruption.
- **Standardized Adapter Contract**: Implements `BaseEcgModelAdapter`, allowing uniform downstream frozen linear probing, alignment analysis, and head-to-head comparison with Model `A` (CIIS).

---

## 2. Model Architecture and Specifications

| Parameter | Specification | Details |
| :--- | :--- | :--- |
| **Model Name / Repository** | `Manhph2211/D-BETA` | Hugging Face Hub / GitHub (`manhph2211/D-BETA`) |
| **Model Revision (Commit SHA)** | `20ff3ccce1759d7d629171e15befafa9a424d2ca` | Pinned git commit |
| **Publication** | ICML 2025 | PMLR 267:49277-49291 |
| **License** | CC BY-NC 4.0 | Research use only, non-commercial, attribution required |
| **Input Modality** | Raw 12-Lead ECG Waveform | `[batch, 12, 5000]` tensor in mV |
| **Sampling Rate & Duration** | 500 Hz, 10.0 seconds | 5,000 samples per lead |
| **Canonical Lead Sequence** | Standard Clinical 12 Leads | `('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')` |
| **Output Representation** | 768-dimensional continuous vector | `pooler_output` / first token multimodal projection |
| **Gated Access Protocol** | Hugging Face Terms Acceptance | Supports local weights via `local_checkpoint_path` or approved `HF_TOKEN` |

---

## 3. Waveform Transformation & Standardization Pipeline

The D-BETA adapter converts raw multi-channel ECG waveforms into standardized input tensors through the following pure pipeline:

```text
Raw WFDB Signal [5000, 12] (500 Hz)
  │
  ▼
Lead Validation & Reordering (Canonical Clinical: I, II, III, aVR, aVL, aVF, V1-V6)
  │
  ▼
Amplitude Unit Scaling (mV)
  │
  ▼
Resampling (500 Hz) & Duration Standardization (5000 samples)
  │
  ▼
Transposition to Channel-First Tensor [12, 5000]
  │
  ▼
Batch Stacking [B, 12, 5000] & PyTorch Tensor Conversion
  │
  ▼
D-BETA ECG Transformer Forward Pass (Eval Mode, torch.no_grad)
  │
  ▼
768-d Frozen ECG Embedding Vector
```

---

## 4. Empirical Benchmark on MIMIC-IV-ECG

A validation benchmark was executed across 100 consecutive records from `~/data/mimic-iv-ecg/1.0/record_list.csv` on a local execution environment (batch size = 16).

### Benchmark Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Total Records Evaluated** | 100 |
| **Valid Embeddings Generated** | 99 (99.0%) |
| **Technical Failures** | 1 (1.0%) |
| **Failure Reason** | Record 93: Waveform contains NaN values |
| **Waveform Load Time** | 0.231 s |
| **Batch Inference Time** | 0.029 s |
| **Throughput** | 3,414.16 records / second |
| **Per-Record Latency** | 0.29 ms / record |
| **Peak Process RSS Memory** | 588.64 MB |

### Embedding Vector Characteristics
| Metric | Value |
| :--- | :--- |
| **Embedding Dimension** | 768 |
| **Consecutive Pairwise Cosine Similarity (Min / Max)** | $-0.5611$ / $0.9966$ |
| **Consecutive Pairwise Cosine Similarity (Median)** | $0.8365$ |
| **Consecutive Pairwise Cosine Similarity (Mean $\pm$ SD)** | $0.6324 \pm 0.4148$ |
| **Consecutive Pairwise Cosine Similarity (10th / 90th %ile)** | $-0.0417$ / $0.9599$ |

---

## 5. Research Guardrails and Pretraining Contamination Disclosure

### Pretraining Contamination
D-BETA was pretrained on paired ECGs and text reports from the MIMIC-IV-ECG database. Consequently, evaluation of D-BETA on MIMIC-IV-ECG records constitutes **in-domain representation probing**, not independent external validation. All subsequent analyses, publications, and figures must clearly maintain this distinction.

### Predictor-Information Firewall
In accordance with repository research rules:
- D-BETA receives only the 12-lead ECG waveform signal.
- No clinical covariates (e.g., age, sex, diagnoses, lab values, medications, encounter metadata) enter the model.
- Supervised downstream linear probing in Stage 8 will use patient-disjoint splits to prevent outcome leakage.

---

## 6. Test Suite and Static Type Checking Verification

The D-BETA implementation is covered by a test suite in `tests/test_dbeta.py` verifying:
- Configuration parameters, immutability, and defaults.
- Waveform preprocessing, channel reordering to standard clinical sequence, and shape standardization (`[12, 5000]`).
- Single and batch inference producing valid 768-dimensional float64 embeddings.
- Robust error handling for empty arrays, NaN values, infinite values, and missing lead channels.
- Batch chunking and validation mask propagation across mixed batches.
- Bitwise determinism under identical inputs.
- Informative `PermissionError` diagnostic handling for gated Hugging Face repositories.
- Local custom checkpoint loading.
- Real MIMIC-IV-ECG waveform integration smoke testing.

**Test Suite Status**: **82 passed** in 4.39s.  
**Static Type Checking**: `basedpyright` passing with **0 errors, 0 warnings, 0 notes**.
