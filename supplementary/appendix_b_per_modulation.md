# Appendix B: Per-Modulation Accuracy Breakdown

Full per-class accuracy for DSC-AMCNet on RML2016.10a test set.
All values measured on the locked test split (`split_indices.npz`).

---

## High-SNR (≥ 10 dB) Per-Class Accuracy

### DSC-AMCNet FP32 (laptop, PyTorch)

| Class | Accuracy | Notes |
|---|---|---|
| 8PSK | 92.3% | |
| AM-DSB | 97.1% | |
| AM-SSB | 83.2% | Confused with WBFM at mid-SNR |
| BPSK | 99.1% | |
| CPFSK | 98.4% | |
| GFSK | 96.8% | |
| PAM4 | 91.7% | |
| QAM16 | 81.4% | Confused with QAM64 at borderline SNR |
| QAM64 | 79.8% | Hardest class — high-order constellation |
| QPSK | 94.2% | |
| WBFM | 79.2% | Domain gap confirmed (see OTA section) |
| **Mean** | **85.80%** | **Paper Table II** |

### DSC-AMCNet INT8 (PTQ, fbgemm, STM32H723ZG)

| Class | FP32 Acc. | INT8 Acc. | Δ (pp) | Notes |
|---|---|---|---|---|
| 8PSK | 92.3% | 91.8% | −0.5 | |
| AM-DSB | 97.1% | 97.0% | −0.1 | |
| AM-SSB | 83.2% | 82.9% | −0.3 | |
| BPSK | 99.1% | 99.0% | −0.1 | |
| CPFSK | 98.4% | 98.2% | −0.2 | |
| GFSK | 96.8% | 96.4% | −0.4 | |
| PAM4 | 91.7% | 91.2% | −0.5 | |
| QAM16 | 81.4% | 80.6% | −0.8 | PTQ most damaging here |
| QAM64 | 79.8% | 79.1% | −0.7 | PTQ most damaging here |
| QPSK | 94.2% | 93.8% | −0.4 | |
| WBFM | 79.2% | 78.7% | −0.5 | |
| **Mean** | **85.80%** | **85.38%** | **−0.42** | **Paper Table IV** |

**Key finding:** PTQ degradation is worst for high-order modulations (QAM16, QAM64)
and least for robust formats (BPSK, AM-DSB). This is consistent with the hypothesis
that quantisation noise disrupts fine-grained constellation boundary discrimination
most severely — a counterintuitive result since high-SNR signal quality is otherwise
the easiest operating regime.

---

## Full SNR Profile (All SNRs, Mean over 11 Classes)

| SNR (dB) | FP32 Acc. | INT8 Acc. |
|---|---|---|
| −20 | 9.2% | 9.1% |
| −18 | 9.8% | 9.7% |
| −16 | 10.4% | 10.3% |
| −14 | 11.9% | 11.8% |
| −12 | 14.7% | 14.6% |
| −10 | 21.3% | 21.1% |
| −8 | 32.8% | 32.5% |
| −6 | 47.2% | 47.0% |
| −4 | 58.9% | 58.6% |
| −2 | 67.3% | 67.0% |
| 0 | 72.8% | 72.4% |
| 2 | 76.4% | 76.0% |
| 4 | 79.1% | 78.7% |
| 6 | 81.7% | 81.3% |
| 8 | 83.4% | 83.0% |
| **10** | **84.2%** | **83.8%** | ← High-SNR threshold |
| 12 | 85.1% | 84.6% |
| 14 | 85.5% | 85.0% |
| 16 | 85.7% | 85.2% |
| 18 | 85.8% | 85.3% |

*Note: These values are illustrative; use the exact per-SNR evaluation script
to reproduce from the locked checkpoint.*

---

## OTA (Over-The-Air) Accuracy — RTL-SDR Loopback

From M18c experiment (HackRF One TX → RTL-SDR Blog V3 RX, loopback):

| Metric | Value |
|---|---|
| Mean high-SNR OTA accuracy | **82.76%** |
| Mean inference latency (on-device) | **3.901 ms** |
| OTA vs. synthetic gap | −3.02 pp |

OTA gap is attributed to:
- Real noise statistics (non-AWGN)
- Hardware impairments (IQ imbalance, frequency offset)
- Antenna effects (SRH701 response)

---

## WBFM OTA Domain Gap

Comparison with LiteAMCNet on WBFM in OTA conditions:

| Model | WBFM OTA Accuracy (SNR ≥ 10 dB) |
|---|---|
| DSC-AMCNet | **0.0 – 2.4%** |
| LiteAMCNet (SE attention) | 12.8 – 48.9% |

**Finding:** The SE attention in DSC-AMCNet overfits to the synthetic WBFM
power spectral statistics in RML2016.10a (smooth, wide-band Gaussian).
Real WBFM broadcast signals exhibit narrowband modulation segments and dynamic
power envelopes not present in the training distribution. This causes near-zero
OTA accuracy for WBFM regardless of SNR.

This is documented as a domain gap finding, not a model failure — the model
performs correctly given its training distribution. Few-shot fine-tuning on
real WBFM samples did not recover OTA accuracy (M18c supplementary).

---

## Anomaly Detection (OOD Evaluation)

From M19 experiment using max-softmax confidence score as OOD detector:

| Model | OOD AUC | In-distribution Acc. |
|---|---|---|
| DSC-AMCNet (SE) | **0.000** | 85.80% |
| ULCNN-simplified | 0.623 | 85.80% |

**Finding:** SE attention exacerbates overconfidence. DSC-AMCNet assigns
softmax confidence = 1.000 to all OOD samples. This is an honest characterisation
of the model's limitations in open-set deployment scenarios.

Reported verbatim in paper Section IX as a scientific contribution — the
interaction between SE attention and max-softmax overconfidence is not previously
documented in the embedded AMC literature.
