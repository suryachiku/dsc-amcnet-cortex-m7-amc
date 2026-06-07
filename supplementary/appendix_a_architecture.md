# Appendix A: DSC-AMCNet Architecture Specification

Full layer-by-layer specification of DSC-AMCNet as deployed in the paper.
Corresponds to `models/dsc_amcnet.py` with `width=64`.

---

## Input

| Property | Value |
|---|---|
| Format | (batch, 2, 128) — 2-channel (I/Q), 128 time samples |
| Dtype | float32 (FP32 training), int8 (deployment) |
| Source | RML2016.10a 128-sample complex IQ windows |
| Normalisation | None applied (raw RML2016.10a values) |

---

## Architecture Overview

```
Input (B, 2, 128)
    │
    ▼
Stem: Conv1d(2→64, k=7, pad=3) → BN → ReLU → MaxPool1d(2)
    │ (B, 64, 64)
    ▼
Stage 1: DSCBlock(64→128, k=5, pad=2) → MaxPool1d(2)
    │ (B, 128, 32)
    ▼
Stage 2: DSCBlock(128→128, k=3) → DSCBlock(128→256, k=3) → MaxPool1d(2)
    │ (B, 256, 16)
    ▼
Stage 3: DSCBlock(256→256, k=3) → DSCBlock(256→256, k=3)
    │ (B, 256, 16)
    ▼
SE Block (channel attention, reduction=4)
    │ (B, 256, 16)
    ▼
Global Average Pool → (B, 256)
    ▼
Linear(256→128) → ReLU → Dropout(0.5)
    ▼
Linear(128→11) → LogSoftmax
    │
Output (B, 11)
```

---

## Layer-by-Layer Table

| Layer | Type | In Shape | Out Shape | Params | Notes |
|---|---|---|---|---|---|
| stem.0 | Conv1d | (B,2,128) | (B,64,128) | 896 | k=7, pad=3, no bias |
| stem.1 | BatchNorm1d | (B,64,128) | (B,64,128) | 128 | |
| stem.2 | ReLU | — | — | 0 | |
| stem.3 | MaxPool1d | (B,64,128) | (B,64,64) | 0 | k=2, stride=2 |
| stage1.0.depthwise | Conv1d | (B,64,64) | (B,64,64) | 320 | k=5, groups=64 |
| stage1.0.bn1 | BatchNorm1d | — | — | 128 | |
| stage1.0.pointwise | Conv1d | (B,64,64) | (B,128,64) | 8,192 | k=1 |
| stage1.0.bn2 | BatchNorm1d | — | — | 256 | |
| stage1.1 | MaxPool1d | (B,128,64) | (B,128,32) | 0 | k=2, stride=2 |
| stage2.0.depthwise | Conv1d | (B,128,32) | (B,128,32) | 384 | k=3, groups=128 |
| stage2.0.pointwise | Conv1d | (B,128,32) | (B,128,32) | 16,384 | k=1 |
| stage2.1.depthwise | Conv1d | (B,128,32) | (B,128,32) | 384 | k=3, groups=128 |
| stage2.1.pointwise | Conv1d | (B,128,32) | (B,256,32) | 32,768 | k=1 |
| stage2.2 | MaxPool1d | (B,256,32) | (B,256,16) | 0 | k=2, stride=2 |
| stage3.0.depthwise | Conv1d | (B,256,16) | (B,256,16) | 768 | k=3, groups=256 |
| stage3.0.pointwise | Conv1d | (B,256,16) | (B,256,16) | 65,536 | k=1 |
| stage3.1.depthwise | Conv1d | (B,256,16) | (B,256,16) | 768 | k=3, groups=256 |
| stage3.1.pointwise | Conv1d | (B,256,16) | (B,256,16) | 65,536 | k=1 |
| se.fc1 | Linear | (B,256) | (B,64) | 16,384 | no bias |
| se.fc2 | Linear | (B,64) | (B,256) | 16,384 | no bias |
| gap | AdaptiveAvgPool1d | (B,256,16) | (B,256,1) | 0 | |
| fc1 | Linear | (B,256) | (B,128) | 32,896 | with bias |
| dropout | Dropout | — | — | 0 | p=0.5 |
| fc2 | Linear | (B,128) | (B,11) | 1,419 | with bias |
| logsoftmax | LogSoftmax | (B,11) | (B,11) | 0 | dim=1 |

**Total trainable parameters: ~258,000** (varies slightly with BN buffers)

---

## DSCBlock Internal Structure

Each `DSCBlock(in_ch, out_ch, k)` expands as:

```
DepthwiseConv1d(in_ch, in_ch, k, groups=in_ch, bias=False)
BatchNorm1d(in_ch)
ReLU
PointwiseConv1d(in_ch, out_ch, k=1, bias=False)
BatchNorm1d(out_ch)
ReLU
```

Depthwise + pointwise decomposition reduces MACs relative to a full `Conv1d(in_ch, out_ch, k)` by a factor of:

```
reduction = 1/out_ch + 1/k
```

For the largest stage (256→256, k=3): ~3× MAC reduction.

---

## SE (Squeeze-and-Excitation) Block

```
Input:  (B, C, L)   — C=256
GAP:    mean over L → (B, C)
FC1:    Linear(C, C//4) → ReLU           [256 → 64]
FC2:    Linear(C//4, C) → Sigmoid        [64 → 256]
Scale:  multiply input × attention        (broadcast over L)
Output: (B, C, L)
```

Reduction ratio = 4. Selected via ablation (Table III of paper).

**⚠️ Important finding:** SE attention causes max-softmax overconfidence on OOD inputs.
All OOD samples (OpenSetRF, non-RML2016 signals) receive softmax confidence = 1.000.
AUC for anomaly detection = 0.000. See Appendix C and paper Section IX.

---

## Quantisation Target

| Property | Value |
|---|---|
| Framework | PyTorch 2.x static quantisation |
| Backend | fbgemm (signed INT8) |
| Calibration | 1000 training samples |
| ONNX opset | 13 |
| X-CUBE-AI version | 10.2.0 (ST Edge AI Core 2.2.0) |
| Flash (Weights) | 51.26 KB |
| RAM (Activations) | 12.09 KB |
| Latency (INT8, 550 MHz) | 1.183 ms |

---

## Width Ablation (Pareto Sweep)

Table III of the paper reports the width multiplier sweep. Summary:

| Width | Flash (KB) | RAM (KB) | High-SNR Acc. | Latency (ms) |
|---|---|---|---|---|
| 32 | 14.1 | 4.2 | 80.4% | 0.41 |
| 48 | 30.8 | 8.1 | 83.7% | 0.79 |
| **64** | **51.26** | **12.09** | **85.80%** | **1.183** |
| 96 | 112.3 | 26.4 | 85.9% | 2.61 |

Width=64 is the Pareto-optimal operating point. Increasing to 96 yields
only 0.1 pp accuracy gain at 2.2× the latency and 2.2× the RAM.
