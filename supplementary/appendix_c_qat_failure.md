# Appendix C: QAT Failure Analysis

Quantization-Aware Training (QAT) consistently degraded DSC-AMCNet accuracy
relative to Post-Training Quantization (PTQ). This appendix documents the
experimental evidence and provides a mechanistic hypothesis.

This is a **negative result** and is reported as such in the paper (Section V-C).
Suppressing negative results weakens reproducibility and misleads future researchers
who may attempt QAT on SE-attention architectures for AMC.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Starting checkpoint | FP32 best (85.80% high-SNR) |
| QAT backend | fbgemm (signed INT8) |
| Fine-tuning optimizer | Adam, lr=1e-4 |
| Fine-tuning schedule | 10 epochs, 30 epochs |
| Batch size | 256 |
| Calibration data | Training split (locked) |
| Evaluation | High-SNR test set (SNR ≥ 10 dB) |

---

## Results

| Method | High-SNR Accuracy | Δ vs FP32 |
|---|---|---|
| FP32 baseline | 85.80% | — |
| PTQ (fbgemm, 1000 calib) | 85.38% | **−0.42 pp** |
| QAT (10 epochs) | ~83.1% | −2.7 pp |
| QAT (30 epochs) | ~82.4% | −3.4 pp |

QAT with 30 epochs of fine-tuning produces worse accuracy than fresh PTQ
with just 1000 calibration samples. This is the opposite of the typical
expectation that gradient-based QAT should outperform static PTQ.

---

## Mechanistic Hypothesis

### 1. SE Attention Sigmoid Saturation

The SE block contains sigmoid activations applied to the channel attention
vector. At high SNR, where DSC-AMCNet achieves its best accuracy, the SE
attention outputs are near-saturated: values cluster near 0 or 1.

During QAT, fake quantization nodes are inserted, adding quantisation noise
to these near-saturated regions. Even small perturbations in the sigmoid
input near the saturation boundary produce large gradient magnitudes (the
chain rule through a saturated sigmoid is unstable). This destabilises the
SE → classifier pathway during fine-tuning.

### 2. PTQ Degradation Is Worst at High SNR

A counterintuitive finding: PTQ accuracy loss peaks at high SNR, not low SNR.
At low SNR, coarse INT8 quantisation has little additional impact because
accuracy is already limited by signal quality. At high SNR, INT8 quantisation
noise becomes the binding constraint, disrupting fine-grained feature
discrimination precisely where the model has learned the most detail.

QAT, by inserting artificial quantisation noise into an already-strained
high-SNR operating regime, amplifies this problem rather than resolving it.

### 3. Training Distribution Mismatch Under Fake-Quant

PTQ calibration uses real training-set statistics to set observer
scale/zero-point. QAT uses fake quantisation: differentiable approximations
to quantisation that perturb activations during the forward pass. The fake-
quant noise does not faithfully reproduce the INT8 rounding characteristics
of fbgemm at inference time, creating a training/deployment mismatch that
is larger for architectures with nonlinear attention paths (SE) than for
pure-convolutional networks.

---

## Comparison with Prior Work

Published QAT results on AMC architectures (e.g., LSTM-based, pure CNN)
typically show QAT outperforming PTQ when:
- The model is sensitive to quantisation (large activation range, many bits).
- Fine-tuning data is plentiful relative to model size.
- The architecture lacks sharp nonlinearities (sigmoid/tanh) in the quantised path.

DSC-AMCNet fails the third condition. SE attention introduces sigmoid gates
that are both sharp and near-saturated at the operating point of interest.

---

## Recommendation

**Use PTQ (fbgemm, 1000 calibration samples) for deployment.**

QAT is not recommended for SE-attention AMC architectures trained on synthetic
datasets. Future work may explore:
- QAT with frozen SE attention (only quantise convolutional stages).
- Knowledge distillation from FP32 teacher to INT8 student.
- Learnable quantisation boundaries (PACT, LSQ) adapted to sigmoid outputs.

---

## Reproducibility

Run the QAT experiment to reproduce Table V:

```bash
python quantization/qat_experiment.py \
    --checkpoint models/checkpoints/dscamcnet_best.pth \
    --data       datasets/RML2016.10a_dict.dat \
    --split      datasets/split_indices.npz \
    --epochs     10
```

Expected output:
```
QAT final epoch: ~83.1%  (−2.7 pp vs FP32)
Conclusion: PTQ is the preferred quantization strategy.
```
