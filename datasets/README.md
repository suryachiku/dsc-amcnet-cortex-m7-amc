# Datasets

## RML2016.10a

**Source:** DeepSig Open Data  
**URL:** https://www.deepsig.ai/datasets  
**File:** `RML2016.10a_dict.dat` (pickled Python dict, ~55 MB)

### Contents

| Property | Value |
|---|---|
| Modulation classes | 11 (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM) |
| SNR range | −20 to +18 dB (step 2 dB) |
| Samples per class/SNR | 1,000 |
| Total samples | 220,000 |
| Sample format | Complex IQ, 128 samples → shape (2, 128) float32 |

### Acquisition

1. Download `RML2016.10a_dict.dat` from [DeepSig](https://www.deepsig.ai/datasets).
2. Place it in this directory:
   ```
   datasets/RML2016.10a_dict.dat
   ```
3. Verify:
   ```bash
   python datasets/prepare_dataset.py \
       datasets/RML2016.10a_dict.dat \
       datasets/split_indices.npz
   ```
   Expected output:
   ```
   Total samples : 220,000
   Unique classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   SNR range     : -20 to 18 dB
   Locked split:
     Train : 154,000
     Val   : 33,000
     Test  : 33,000
   ```

## Dataset Split (split_indices.npz)

The train/val/test split is **locked** and archived on Zenodo:

> **DOI:** https://doi.org/10.5281/zenodo.XXXXXXX

Download `split_indices.npz` from Zenodo and place it in this directory.

The split was generated with a fixed random seed (seed = 42) and stratified
sampling to ensure equal class/SNR representation across splits. It must not
be regenerated — doing so will produce different accuracy numbers than those
reported in the paper.

### Why is the split locked?

Without a fixed split, two researchers training the same model may evaluate
on different samples, making results incomparable. Archiving the exact index
arrays on Zenodo allows anyone to independently verify all claims in the paper
using precisely the test set the authors used.

## RML2018.01a (used for comparison only)

RML2018.01a is referenced in related work comparisons (Table I) but is **not
used for training or evaluation** in this paper's experiments. If you intend
to compare against models trained on RML2018.01a, note that direct accuracy
comparisons are invalid due to different class sets and sample counts.
