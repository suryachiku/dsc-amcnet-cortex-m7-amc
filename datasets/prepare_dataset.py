"""
prepare_dataset.py
==================
Dataset loading utilities for RML2016.10a.

Loads the raw .dat file and the locked split_indices.npz to produce
consistent train/val/test subsets for all experiments.

The locked split (split_indices.npz) is archived on Zenodo alongside
model weights. Download from: https://doi.org/10.5281/zenodo.XXXXXXX

Usage
-----
    from datasets.prepare_dataset import load_splits, MODULATIONS, SNR_VALUES
    train_X, train_y, val_X, val_y, test_X, test_y = load_splits(
        data_path="datasets/RML2016.10a_dict.dat",
        split_path="datasets/split_indices.npz",
    )
"""

import pickle
import numpy as np
from pathlib import Path

# RML2016.10a canonical modulation class ordering
# (alphabetical, as used in all experiments)
MODULATIONS = [
    "8PSK",     # class 0
    "AM-DSB",   # class 1
    "AM-SSB",   # class 2
    "BPSK",     # class 3
    "CPFSK",    # class 4
    "GFSK",     # class 5
    "PAM4",     # class 6
    "QAM16",    # class 7
    "QAM64",    # class 8
    "QPSK",     # class 9
    "WBFM",     # class 10
]

# SNR values present in RML2016.10a
SNR_VALUES = list(range(-20, 20, 2))   # -20, -18, ..., +18 dB

# High-SNR threshold used for accuracy reporting in paper
HIGH_SNR_THRESHOLD = 10    # dB


def load_raw(data_path: str):
    """
    Load RML2016.10a and return flat arrays.

    Returns
    -------
    X   : np.ndarray, shape (N, 2, 128), dtype float32
    y   : np.ndarray, shape (N,),       dtype int64
    snr : np.ndarray, shape (N,),       dtype int32   (dB)
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Download RML2016.10a from https://www.deepsig.ai/datasets"
        )

    with open(data_path, "rb") as f:
        data_dict = pickle.load(f, encoding="latin1")

    samples, labels, snrs = [], [], []
    for (mod, snr), iq in data_dict.items():
        if mod not in MODULATIONS:
            continue
        label = MODULATIONS.index(mod)
        for sample in iq:
            samples.append(sample)
            labels.append(label)
            snrs.append(snr)

    X   = np.array(samples, dtype=np.float32)
    y   = np.array(labels,  dtype=np.int64)
    snr = np.array(snrs,    dtype=np.int32)
    return X, y, snr


def load_splits(data_path: str, split_path: str):
    """
    Load dataset and apply the locked train/val/test split.

    Returns
    -------
    train_X, train_y, val_X, val_y, test_X, test_y
    (numpy arrays, same dtype as load_raw)
    """
    X, y, _ = load_raw(data_path)

    split = np.load(split_path)
    train_idx = split["train_indices"]
    val_idx   = split["val_indices"]
    test_idx  = split["test_indices"]

    assert max(train_idx.max(), val_idx.max(), test_idx.max()) < len(X), \
        "split_indices.npz references out-of-range sample indices — wrong dataset?"

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


def load_by_snr(data_path: str, split_path: str, snr_min: int = 10):
    """
    Return test-set samples filtered to SNR >= snr_min.
    Used for high-SNR accuracy evaluation (paper Table II).
    """
    X, y, snr = load_raw(data_path)
    split = np.load(split_path)
    test_idx = split["test_indices"]
    X_test, y_test, snr_test = X[test_idx], y[test_idx], snr[test_idx]

    mask = snr_test >= snr_min
    return X_test[mask], y_test[mask], snr_test[mask]


if __name__ == "__main__":
    import sys

    data_path  = sys.argv[1] if len(sys.argv) > 1 else "datasets/RML2016.10a_dict.dat"
    split_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/split_indices.npz"

    X, y, snr = load_raw(data_path)
    print(f"Total samples : {len(X):,}")
    print(f"Unique classes: {np.unique(y).tolist()} → {MODULATIONS}")
    print(f"SNR range     : {snr.min()} to {snr.max()} dB")

    train_X, train_y, val_X, val_y, test_X, test_y = load_splits(data_path, split_path)
    print(f"\nLocked split:")
    print(f"  Train : {len(train_X):,}")
    print(f"  Val   : {len(val_X):,}")
    print(f"  Test  : {len(test_X):,}")
