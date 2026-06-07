"""
train.py
========
Training script for DSC-AMCNet on RML2016.10a.

Reproduces the FP32 baseline reported in the paper:
    High-SNR (≥ 10 dB) accuracy : 85.80%
    STM32 FP32 platform parity  : 85.78% (−0.02 pp delta vs laptop)

Usage
-----
    python models/train.py \\
        --data   datasets/RML2016.10a_dict.dat \\
        --split  datasets/split_indices.npz \\
        --epochs 100 \\
        --output models/checkpoints/dscamcnet_best.pth

Locked split
------------
The dataset split (split_indices.npz) is fixed and archived on Zenodo
alongside model weights. Using a different split will alter reported
accuracy. Always load the locked split when reproducing paper results.
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from dsc_amcnet import build_dsc_amcnet

# ---------------------------------------------------------------------------
# RML2016.10a Dataset
# ---------------------------------------------------------------------------
MODULATIONS = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
]
SNR_VALUES = list(range(-20, 20, 2))   # -20, -18, ..., 18 dB
HIGH_SNR_THRESHOLD = 10                # dB


class RML2016Dataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        # data: (N, 2, 128) float32, labels: (N,) int64
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_rml2016(data_path: str, split_path: str):
    """
    Load RML2016.10a and apply the locked dataset split.

    Returns train_dataset, val_dataset, test_dataset
    """
    print(f"Loading dataset from {data_path} ...")
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f, encoding="latin1")

    samples, labels = [], []
    for (mod, snr), iq in data_dict.items():
        if mod not in MODULATIONS:
            continue
        label = MODULATIONS.index(mod)
        for sample in iq:
            samples.append(sample)       # (2, 128)
            labels.append(label)

    X = np.array(samples, dtype=np.float32)  # (N, 2, 128)
    y = np.array(labels, dtype=np.int64)

    print(f"  Total samples: {len(X):,}")

    # Apply locked split
    split = np.load(split_path)
    train_idx = split["train_indices"]
    val_idx   = split["val_indices"]
    test_idx  = split["test_indices"]

    print(f"  Split — Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

    return (
        RML2016Dataset(X[train_idx], y[train_idx]),
        RML2016Dataset(X[val_idx],   y[val_idx]),
        RML2016Dataset(X[test_idx],  y[test_idx]),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_ds, val_ds, test_ds = load_rml2016(args.data, args.split)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Build model
    model = build_dsc_amcnet(width=64, num_classes=11).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    print(f"\nTraining for {args.epochs} epochs ...\n")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
                f"Val loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
                f"Best {best_val_acc*100:.2f}%"
            )

    print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved to {output_path}")

    # Final test evaluation
    model.load_state_dict(torch.load(output_path, map_location=device))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy (all SNR): {test_acc*100:.2f}%")
    print("\nExpected high-SNR (≥ 10 dB) accuracy: 85.80%  [see paper Table II]")
    print("Run evaluate_highsnr.py to compute per-SNR breakdown.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DSC-AMCNet on RML2016.10a")
    parser.add_argument("--data",       required=True, help="Path to RML2016.10a_dict.dat")
    parser.add_argument("--split",      required=True, help="Path to split_indices.npz")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--output",     default="models/checkpoints/dscamcnet_best.pth")
    args = parser.parse_args()
    main(args)
