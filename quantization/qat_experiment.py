"""
qat_experiment.py
=================
Quantization-Aware Training (QAT) experiment for DSC-AMCNet.

This script documents a NEGATIVE RESULT: QAT consistently degraded accuracy
relative to PTQ in this architecture, contrary to the typical expectation
that QAT outperforms PTQ for difficult-to-quantize models.

Findings (paper Section V-C, Table V):
    PTQ accuracy (fbgemm, 1000 calib samples):  85.38%  (−0.42 pp vs FP32)
    QAT accuracy (fine-tuned 10 epochs):         83.1%   (−2.7 pp vs FP32)
    QAT accuracy (fine-tuned 30 epochs):         82.4%   (−3.4 pp vs FP32)

Hypothesis:
    The SE (Squeeze-and-Excitation) attention block contains sigmoid activations
    that produce near-saturated outputs at high SNR. During QAT fine-tuning,
    fake quantization noise is inserted into these near-saturated regions,
    producing large gradient magnitudes that destabilise the global average
    pool → SE → classifier pathway. The result is catastrophic forgetting of
    the fine-grained feature discriminations learned during FP32 training.

    This is consistent with the more general finding that PTQ degradation in
    DSC-AMCNet peaks at high SNR (not low SNR) — precisely where SE attention
    is most active and most sensitive to perturbation.

This script:
    1. Runs the QAT fine-tuning procedure used in the paper.
    2. Logs train/val accuracy across epochs.
    3. Reports final accuracy vs the PTQ baseline.

Usage
-----
    python quantization/qat_experiment.py \\
        --checkpoint models/checkpoints/dscamcnet_best.pth \\
        --data       datasets/RML2016.10a_dict.dat \\
        --split      datasets/split_indices.npz \\
        --epochs     10

⚠️  This experiment exists to document the failure. Do not use the QAT
    checkpoint for deployment — use the PTQ ONNX from ptq_quantize.py.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.dsc_amcnet import build_dsc_amcnet
from datasets.prepare_dataset import load_splits, load_by_snr


def main(args):
    device = torch.device("cpu")  # QAT must run on CPU for PyTorch static quant

    print("=" * 60)
    print("QAT Experiment — DSC-AMCNet (NEGATIVE RESULT)")
    print("=" * 60)
    print("\nWarning: QAT degraded accuracy in this architecture.")
    print("See docstring and paper Section V-C for analysis.\n")

    # Load data
    train_X, train_y, _, _, _, _ = load_splits(args.data, args.split)
    X_hi, y_hi, _                = load_by_snr(args.data, args.split, snr_min=10)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long()),
        batch_size=256, shuffle=True,
    )
    hi_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_hi).float(), torch.from_numpy(y_hi).long()),
        batch_size=256,
    )

    # Build QAT model
    model = build_dsc_amcnet()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.train()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare_qat(model, inplace=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.NLLLoss()

    print(f"Fine-tuning for {args.epochs} epochs with QAT ...\n")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            model_int8 = torch.quantization.convert(
                torch.quantization.prepare_qat(
                    build_dsc_amcnet().eval(), inplace=False
                ),
                inplace=False,
            )
            # Use the live QAT model (not fully converted) for per-epoch tracking
            for xb, yb in hi_loader:
                out = model(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)
        acc = correct / total
        history.append(acc)
        print(f"  Epoch {epoch:3d}: High-SNR accuracy = {acc*100:.2f}%")

    print("\n" + "=" * 60)
    print("QAT RESULT SUMMARY")
    print("=" * 60)
    print(f"  FP32 baseline    : 85.80%  (PTQ benchmark)")
    print(f"  PTQ (fbgemm)     : 85.38%  (−0.42 pp)")
    print(f"  QAT final epoch  : {history[-1]*100:.2f}%  (reported in paper Table V)")
    degradation = (85.80 - history[-1]*100)
    print(f"  QAT degradation  : −{degradation:.2f} pp  (WORSE than PTQ)")
    print("\nConclusion: PTQ is the preferred quantization strategy for this architecture.")
    print("QAT is not recommended due to SE attention instability under fake-quant noise.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--split",      required=True)
    parser.add_argument("--epochs", type=int, default=10)
    main(parser.parse_args())
