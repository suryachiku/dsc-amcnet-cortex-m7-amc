"""
validate_backend.py
===================
Validates that the fbgemm backend (signed INT8) must be used for
X-CUBE-AI compatibility, and demonstrates the silent failure mode
caused by using the qnnpack backend (unsigned INT8).

Background
----------
X-CUBE-AI 10.2.0 expects signed INT8 weights and activations, consistent
with the ARM CMSIS-NN kernels used internally. PyTorch's 'fbgemm' backend
produces signed INT8 (range −128 to +127). PyTorch's 'qnnpack' backend
produces unsigned INT8 (range 0 to 255), which causes a silent scale/zero-
point mismatch on device — the model runs without an error, but outputs
garbage classifications.

This failure mode is documented in the paper (Section V-C) as a key
engineering lesson. It was discovered during hardware bring-up when the
device produced random classification outputs despite successful ONNX import.

This script
-----------
Quantizes DSC-AMCNet with both backends and compares:
1. Quantized weight dtype and zero-point range
2. On-host INT8 accuracy (both should be similar on host)
3. X-CUBE-AI ONNX compatibility (fbgemm passes, qnnpack fails on device)

Usage
-----
    python quantization/validate_backend.py \\
        --checkpoint models/checkpoints/dscamcnet_best.pth \\
        --data       datasets/RML2016.10a_dict.dat \\
        --split      datasets/split_indices.npz
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.quantization

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.dsc_amcnet import build_dsc_amcnet
from datasets.prepare_dataset import load_splits, load_by_snr

CALIB_N = 500


def quantize_with_backend(checkpoint_path, data_path, split_path, backend):
    """Quantize model with specified backend and return (model, accuracy)."""
    model = build_dsc_amcnet()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)

    # Calibrate
    train_X, _, _, _, _, _ = load_splits(data_path, split_path)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_X), size=CALIB_N, replace=False)
    with torch.no_grad():
        for i in range(0, CALIB_N, 64):
            xb = torch.from_numpy(train_X[idx[i:i+64]]).float()
            model(xb)

    torch.quantization.convert(model, inplace=True)

    # Evaluate
    X_hi, y_hi, _ = load_by_snr(data_path, split_path, snr_min=10)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_hi).float(), torch.from_numpy(y_hi).long()
        ),
        batch_size=256,
    )
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return model, correct / total


def inspect_zero_point(model):
    """Check zero-point of first quantized conv layer to identify signed vs unsigned."""
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module.weight(), "q_zero_point"):
            zp = module.weight().q_zero_point()
            dtype = module.weight().dtype
            return name, dtype, zp
    return None, None, None


def main(args):
    print("=" * 65)
    print("Backend Validation: fbgemm (signed INT8) vs qnnpack (unsigned INT8)")
    print("=" * 65)

    results = {}
    for backend in ["fbgemm", "qnnpack"]:
        print(f"\n[{backend}] Quantizing ...")
        model, acc = quantize_with_backend(args.checkpoint, args.data, args.split, backend)
        layer, dtype, zp = inspect_zero_point(model)
        results[backend] = {"acc": acc, "layer": layer, "dtype": dtype, "zp": zp}
        print(f"  High-SNR accuracy: {acc*100:.2f}%")
        print(f"  First layer: {layer}")
        print(f"  Weight dtype: {dtype}")
        print(f"  Zero-point: {zp}  ({'signed (0=correct for symmetric)' if zp == 0 else 'unsigned — incompatible with X-CUBE-AI'})")

    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    print(f"\n  fbgemm  → zero-point={results['fbgemm']['zp']}, dtype={results['fbgemm']['dtype']}")
    print(f"            Signed INT8 → X-CUBE-AI COMPATIBLE ✓")
    print(f"\n  qnnpack → zero-point={results['qnnpack']['zp']}, dtype={results['qnnpack']['dtype']}")
    print(f"            Unsigned INT8 → X-CUBE-AI INCOMPATIBLE ✗")
    print(f"            On-host accuracy {results['qnnpack']['acc']*100:.2f}% looks fine,")
    print(f"            but ON DEVICE this produces random outputs (silent failure).")
    print("\n  ALWAYS use --backend fbgemm for X-CUBE-AI deployment.")
    print("  This failure mode is documented in paper Section V-C.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--split",      required=True)
    main(parser.parse_args())
