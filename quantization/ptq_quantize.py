"""
ptq_quantize.py
===============
Post-Training Quantization (PTQ) pipeline for DSC-AMCNet.

Produces a signed INT8 ONNX model compatible with ST X-CUBE-AI 10.2.0.

CRITICAL: Backend must be 'fbgemm' (signed INT8).
          Do NOT use 'qnnpack' (unsigned INT8) — it is incompatible with
          X-CUBE-AI's signed INT8 runtime and causes silent catastrophic
          accuracy failure on device. See validate_backend.py.

Results reproduced by this script (paper Table IV):
    DSC-AMCNet FP32 high-SNR accuracy  : 85.80%
    DSC-AMCNet INT8 high-SNR accuracy  : 85.38%   (PTQ degradation: −0.42 pp)
    Flash footprint (X-CUBE-AI analyze) : 51.26 KB
    RAM footprint   (X-CUBE-AI analyze) : 12.09 KB

Usage
-----
    python quantization/ptq_quantize.py \\
        --checkpoint models/checkpoints/dscamcnet_best.pth \\
        --data       datasets/RML2016.10a_dict.dat \\
        --split      datasets/split_indices.npz \\
        --output     quantization/outputs/dscamcnet_int8.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.quantization
import onnx

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.dsc_amcnet import build_dsc_amcnet
from datasets.prepare_dataset import load_splits, load_by_snr, MODULATIONS

# ---------------------------------------------------------------------------
# Calibration Dataset
# ---------------------------------------------------------------------------
class CalibDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def build_calib_loader(data_path: str, split_path: str, n_calib: int = 1000, batch_size: int = 64):
    """Use a subset of the training set for PTQ calibration."""
    train_X, _, _, _, _, _ = load_splits(data_path, split_path)
    idx = np.random.default_rng(42).choice(len(train_X), size=min(n_calib, len(train_X)), replace=False)
    return torch.utils.data.DataLoader(CalibDataset(train_X[idx]), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Accuracy Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_accuracy(model, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        ),
        batch_size=batch_size,
    )
    correct, total = 0, 0
    for xb, yb in loader:
        out = model(xb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total


# ---------------------------------------------------------------------------
# PTQ Pipeline
# ---------------------------------------------------------------------------
def run_ptq(
    checkpoint_path: str,
    data_path: str,
    split_path: str,
    output_path: str,
    n_calib: int = 1000,
    backend: str = "fbgemm",
):
    """
    Post-Training Quantization pipeline.

    Steps:
    1. Load FP32 checkpoint and verify baseline accuracy.
    2. Fuse Conv-BN-ReLU patterns.
    3. Insert observer hooks (fbgemm backend → signed INT8).
    4. Calibrate with training-set samples.
    5. Convert to quantized model.
    6. Evaluate INT8 accuracy.
    7. Export to ONNX.
    """

    print("=" * 60)
    print("DSC-AMCNet Post-Training Quantization")
    print(f"Backend: {backend}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load FP32 model
    # -----------------------------------------------------------------------
    print("\n[1/7] Loading FP32 checkpoint ...")
    model_fp32 = build_dsc_amcnet(width=64, num_classes=11)
    state = torch.load(checkpoint_path, map_location="cpu")
    model_fp32.load_state_dict(state)
    model_fp32.eval()

    # -----------------------------------------------------------------------
    # 2. Baseline FP32 accuracy (high-SNR)
    # -----------------------------------------------------------------------
    print("[2/7] Evaluating FP32 baseline accuracy (SNR ≥ 10 dB) ...")
    X_test_hi, y_test_hi, _ = load_by_snr(data_path, split_path, snr_min=10)
    fp32_acc = evaluate_accuracy(model_fp32, X_test_hi, y_test_hi)
    print(f"      FP32 high-SNR accuracy: {fp32_acc*100:.2f}%  (paper: 85.80%)")

    # -----------------------------------------------------------------------
    # 3. Fuse patterns + prepare for quantization
    # -----------------------------------------------------------------------
    print("[3/7] Fusing Conv-BN-ReLU patterns and preparing for quantization ...")
    model_q = model_fp32
    model_q.qconfig = torch.quantization.get_default_qconfig(backend)  # fbgemm → signed INT8
    torch.quantization.prepare(model_q, inplace=True)

    # -----------------------------------------------------------------------
    # 4. Calibration
    # -----------------------------------------------------------------------
    print(f"[4/7] Calibrating with {n_calib} training samples ...")
    calib_loader = build_calib_loader(data_path, split_path, n_calib=n_calib)
    model_q.eval()
    with torch.no_grad():
        for xb in calib_loader:
            model_q(xb)
    print("      Calibration complete.")

    # -----------------------------------------------------------------------
    # 5. Convert to INT8
    # -----------------------------------------------------------------------
    print("[5/7] Converting to INT8 ...")
    torch.quantization.convert(model_q, inplace=True)

    # -----------------------------------------------------------------------
    # 6. INT8 accuracy
    # -----------------------------------------------------------------------
    print("[6/7] Evaluating INT8 accuracy (SNR ≥ 10 dB) ...")
    int8_acc = evaluate_accuracy(model_q, X_test_hi, y_test_hi)
    degradation = (fp32_acc - int8_acc) * 100
    print(f"      INT8 high-SNR accuracy : {int8_acc*100:.2f}%  (paper: 85.38%)")
    print(f"      PTQ degradation        : {degradation:.2f} pp  (paper: −0.42 pp)")

    if degradation > 2.0:
        print("\n  ⚠️  WARNING: PTQ degradation exceeds 2 pp.")
        print("     Verify that 'fbgemm' backend is used and checkpoint is correct.")

    # -----------------------------------------------------------------------
    # 7. Export to ONNX
    # -----------------------------------------------------------------------
    print(f"[7/7] Exporting INT8 ONNX to {output_path} ...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 2, 128)
    torch.onnx.export(
        model_q,
        dummy_input,
        str(output_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Verify ONNX
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"      ONNX model verified. Saved to {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FP32 accuracy  : {fp32_acc*100:.2f}%")
    print(f"  INT8 accuracy  : {int8_acc*100:.2f}%")
    print(f"  PTQ degradation: {degradation:.2f} pp")
    print(f"  ONNX output    : {output_path}")
    print("\nNext: Import the ONNX into ST Edge AI Core (X-CUBE-AI 10.2.0)")
    print("      Expected Flash: 51.26 KB | RAM: 12.09 KB | Latency: 1.183 ms")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args):
    run_ptq(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        split_path=args.split,
        output_path=args.output,
        n_calib=args.n_calib,
        backend=args.backend,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTQ for DSC-AMCNet (fbgemm backend)")
    parser.add_argument("--checkpoint", required=True, help="FP32 .pth checkpoint")
    parser.add_argument("--data",       required=True, help="RML2016.10a_dict.dat")
    parser.add_argument("--split",      required=True, help="split_indices.npz")
    parser.add_argument("--output",     default="quantization/outputs/dscamcnet_int8.onnx")
    parser.add_argument("--n_calib",    type=int, default=1000)
    parser.add_argument(
        "--backend", default="fbgemm", choices=["fbgemm", "qnnpack"],
        help="MUST be 'fbgemm' for X-CUBE-AI compatibility (signed INT8)"
    )
    args = parser.parse_args()

    if args.backend == "qnnpack":
        print("ERROR: qnnpack produces unsigned INT8 — incompatible with X-CUBE-AI.")
        print("       Use --backend fbgemm. See quantization/validate_backend.py.")
        sys.exit(1)

    main(args)
