# Pre-trained Checkpoints

Pre-trained model weights and the locked dataset split are archived on Zenodo
to ensure long-term, citable availability independent of this repository.

## Download

**Zenodo Archive:** https://doi.org/10.5281/zenodo.XXXXXXX

Download and place files as follows:

```
models/checkpoints/
├── dscamcnet_best.pth         ← DSC-AMCNet FP32 weights (from Zenodo)
├── dscamcnet_int8.onnx        ← INT8 ONNX (fbgemm backend, X-CUBE-AI compatible)
└── ulcnn_simplified_fp32.pth  ← ULCNN-simplified baseline weights
datasets/
└── split_indices.npz          ← Locked train/val/test split (from Zenodo)
```

## Integrity Verification

After downloading, verify SHA-256 checksums:

| File | SHA-256 |
|------|---------|
| `dscamcnet_best.pth` | `[checksum here after generation]` |
| `dscamcnet_int8.onnx` | `[checksum here after generation]` |
| `split_indices.npz` | `[checksum here after generation]` |

Verify on Linux/macOS:
```bash
sha256sum models/checkpoints/dscamcnet_best.pth
sha256sum datasets/split_indices.npz
```

Verify on Windows (PowerShell):
```powershell
Get-FileHash models\checkpoints\dscamcnet_best.pth -Algorithm SHA256
Get-FileHash datasets\split_indices.npz -Algorithm SHA256
```

## Notes

- The `split_indices.npz` is **locked** — it defines the exact train/val/test partition
  used in all paper experiments. Using a different split will produce different accuracy
  numbers. This is expected.
- The INT8 ONNX was generated with the `fbgemm` backend (signed INT8). Loading it
  with `qnnpack` (unsigned INT8) will cause silent accuracy failure on device.
  See `quantization/validate_backend.py` for verification.
