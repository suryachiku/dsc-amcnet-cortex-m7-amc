"""
analyze_flash_ram.py
====================
Parses ST Edge AI Core (stedgeai) Analyze output and extracts Flash and
RAM footprint figures for the paper.

The `stedgeai analyze` command produces a report like:
    Network memory footprints:
     Weights (ro)   :    51,264 bytes (51.26 KiB)
     Activations    :    12,380 bytes (12.09 KiB)

This script:
1. Runs `stedgeai analyze` on the ONNX.
2. Parses Flash (weights) and RAM (activations) from the output.
3. Verifies against expected paper values.
4. Writes a JSON report.

Usage
-----
    # Run analysis and parse:
    python hardware_characterization/analyze_flash_ram.py \\
        --onnx  quantization/outputs/dscamcnet_int8.onnx \\
        --target stm32h7 \\
        --output results/flash_ram_report.json

    # Parse a pre-existing stedgeai log file:
    python hardware_characterization/analyze_flash_ram.py \\
        --log results/stedgeai_analyze_output.txt \\
        --output results/flash_ram_report.json

Paper values (locked, silicon-measured):
    DSC-AMCNet INT8  : Flash = 51.26 KB, RAM = 12.09 KB
    ULCNN-simp INT8  : Flash = 44.26 KB, RAM = 36.78 KB
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Expected values from paper (Table IV)
EXPECTED = {
    "dscamcnet": {"flash_kb": 51.26, "ram_kb": 12.09},
    "ulcnn":     {"flash_kb": 44.26, "ram_kb": 36.78},
}
TOLERANCE_KB = 0.5   # Acceptable delta (rounding differences across tool versions)


def run_stedgeai(onnx_path: str, target: str) -> str:
    """Run stedgeai analyze and capture output."""
    cmd = [
        "stedgeai", "analyze",
        "--target", target,
        "--name", "network",
        str(onnx_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout + result.stderr
    except FileNotFoundError:
        print("ERROR: 'stedgeai' not found in PATH.")
        print("Install ST Edge AI Core 2.2.0 and add it to PATH.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: stedgeai timed out after 120 s.")
        sys.exit(1)


# Patterns for different stedgeai output formats
FLASH_PATTERNS = [
    re.compile(r"Weights\s*\(ro\)\s*:\s*([\d,]+)\s*bytes\s*\(([\d.]+)\s*KiB\)", re.IGNORECASE),
    re.compile(r"ROM\s*:\s*([\d,]+)\s*bytes\s*\(([\d.]+)\s*KiB\)", re.IGNORECASE),
    re.compile(r"weights.*?:\s*([\d,]+)\s*bytes", re.IGNORECASE),
]
RAM_PATTERNS = [
    re.compile(r"Activations\s*:\s*([\d,]+)\s*bytes\s*\(([\d.]+)\s*KiB\)", re.IGNORECASE),
    re.compile(r"RAM\s*:\s*([\d,]+)\s*bytes\s*\(([\d.]+)\s*KiB\)", re.IGNORECASE),
    re.compile(r"activations.*?:\s*([\d,]+)\s*bytes", re.IGNORECASE),
]


def parse_output(text: str) -> dict:
    flash_bytes, flash_kb = None, None
    ram_bytes, ram_kb = None, None

    for pat in FLASH_PATTERNS:
        m = pat.search(text)
        if m:
            flash_bytes = int(m.group(1).replace(",", ""))
            flash_kb = float(m.group(2)) if len(m.groups()) > 1 else flash_bytes / 1024
            break

    for pat in RAM_PATTERNS:
        m = pat.search(text)
        if m:
            ram_bytes = int(m.group(1).replace(",", ""))
            ram_kb = float(m.group(2)) if len(m.groups()) > 1 else ram_bytes / 1024
            break

    return {
        "flash_bytes": flash_bytes,
        "flash_kb": flash_kb,
        "ram_bytes": ram_bytes,
        "ram_kb": ram_kb,
    }


def verify(parsed: dict, model_key: str = "dscamcnet"):
    if model_key not in EXPECTED:
        print(f"No expected values for '{model_key}'. Skipping verification.")
        return

    exp = EXPECTED[model_key]
    ok = True

    for metric, exp_val in exp.items():
        got_val = parsed.get(metric)
        if got_val is None:
            print(f"  ⚠️  Could not parse {metric}")
            ok = False
            continue
        delta = abs(got_val - exp_val)
        status = "✓" if delta <= TOLERANCE_KB else "✗"
        print(f"  {metric}: {got_val:.2f} KB  (paper: {exp_val:.2f} KB, Δ={delta:.2f} KB) {status}")
        if delta > TOLERANCE_KB:
            ok = False

    if ok:
        print("\n  All values within tolerance. Paper claims verified. ✓")
    else:
        print("\n  ⚠️  Values outside tolerance. Recheck ONNX backend and X-CUBE-AI version.")


def main(args):
    if args.log:
        print(f"Parsing existing log: {args.log}")
        with open(args.log) as f:
            text = f.read()
    else:
        text = run_stedgeai(args.onnx, args.target)

    print("\n--- Parsed Footprint ---")
    parsed = parse_output(text)

    if parsed["flash_kb"] is None:
        print("ERROR: Could not parse Flash from output.")
        print("Raw output preview:\n", text[:2000])
        sys.exit(1)

    print(f"  Flash : {parsed['flash_kb']:.2f} KB  ({parsed['flash_bytes']:,} bytes)")
    print(f"  RAM   : {parsed['ram_kb']:.2f} KB  ({parsed['ram_bytes']:,} bytes)")

    print("\n--- Paper Verification ---")
    verify(parsed, model_key=args.model)

    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "onnx": str(args.onnx) if args.onnx else None,
        "target": args.target if args.target else None,
        **parsed,
        "expected": EXPECTED.get(args.model),
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ST Edge AI Core Flash/RAM analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx",  help="ONNX model path (runs stedgeai)")
    group.add_argument("--log",   help="Pre-existing stedgeai output log file")
    parser.add_argument("--target",  default="stm32h7")
    parser.add_argument("--model",   default="dscamcnet", choices=["dscamcnet", "ulcnn"])
    parser.add_argument("--output",  default="results/flash_ram_report.json")
    main(parser.parse_args())
