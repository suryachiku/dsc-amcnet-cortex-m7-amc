"""
measure_latency.py
==================
Reads DWT latency reports from the STM32H723ZG over UART and logs to CSV.

Parses the benchmark output produced by DWT_BenchmarkN() in dwt_measure.c:
    [DWT] N=1000  mean=1.183 ms  min=1.181 ms  max=1.187 ms

Usage
-----
    python hardware_characterization/measure_latency.py \\
        --port   COM7         (Windows) or /dev/ttyACM0 (Linux)
        --baud   115200
        --output results/latency_dscamcnet_int8.csv

Produces a CSV with columns: timestamp, mean_ms, min_ms, max_ms, n_inferences

Paper result:
    DSC-AMCNet INT8 : mean = 1.183 ms @ 550 MHz
    ULCNN-simp INT8 : mean = 2.104 ms @ 550 MHz
    Speedup         : 1.78×
"""

import argparse
import re
import time
import csv
from datetime import datetime
from pathlib import Path

try:
    import serial
except ImportError:
    raise ImportError("Install pyserial: pip install pyserial")


# DWT output pattern from DWT_BenchmarkN()
DWT_PATTERN = re.compile(
    r"\[DWT\]\s+N=(\d+)\s+mean=([\d.]+) ms\s+min=([\d.]+) ms\s+max=([\d.]+) ms"
)


def parse_dwt_line(line: str):
    """Return (n, mean_ms, min_ms, max_ms) or None if line doesn't match."""
    m = DWT_PATTERN.search(line)
    if m:
        return int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
    return None


def read_and_log(port: str, baud: int, output: str, timeout: float = 60.0):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {port} @ {baud} baud ...")
    with serial.Serial(port, baud, timeout=1) as ser:
        print(f"Connected. Logging to {output_path}")
        print("Waiting for DWT output ... (Ctrl+C to stop)\n")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "n_inferences", "mean_ms", "min_ms", "max_ms"])

            start = time.time()
            while time.time() - start < timeout:
                try:
                    raw = ser.readline()
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue

                if line:
                    print(f"  {line}")

                result = parse_dwt_line(line)
                if result:
                    n, mean_ms, min_ms, max_ms = result
                    ts = datetime.now().isoformat()
                    writer.writerow([ts, n, mean_ms, min_ms, max_ms])
                    f.flush()
                    print(f"\n  ✓ Recorded: mean={mean_ms:.3f} ms | min={min_ms:.3f} ms | max={max_ms:.3f} ms\n")

    print(f"\nLogging complete. Results saved to {output_path}")
    summarize(output_path)


def summarize(csv_path: Path):
    import statistics
    means = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            means.append(float(row["mean_ms"]))

    if not means:
        print("No DWT measurements recorded.")
        return

    print("\n" + "=" * 50)
    print("LATENCY SUMMARY")
    print("=" * 50)
    print(f"  Measurements   : {len(means)}")
    print(f"  Overall mean   : {statistics.mean(means):.3f} ms")
    print(f"  Std dev        : {statistics.stdev(means):.4f} ms" if len(means) > 1 else "")
    print(f"\n  Paper value    : 1.183 ms (DSC-AMCNet INT8, 550 MHz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STM32 DWT Latency Logger")
    parser.add_argument("--port",    required=True, help="Serial port (e.g. COM7 or /dev/ttyACM0)")
    parser.add_argument("--baud",    type=int, default=115200)
    parser.add_argument("--output",  default="results/latency_log.csv")
    parser.add_argument("--timeout", type=float, default=120.0, help="Stop after this many seconds")
    args = parser.parse_args()
    read_and_log(args.port, args.baud, args.output, args.timeout)
