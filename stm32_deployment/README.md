# STM32 Deployment

Step-by-step instructions to deploy DSC-AMCNet INT8 on the NUCLEO-H723ZG
using STM32CubeIDE 1.15.x and X-CUBE-AI 10.2.0 (ST Edge AI Core 2.2.0).

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| STM32CubeIDE | 1.15.x | Available at st.com |
| X-CUBE-AI | 10.2.0 | Install via CubeMX Embedded Software |
| ST Edge AI Core | 2.2.0 | CLI: `stedgeai` |
| STM32CubeMX | 6.11.x | Bundled with CubeIDE |
| Arm GCC | 12.3 | Bundled with CubeIDE |

**Board:** NUCLEO-H723ZG (STM32H723ZG, Cortex-M7, 550 MHz, 1 MB Flash, 564 KB RAM)

---

## Step 1: Verify the ONNX Model

Before importing into X-CUBE-AI, confirm the ONNX was generated with the
`fbgemm` backend (signed INT8):

```bash
python quantization/validate_backend.py \
    --checkpoint models/checkpoints/dscamcnet_best.pth \
    --data datasets/RML2016.10a_dict.dat \
    --split datasets/split_indices.npz
```

Expected output:
```
fbgemm  → zero-point=0, dtype=torch.qint8   ← COMPATIBLE ✓
qnnpack → zero-point=128, dtype=torch.quint8 ← INCOMPATIBLE ✗
```

---

## Step 2: Analyze with ST Edge AI Core CLI

```bash
stedgeai analyze \
    --target stm32h7 \
    --name dscamcnet \
    quantization/outputs/dscamcnet_int8.onnx
```

Expected output (paper Table IV):
```
Flash usage  : 51.26 KB
RAM usage    : 12.09 KB
MACC count   : ~1.2M
```

If Flash or RAM differs, recheck the ONNX (wrong backend or architecture variant).

---

## Step 3: Create STM32CubeIDE Project

1. Open STM32CubeIDE → File → New → STM32 Project
2. Select board: `NUCLEO-H723ZG`
3. Project name: `DSC_AMCNet_Deployment`
4. Language: C, Executable

---

## Step 4: Configure Clock (550 MHz)

In CubeMX (double-click the `.ioc` file):

1. **RCC:** HCLK = 550 MHz (PLL1, HSE 8 MHz input)
2. **USART3:** Enable, Mode = Asynchronous, Baud = 115200
   - PA8 = USART3_TX, PB11 = USART3_RX
   - Redirect `printf` via USART3: add `syscalls.c` with `__io_putchar`
3. **Generate code** (Project → Generate Code)

---

## Step 5: Import X-CUBE-AI Model

1. In CubeMX: **Additional Software → X-CUBE-AI**
2. Enable X-CUBE-AI, add Network:
   - Model file: `quantization/outputs/dscamcnet_int8.onnx`
   - Network name: `network`
   - Compression: None (already INT8)
3. Click **Analyze** — verify 51.26 KB Flash, 12.09 KB RAM
4. Click **Generate Code**

This produces:
```
Middlewares/ST/AI/
├── Inc/
│   ├── ai_datatypes_defines.h
│   ├── network.h
│   └── network_data.h
└── Src/
    ├── network.c
    └── network_data.c
```

---

## Step 6: Add DSC-AMCNet Firmware Files

Copy the firmware sources into the project:

```
Core/Src/
├── inference_main.c   ← from stm32_deployment/firmware/
├── dwt_measure.c      ← from stm32_deployment/firmware/
Core/Inc/
├── dwt_measure.h      ← create (declarations for DWT_Init, DWT_Start, etc.)
```

In `Core/Src/main.c`, add to the main loop:
```c
#include "inference_main.h"
// ...
int main(void) {
    HAL_Init();
    SystemClock_Config();   // ensures 550 MHz
    MX_USART3_UART_Init();
    inference_main();       // does not return
}
```

---

## Step 7: Build and Flash

```
Project → Build Project   (Ctrl+B)
Run → Debug               (F11) or Run → Run (Ctrl+F11)
```

Flash should be under 160 KB total (model + firmware + HAL).

---

## Step 8: Read Latency Output

Connect USART3 (115200 8N1) via USB-UART or STLink VCP:

```
=== DSC-AMCNet Inference on STM32H723ZG ===
    Core: Cortex-M7 @ 550 MHz
    Model: DSC-AMCNet INT8 (X-CUBE-AI 10.2.0)

[AI] DSC-AMCNet INT8 initialised.
[AI] Flash: 51.26 KB | RAM: 12.09 KB

[BENCH] Running 1000 inferences (10 warm-up discarded) ...
[DWT] N=1000  mean=1.183 ms  min=1.181 ms  max=1.187 ms

[RESULT] DSC-AMCNet INT8 Latency
         Mean : 1.183 ms  (paper: 1.183 ms)  ✓
```

---

## Step 9: Record Latency with Python Logger

Use `hardware_characterization/measure_latency.py` to log results to file:

```bash
python hardware_characterization/measure_latency.py \
    --port COM7 \
    --baud 115200 \
    --output results/latency_log.csv
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Random classification output | Wrong ONNX backend (qnnpack) | Re-export with fbgemm |
| Flash overflow | Wrong ONNX (FP32 instead of INT8) | Re-run ptq_quantize.py |
| Latency >> 1.183 ms | Clock not 550 MHz | Check SystemClock_Config |
| DWT reads 0 | CoreDebug TRCENA not set | Ensure DWT_Init() called first |
| No UART output | Printf not redirected | Add `__io_putchar` to syscalls.c |
