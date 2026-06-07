/**
 * dwt_measure.c
 * =============
 * DWT (Data Watchpoint and Trace) cycle-accurate inference timing.
 *
 * Provides microsecond-resolution latency measurement on ARM Cortex-M7.
 * Used to obtain the silicon-measured latency values reported in the paper.
 *
 * Hardware: STM32H723ZG Nucleo
 *   - Core clock: 550 MHz (HCLK configured in SystemClock_Config)
 *   - DWT_CYCCNT increments every CPU cycle
 *   - Resolution: 1/550e6 s ≈ 1.82 ns per count
 *
 * Measured results (reported in paper Table IV):
 *   DSC-AMCNet INT8  : 1,183 µs  (650,650 cycles @ 550 MHz)
 *   ULCNN-simp INT8  : 2,104 µs  (1,157,200 cycles @ 550 MHz)
 *   Speedup          : 1.78×
 *
 * Usage
 * -----
 * 1. Call DWT_Init() once at startup (after SystemClock_Config).
 * 2. Call DWT_Start() immediately before ai_run().
 * 3. Call DWT_Stop() immediately after ai_run().
 * 4. Read latency via DWT_GetMicroseconds().
 *
 * Include:  "dwt_measure.h"
 * Link with: STM32H7xx HAL, X-CUBE-AI runtime
 */

#include "dwt_measure.h"
#include "stm32h7xx_hal.h"
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Module-private state
 * ---------------------------------------------------------------------- */
static uint32_t dwt_start_cycles = 0;
static uint32_t dwt_stop_cycles  = 0;
static uint32_t dwt_elapsed      = 0;

/* -------------------------------------------------------------------------
 * DWT_Init
 * Enables the DWT cycle counter.
 * Must be called once before any measurement.
 * ---------------------------------------------------------------------- */
void DWT_Init(void)
{
    /* CoreDebug: enable trace */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

    /* Reset and enable cycle counter */
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

/* -------------------------------------------------------------------------
 * DWT_Start
 * Capture start timestamp. Call immediately before inference.
 * ---------------------------------------------------------------------- */
void DWT_Start(void)
{
    dwt_start_cycles = DWT->CYCCNT;
}

/* -------------------------------------------------------------------------
 * DWT_Stop
 * Capture stop timestamp and compute elapsed cycles.
 * Call immediately after inference.
 * ---------------------------------------------------------------------- */
void DWT_Stop(void)
{
    dwt_stop_cycles = DWT->CYCCNT;

    /* Handle 32-bit counter wrap-around */
    if (dwt_stop_cycles >= dwt_start_cycles)
    {
        dwt_elapsed = dwt_stop_cycles - dwt_start_cycles;
    }
    else
    {
        dwt_elapsed = (UINT32_MAX - dwt_start_cycles) + dwt_stop_cycles + 1U;
    }
}

/* -------------------------------------------------------------------------
 * DWT_GetCycles
 * Returns raw elapsed cycle count from last Start/Stop pair.
 * ---------------------------------------------------------------------- */
uint32_t DWT_GetCycles(void)
{
    return dwt_elapsed;
}

/* -------------------------------------------------------------------------
 * DWT_GetMicroseconds
 * Converts elapsed cycles to microseconds using the system clock.
 *
 * Formula: µs = cycles / (HCLK_MHz)
 *
 * For STM32H723ZG at 550 MHz:
 *   1 µs = 550 cycles
 *   1,183 µs = 650,650 cycles  (DSC-AMCNet INT8)
 * ---------------------------------------------------------------------- */
float DWT_GetMicroseconds(void)
{
    uint32_t hclk_mhz = HAL_RCC_GetHCLKFreq() / 1000000U;
    return (float)dwt_elapsed / (float)hclk_mhz;
}

/* -------------------------------------------------------------------------
 * DWT_PrintResult
 * Prints measurement via UART (Redirect printf to USART3 in CubeIDE).
 * ---------------------------------------------------------------------- */
void DWT_PrintResult(const char *label)
{
    float us = DWT_GetMicroseconds();
    float ms = us / 1000.0f;
    printf("[DWT] %s: %lu cycles | %.2f µs | %.3f ms\r\n",
           label, (unsigned long)dwt_elapsed, us, ms);
}

/* -------------------------------------------------------------------------
 * DWT_BenchmarkN
 * Runs inference N times, reports min / mean / max.
 * Used to obtain stable latency figures for the paper.
 *
 * Paper methodology:
 *   N = 1000 inference passes per model
 *   Reported value = mean of 1000 measurements
 *   First 10 passes discarded (cache warm-up)
 * ---------------------------------------------------------------------- */
void DWT_BenchmarkN(
    void (*inference_fn)(void),   /* pointer to ai_run() wrapper */
    uint32_t N,
    float *out_mean_us,
    float *out_min_us,
    float *out_max_us)
{
    float sum = 0.0f;
    float mn  = 1e9f;
    float mx  = 0.0f;

    /* Warm-up passes (not counted) */
    for (uint32_t i = 0; i < 10; i++) { inference_fn(); }

    for (uint32_t i = 0; i < N; i++)
    {
        DWT_Start();
        inference_fn();
        DWT_Stop();

        float us = DWT_GetMicroseconds();
        sum += us;
        if (us < mn) mn = us;
        if (us > mx) mx = us;
    }

    *out_mean_us = sum / (float)N;
    *out_min_us  = mn;
    *out_max_us  = mx;

    printf("[DWT] N=%lu  mean=%.3f ms  min=%.3f ms  max=%.3f ms\r\n",
           (unsigned long)N,
           *out_mean_us / 1000.0f,
           *out_min_us  / 1000.0f,
           *out_max_us  / 1000.0f);
}
