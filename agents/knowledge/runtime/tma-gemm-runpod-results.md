# Repeated GEMM With IO Admission

2026-10-16, script commit `2319f4d`, RunPod `/vdcores`, branch `zj/bw-isolation`.
H100 PCIe: 114 SMs, 81,559 MiB. Conda base Python 3.14, PyTorch `2.14.1+cu130`,
CUDA nvcc 13.0.88. Remote started clean on main; benchmark branch was fetched
and checked out. No competing compute process reported. All GPU runs sequential.

`make pyext` succeeded, installing the extension into Conda base. Default full
handler build: 255 registers/thread, 176-byte stack, 52/144 bytes spill stores/
loads, 14,656 bytes static shared memory. No CUDA/runtime changes were needed.

## Workload and validation

Existing BF16 `Gemm_M64N128K64` executes an independent C=A@B.T per worker;
A=64xK, B=128xK, C=64x128. `LoopC`/`LoopM` repeat it inside one kernel launch.
All modes use identical operands, work counts and compiled binary. Full store
completion is included via per-worker output barriers and a final 16-byte load;
those extra reads are counted. Rates use device profile start-to-end windows.

Small benchmark (`--sms 2 --k 256 --passes 3 --warmup 0 --repeats 1`) completed
all modes. An independent small output check with disabled admission and cap
100 GB/s passed against FP32 PyTorch batched GEMM (rtol=0.02, atol=0.02): mean
absolute error 0.01806, max 0.12495 in both modes. No test files were added.

## Larger measurement

```bash
python app/python/bandwidth_gemm.py --k 16384 --passes 32 --json output/gemm_large_20261016.json
```

114 workers, 684-MiB input working set, 32 passes, two warmups and five measured
launches per mode. Per launch: 979.253 GFLOP, 22.951 GB logical reads and 0.060 GB
logical writes. Medians:

| Mode | IO cap (GB/s) | Device ms | TFLOP/s | Logical TMA GB/s | Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disabled | Unlimited | 14.461 | 67.715 | 1591.21 | — |
| High cap, 10x | 15912.08 | 15.180 | 64.510 | 1515.89 | 4.97% |
| 1x baseline IO | 1591.21 | 14.975 | 65.391 | 1536.59 | 3.55% |
| 0.75x | 1193.41 | 19.370 | 50.555 | 1187.96 | 33.94% |
| 0.5x | 795.60 | 29.066 | 33.690 | 791.68 | 100.99% |

Disabled samples range 14.398–14.581 ms; high-cap samples 15.035–15.264 ms.
The larger configuration provides a 14–29 ms device interval across all modes.

## Default configuration

K=4096, eight passes, same 114 workers, two warmups/five samples. Input 171 MiB;
61.203 GFLOP and 1.449 GB logical IO per launch.

| Mode | IO cap (GB/s) | Device ms | TFLOP/s | Logical TMA GB/s | Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disabled | Unlimited | 0.995 | 61.493 | 1456.25 | — |
| High cap, 10x | 14562.46 | 1.023 | 59.842 | 1417.15 | 2.76% |
| 1x | 1456.25 | 1.019 | 60.065 | 1422.45 | 2.38% |
| 0.75x | 1092.18 | 1.339 | 45.716 | 1082.62 | 34.51% |
| 0.5x | 728.12 | 2.020 | 30.295 | 717.43 | 102.98% |

## Interpretation and artifacts

High cap is intended to avoid throttling; no admission-denial counters were
collected, so it is not proven to be entirely nonbinding. Its roughly 5% larger-
configuration slowdown is an observed end-to-end GEMM effect, not a measurement
isolating CUDA-core contention alone. Binding-cap slowdowns include operand
starvation. Same-binary comparisons hold register reservation constant; they
do not measure compile-time register-footprint costs. This is batched skinny
GEMM, not one large square GEMM or a model. Logical TMA bytes are not physical
DRAM counters, and a large working set does not prove zero L2 hits.

Three JSON files and five logs are saved locally in ignored `output/`, and
remotely under `output/` (JSON) / `.agentlog/` (logs):

- `gemm_small_20261016.json`, `gemm_default_20261016.json`, `gemm_large_20261016.json`.
- `gemm_build_20261016.log`, `gemm_output_check_20261016.log`,
  `gemm_small_20261016.log`, `gemm_default_20261016.log`, `gemm_large_20261016.log`.

Export checksum and all trial FLOP/byte/time calculations were verified locally.
