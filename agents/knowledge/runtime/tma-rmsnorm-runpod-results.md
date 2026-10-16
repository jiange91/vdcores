# CUDA-core RMSNorm With IO Admission

2026-10-16, script commit `7d0a982`, RunPod `/vdcores`, Conda base. H100 PCIe,
114 workers, same full runtime binary used for the GEMM measurements. Installed
extension already contains `OP_RMS_NORM_F16_K_4096_SMEM`; no CUDA changes/rebuild.
No competing compute process reported; all benchmark launches sequential.

## Workload

`app/python/bandwidth_rmsnorm.py` uses the existing 128-thread compute handler.
It performs packed BF16 arithmetic, FP32 reductions and reciprocal square root;
no Tensor Core operations. Each operation normalizes one 4096-element BF16 row,
loading 8 KiB of weights on port 0 and 8 KiB of input on port 1, then storing
8 KiB of output. Each worker reuses its weight vector across distinct rows.
Existing barriers and a final 16-byte load include full output completion.

```bash
python app/python/bandwidth_rmsnorm.py --json output/rmsnorm_20261016.json
```

Defaults: 512 distinct rows/worker, eight on-device passes, 456-MiB input and
456-MiB output. Per launch: 1.913 G normalized elements, 7.650 GB logical reads
and 3.825 GB logical writes. Three calibration launches after five warmups set
the cap reference (1033.726 GB/s). Each mode receives five warmups, followed by
ten rounds of cyclically rotated mode order. All modes reuse the same operands,
compiled binary and work counts. Caps divide equally among workers.

## Results

| Mode | IO cap GB/s | Median device ms | G elements/s | Logical TMA GB/s | Median paired slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disabled | Unlimited | 11.102 | 172.274 | 1033.64 | — |
| High cap, 1000x | 1033726.18 | 11.195 | 170.841 | 1025.05 | 0.82% |
| 1x | 1033.73 | 11.196 | 170.833 | 1025.00 | 0.84% |
| 0.75x | 775.29 | 14.805 | 129.190 | 775.14 | 33.35% |
| 0.5x | 516.86 | 22.198 | 86.159 | 516.96 | 99.95% |

Paired slowdown is the median of per-round time ratios against disabled.
Disabled samples span 11.096–11.112 ms; high-cap samples span 11.187–11.200 ms.
The high-cap effect is approximately 0.8% here, similar to the latest interleaved
GEMM result. It is an end-to-end effect, not an isolated measurement of compute
interference. Binding caps also include input starvation. Admission-denial and
hardware stall counters were not collected; GPU counter access was previously
blocked by `ERR_NVGPUCTRPERM`. No clock control was applied.

Bandwidth counts admitted logical TMA bytes, not measured DRAM traffic. In
particular, reused weight vectors may hit L2; a large input does not establish
zero cache hits. This workload does not establish saturated CUDA-core throughput.

## Verification and artifacts

Local `py_compile` and diff check passed. A one-off small check used two workers,
four distinct rows and three passes, with two launches each at disabled,
1000 GB/s and 5 GB/s. All outputs passed against FP32 RMSNorm at rtol=0.03,
atol=0.03. Mean absolute error was 0.001488 and maximum 0.054057, identical
across modes. The existing handler accumulates part of the sum in BF16, so it
does not promise FP32-reference accuracy. No test files were added.

Local ignored `output/` contains `rmsnorm_20261016.json`,
`rmsnorm_20261016.log`, and `rmsnorm_output_check_20261016.log`. Remote JSON is
under `output/`; logs are under `.agentlog/`. Export checksum verified locally.
