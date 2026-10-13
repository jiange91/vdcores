# TMA Bandwidth: RunPod H100 PCIe

Measured 2026-10-13 on implementation commit `e4ad2ea`, branch `zj/bw-isolation`,
checkout `/vdcores`. Hardware: NVIDIA H100 PCIe, 114 physical SMs, 81,559 MiB.
Conda base: PyTorch `2.14.1+cu130`, CUDA 13.0. No competing compute process was
reported before the runs. GPU benchmarks ran sequentially.

`make pyext` succeeded. The full build used 255 registers, a 176-byte stack frame,
52 bytes of spill stores and 144 bytes of spill loads. These are compiler reports,
not measured limiter overhead; both benchmark cases use that same kernel.

## Requested comparison

```bash
python app/python/bandwidth_load.py --json output/runpod_bandwidth_load_20261013.json
```

114 workers, 57 per component; 8-KiB tiles; 1,024 unique tiles per worker;
8 passes per launch; 912-MiB input; 7.650 GB of logical loads per launch;
64-KiB local burst allowance. Two warmups and five measured launches per case.
Both load ports are active; no workload stores. Rates below are median decimal
GB/s, computed from logical payload and device timestamps.

| Case | Scope | Cap (GB/s) | Achieved (GB/s) | Duration (ms) |
| --- | --- | ---: | ---: | ---: |
| Best effort | All workers | Unlimited | 1848.87 | 4.138 |
| 50/50 | Component 0 | 924.43 | 922.49 | 4.147 |
| 50/50 | Component 1 | 924.43 | 922.51 | 4.147 |
| 50/50 | All workers | 1848.87 | 1844.99 | 4.147 |

The partitioned run retains 99.79% of the disabled kernel's aggregate throughput.
Component rates use their own active windows; aggregate rates use the entire
worker set. This symmetric workload naturally gives each component about half
the best-effort rate, so the comparison alone does not demonstrate throttling
or interference isolation under unequal demand.

## Lower-cap calibration

Reused the benchmark's `build_launcher`/`measure` helpers, with total cap set to
half the best-effort reference: each component receives 25% of that reference.
Same input shape and passes; one warmup and three measured launches.

| Scope | Cap (GB/s) | Achieved (GB/s) | Duration (ms) |
| --- | ---: | ---: | ---: |
| Component 0 | 462.22 | 462.44 | 8.272 |
| Component 1 | 462.22 | 462.43 | 8.272 |
| All workers | 924.43 | 924.85 | 8.272 |

The doubled duration demonstrates rate-dependent throttling. Slightly exceeding
the rate is consistent with the initial local burst credits; these are rate-plus-
burst caps, not strict constant instantaneous rates.

## Existing tma1d comparison

The unchanged `app/python/tma1d.py` completed on the rebuilt kernel at 132 worker
blocks: 1.098400 ms and **938.87 GiB/s**, printed under a `GB/s` label. This is
about **1008.10 decimal GB/s**. Running the same script with only its worker count
changed to 114, without editing the file, gave 0.523360 ms and **1701.74 GiB/s**
(about **1827.23 decimal GB/s**). Both are one-iteration results.

The original 132-block setting exceeds the device's 114 SMs. Its lower result
therefore includes oversubscription; it should not be interpreted as the GPU's
bandwidth ceiling. The new script uses 114 resident workers, two load ports,
longer runs, and decimal units, so its result is not directly comparable to the
old number without accounting for these differences.

## Artifacts and limits

- Remote main result: `/vdcores/output/runpod_bandwidth_load_20261013.json`.
- Remote calibration: `/vdcores/output/runpod_bandwidth_load_25pct_20261013.json`.
- Local copies are under the checkout's ignored `output/` directory, together
  with build and original-script logs.
- Metrics are logical TMA load throughput, assuming cache misses; DRAM counters
  were not collected. A large working set does not prove a zero cache-hit rate.
- No numerical payload check, store/reduction run, unequal-demand isolation
  experiment, or deadline evaluation is included in these measurements.

## Component-count sweep

Measured subsequently on benchmark commit `7749380`, same GPU and parameters,
two warmups and five measured launches per case. No competing compute process
was reported. Python-only change; the existing extension was reused.

```bash
python app/python/bandwidth_load.py --components 2 4 6 8 --json output/runpod_bandwidth_scaling_20261013.json
```

Fresh best-effort reference: **1829.79 GB/s**, 4.181 ms. Every component receives
1/N of this rate, divided equally over its own workers. The first N-1 groups
receive floor(114/N) workers; the last receives the remainder. Worker indices
are block indices, not physical SM affinity; the launch has 114 worker blocks.

| Components | Workers per component | Cap each (GB/s) | Achieved per component (GB/s) | Aggregate (GB/s) | % reference | Finish (ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 2 | 57, 57 | 914.90 | 912.83–912.89 | 1825.62 | 99.8% | 4.191 |
| 4 | 28, 28, 28, 30 | 457.45 | First three: 449.24–449.54; last: 457.06 | 1736.73 | 94.9% | 4.405 |
| 6 | 19 × 6 | 304.97 | 304.31–304.39 | 1825.77 | 99.8% | 4.190 |
| 8 | 14 × 7, 16 | 228.72 | First seven: 224.68–224.94; last: 228.68 | 1629.32 | 89.0% | 4.695 |

Equal work per worker gives the last group more bytes in the 4/8 cases, despite
its equal aggregate bandwidth cap. The ideal full-completion-window throughput
relative to the reference is 114/(N × largest group): 95.0% for four components
and 89.06% for eight. Observed aggregate rates closely follow these tails;
they should not be attributed to scheduler overhead. Per-component rates use
each group's own active window and need not sum to the aggregate rate.

This checks throughput with 2–8 disjoint worker groups and local budgets. It
does not exercise same-SM multiplexing or establish isolation under unequal
demand. Raw per-worker timestamps and all trials are retained in
`output/runpod_bandwidth_scaling_20261013.json`, locally and on RunPod.

### Four/eight-component rerun

Reran `--components 4 8` on the same checkout with unchanged defaults, two
warmups/five measured launches, no competing compute process. Fresh reference:
1829.54 GB/s. Four components: cap 457.39 GB/s each, first three achieved
449.40–449.63 and last 457.08; aggregate 1736.82 GB/s (94.9%), 4.405 ms.
Eight components: cap 228.69 each, first seven achieved 224.68–224.93 and last
228.65; aggregate 1629.08 GB/s (89.0%), 4.696 ms. Aggregate results differ from
the previous sweep by less than 0.02%; the completion-tail observation reproduces.
Raw results: `output/runpod_bandwidth_scaling_4_8_rerun_20261013.json` locally
and on RunPod. Remote log: `.agentlog/bandwidth_scaling_4_8_rerun.log`.
