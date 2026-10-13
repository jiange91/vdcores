# Independent Worker and TMA Bandwidth Allocations

2026-10-13, benchmark commit `d7a5fc8`, RunPod `/vdcores`, H100 PCIe with 114 SMs.
No competing compute process was reported before the sequence. Python-only
changes reused the previously built full Hopper runtime. No tests added.

## Experiment

`app/python/bandwidth_partition.py` sweeps worker ratios and bandwidth ratios
independently. The first group's worker count is rounded down; the last gets
the remainder. These are resident worker-block counts, not physical SM affinity.
Two TMA-only streams use both load ports; dummy consumers free completed loads.
No compute-heavy kernel, model inference, or compute throughput is evaluated.

Traffic follows BW weights, so ideal completion times match even with unequal
caps. Integer load-pair counts are rounded up to preserve exact ratios and
uniform work within each group. Code generation and measurement use the same
per-worker count plan. Each point includes identical work with rate admission
disabled as its SM-only comparison; this is a disjoint-worker baseline, not an
MPS/MIG benchmark. Both measured cases use the same compiled runtime.

Defaults: 32-KiB transfers, target 256 tiles/worker/pass, 32 passes, 64-KiB
burst/worker, two warmups and five measured launches, all sequential. Actual
traffic differs slightly between points because of rounding and is retained
in JSON. Metrics are median decimal logical TMA GB/s, assuming cache misses;
not DRAM counters or a proof of zero cache hits.

## Capacity calibration

| Transfer size | Workers | Isolated best effort (GB/s) |
| ---: | ---: | ---: |
| 8 KiB | 28 | 456.19 |
| 8 KiB | 57 | 929.59 |
| 8 KiB | 86 | 1402.33 |
| 32 KiB | 28 | 1817.98 |
| 32 KiB | 57 | 1956.13 |
| 32 KiB | 114 | 1994.60 |

The 8-KiB load program limits how much traffic a small group can offer. Using
32-KiB loads allows 28 workers to offer far more than 25% of the all-worker
reference. This is a changed TMA workload, not an improvement attributed to the
limiter. Calibration data is retained separately; 65,520-byte trials also exist
in the tile-capacity artifact and reached about 1962–1985 GB/s at 28/57/114 workers.

## Independent caps with headroom

```bash
python app/python/bandwidth_partition.py --json output/runpod_bandwidth_partition_70pct_20261013.json
```

Reference **1994.19 GB/s**; total caps **1395.94 GB/s**, 70% of reference. For BW
50/50, caps are 697.97/697.97; for 25/75 they are 348.98/1046.95, and reversed
for 75/25. BW percentages below are of the total admitted budget, not the full
device reference.

| Workers A/B | BW budget A/B | Achieved A/B (GB/s) | Aggregate (GB/s) |
| --- | --- | --- | ---: |
| 57/57 | 50/50 | 697.96/697.96 | 1395.93 |
| 57/57 | 25/75 | 349.07/1046.90 | 1395.86 |
| 57/57 | 75/25 | 1046.90/349.06 | 1395.86 |
| 28/86 | 50/50 | 697.92/698.04 | 1395.85 |
| 28/86 | 25/75 | 348.98/1046.93 | 1395.91 |
| 28/86 | 75/25 | 1045.76/349.11 | 1394.35 |
| 85/29 | 50/50 | 698.04/697.89 | 1395.78 |
| 85/29 | 25/75 | 349.10/1046.11 | 1394.82 |
| 85/29 | 75/25 | 1046.94/348.98 | 1395.92 |

All component median rates are within **0.114%** of their configured caps.
This demonstrates independently controlled TMA rate caps at different worker
allocations for this workload. The 85/29 worker, 25/75 bandwidth point illustrates
an allocation suitable for a compute-heavy A and memory-heavy B, but both
measured components are load-only; no mixed-workload benefit is established.

SM-only allocation does not force bandwidth proportional to SM count. Its
limitation here is the absence of an explicit per-component BW cap. All matched
SM-only rates are retained in JSON. Component rates use their own active windows,
which may include a period after the peer finishes; their sum must not be
interpreted as concurrent GPU bandwidth. Aggregate uses the full worker window.

## Full-budget stress sweep

```bash
python app/python/bandwidth_partition.py --total-cap-fraction 1 --json output/runpod_bandwidth_partition_full_20261013.json
```

Reference and total caps **1994.12 GB/s**; caps are 997.06/997.06 for 50/50 and
498.53/1495.59 for 25/75, reversed for 75/25.

| Workers A/B | BW budget A/B | Achieved A/B (GB/s) | Aggregate (GB/s) |
| --- | --- | --- | ---: |
| 57/57 | 50/50 | 992.25/992.12 | 1984.21 |
| 57/57 | 25/75 | 498.47/1423.14 | 1897.52 |
| 57/57 | 75/25 | 1446.12/498.52 | 1928.16 |
| 28/86 | 50/50 | 921.62/996.88 | 1843.23 |
| 28/86 | 25/75 | 495.66/1489.05 | 1982.63 |
| 28/86 | 75/25 | 1372.67/498.64 | 1830.23 |
| 85/29 | 50/50 | 996.89/902.21 | 1804.42 |
| 85/29 | 25/75 | 498.63/1392.23 | 1856.31 |
| 85/29 | 75/25 | 1489.05/497.39 | 1985.40 |

Some rates fall below caps by up to **9.51%**. Local admission controls offered
traffic but does not reserve hardware service; isolated capacity is insufficient
to predict colocated service. Do not claim guaranteed BW or SM-independent
achievable capacity from the headroom result. Residual mechanisms are not
diagnosed here; neither latency isolation nor P99 performance was evaluated.

## Artifacts

Local and remote ignored `output/` contain:

- `runpod_bandwidth_partition_70pct_20261013.json`
- `runpod_bandwidth_partition_full_20261013.json`
- `runpod_bandwidth_capacity_20261013.json`
- `runpod_bandwidth_capacity_tiles_20261013.json`

Remote logs: `.agentlog/bandwidth_partition_{70pct,full}.log` and
`.agentlog/bandwidth_capacity{,_tiles}.log`. Export checksum was verified locally;
all trials have exact BW-weighted traffic ratios and matching aggregate byte totals.
