# TMA Bandwidth Isolation

Design discussion, 2026-10-12; local rate/burst implementation added 2026-10-13.
No tests are included, as requested. CUDA compilation and the load-only benchmark
were verified on RunPod's H100 PCIe on 2026-10-13; see
[run results](tma-bandwidth-runpod-results.md). Repeated GEMM loads/stores were
also exercised on 2026-10-16; reductions and general workload isolation remain
unverified. The local Mac has no CUDA environment.

Current agreed implementation scope: fixed per-launch local rate/burst budgets only. Do not implement live updates or explicit outstanding-transfer tracking. Their discussion below is background, not a requirement for the first version.

## Initial scope

All workload data movement uses TMA and reads miss L2. Start with ordinary loads/stores of known tile sizes; TMA reductions need separate read/modify/write accounting. Runtime metadata traffic remains overhead. Use multiple statically assigned component streams within one `dae2` launch; the current host launch API synchronizes the device and is not a concurrent serving interface.

## Contract and accounting

The contract below includes later outstanding-traffic and reconfiguration controls. The first implementation should expose only fixed rate and burst settings.

Each component owns one GPU-wide admission contract:

- `rate`: admitted transfer bytes per second, shared across reads and writes;
- `burst_bytes`: maximum accumulated rate credit;
- separate read/write outstanding-byte limits, to avoid one direction exhausting all outstanding capacity;
- a configuration generation for consistent rate/burst updates.

All blocks and both load ports for that component share the contract. Worker count is not multiplied into the configured rate. Assign a component ID through a worker-to-component table initially; no per-operation tag is needed while a worker's instruction streams belong to one component.

Use continuous-refill byte admission: available rate credit grows with time up to the burst allowance. Charge transfer bytes just before issuing a ready TMA operation. Outstanding credit returns only after completion. A globally contended atomic on every TMA is a correctness baseline, not the preferred fast path.

The intended admitted-byte envelope is `A_i(s,t) <= burst_i + rate_i * (t-s)`, subject to bounded grant-to-issue delay. Do not grant large leases or future permits that can accumulate and issue together later. Burst capacity must accommodate the largest operation. Accounting must classify opcodes: output-slot allocation, control operations, and register carriers are not transfers.

## Minimal first-version API

Implemented launcher configuration method:

```python
dae.set_bandwidth(
    "slow",
    num_sms=64,
    base_sm=0,
    bandwidth_gb_s=400.0,
    burst_bytes_per_sm=64 * 1024,
)
```

Keep the existing `schedule.place(num_sms, base_sm)` interface for instruction placement. The rate setting applies to every TMA transfer on the configured workers, not to an instruction tag. The name is for host-side configuration/reporting only. Worker indices are logical block indices, not physical SM affinity.

- `bandwidth_gb_s` is the component's aggregate read-plus-write admission cap, in decimal GB/s (`1 GB/s = 10^9 bytes/s`). Divide by the configured worker count, not by the number currently issuing.
- `burst_bytes_per_sm` is a local allowance, defaulting to 64 KiB. Aggregate burst allowance is worker count times this value. Validate it against the largest charged transfer on the configured workers.
- Omitted configuration or `bandwidth_gb_s=None` disables admission control. Reject zero, negative, or nonfinite rates; zero is not a pause mechanism in this version.
- Reject overlapping named component ranges and out-of-range/nonpositive worker counts. A named configuration can be replaced before launch; configuration remains immutable during execution.

Python materializes one configuration record per worker: enabled flag, divided rate (or precomputed pacing coefficient), and local burst allowance. Pass the table through the launcher/extension/runtime/kernel interfaces. Each block reads its record once at initialization and maintains its limiter state in shared memory. No GPU component table or `MInst`/`CInst` changes are required. Every relevant load/store issue path uses the same block-local admission helper.

Outstanding-byte limits, live configuration changes, and borrowing are outside this implementation. Completion handling is unchanged.

### Implementation and checks

- `python/dae/bandwidth.py` validates named disjoint ranges and packs 16-byte per-worker records. `Launcher.set_bandwidth` configures them; launch validates built and queued memory instructions before execution. Known raw-address paths are rejected on limited workers. Existing disabled launches keep the original positional extension call.
- `include/dae/bandwidth.cuh` maintains one shared-memory `BlockBandwidthLimiter` per block. The two load lanes and store lane use a block-scoped acquire/release atomic lock around refill/charge arithmetic only. Waiting for credits happens outside the lock. A disabled worker has a null limiter pointer.
- `include/dae/bandwidth.cuh` also contains the config ABI and token-bucket arithmetic. Refill uses globaltimer nanoseconds and double byte credit; each launch begins with full local burst credit. This is a per-launch contract, not persistent accounting across host launches. `runtime.cuh` forward-declares `BandwidthConfig`; the Torch binding includes its full definition.
- `include/dae/bandwidth_ops.inc` is the common opcode/cost registry for CUDA charging and Python burst validation. Ordinary TMA transfers charge payload bytes. TMA reductions charge `2 * size` as a nominal destination read plus write; this is not a DRAM-traffic measurement. Control/register operations and output allocation charge zero.
- `src/runtime.cu`, `src/torch_runtime.cu`, and `include/dae/dae2.cuh` pass an optional config table to the kernel and initialize local state once. No per-TMA global budget atomic or global policy read is added.
- No tests are added. The separately requested `app/python/bandwidth_load.py` benchmark compares disabled admission with two 50% bandwidth partitions, keeping the current kernel and workload unchanged. It uses both load ports, no workload stores, and device start/end timestamps; results are logical TMA throughput rather than DRAM counters. The cap is half the median best-effort aggregate rate. Equal offered load evaluates the balanced case, not isolation under unequal demand.

The balanced load-only run retained 99.8% of the disabled kernel's aggregate throughput, and a lower-cap calibration demonstrated throttling. These results do not establish fair scheduling, interference isolation for unequal workloads, or general overhead. The local admission lock is a simple first version, not a claim of minimum possible overhead or hard real-time guarantees.

## Global policy, local enforcement

Global-memory address space does not imply an HBM access on every update. Device-scoped atomic communication is coherent at L2; it does not require bypassing L2. Use device-scope atomics for shared counters, and release/acquire publication for policy/mailbox state. Plain loads/stores or `volatile` are not a replacement. Scope the cache-miss workload assumption to tensor traffic, not control metadata. See [CUDA thread scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#thread-scopes) and [libcu++ message passing](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#example-message-passing).

For the first static-worker experiment, shard each component contract across its blocks: `sum_j rate_ij = rate_i`, `sum_j burst_ij <= burst_i`, and sum outstanding allowances within the component limits. Each block maintains shared-memory buckets coordinated among its two load issuers and store issuer. The per-TMA path checks local state and time; it performs no cross-SM atomic. Immutable initial policy can be loaded once. This preserves the aggregate admission envelope, but idle blocks strand their assigned credits.

Keep one component per block initially. The existing load/store issuers are per block, so this avoids cross-component software FIFO blocking while still exercising shared hardware-bandwidth interference. Multiplexing components within a block is a separate change: it needs component-specific streams/ready queues, fair or priority-based selection before issue, and completion handling that does not block unrelated ready work. More issuer warps alone would not remove the common compute FIFO or allocator/PC blocking.

For elastic enforcement, use bounded credit leases: blocks acquire a batch of byte credits from the component's global budget and spend them locally. Credits already leased are deducted globally and cannot be granted again. This amortizes cross-SM coordination, but delayed local spending widens the actual-issue burst envelope: if at most `L_i` bytes remain leased, the conservative envelope becomes `rate_i * T + global_burst_i + L_i`. Count that allowance inside the desired burst contract, or use time-valid grants and local pacing to tighten it. Large leases improve amortization but worsen transient interference and reclamation.

Dynamic updates must account for old quotas/leases and outstanding transfers. Policy publication alone does not instantly revoke locally cached authority. Use bounded lease validity and a transition protocol; do not combine an old allocation on some blocks with a newly increased allocation on others and assume the aggregate cap holds. Likewise, global outstanding-byte accounting should be sharded or leased rather than adding a global atomic on every completion.

## Integration boundaries

- Loads: after the global dependency wait, before the TMA opcode switch in `include/dae/pipeline/ldwarp.cuh`.
- Stores: after compute's writeback request, before the store/reduction switch in `include/dae/pipeline/stwarp.cuh`.
- Preserve current queue ordering, TMA transaction registration, and dependency release protocol.
- Track read completion through the existing TMA-backed `m2c` barrier. A thread-0 hook after successful compute consumption can conservatively release read-outstanding bytes; later accounting reflects consumer delay as well as transfer delay.
- Track writes with full bulk-group completion. `cp.async.bulk.wait_group.read` only waits for source reads and cannot release a credit intended to cover outstanding destination writes. A simple initial full wait may reduce overlap and must be compared with baseline.
- Keep fair admission between readers and writers, with progress preference for writes that free slots/release dependencies. They still consume rate credit; do not exempt them as unlimited traffic. Avoid waiting while holding an admission lock or an outstanding reservation.

## Isolation and adaptation

Measure conservative sustainable bandwidth `C` with cache-miss transfers and the relevant read/write mix. Initially choose fixed rates with `sum(rate_i) <= eta * C`, where `eta < 1` is measured headroom. Small aggregate burst allowances reduce synchronized traffic spikes. Outstanding limits restrict the work that remains after a rate reduction; size them to preserve pipeline progress and useful overlap.

This controls offered traffic, not the hardware memory controller's arbitration. Treat predictable service/latency under colocation as an empirical claim until a service model is validated; an admission cap is not a minimum bandwidth guarantee.

Add borrowing after fixed-rate isolation works: a device controller can lend unused capacity by updating rates, keeping their sum within capacity. Keep reserved shares distinct from temporary loans. A returning component regains its share only after configuration publication and already-issued loan traffic drains. On rate reduction, do not reset the bucket to a full burst or forget outstanding work.

## First validation

`app/python/bandwidth_gemm.py` (added 2026-10-16) implements a simple repeated
GEMM experiment using existing `Gemm_M64N128K64`, `LoopC` and `LoopM`: independent
64x128xK products per worker, no new CUDA handlers. Disabled admission establishes
a logical read+write reference; the same work is rerun with a high cap (default
10x) and fractional caps (1x/0.75x/0.5x). Per-worker output-store barriers count
all passes; a final barrier-gated 16-byte load and Dummy consumer put full output
completion inside the device profile window, with those extra read bytes counted.
Reports median device time/TFLOP/s/logical GB/s/slowdown and optional raw-trial JSON.
High cap is a nonbinding candidate, not verified by admission-wait counters;
binding runs combine operand starvation with software overhead. Compilation,
small output validation and GPU execution succeeded on RunPod H100 on 2026-10-16;
see [GEMM results](tma-gemm-runpod-results.md). No tests or runtime changes.

Run two independent TMA-only components inside one launch. Sweep rate, burst, worker count, and outstanding limits. Check the GPU-wide admitted-byte envelope, achieved read/write bandwidth, component latency/goodput, and slowdown of a fixed-budget component when its neighbor increases demand. Compare disabled admission with baseline to isolate bookkeeping overhead. Then measure configuration-change settling time and borrowing/reclamation.

Use measured DRAM bytes rather than interpreting no-L2-read-hits as instantaneous HBM service. Alignment, write buffering, and asynchronous completion can still separate issue time from physical memory traffic.
