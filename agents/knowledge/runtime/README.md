# Runtime Knowledge Index

Start here for the VDCores runtime and VM model.

## Recommended Load Order

1. [vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md)
2. [vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md)
3. [vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md)

Then pull in narrower notes as needed:

- [memory-core-performance-knobs.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/memory-core-performance-knobs.md)
- [gemm-scheduler.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/gemm-scheduler.md)
- [memory-admission-boundaries.md](memory-admission-boundaries.md): current transfer/dependency paths, direct-global-memory bypasses, and boundaries for future bandwidth admission control
- [tma-bandwidth-isolation-design.md](tma-bandwidth-isolation-design.md): implemented local rate/burst API, accounting scope, and validation; outstanding/live controls are outside the current implementation
- [tma-bandwidth-runpod-results.md](tma-bandwidth-runpod-results.md): H100 PCIe build and load-only best-effort/partitioned measurements, including worker-count and unit caveats
- [tma-bandwidth-partition-results.md](tma-bandwidth-partition-results.md): independent worker/BW ratio sweeps, matched SM-only baselines, transfer-size calibration, and full-budget limits
- [tma-bandwidth-register-comparison.md](tma-bandwidth-register-comparison.md): matched main/limiter compiler reports; full builds both use 255 regs/thread, load-only builds use 38/48

## What Each Note Covers

- [vdcores-vm-model.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-vm-model.md):
  per-SM runtime layout, register state, queues, allocator, slot lifetime, and `GROUP` / accumulate behavior
- [vdcores-operator-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-operator-semantics.md):
  instruction-field meanings and operator-by-operator state transitions
- [vdcores-queues.md](/home1/11362/depctg/vdcores/agents/knowledge/runtime/vdcores-queues.md):
  queue protocol and deadlock-oriented guidance
