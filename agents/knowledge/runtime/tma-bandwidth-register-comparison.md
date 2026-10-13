# Limiter Register Footprint

2026-10-13, RunPod H100, CUDA `nvcc` 13.0.88. Compared fetched `origin/main`
(`fd75b8d`) against bandwidth branch `d47bf20`. Separate detached checkouts
kept main out of the installed Python environment. Fresh `make -B runtime.o`
builds used identical default flags (sm90a, O3, fast math, lineinfo, NDEBUG).
Selected-op include files matched byte-for-byte for each comparison.

| Selected compute handlers | Main regs/thread | Limiter regs/thread | Main spills store/load (bytes) | Limiter spills store/load (bytes) |
| --- | ---: | ---: | --- | --- |
| Default full, 26 handlers | 255 | 255 | 52 / 144 | 52 / 144 |
| OP_DUMMY, OP_LOOPC, OP_TERMINATEC | 38 | 48 | 0 / 0 | 0 / 0 |

Full builds both report a 176-byte stack frame and 16 barriers. Static shared
memory: main 14,624 bytes; limiter 14,656. Small builds both report an 80-byte
stack frame and 9 barriers. Static shared memory: main 14,368; limiter 14,400.
These are compiler reports; spill byte counts are not measured runtime traffic.
Stack frames can exist even when the compiler reports zero spills.

The full kernel's 255-register allocation and spills predate the limiter.
The small-build comparison exposes ten more reported registers/thread with
limiter support. This is a kernel-wide compiler figure, not ten registers used
by every lane at runtime or a per-LD-warp count. Hardware allocation is granular.
Enabled/disabled admission in the same compiled kernel has the same reserved
footprint. Extra runtime arithmetic, shared loads/atomics and retries remain.

Compute and memory warps are separate execution contexts within the same CTA,
not dedicated physical processors. Register values are private, but allocations
consume physical SM/subpartition register capacity. Warps can share scheduler
issue capacity and execution pipelines; all subpartitions share L1/shared memory.
Different tensor/ALU pipelines can overlap, so scheduling independence does not
imply either zero interference or one-for-one compute slowdown. Current runtime
has no active `setmaxnreg` redistribution. Actual compute throughput loss remains
unmeasured. See [NVIDIA hardware model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#hardware-model)
and [Hopper occupancy](https://docs.nvidia.com/cuda/hopper-tuning-guide/#occupancy).

Both full builds and both small builds succeeded. No extension installation,
GPU performance run, runtime source change, or tests were performed. Primary
`/vdcores` retained the full generated handler set and feature branch; the two
temporary checkouts were removed after saving logs. Local ignored `output/`
contains `runpod_register_{main,bw}{,_load_only}_build_20261013.log`; remote
logs remain in `/vdcores/.agentlog/register_{main,bw}{,_load_only}_build.log`.
