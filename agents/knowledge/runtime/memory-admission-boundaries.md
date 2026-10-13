# Memory Admission Boundaries

Source inspection on 2026-10-12, baseline commit `fd75b8d`. No runtime changes or GPU measurements.

## Execution model

- `python/dae/launcher.py:SMInstructionBuilder.add` splits a composed schedule into separate compute and memory streams per logical worker. `Schedule.place` selects worker indices; it is not physical SM affinity.
- `src/runtime.cu:launch_dae` launches `numSMs` blocks of 256 threads and synchronizes the device before and after launch. The current host API is blocking, not a continuously serving executor.
- `include/dae/dae2.cuh` uses `blockIdx.x` as the worker index. Threads 0–127 execute compute instructions; 128–159 allocate/interpret memory instructions; lane 0 of warps 5, 6, and 7 executes stores, load port 0, and load port 1 respectively.
- Current `context.cuh` provides 24 normal 8-KiB slots, 9 special metadata slots, 512 instructions per stream, and 32-entry queues. `dae2LoadInstructions=true` copies both streams into shared memory at launch.
- Compute dispatch runs a whole operator handler before returning to its PC loop. A GEMV/attention instruction can contain many tile iterations; there is no generic task switch at each tensor instruction.

## Transfer and dependency protocol

1. `pipeline/allocwarp.cuh` rewrites addresses/resource IDs, allocates a contiguous shared-memory slot span, and records `MInst` in `st_insts[lead_slot]`.
2. It writes a placeholder into `m2c` and sends `LdCmd {slot, bar, opcode}` through the selected `m2ld` queue. `LdCmd.bar` is the **m2c queue position**, not the global dependency-barrier ID.
3. `pipeline/ldwarp.cuh` optionally waits for the global read dependency, issues a TMA load, registers expected transaction bytes, and arrives on the corresponding `m2c` barrier. Output allocations have no load: this arrival only makes the output slot available to compute.
4. Compute consumes `m2c` in FIFO order. Its 128 participants plus the loader's arrival and, for loads, TMA transaction completion gate consumption. The two ports can issue/complete independently, but compute does not skip a blocked FIFO entry.
5. Compute waits for asynchronous tensor work before releasing its input buffers. Plain `c2m.push` frees normal slots immediately; `push<..., true>` instead queues writeback/completion work.
6. `pipeline/stwarp.cuh` issues TMA stores/reductions. With `BARRIER`, it waits for full bulk-group completion and decrements `bars[inst.bar()]`. Otherwise it waits for shared-source reads to finish before freeing the slot; that wait is not full global-store completion.

Global `bars[]` are initialized countdowns. Loads with `BARRIER` and `OP_ISSUE_BARRIER` poll until zero; producer writebacks decrement after completion. They are separate from the per-block queue barriers. `ResourceGroup`/`GROUP` advance descriptor and barrier IDs across iterations; repeated schedules must provide fresh or explicitly restored barrier state.

For example, `app/python/global_barrier.py` connects a first projection's stores to a second projection's loads with one counter initialized to the producer count. This illustrates the protocol; the example's operator selection was not execution-tested.

This is a prebuilt instruction-stream executor with blocking dependency waits, not an arbitrary ready-node DAG scheduler. Adding a delay at transfer issue preserves stream order, but adds backpressure through occupied slots and FIFO queues. Existing slot/queue reuse assumptions must survive any scheduling change.

## Head-of-line blocking scope

The two load issuers and one store issuer are **per block**, not GPU-wide hardware-channel allocations. Loads can issue additional asynchronous transfers without waiting for each prior transfer to finish. Component count alone therefore does not create a two-component limit. With disjoint worker sets, components have independent software queues and issuers, though they still contend for hardware bandwidth.

If multiple components share one block's streams, an unready or throttled load can block later commands on its port; the common compute FIFO can also block on the earliest token. The store executor waits after each issued store before fetching the next writeback, so cross-component store serialization can appear with two components already. Supporting same-block multiplexing requires separate component readiness/queue state and nonblocking issue selection, not merely more load/store warps.

## Boundaries relevant to future rate limiting

- **Reads:** actual issue is in `pipeline/ldwarp.cuh`, after the global dependency wait and before the load opcode switch. Both load ports need coordinated accounting.
- **Writes:** actual issue is in `pipeline/stwarp.cuh`, before the store/reduce switch. Allocating an output slot is not issuing a write and should not consume its transfer credit.
- **Bytes:** `MInst.size` is logical transfer bytes for TMA operations, but is a repeat count/register ID/etc. for other opcodes. Accounting must classify opcodes, not charge every instruction. Descriptor-backed transfer size must agree with the descriptor tile; reductions also access the destination.
- **Scope:** no task ID, bandwidth budget, burst limit, or shared admission state exists in the current launch interface. A per-task GPU-wide cap must coordinate all participating blocks and both ports; independent equal caps can strand capacity and allow synchronized bursts.
- **Bypasses:** `RawAddress` only transports a pointer. Argmax and split-attention LSE paths issue ordinary global accesses in compute code (`include/task/argmax.cuh`, `include/task/attention.cuh`). Token reads, barrier polling, and runtime metadata also generate traffic. A TMA-only limiter would not cover them.
- **Effect:** logical admitted bytes are not measured HBM bytes because of caching, transaction behavior, and in-flight work. Limiting issue does not cancel issued transfers or provide a minimum service guarantee by itself.
- **Progress:** throttled stores retain slots and delay downstream barriers. Credit policy must allow dependency-producing writes to progress; zero budgets or priority policies can otherwise block consumers indefinitely.

## Inspection limits

The generated selection currently says `source=default:full`; `dynamic_compute_handlers.inc` is empty. Older VM notes contain historical compiled-mode and selection observations; do not assume those describe this launch path. No `nvcc` or `nvidia-smi` was found on the inspection host, so the findings are source-level only.
