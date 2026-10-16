"""Repeated BF16 GEMM: admission disabled, high cap, and a bandwidth-cap sweep.

Each worker computes C[64,128] = A[64,K] @ B[128,K].T using the existing
GEMM operator. All cases use identical work and the same compiled runtime.
Reported bandwidth counts logical TMA reads + writes, not DRAM counters.
"""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sms", type=int, default=None, help="Workers; default: available SMs, up to 132")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--passes", type=int, default=8, help="GEMMs per worker inside each launch")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5, help="Measured launches per mode")
    parser.add_argument("--high-cap-multiplier", type=float, default=10,
                        help="High cap / baseline logical GB/s; intended to avoid throttling")
    parser.add_argument("--cap-fractions", type=float, nargs="+", default=[1.0, 0.75, 0.5],
                        help="Sweep caps as fractions of baseline logical GB/s")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.sms is not None and not 1 <= args.sms <= 132:
        parser.error("sms must be in [1, 132]")
    if not 64 <= args.k <= 65472 or args.k % 64:
        parser.error("k must be a multiple of 64 in [64, 65472]")
    if not 1 <= args.passes <= 65535 or args.warmup < 0 or args.repeats < 1:
        parser.error("Invalid passes, warmup, or repeats")
    if not isfinite(args.high_cap_multiplier) or args.high_cap_multiplier <= 1:
        parser.error("high-cap-multiplier must be finite and greater than 1")
    if any(not isfinite(fraction) or not 0 < fraction <= 1 for fraction in args.cap_fractions):
        parser.error("cap-fractions must be in (0, 1]")
    return args


def build_launcher(args, a, b, c, cap=None):
    from dae.launcher import Launcher
    from dae.instructions import (
        Dummy, Gemm_M64N128K64, IssueBarrier, LoopC, LoopM, RepeatM, TmaLoad1D, TmaTensor,
    )
    from dae.tma_utils import Major

    dae = Launcher(args.sms)
    load_a = TmaTensor(dae, a).wgmma_load(64, 64, Major.K)
    load_b = TmaTensor(dae, b).wgmma_load(128, 64, Major.K)
    store_c = TmaTensor(dae, c).wgmma_store(64, 128, Major.K)
    done = [dae.new_bar(args.passes) for _ in range(args.sms)]

    def program(worker):
        return [
            Gemm_M64N128K64(args.k // 64),
            RepeatM.on(args.k // 64,
                       (load_b.cord(worker * 128, 0), load_b.cord2tma(0, 64)),
                       (load_a.cord(worker * 64, 0).port(1), load_a.cord2tma(0, 64))),
            store_c.cord(worker * 64, 0).bar(done[worker]),
            LoopC(args.passes, 0),
            LoopM(args.passes, 0),
            # A final load, gated by all output stores, makes the compute-end
            # timestamp cover completed writes too. Count its 16 bytes below.
            IssueBarrier(done[worker]),
            TmaLoad1D(c[worker * 64], bytes=16),
            Dummy(1),
        ]

    dae.s(program)
    if cap is not None:
        dae.set_bandwidth("gemm", num_sms=args.sms, bandwidth_gb_s=cap)
    return dae


def measure(dae, args, name, flops, total_bytes):
    print(f"Running {name}", flush=True)
    for _ in range(args.warmup):
        dae.launch()
    trials = []
    for _ in range(args.repeats):
        dae.launch()
        times = dae.profile[:, :2].cpu().tolist()
        if any(end <= start for start, end in times):
            raise RuntimeError("Invalid device timestamps")
        duration_ns = max(end for _, end in times) - min(start for start, _ in times)
        trials.append({"ms": duration_ns / 1e6, "tflops": flops / duration_ns / 1e3,
                       "gb_s": total_bytes / duration_ns})
    return {"trials": trials, **{key: median(t[key] for t in trials) for key in trials[0]}}


def main():
    args = parse_args()
    import torch
    from dae import runtime

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and the rebuilt Hopper runtime")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    if device.major != 9:
        raise RuntimeError("The current build targets Hopper (sm_90a)")
    required = {"OP_GEMM_M64N128K64", "OP_DUMMY", "OP_LOOPC", "OP_TERMINATEC"}
    if not hasattr(runtime, "bandwidth_transfer_costs") or required - set(runtime.supported_compute_ops):
        raise RuntimeError("Rebuild with make pyext and handlers: " + ",".join(sorted(required)))
    if args.sms is None:
        args.sms = min(device.multi_processor_count, 132)
    if args.sms > device.multi_processor_count:
        raise ValueError("Workers must fit the GPU")

    torch.manual_seed(0)
    a = torch.randn((args.sms * 64, args.k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((args.sms * 128, args.k), dtype=torch.bfloat16, device="cuda")
    c = torch.empty((args.sms * 64, 128), dtype=torch.bfloat16, device="cuda")
    read_bytes = args.sms * (args.passes * (64 + 128) * args.k * 2 + 16)
    write_bytes = args.sms * args.passes * 64 * 128 * 2
    flops = args.sms * args.passes * 2 * 64 * 128 * args.k
    print(f"{device.name}; {args.sms} workers; GEMM 64x128x{args.k}; {args.passes} passes")
    print(f"Input working set: {(a.numel() + b.numel()) * 2 / 1024**2:.1f} MiB; "
          f"logical reads/writes per launch: {read_bytes / 1e9:.3f}/{write_bytes / 1e9:.3f} GB")

    baseline = measure(build_launcher(args, a, b, c), args, "disabled", flops, read_bytes + write_bytes)
    results = [{"mode": "disabled", "cap_gb_s": None, **baseline}]
    modes = [("high_cap", args.high_cap_multiplier)]
    modes += [(f"cap_{fraction:g}x", fraction) for fraction in args.cap_fractions]
    for name, multiplier in modes:
        cap = baseline["gb_s"] * multiplier
        result = measure(build_launcher(args, a, b, c, cap), args, name, flops, read_bytes + write_bytes)
        results.append({"mode": name, "cap_gb_s": cap, **result})

    print("\n| mode | IO cap GB/s | device ms | TFLOP/s | logical TMA GB/s | slowdown |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        result["slowdown_percent"] = 100 * (result["ms"] / baseline["ms"] - 1)
        cap = "unlimited" if result["cap_gb_s"] is None else f"{result['cap_gb_s']:.2f}"
        print(f"| {result['mode']} | {cap} | {result['ms']:.3f} | {result['tflops']:.3f} | "
              f"{result['gb_s']:.2f} | {result['slowdown_percent']:+.2f}% |")
    print("Medians across launches; same compiled kernel in every mode. Logical TMA bytes include "
          "reads, writes and the final 16-byte completion load per worker; these are not DRAM counters.")
    print("High cap is intended to be nonbinding, but no admission-wait counter is collected. "
          "Lower-cap slowdowns also include operand starvation, not just limiter logic.")
    if args.json:
        parameters = vars(args).copy()
        parameters.pop("json")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"device": device.name, "parameters": parameters,
                                       "read_bytes": read_bytes, "write_bytes": write_bytes,
                                       "flops": flops, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
