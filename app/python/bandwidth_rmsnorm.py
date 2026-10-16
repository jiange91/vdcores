"""Existing CUDA-core RMSNorm with TMA IO and interleaved bandwidth-cap modes.

One 4096-element BF16 row per operation; no Tensor Core compute. Each worker
streams distinct input/output rows and reloads its weight vector for each row.
Bandwidth is logical TMA reads + writes, including cached weights, not HBM IO.
"""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sms", type=int, default=None)
    parser.add_argument("--rows", type=int, default=512, help="Distinct rows per worker")
    parser.add_argument("--passes", type=int, default=8, help="On-device passes over those rows")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10, help="Interleaved measurement rounds")
    parser.add_argument("--high-cap-multiplier", type=float, default=1000)
    parser.add_argument("--cap-fractions", type=float, nargs="+", default=[1, 0.75, 0.5])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.sms is not None and not 1 <= args.sms <= 132:
        parser.error("sms must be in [1, 132]")
    if not 1 <= args.rows <= 65535 or not 1 <= args.passes <= 65535:
        parser.error("rows and passes must be in [1, 65535]")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    if not isfinite(args.high_cap_multiplier) or args.high_cap_multiplier <= 1:
        parser.error("high-cap-multiplier must be finite and greater than 1")
    if any(not isfinite(f) or not 0 < f <= 1 for f in args.cap_fractions):
        parser.error("cap-fractions must be in (0, 1]")
    return args


def build_launcher(args, weights, inputs, outputs, cap=None):
    from dae.launcher import Launcher
    from dae.instructions import (
        Dummy, IssueBarrier, LoopC, LoopM, RepeatM,
        RMS_NORM_F16_K_4096_SMEM, TmaLoad1D, TmaStore1D,
    )

    dae = Launcher(args.sms)
    done = [dae.new_bar(args.rows * args.passes) for _ in range(args.sms)]

    def program(worker):
        return [
            RMS_NORM_F16_K_4096_SMEM(1, 1e-5),
            LoopC(args.rows, 0),
            LoopC(args.passes, 0, reg=1),
            RepeatM.on(args.rows,
                       (TmaLoad1D(weights[worker]), 0),
                       (TmaLoad1D(inputs[worker, 0]).port(1), 8192),
                       (TmaStore1D(outputs[worker, 0]).bar(done[worker]), 8192)),
            LoopM(args.passes, 0),
            # Include full output-store completion in the device timing window.
            IssueBarrier(done[worker]),
            TmaLoad1D(outputs[worker, 0], bytes=16),
            Dummy(1),
        ]

    dae.s(program)
    if cap is not None:
        dae.set_bandwidth("rmsnorm", num_sms=args.sms, bandwidth_gb_s=cap)
    return dae


def measure(dae, total_bytes, elements):
    dae.launch()
    times = dae.profile[:, :2].cpu().tolist()
    if any(end <= start for start, end in times):
        raise RuntimeError("Invalid device timestamps")
    ns = max(end for _, end in times) - min(start for start, _ in times)
    return {"ms": ns / 1e6, "gb_s": total_bytes / ns, "gelem_s": elements / ns}


def main():
    args = parse_args()
    import torch
    from dae import runtime

    device = torch.cuda.get_device_properties(0)
    required = {"OP_RMS_NORM_F16_K_4096_SMEM", "OP_DUMMY", "OP_LOOPC", "OP_TERMINATEC"}
    if device.major != 9 or required - set(runtime.supported_compute_ops):
        raise RuntimeError("Requires Hopper and make pyext with RMSNorm, Dummy and LoopC handlers")
    args.sms = args.sms or min(device.multi_processor_count, 132)
    if args.sms > device.multi_processor_count:
        raise ValueError("Workers must fit the GPU")
    torch.manual_seed(0)
    weights = torch.randn((args.sms, 4096), dtype=torch.bfloat16, device="cuda")
    inputs = torch.randn((args.sms, args.rows, 4096), dtype=torch.bfloat16, device="cuda")
    outputs = torch.empty_like(inputs)
    elements = inputs.numel() * args.passes
    read_bytes = elements * 4 + args.sms * 16
    write_bytes = elements * 2
    total_bytes = read_bytes + write_bytes
    print(f"{device.name}; {args.sms} workers; {args.rows} rows/worker; {args.passes} passes", flush=True)
    print(f"Input: {inputs.numel() * 2 / 1024**2:.1f} MiB; logical reads/writes: "
          f"{read_bytes / 1e9:.3f}/{write_bytes / 1e9:.3f} GB", flush=True)

    baseline = build_launcher(args, weights, inputs, outputs)
    for _ in range(args.warmup):
        baseline.launch()
    reference = median(measure(baseline, total_bytes, elements)["gb_s"] for _ in range(3))
    modes = [("disabled", None), ("high_cap", reference * args.high_cap_multiplier)]
    modes += [(f"cap_{f:g}x", reference * f) for f in args.cap_fractions]
    launchers = [baseline] + [build_launcher(args, weights, inputs, outputs, cap) for _, cap in modes[1:]]
    for dae in launchers:
        for _ in range(args.warmup):
            dae.launch()
    rounds = []
    for i in range(args.repeats):
        order = list(range(i % len(modes), len(modes))) + list(range(i % len(modes)))
        samples = {modes[j][0]: measure(launchers[j], total_bytes, elements) for j in order}
        rounds.append({"order": [modes[j][0] for j in order], "results": samples})
        print(f"Finished round {i + 1}/{args.repeats}", flush=True)

    results = []
    print("\n| mode | cap GB/s | device ms | G elements/s | logical TMA GB/s | paired slowdown |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, cap in modes:
        summary = {key: median(r["results"][name][key] for r in rounds)
                   for key in ("ms", "gb_s", "gelem_s")}
        summary["paired_slowdown_percent"] = median(
            100 * (r["results"][name]["ms"] / r["results"]["disabled"]["ms"] - 1) for r in rounds)
        results.append({"mode": name, "cap_gb_s": cap, **summary})
        cap_str = "unlimited" if cap is None else f"{cap:.2f}"
        print(f"| {name} | {cap_str} | {summary['ms']:.3f} | {summary['gelem_s']:.3f} | "
              f"{summary['gb_s']:.2f} | {summary['paired_slowdown_percent']:+.2f}% |")
    print("Same binary and operands. Rotating mode order; slowdown uses paired per-round ratios. "
          "High cap is intended to avoid throttling; admission waits are not counted. "
          "Logical TMA bandwidth includes reloaded weights, which may hit L2.")
    if args.json:
        parameters = vars(args).copy()
        parameters.pop("json")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"device": device.name, "parameters": parameters,
                                       "read_bytes": read_bytes, "write_bytes": write_bytes,
                                       "elements": elements, "reference_gb_s": reference,
                                       "results": results, "rounds": rounds}, indent=2) + "\n")


if __name__ == "__main__":
    main()
