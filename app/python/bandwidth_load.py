"""Compare unlimited TMA loads with equal bandwidth partitions.

Only workload loads are issued: no TMA stores or tensor computation. The current
runtime's dummy consumer waits for completed loads and releases their slots.
Reported GB/s uses requested load bytes and device timestamps, not DRAM counters.
"""

import argparse
import json
from pathlib import Path
from statistics import median


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sms", type=int, default=None,
                        help="Total worker blocks; default: available SMs, up to 132")
    parser.add_argument("--components", type=int, nargs="+", default=[2],
                        help="Component counts to run sequentially, e.g. 2 4 6 8; remainder workers go to last")
    parser.add_argument("--tile-bytes", type=int, default=8192)
    parser.add_argument("--loads-per-sm", type=int, default=1024,
                        help="Even number of unique load tiles per worker per pass")
    parser.add_argument("--passes", type=int, default=8,
                        help="Device-side passes over the input per launch")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--burst-bytes-per-sm", type=int, default=65536)
    parser.add_argument("--json", type=Path, default=None, help="Optional output path")
    args = parser.parse_args()
    if any(count < 2 or count > 132 for count in args.components):
        parser.error("components must be in [2, 132]")
    if len(set(args.components)) != len(args.components):
        parser.error("component counts must be unique")
    if args.sms is not None and not max(args.components) <= args.sms <= 132:
        parser.error("sms must be between the largest component count and 132")
    if not 0 < args.tile_bytes <= 65520 or args.tile_bytes % 16:
        parser.error("tile-bytes must be a positive multiple of 16, at most 65520")
    if not 2 <= args.loads_per_sm <= 65534 or args.loads_per_sm % 2:
        parser.error("loads-per-sm must be even and in [2, 65534]")
    if not 1 <= args.passes <= 65535:
        parser.error("passes must be in [1, 65535]")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")
    if not args.tile_bytes <= args.burst_bytes_per_sm <= 2**53:
        parser.error("burst-bytes-per-sm must cover a tile and be at most 2**53")
    return args


def component_ranges(sms, components):
    """Contiguous worker ranges; only the last receives the remainder."""
    width = sms // components
    if width < 1:
        raise ValueError("Each component needs at least one worker")
    return {
        f"component_{index}": (index * width, (index + 1) * width if index < components - 1 else sms)
        for index in range(components)
    }


def build_launcher(source, args, total_cap_gb_s=None, components=2):
    from dae.launcher import Launcher
    from dae.instructions import Dummy, LoopC, LoopM, RepeatM, TmaLoad1D

    dae = Launcher(args.sms)

    def program(worker):
        load0 = TmaLoad1D(source[worker], bytes=args.tile_bytes)
        load1 = load0.cord(args.tile_bytes).port(1)
        return [
            Dummy(args.loads_per_sm),
            RepeatM.on(
                args.loads_per_sm // 2,
                (load0, 2 * args.tile_bytes),
                (load1, 2 * args.tile_bytes),
            ),
            # Both streams restart at PC 0; REPEAT resets the address offsets.
            LoopC(args.passes, 0),
            LoopM(args.passes, 0),
        ]

    dae.s(program)
    if total_cap_gb_s is not None:
        for name, (begin, end) in component_ranges(args.sms, components).items():
            dae.set_bandwidth(name, num_sms=end - begin, base_sm=begin,
                              bandwidth_gb_s=total_cap_gb_s / components,
                              burst_bytes_per_sm=args.burst_bytes_per_sm)
    return dae


def measure(dae, args, label, components=2):
    for iteration in range(args.warmup):
        print(f"[launch] {label} warmup {iteration + 1}/{args.warmup}", flush=True)
        dae.launch()

    bytes_per_worker = args.loads_per_sm * args.passes * args.tile_bytes
    scopes = {"all": (0, args.sms)}
    if components:
        scopes.update(component_ranges(args.sms, components))
    trials = []
    for iteration in range(args.repeats):
        print(f"[launch] {label} measured {iteration + 1}/{args.repeats}", flush=True)
        dae.launch()
        timestamps = dae.profile[:, :2].cpu().tolist()
        trial = {}
        for scope, (begin, end) in scopes.items():
            selected = timestamps[begin:end]
            if any(stop <= start for start, stop in selected):
                raise RuntimeError("Invalid device timestamps; increase workload size or check runtime execution")
            start_ns = min(start for start, _ in selected)
            end_ns = max(stop for _, stop in selected)
            duration_ns = end_ns - start_ns
            load_bytes = (end - begin) * bytes_per_worker
            trial[scope] = {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": duration_ns / 1e6,
                "load_bytes": load_bytes,
                # bytes/ns is numerically decimal GB/s.
                "gb_s": load_bytes / duration_ns,
            }
        trial["worker_timestamps_ns"] = timestamps
        trials.append(trial)
    summary = {
        scope: {
            "gb_s": median(t[scope]["gb_s"] for t in trials),
            "duration_ms": median(t[scope]["duration_ms"] for t in trials),
        }
        for scope in scopes
    }
    return {"summary": summary, "trials": trials}


def main():
    args = parse_args()
    # Keep --help available without importing the CUDA extension or Torch.
    import torch
    from dae import runtime

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and the rebuilt Hopper dae.runtime extension")
    if not hasattr(runtime, "bandwidth_transfer_costs"):
        raise RuntimeError("Rebuild dae.runtime with make pyext before running this benchmark")
    required = {"OP_DUMMY", "OP_LOOPC", "OP_TERMINATEC"}
    missing = required - set(runtime.supported_compute_ops)
    if missing:
        raise RuntimeError(
            f"Missing compute handlers {sorted(missing)}; rebuild with "
            "make pyext DAE_COMPUTE_OPS=OP_DUMMY,OP_LOOPC,OP_TERMINATEC")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.major != 9:
        raise RuntimeError("The current build targets Hopper (sm_90a)")
    if args.sms is None:
        args.sms = min(properties.multi_processor_count, 132)
    if args.sms < max(args.components) or args.sms > properties.multi_processor_count:
        raise ValueError("Worker count must fit the GPU and allow nonempty partitions")

    working_bytes = args.sms * args.loads_per_sm * args.tile_bytes
    print(f"Device: {properties.name}; workers: {args.sms}; component sweep: {args.components}")
    print(f"Working set: {working_bytes / 1024**2:.1f} MiB; "
          f"load traffic per launch: {working_bytes * args.passes / 1e9:.3f} GB")
    print("Both load ports active; no workload stores. Using current kernel for all cases.")
    torch.manual_seed(0)
    source = torch.empty((args.sms, args.loads_per_sm * args.tile_bytes),
                         dtype=torch.uint8, device="cuda").random_(0, 256)
    torch.cuda.synchronize()

    best_effort = measure(build_launcher(source, args), args, "best_effort", components=0)
    reference = best_effort["summary"]["all"]["gb_s"]
    print(f"Best-effort reference: {reference:.2f} GB/s", flush=True)
    partitioned = {}
    rows = [("best_effort", "all", None, best_effort)]
    for count in args.components:
        ranges = component_ranges(args.sms, count)
        cap = reference / count
        print(f"Components: {count}; workers: {[end - begin for begin, end in ranges.values()]}; "
              f"cap per component: {cap:.2f} GB/s", flush=True)
        result = measure(build_launcher(source, args, reference, count), args, f"split_{count}", count)
        partitioned[str(count)] = {
            "component_cap_gb_s": cap,
            "partitions": {name: {"base_sm": begin, "num_sms": end - begin}
                           for name, (begin, end) in ranges.items()},
            **result,
        }
        rows += [(f"split_{count}", scope, cap, result) for scope in ranges]
        rows.append((f"split_{count}", "all", reference, result))

    print("\n| case | scope | cap (GB/s) | achieved (GB/s) | % best effort | duration (ms) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for case, scope, cap, result in rows:
        stats = result["summary"][scope]
        cap_text = "unlimited" if cap is None else f"{cap:.2f}"
        print(f"| {case} | {scope} | {cap_text} | {stats['gb_s']:.2f} | "
              f"{100 * stats['gb_s'] / reference:.1f}% | {stats['duration_ms']:.3f} |")
    print("\nMedians across measured launches. Component rates use each component's own "
          "start-to-finish window; aggregate rates use the whole worker set.")
    print("Work is equal per worker. The last component may have more workers and bytes "
          "but the same bandwidth cap, creating a longer tail when worker counts do not divide evenly.")
    print("GB/s is logical TMA-load throughput, assuming cache misses; it is not a DRAM-counter "
          "measurement. Use a working set well beyond L2. Initial burst credits can raise "
          "short-run average rates above the configured cap.")
    if args.json:
        parameters = vars(args).copy()
        parameters.pop("json")
        payload = {
            "metric": "logical_tma_load_gb_s",
            "device": properties.name,
            "parameters": parameters,
            "working_set_bytes": working_bytes,
            "best_effort_reference_gb_s": reference,
            "best_effort": best_effort,
            "partitioned": partitioned,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
