"""Sweep independent two-component worker and TMA-bandwidth partitions.

Both components issue TMA loads, not compute-heavy kernels. Traffic is
proportional to the requested bandwidth split so ideal completion times match.
Each capped case is compared with identical work using SM-only allocation.
"""

import argparse
import json
from math import gcd, lcm
from pathlib import Path

from bandwidth_load import build_launcher, measure


def ratio(value):
    try:
        first, second = map(int, value.split(":"))
        if first <= 0 or second <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Use two positive integer weights, e.g. 1:3") from None
    divisor = gcd(first, second)
    return first // divisor, second // divisor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sm-splits", type=ratio, nargs="+", default=[(1, 1), (1, 3), (3, 1)])
    parser.add_argument("--bw-splits", type=ratio, nargs="+", default=[(1, 1), (1, 3), (3, 1)])
    parser.add_argument("--total-cap-fraction", type=float, default=0.7,
                        help="Total component caps / measured all-worker reference (default: 0.7)")
    parser.add_argument("--sms", type=int, default=None)
    parser.add_argument("--tile-bytes", type=int, default=32768,
                        help="TMA payload bytes (default: 32768, calibrated for high BW with fewer workers)")
    parser.add_argument("--loads-per-sm", type=int, default=256,
                        help="Target average loads/worker/pass; rounded up to preserve exact BW-weighted traffic")
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--burst-bytes-per-sm", type=int, default=65536)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not 0 < args.total_cap_fraction <= 1:
        parser.error("total-cap-fraction must be in (0, 1]")
    if args.sms is not None and not 2 <= args.sms <= 132:
        parser.error("sms must be in [2, 132]")
    if not 0 < args.tile_bytes <= 65520 or args.tile_bytes % 16:
        parser.error("tile-bytes must be a positive multiple of 16, at most 65520")
    if not 2 <= args.loads_per_sm <= 65534 or args.loads_per_sm % 2:
        parser.error("loads-per-sm must be even and in [2, 65534]")
    if not 1 <= args.passes <= 65535 or args.warmup < 0 or args.repeats < 1:
        parser.error("Invalid passes, warmup, or repeats")
    if not args.tile_bytes <= args.burst_bytes_per_sm <= 2**53:
        parser.error("burst-bytes-per-sm must cover a tile and be at most 2**53")
    if len(set(args.sm_splits)) != len(args.sm_splits) or len(set(args.bw_splits)) != len(args.bw_splits):
        parser.error("Split ratios must be unique")
    return args


def workload(args, sm_weights, bw_weights):
    boundary = args.sms * sm_weights[0] // sum(sm_weights)
    sizes = [boundary, args.sms - boundary]
    if min(sizes) < 1:
        raise ValueError("SM split leaves an empty component")
    # pairs_i = weight_i * quantum must divide evenly over that group's workers.
    quantum = lcm(*(size // gcd(size, weight) for size, weight in zip(sizes, bw_weights)))
    divisor = 2 * sum(bw_weights) * quantum
    quantum *= (args.sms * args.loads_per_sm + divisor - 1) // divisor
    group_loads = [2 * quantum * weight // size for size, weight in zip(sizes, bw_weights)]
    if not all(2 <= count <= 65534 for count in group_loads):
        raise ValueError("Weighted per-worker loads exceed instruction limits; reduce --loads-per-sm")
    scopes = {"component_0": (0, boundary), "component_1": (boundary, args.sms)}
    loads = [group_loads[0]] * sizes[0] + [group_loads[1]] * sizes[1]
    return scopes, loads


def main():
    args = parse_args()
    import torch
    from dae import runtime

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.major != 9:
        raise RuntimeError("The current build targets Hopper (sm_90a)")
    required = {"OP_DUMMY", "OP_LOOPC", "OP_TERMINATEC"}
    if not hasattr(runtime, "bandwidth_transfer_costs") or required - set(runtime.supported_compute_ops):
        raise RuntimeError("Rebuild dae.runtime with the bandwidth implementation and load benchmark handlers")
    if args.sms is None:
        args.sms = min(properties.multi_processor_count, 132)
    if args.sms > properties.multi_processor_count:
        raise ValueError("Workers must fit the GPU")
    plans = [(sm, bw, *workload(args, sm, bw)) for sm in args.sm_splits for bw in args.bw_splits]
    source_loads = max(args.loads_per_sm, *(max(loads) for _, _, _, loads in plans))
    torch.manual_seed(0)
    source = torch.empty((args.sms, source_loads * args.tile_bytes),
                         dtype=torch.uint8, device="cuda").random_(0, 256)
    torch.cuda.synchronize()
    reference_result = measure(build_launcher(source, args), args, "reference", components=0)
    reference = reference_result["summary"]["all"]["gb_s"]
    total_cap = reference * args.total_cap_fraction
    print(f"Device: {properties.name}; workers: {args.sms}; reference: {reference:.2f} GB/s; "
          f"total cap: {total_cap:.2f} GB/s", flush=True)
    cases = []
    for sm, bw, scopes, loads in plans:
        label = f"sm_{sm[0]}_{sm[1]}_bw_{bw[0]}_{bw[1]}"
        caps = [total_cap * weight / sum(bw) for weight in bw]
        budgets = [(name, begin, end, cap) for (name, (begin, end)), cap in zip(scopes.items(), caps)]
        print(f"{label}; workers: {[end-begin for begin,end in scopes.values()]}; caps: {caps}", flush=True)
        uncapped = measure(build_launcher(source, args, worker_loads=loads),
                           args, label + "_sm_only", worker_loads=loads, component_scopes=scopes)
        capped = measure(build_launcher(source, args, worker_loads=loads, component_budgets=budgets),
                         args, label + "_capped", worker_loads=loads, component_scopes=scopes)
        cases.append({
            "label": label,
            "sm_weights": sm,
            "bw_weights": bw,
            "partitions": {name: {"base_sm": begin, "num_sms": end-begin, "cap_gb_s": cap}
                           for name, begin, end, cap in budgets},
            "sm_only": uncapped,
            "capped": capped,
        })

    print("\n| workers A/B | BW split A/B | cap A/B GB/s | SM-only A/B GB/s | capped A/B GB/s | capped aggregate GB/s |")
    print("| --- | --- | --- | --- | --- | ---: |")
    for case in cases:
        parts = list(case["partitions"].values())
        names = list(case["partitions"])
        caps = "/".join(f"{p['cap_gb_s']:.2f}" for p in parts)
        sizes = "/".join(str(p['num_sms']) for p in parts)
        raw = "/".join(f"{case['sm_only']['summary'][name]['gb_s']:.2f}" for name in names)
        limited = "/".join(f"{case['capped']['summary'][name]['gb_s']:.2f}" for name in names)
        print(f"| {sizes} | {case['bw_weights'][0]}:{case['bw_weights'][1]} | {caps} | {raw} | {limited} | "
              f"{case['capped']['summary']['all']['gb_s']:.2f} |")
    print("Component rates use their own active windows; aggregate uses the full completion window. "
          "SM-only rates can include a period after the other component finishes; they are not simultaneous shares.")
    print("Workload bytes follow BW weights, not SM weights. Rates count actual rounded load pairs. "
          "Both components are load-only; no compute throughput is measured. Logical TMA GB/s is not a DRAM counter.")
    if args.json:
        parameters = vars(args).copy()
        parameters.pop("json")
        payload = {
            "metric": "logical_tma_load_gb_s",
            "device": properties.name,
            "parameters": parameters,
            "workload_balance": "bytes_proportional_to_bandwidth_weights",
            "source_allocation_bytes": source.numel(),
            "best_effort_reference_gb_s": reference,
            "total_cap_gb_s": total_cap,
            "reference": reference_result,
            "cases": cases,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
