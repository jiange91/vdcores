"""Launch-time bandwidth partitions; independent of Torch/CUDA for validation."""

from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class BandwidthPartition:
    name: str
    num_sms: int
    base_sm: int
    bandwidth_gb_s: float | None
    burst_bytes_per_sm: int


class BandwidthPartitions:
    def __init__(self, num_sms: int):
        self.num_sms = num_sms
        self.partitions: dict[str, BandwidthPartition] = {}

    def set(self, name, *, num_sms, base_sm=0, bandwidth_gb_s=None,
            burst_bytes_per_sm=64 * 1024):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Bandwidth partition name must be a nonempty string")
        for label, value in (("num_sms", num_sms), ("base_sm", base_sm),
                             ("burst_bytes_per_sm", burst_bytes_per_sm)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer")
        if num_sms <= 0 or base_sm < 0 or base_sm + num_sms > self.num_sms:
            raise ValueError("Bandwidth partition must be within the launch's worker range")
        # Credit is represented as a double; keep integer byte counts exact.
        if not 0 < burst_bytes_per_sm <= 2**53:
            raise ValueError("burst_bytes_per_sm must be in [1, 2**53]")
        if bandwidth_gb_s is not None:
            if isinstance(bandwidth_gb_s, bool):
                raise TypeError("bandwidth_gb_s must be a positive number or None")
            bandwidth_gb_s = float(bandwidth_gb_s)
            local_rate = bandwidth_gb_s / num_sms
            if not math.isfinite(local_rate) or local_rate <= 0:
                raise ValueError("bandwidth_gb_s must give a finite, positive per-worker rate")
        for other in self.partitions.values():
            if other.name != name and max(base_sm, other.base_sm) < min(
                    base_sm + num_sms, other.base_sm + other.num_sms):
                raise ValueError(f"Bandwidth partition overlaps {other.name!r}")
        self.partitions[name] = BandwidthPartition(
            name, num_sms, base_sm, bandwidth_gb_s, burst_bytes_per_sm)

    @property
    def enabled(self):
        return any(p.bandwidth_gb_s is not None for p in self.partitions.values())

    def pack(self, memory_streams, transfer_costs, unsupported_ops):
        """Pack BandwidthConfig {double bytes_per_ns; uint64 burst_bytes} rows.

        Opcode keys omit the six runtime flag bits. Streams contain instructions
        with opcode/size fields, including already-built and newly queued ones.
        """
        if not self.enabled:
            return None
        if len(memory_streams) != self.num_sms:
            raise ValueError("Expected one memory stream per worker")
        records = [(0.0, 0)] * self.num_sms
        for partition in self.partitions.values():
            if partition.bandwidth_gb_s is None:
                continue
            # Decimal GB/s is numerically bytes/ns.
            rate = partition.bandwidth_gb_s / partition.num_sms
            for worker in range(partition.base_sm, partition.base_sm + partition.num_sms):
                for inst in memory_streams[worker]:
                    kind = int(inst.opcode) >> 6
                    if kind in unsupported_ops:
                        raise ValueError(
                            f"Bandwidth partition {partition.name!r}, worker {worker}: "
                            "raw-address traffic bypasses the TMA limiter")
                    multiplier = transfer_costs.get(kind, 0)
                    if multiplier:
                        charged_bytes = int(inst.size) * multiplier
                        if charged_bytes <= 0 or charged_bytes > partition.burst_bytes_per_sm:
                            raise ValueError(
                                f"Bandwidth partition {partition.name!r}, worker {worker}: "
                                f"transfer cost {charged_bytes} exceeds the local burst "
                                "or is nonpositive; increase burst_bytes_per_sm")
                records[worker] = (rate, partition.burst_bytes_per_sm)
        return b"".join(struct.pack("<dQ", rate, burst) for rate, burst in records)
