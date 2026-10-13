#pragma once

#include "context.cuh"
#include <cuda/atomic>

// ABI shared with python/dae/bandwidth.py. Zero rate disables the limiter.
struct alignas(8) BandwidthConfig {
  double bytes_per_ns;
  uint64_t burst_bytes;
};
static_assert(sizeof(BandwidthConfig) == 16);

// Callers serialize access; this class has no synchronization of its own.
struct BandwidthBucket {
  uint64_t last_refill_ns;
  double credit_bytes;

  __device__ __forceinline__ void init(const BandwidthConfig &config, uint64_t now) {
    last_refill_ns = now;
    credit_bytes = static_cast<double>(config.burst_bytes);
  }

  __device__ __forceinline__ bool try_take(const BandwidthConfig &config, uint32_t bytes,
                                         uint64_t now) {
    if (now > last_refill_ns) {
      const double refill = static_cast<double>(now - last_refill_ns) * config.bytes_per_ns;
      const double available = credit_bytes + refill;
      const double capacity = static_cast<double>(config.burst_bytes);
      credit_bytes = available < capacity ? available : capacity;
      last_refill_ns = now;
    }
    if (credit_bytes < bytes)
      return false;
    credit_bytes -= bytes;
    return true;
  }

  __device__ __forceinline__ uint32_t retry_ns(const BandwidthConfig &config, uint32_t bytes,
                                             uint32_t max_sleep_ns) const {
    const double wait = (static_cast<double>(bytes) - credit_bytes) / config.bytes_per_ns;
    if (wait >= max_sleep_ns)
      return max_sleep_ns;
    return wait > 0 ? static_cast<uint32_t>(wait) + 1 : 1;
  }
};

__device__ __forceinline__ uint32_t tma_load_bytes(const MInst &inst) {
  switch (op(inst.opcode)) {
    #define DAE_BW_LOAD(name, multiplier) case op(name): return uint32_t(inst.size) * multiplier;
    #define DAE_BW_STORE(name, multiplier)
    #define DAE_BW_UNSUPPORTED(name)
    #include "bandwidth_ops.inc"
    #undef DAE_BW_LOAD
    #undef DAE_BW_STORE
    #undef DAE_BW_UNSUPPORTED
    default: return 0;
  }
}

__device__ __forceinline__ uint32_t tma_store_bytes(const MInst &inst) {
  switch (op(inst.opcode)) {
    #define DAE_BW_LOAD(name, multiplier)
    #define DAE_BW_STORE(name, multiplier) case op(name): return uint32_t(inst.size) * multiplier;
    #define DAE_BW_UNSUPPORTED(name)
    #include "bandwidth_ops.inc"
    #undef DAE_BW_LOAD
    #undef DAE_BW_STORE
    #undef DAE_BW_UNSUPPORTED
    default: return 0;
  }
}

// One shared-memory instance per block, used by both LD lanes and the ST lane.
struct BlockBandwidthLimiter {
  BandwidthConfig config;
  BandwidthBucket bucket;
  unsigned int lock;

  __device__ __forceinline__ void init(const BandwidthConfig &cfg, uint64_t now) {
    config = cfg;
    bucket.init(cfg, now);
    lock = 0;
  }

  __device__ __forceinline__ void admit(uint32_t bytes) {
    if (bytes == 0)
      return;
    // Python validates this before launch. Trap rather than hang for malformed
    // configurations supplied directly through the low-level launch interface.
    if (bytes > config.burst_bytes || !(config.bytes_per_ns > 0)) {
      asm volatile("trap;");
      return;
    }
    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> gate(lock);
    for (;;) {
      unsigned int expected = 0;
      uint32_t sleep_ns = 16;
      if (gate.compare_exchange_strong(expected, 1, cuda::memory_order_acquire,
                                       cuda::memory_order_relaxed)) {
        const bool admitted = bucket.try_take(config, bytes, cuda::ptx::get_sreg_globaltimer());
        if (!admitted)
          sleep_ns = bucket.retry_ns(config, bytes, 1024);
        gate.store(0, cuda::memory_order_release);
        if (admitted)
          return;
      }
      // Never wait for refill while holding the shared admission lock.
      __nanosleep(sleep_ns);
    }
  }
};
