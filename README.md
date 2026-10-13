# VDCores

[![CUDA](https://img.shields.io/badge/CUDA-13.0-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Extension-ee4c2c?logo=pytorch)](https://pytorch.org/)

VDCores is a research runtime and programming interface for modern asynchronous GPUs. It decouples GPU kernels into async executing memory and compute virtual cores and reconnects them through explicit dependencies.

Decoupling brings three key benefits:

**Asynchrony with Simplicity.** Kernel writers can focus on the computation logic while the runtime handles dynamic memory allocation, data movement, and dependency tracking under the hood. See a compact example in [`GEMV on VDCores`](include/task/gemv.cuh).

**Overlapping for free.** The VDCores runtime automatically exploits prefetching, overlap, and scheduling opportunities. Say goodbye to manually fusing stages or hand-engineering overlap for every single workload.

**Compose to adapt, on the fly.** Recompose VDCores memory, compute, and control blocks to explore schedules and swap execution plans. Adapt to changing resources or input batches by changing how VDCores instructions are connected.

Learn more about VDCores in our [paper](https://arxiv.org/abs/2605.03190) and
[blog post](https://mlsys.wuklab.io/posts/vdcores/).

## Llama 3.1-8B-Instruct Decoding Demo

The repository includes a decoding demo for `meta-llama/Llama-3.1-8B` in [`app/python/llama3/sched.py`](app/python/llama3/sched.py).

![VDCores Performance on LLama3.1](assets/llama3_8b_decoding_performance_bf16_handdraw_flat.png)




VDCores currently supports Hopper GPUs with 132SMs (eg, GH200, H100 NVL) . For the cleanest setup, we recommend starting from a fresh environment with CUDA 13.0, following [`setup.sh`](setup.sh) as the reference setup path.

Typical setup:

```bash
# 1) Build the runtime object and Python extension
make pyext

# 2) Provide a Hugging Face token for gated model access
export HF_TOKEN=...

# 3) Optimize runtime and run demo
python app/python/llama3/sched.py -w
make pyext
python app/python/llama3/sched.py -N 256 "Write a hello world in Python."
```

Notes:

- A clean environment with CUDA 13.0 is recommended. If you are setting up from scratch, use [`setup.sh`](setup.sh) as the reference.
- The build is currently configured for `sm_90a` in [`Makefile`](Makefile) and [`setup.py`](setup.py), so only Hopper-class GPUs are currently supported.
- The Python extension is packaged as `dae` and links [`src/torch_runtime.cu`](src/torch_runtime.cu) with [`src/runtime.cu`](src/runtime.cu) via `runtime.o`.

## Getting Started

### Fixed TMA Bandwidth Caps

Configure a component's aggregate read-plus-write cap before launching:

```python
dae.set_bandwidth("slow", num_sms=64, base_sm=0, bandwidth_gb_s=400)
dae.set_bandwidth("fast", num_sms=64, base_sm=64, bandwidth_gb_s=200)
dae.s(slow_schedule.place(64), fast_schedule.place(64, base_sm=64))
dae.launch()
```

Rates use decimal GB/s and are divided equally among the configured worker
blocks. Each block shares one token bucket between its two load issuers and
store issuer. The optional `burst_bytes_per_sm` defaults to 64 KiB; each launch
starts with a full bucket. Omitted settings or `bandwidth_gb_s=None` disable the
limiter. Settings apply to the next launch; there are no live updates.

This controls admitted TMA traffic, not measured HBM service. Known raw-address
bypasses on limited workers are rejected. TMA reductions charge twice their
payload size as a read/modify/write accounting model. Burst allowances must
cover the largest charged operation. Instruction placement still uses
`Schedule.place`; worker indices do not provide physical SM affinity.

Rebuild the extension with `make pyext` before using the new API.

For a TMA-load-only comparison, run:

```bash
python app/python/bandwidth_load.py --json output/bandwidth_load.json
# Sequential component-count sweep:
python app/python/bandwidth_load.py --components 2 4 6 8 --json output/bandwidth_scaling.json
```

The script measures best-effort throughput with admission disabled, then gives
each of N components 1/N of that measured rate. Workers are split evenly with
the remainder assigned to the last component; the default is two components.
All cases use the current kernel, identical input, and both load ports. Work is
equal per worker, so extra workers in the last component cause a completion tail
under equal component caps. It reports median aggregate
and per-component GB/s using device timestamps. There are no workload stores;
the dummy consumer only releases completed load buffers. Reported bytes are
logical TMA traffic, not DRAM counters. Keep the working set larger than L2.

### Runtime Layout

The codebase is organized around three layers:

- `include/dae/` and `src/`: the core runtime, virtual core abstractions, queues, allocators, launcher plumbing, and CUDA backend. Good entry points are [`include/dae/runtime.cuh`](include/dae/runtime.cuh), [`include/dae/virtualcore.cuh`](include/dae/virtualcore.cuh), [`src/runtime.cu`](src/runtime.cu), and [`src/torch_runtime.cu`](src/torch_runtime.cu).
- `include/task/`: kernel task building blocks such as attention, GEMV, RMSNorm, RoPE, SiLU, WGMMA, and argmax. Start with [`include/task/attention.cuh`](include/task/attention.cuh), [`include/task/gemv.cuh`](include/task/gemv.cuh), and [`include/task/rms_norm.cuh`](include/task/rms_norm.cuh).
- `python/dae/` and `app/python/`: Python-side model building and schedule composition. Start with [`python/dae/launcher.py`](python/dae/launcher.py), [`python/dae/schedule.py`](python/dae/schedule.py), and [`python/dae/model.py`](python/dae/model.py). End-to-end examples live in [`app/python/llama3/`](app/python/llama3) and [`app/python/qwen3/`](app/python/qwen3).

If you are new to the repository (as a model programmer want to play with schedules), a practical path is:

1. Build the extension with `make pyext`.
2. Read a small application example in [`app/python/`](app/python/) or jump directly to full LLM inference example [`app/python/llama3/sched.py`](app/python/llama3/sched.py).
3. Follow how Python schedules map to task primitives and runtime instructions through `launcher.py`, `schedule.py`, and the task headers.

## Contact and Reference

Contacts:

- Zhiyuan Guo, zhiyuang@cornell.edu
- Zijian He, zih015@ucsd.edu

Reference:

- Zijian He, Adrian Sampson, Yiying Zhang, Zhiyuan Guo, "VDCores: Resource Decoupled Programming and Execution for Asynchronous GPU", arXiv 2026, https://arxiv.org/abs/2605.03190
- Zhiyuan Guo, Zijian He, Adrian Sampson, and Yiying Zhang, “VDCores: A Runtime for Modern Async GPUs.” https://mlsys.wuklab.io/posts/vdcores/
