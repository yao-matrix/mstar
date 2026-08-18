---
name: xpu-model-enabling
description: Enable, port, debug, and benchmark model serving on Intel XPU. Use when adapting accelerator-oriented PyTorch inference code to XPU, selecting XCCL settings, sharding models to fit memory, integrating XPU kernels, migrating KV state, or validating text and image generation performance.
---

# Intel XPU Model Enabling

## Preserve Model Semantics

Treat Intel XPU as a general-purpose accelerator, not as a reason to create a
different model workflow.

- Keep graph Walks, cache labels, request transitions, and model semantics
  identical across CUDA and XPU when possible.
- Put hardware differences behind runtime interfaces such as attention,
  collectives, and KV transport backends.
- Prefer a portable correctness backend now and replace it with device IPC or a
  faster kernel later without changing model integration code.
- Before adding model-specific recomputation, check whether existing state can
  be moved through host-staged shared memory.

## Plan From Capacity

- Measure model weights, KV cache, intermediate tensors, and kernel workspace
  before changing device APIs.
- Shard the largest component first. For BAGEL on 24 GB devices, TP=2 fits one
  LLM replica; place ViT/VAE on a separate XPU.
- Avoid layouts that place an unsharded copy of an oversized component on a
  worker, including seemingly lightweight join nodes that initialize the full
  model.
- Start with conservative sequence and page limits, then increase them from
  measured headroom.

## Port Accelerator Runtime Code

- Prefer `torch.accelerator.set_device_index`, `torch.Event`,
  `torch.accelerator.current_stream`, and `torch.accelerator.synchronize`.
- Capture the execution stream before launching an engine and record completion
  on that same stream.
- Use `torch.amp.autocast(device_type)` and device-aware dtype APIs.
- Keep explicit device branches only for backend-specific capabilities such as
  CUDA IPC or XPU-only kernels.
- Detect the accelerator in orchestration code, then pass concrete `xpu:N`
  devices to workers for rank placement.
- Set accelerator and collective environment variables in the launching shell,
  not from Python.

## Configure XCCL

- Select the backend with `dist.get_default_backend_for_device`; XPU resolves
  to `xccl`.
- Pass `device_id` to `init_process_group` so each process group is bound to the
  correct XPU.
- Initialize every worker in the default world, then create TP/SP subgroups
  collectively.
- Use native `dist.barrier()`. The first call may be slow because it lazily
  initializes the communicator; do not replace it without a minimal
  reproduction proving a backend defect.
- Prefer oneCCL OFI/TCP for independently spawned single-node workers:

```bash
export LD_LIBRARY_PATH=/opt/venv/lib:${LD_LIBRARY_PATH}
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER_PATH=/opt/venv/lib
export FI_PROVIDER=tcp
```

Do not assume `FI_PROVIDER=shm` accelerates XPU collectives. In the tested
setup, oneCCL used Level Zero/SYCL for the device data path; forcing libfabric
SHM did not improve all-reduce latency and increased startup time.

## Integrate Paged Attention

- Preserve the cache allocator, labels, snapshots, page tables, and request
  lifecycle. Replace only attention execution.
- Standard PyTorch SDPA does not accept a paged KV-cache table.
- Match kernels to the actual cache layout:

```text
[layers, pages, 2, page_size, local_kv_heads, head_dim]
```

- Probe exact local TP geometry, dtype, tensor strides, page size, and causal
  mode. For BAGEL TP=2 this includes 14 query heads, 2 KV heads, head dimension
  128, and page size 64.
- Pass block tables, cumulative query offsets, and host KV lengths to the XPU
  paged-attention backend.

## Move KV State Portably

Separate graph-edge tensor transport from KV-cache migration; enabling SHM for
ordinary tensors does not automatically make paged KV state transferable.

For local multi-process migration:

- Use CUDA IPC on CUDA when available.
- Use host-staged SHM for CPU/XPU correctness when device IPC is unavailable.
- Publish only occupied pages, not the maximum cache allocation.
- Version publications by page indexes and sequence length. Page indexes can
  remain unchanged while new tokens modify a partially filled final page.
- Preserve TP rank alignment: source rank `r` transfers its local KV-head shard
  to destination rank `r`, and reject mismatched TP world sizes.
- Copy the precise page/token ranges described by the cache planner.
- Publish atomically so consumers never observe partial files.
- Scope segments to request and label, and remove them during request cleanup.
- Gate publication with the existing store-write policy so colocated serving
  does not stage unused data.
- Keep the transfer interface stable so future XPU IPC can replace SHM without
  changing model Walks.

Host staging can be competitive when migration occurs once before a long
iterative phase. Measure transition cost separately from the subsequent loop.

## Discover And Validate XPU Operators

Inspect registered operators when wrappers are incomplete:

```python
torch._C._dispatch_get_all_op_names()
torch._C._dispatch_find_schema_or_throw(name, "").schema()
```

- Probe each operator using the model's exact inputs.
- Use fused kernels for RMSNorm, RoPE, and top-k/top-p sampling when available.
- Normalize inputs once during planning, such as converting position IDs to
  `int64`.
- Cache reusable RoPE cosine/sine tables instead of recomputing them per layer.
- Verify tensor contiguity and strides; microbenchmarks with contiguous tensors
  may miss model-view constraints.

## Handle Missing Kernel Specializations

- Treat fallback warnings as build-configuration evidence and record the exact
  missing tuple.
- BAGEL image generation requires:

```text
128,true,false,false,false,false
```

- Use `vllm-xpu-kernels` commit
  `95d80c7a1d4bc06360fcd3b92deffa36da7eadca` or newer.
- Build with a matching oneAPI DPC++ compiler and
  `--no-build-isolation --no-deps` to preserve the PyTorch environment.
- Run an exact-shape XPU probe after rebuilding. A successful wheel build does
  not prove the specialization was instantiated.

## Validate In Layers

1. Verify imports and operator registration.
2. Run exact-shape kernel probes.
3. Validate TP shapes and checkpoint loading.
4. Exercise the full XCCL world and every subgroup.
5. Wait for server readiness and check health.
6. Run deterministic text generation.
7. Run image generation and editing when supported.
8. Decode and inspect output images.
9. Measure per-device memory and client-observed turnaround.
10. Repeat requests to detect cache leaks or stale state.
11. Verify SHM or IPC resources are cleaned after request completion.
12. Compare same-seed sequential and parallel outputs within BF16 tolerance.

Count only HTTP 200 responses as performance samples. Fast HTTP 500 responses
measure error latency. Compare repeated runs before attributing small timing
changes to code, and distinguish cold startup, first request, and warm repeat.
