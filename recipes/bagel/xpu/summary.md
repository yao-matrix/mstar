# BAGEL Intel XPU Activity Report

## Objective

Enable `mstar serve bagel` on Intel XPUs with 24 GB of VRAM per device, then
extend BAGEL's existing CFG-parallel execution without introducing an
XPU-specific model workflow.

A single device cannot hold the BAGEL LLM. The validated deployments use TP=2
for each LLM replica and place ViT/VAE on a separate XPU.

## Runtime and tensor parallelism

- Added automatic accelerator detection through `torch.accelerator`.
- Used generic accelerator APIs for device selection, streams, events, and
  synchronization where supported.
- Selected PyTorch's native `xccl` backend for Intel XPU collectives.
- Used native `dist.barrier()` with process groups bound to concrete devices.
- Converted BAGEL embeddings, QKV/output projections, MLPs, and LM head to
  mstar tensor-parallel primitives.
- Preserved checkpoint names and loaded TP=2 shards within 24 GB per XPU.
- Added `configs/bagel_xpu_tp2.yaml` for three-device sequential CFG serving.
- Added `configs/bagel_xpu_cfg_tp2.yaml` for seven-device parallel CFG serving.

Required launch environment:

```bash
export LD_LIBRARY_PATH=/opt/venv/lib:${LD_LIBRARY_PATH}
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER_PATH=/opt/venv/lib
export FI_PROVIDER=tcp
export HF_HUB_OFFLINE=1
```

The MPI transport was unsuitable for independently spawned workers. For
single-node collectives, forcing the libfabric SHM provider did not improve
all-reduce latency and increased communicator startup time. OFI/TCP remained
the best tested bootstrap/provider setting; oneCCL used Level Zero/SYCL for
the XPU collective data path.

## XPU execution backend

- Added paged attention through `vllm-xpu-kernels` while retaining mstar's
  existing paged allocator, labels, snapshots, and page tables.
- Used 64-token pages, supported by the XPU paged-decode kernels.
- Integrated fused XPU RMSNorm, cached RoPE, top-k/top-p sampling, and paged
  prefill/decode.
- Kept CUDA FlashInfer behavior intact.
- Removed eager model-registry imports that initialized unrelated CUDA-only
  dependencies during XPU startup.
- Fixed BAGEL MoT merge-buffer dtype handling under BF16 autocast.

The initially installed XPU kernel wheel lacked BAGEL's non-causal paged
chunk-prefill tuple:

```text
128,true,false,false,false,false
```

A custom wheel proved the specialization, and upstream later added it in
`vllm-xpu-kernels` commit
`95d80c7a1d4bc06360fcd3b92deffa36da7eadca`. The latest tested package was:

```text
vllm-xpu-kernels 0.1.dev5+g95d80c7a1
```

## CFG parallelism and KV migration

M* already had the BAGEL `image_gen_cfg` Walk, three parallel LLM branches,
cache labels, and CFG join. The first XPU prototype constructed conditioning
caches independently on every TP group because CUDA IPC was unavailable. That
worked, but it made BAGEL's control flow accelerator-specific.

The final design preserves the original model workflow:

```text
main prefill
    -> migrate cfg_text and cfg_img KV pages
    -> existing image_gen_cfg Walk
```

Only the local KV transport backend differs:

```text
CUDA: device -> CUDA IPC -> device
XPU/CPU: device -> host shared memory -> device
```

The generic SHM backend:

- Packs only occupied pages for each request and cache label.
- Publishes atomically to a shared-memory filesystem.
- Republishes when either page indexes or sequence length change, covering
  appends within a partially filled final page.
- Uses mstar's existing TP metadata to map source rank `r` to destination rank
  `r` when TP world sizes match.
- Copies only requested page/token ranges into destination pages.
- Removes request-scoped files during KV-cache cleanup.
- Is selected for non-CUDA local transport; the obsolete local-only backend was
  removed.
- Publishes only under `StoreWritePolicy.ALWAYS`, so colocated serving avoids
  unnecessary staging.

The model-side CFG delta is intentionally small: TP-enable both CFG LLM node
names, make `combine_cfg` a parameterless arithmetic join, and add the TP=2
placement configuration. The H100 configuration and CUDA IPC path remain
unchanged.

## Validation

- Sequential topology: three workers, one TP=2 LLM group.
- CFG-parallel topology: seven workers, three independent TP=2 LLM groups.
- Text generation returned `OK!`.
- 1024x1024 text-to-image and image editing completed successfully.
- Repeated requests left `/dev/shm/mstar_kv` empty after cleanup.
- Same-seed sequential/parallel output comparison: 43.82 dB PSNR and 0.51
  mean pixel error.
- Focused BAGEL/cache regressions: 17 passed.
- Ruff, Python compilation, and diff checks passed.

## Benchmark

| Build or topology | Turnaround |
| --- | ---: |
| Missing specialization, reference fallback | 417.24 s |
| Dedicated BAGEL kernel, TP=4 | 192.16 s |
| Upstream kernel, sequential TP=2 | 177.86 s |
| CFG parallel, local-prefill prototype | 65.89 s |
| CFG parallel, SHM KV migration | 66.70 s |
| CFG parallel, SHM warm repeat | 60.01 s |
| CFG parallel image editing | 80.16 s |

SHM-based CFG parallelism delivered a 2.67x speedup over the corresponding
177.85-second sequential TP=2 comparison. Host staging added about 0.8 seconds
relative to the local-prefill prototype while removing model-specific control
flow and retaining future compatibility with a device IPC backend.

## Pull request structure

- Runtime accelerator support: merged foundation.
- BAGEL TP=2 XPU serving: upstream PR #220.
- CFG-parallel TP=2 plus generic SHM KV migration: upstream PR #221, stacked on
  #220.
- Benchmark and reusable skill documentation: separate stacked documentation
  PR.

## Remaining work

- Replace host staging with XPU device IPC when PyTorch and the runtime expose a
  stable cross-process mechanism; keep the same `KVTransferEngine` contract.
- Make SHM copies asynchronous and reuse a bounded arena instead of serialized
  per-label files if migration becomes a significant transition cost.
- Add automated multi-XPU integration coverage for cache migration and image
  generation.
- Consume a released `vllm-xpu-kernels` wheel containing the required tuple.
- Investigate XPU graph capture or compilation after correctness and operator
  coverage stabilize.
- Optimize ViT attention, which currently uses PyTorch SDPA fallback.
- Improve shutdown handling for interrupted multi-process runs.
