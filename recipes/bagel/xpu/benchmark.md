# BAGEL Intel XPU Benchmark Recipe

## Tested hardware and runtime

- Hardware: Intel Arc Pro B60, 24 GB VRAM per device
- Collectives: PyTorch XCCL over oneCCL OFI/TCP
- KV cache: paged, 64-token pages, 8192-token sequence limit
- Attention: `vllm-xpu-kernels`
- Benchmark batch size: 1
- Latest tested kernel package: `vllm-xpu-kernels 0.1.dev5+g95d80c7a1`

Two deployment profiles were validated:

| Profile | Configuration | Placement |
| --- | --- | --- |
| Sequential CFG | `configs/bagel_xpu_tp2.yaml` | XPU 0 ViT/VAE; XPU 1-2 LLM TP=2 |
| Parallel CFG | `configs/bagel_xpu_cfg_tp2.yaml` | XPU 0 ViT/VAE; three TP=2 LLM replicas on XPU 1-6 |

The parallel profile reuses BAGEL's existing `image_gen_cfg` Walk. Prefill runs
on the main replica, and labeled KV pages migrate to the two CFG replicas using
host-staged shared memory. CUDA deployments continue to use CUDA IPC for the
same logical operation.

## Required environment

Set runtime variables in the launching shell, not in Python:

```bash
export LD_LIBRARY_PATH=/opt/venv/lib:${LD_LIBRARY_PATH}
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER_PATH=/opt/venv/lib
export FI_PROVIDER=tcp
export HF_HUB_OFFLINE=1
```

`HF_HUB_OFFLINE=1` assumes the BAGEL checkpoint is already cached.

## BAGEL kernel support

BAGEL image generation requires this non-causal paged chunk-prefill
specialization:

```text
128,true,false,false,false,false
```

It means head size 128, paged cache, non-causal attention, and no local, sink,
or LSE mode. The specialization is present in `vllm-xpu-kernels` main as of
commit `95d80c7a1d4bc06360fcd3b92deffa36da7eadca`.

Build and install that revision without replacing the working PyTorch stack:

```bash
source /opt/intel/oneapi/setvars.sh --force
export VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf
export VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf
export MAX_JOBS=16

/opt/venv/bin/pip wheel \
  --no-build-isolation \
  --no-deps \
  --no-cache-dir \
  --wheel-dir /tmp/vllm-xpu-kernels-wheel .

/opt/venv/bin/pip install \
  --no-deps \
  --force-reinstall \
  /tmp/vllm-xpu-kernels-wheel/vllm_xpu_kernels-*.whl
```

After installation, probe the exact model geometry. A successful wheel build
does not prove that the required specialization was instantiated.

## Start sequential CFG serving

```bash
mstar serve bagel \
  --config configs/bagel_xpu_tp2.yaml \
  --tensor-comm-protocol SHM \
  --host 127.0.0.1 \
  --port 8010 \
  --log-level INFO
```

## Start parallel CFG serving

The parallel configuration requires seven visible XPUs:

```bash
mstar serve bagel \
  --config configs/bagel_xpu_cfg_tp2.yaml \
  --tensor-comm-protocol SHM \
  --host 127.0.0.1 \
  --port 8010 \
  --log-level INFO
```

Wait for all workers to report ready, then check:

```bash
curl -sS http://127.0.0.1:8010/health
```

## Text smoke test

```bash
curl -sS --max-time 180 \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bagel",
    "messages": [{"role": "user", "content": "Reply with OK"}],
    "max_tokens": 2,
    "temperature": 0
  }' \
  http://127.0.0.1:8010/v1/chat/completions
```

Expected response content:

```text
OK!
```

## Image turnaround benchmark

This sends one 1024x1024 image request and measures client-observed turnaround,
including generation, encoding, serialization, and response transfer:

```bash
curl -sS --max-time 1800 \
  -o bagel-image-response.json \
  -w 'HTTP %{http_code}\nTurn-around: %{time_total}s\nResponse bytes: %{size_download}\n' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bagel",
    "prompt": "A red sports car parked beside a mountain lake at sunrise",
    "n": 1,
    "size": "1024x1024"
  }' \
  http://127.0.0.1:8010/v1/images/generations
```

Decode the returned image:

```bash
python -c "import base64,json,pathlib; d=json.loads(pathlib.Path('bagel-image-response.json').read_text()); pathlib.Path('bagel-image.png').write_bytes(base64.b64decode(d['data'][0]['b64_json']))"
```

Count only HTTP 200 responses as benchmark samples. Verify that the decoded
file is a valid 1024x1024 RGB PNG.

## KV migration checks

For parallel CFG, shared-memory files are request-scoped and should be removed
when the request cache is released:

```bash
find /dev/shm/mstar_kv -maxdepth 1 -type f -print
```

The directory should be empty after request completion. Publication is gated
by `StoreWritePolicy.ALWAYS`; ordinary colocated serving does not stage KV
files. Set `MSTAR_KV_SHM_DIR` in the shell to use a different shared filesystem.

## Results

| Configuration | Turnaround | Result |
| --- | ---: | --- |
| Missing chunk-prefill specialization | 417.24 s | Reference attention fallback |
| Dedicated BAGEL kernel, TP=4 | 192.16 s | HTTP 200 |
| Sequential CFG, TP=2 | 174.84-177.86 s | HTTP 200 |
| Parallel CFG, local-prefill prototype | 65.89 s | HTTP 200 |
| Parallel CFG, SHM KV migration | 66.70 s | HTTP 200 |
| Parallel CFG, SHM migration, warm repeat | 60.01 s | HTTP 200 |
| Parallel CFG image editing | 80.16 s | HTTP 200 |

The dedicated attention specialization reduced the original fallback runtime
by approximately 54%. TP=2 was about 9% faster than TP=4 and used about
19,459 MiB per LLM device after generation.

Parallel CFG with SHM migration was 2.67x faster than the latest 177.85-second
sequential comparison. It was about 0.8 seconds slower than the local-prefill
prototype while preserving the same BAGEL graph and cache semantics across
CUDA and XPU. Same-seed parallel-versus-sequential output comparison measured
43.82 dB PSNR and 0.51 mean pixel error, consistent with BF16 execution-order
differences. Server startup and checkpoint loading are excluded from all
request timings.
