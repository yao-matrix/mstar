# BAGEL Intel XPU Benchmark Recipe

## August benchmark hardware and runtime

- Hardware: Intel Arc Pro B60, 24 GB VRAM per device
- PyTorch: `2.15.0.dev20260824+xpu`
- Collectives: PyTorch XCCL over oneCCL `2022.1.2`, OFI/TCP bootstrap
- KV cache: paged, 64-token pages, 8192-token sequence limit
- Attention: `vllm-xpu-kernels`
- Benchmark batch size: 1
- Kernel package used for the August benchmarks:
  `vllm-xpu-kernels 0.1.14.dev15+gcd0ba52.d20260824`

Two profiles were validated on that revision:

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

With the August runtime, one physical card gave incorrect paged-attention
results. An exact eager kernel probe passed on physical devices 0-5 and 7
but failed on physical device 6, while a two-rank
XCCL test involving that card passed. The later September run used physical
devices 0-6 successfully; re-probe a suspect card after changing the runtime.
The August workaround was:

```bash
export ZE_AFFINITY_MASK=0,1,2,3,4,5,7
```

Logical device indexes remain contiguous after applying the mask.

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
when the request cache is released. The August implementation used a fixed
directory:

```bash
find /dev/shm/mstar_kv -maxdepth 1 -type f -print
```

The directory should be empty after request completion. On the newer PR #221
implementation, each deployment has a private `mstar_kv_*` directory. Check
for remaining snapshot files with:

```bash
find /dev/shm -maxdepth 2 -type f -name 'mstar_kv_*.pt' -print
```

Set `MSTAR_KV_SHM_DIR` in the shell to use a different shared filesystem.

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
| Parallel CFG eager, current runtime control | 67.32 s | Valid image |
| Parallel CFG XPUGraph, corrected capture | 66.39 s | Valid image |

The dedicated attention specialization reduced the original fallback runtime
by approximately 54%. TP=2 was about 9% faster than TP=4 and used about
19,459 MiB per LLM device after generation.

Parallel CFG with SHM migration was 2.67x faster than the corresponding
177.85-second sequential comparison. It was about 0.8 seconds slower than the local-prefill
prototype while preserving the same BAGEL graph and cache semantics across
CUDA and XPU. Same-seed parallel-versus-sequential output comparison measured
43.82 dB PSNR and 0.51 mean pixel error, consistent with BF16 execution-order
differences. Server startup and checkpoint loading are excluded from all
request timings.

The corrected XPUGraph result was only about 1.4% faster than the matching
eager control, which is within run-to-run noise. Treat graph capture as a
correctness experiment on this stack, not as a demonstrated performance win.
The graph-safe implementation must preserve timestep frequency calculations in
FP32; precomputing them in the model BF16 dtype corrupts both eager and captured
outputs. Always compare graph mode with an eager run from the same code and
runtime, using the same seed and healthy physical devices.

## September 28 validation of upstream PR #221

This later run used
[PR #221 head `98c218d`](https://github.com/mstar-project/mstar/pull/221/commits/98c218dc4b0d840ce461832ac863bf121d5ef38a),
which has a different resource configuration and device placement from the
historical branch documented above. Run the following from that PR's checkout,
with its BAGEL weights already cached:

```bash
ZE_AFFINITY_MASK=0,1,2,3,4,5,6 HF_HUB_OFFLINE=1 \
python -m mstar.api_server.entrypoint \
  --config configs/bagel_xpu_cfg_tp2.yaml \
  --cache-dir /root/.cache/huggingface/hub \
  --socket-path-prefix /tmp/mstar_bagel_xpu_cfg_tp2/ \
  --upload-dir /tmp/mstar_uploads_bagel_xpu_cfg_tp2/ \
  --port 8011 \
  --tensor-comm-protocol SHM \
  --timeout 600 \
  --log-stats-file /tmp/mstar-bagel-xpu-cfg-tp2-stats.jsonl \
  --log-level INFO
```

That configuration places LLM TP=2 on logical XPUs 0-1, `cfg_text` TP=2 on
2-3, `cfg_img` TP=2 on 4-5, and ViT/VAE on 6. All seven workers reached
ready state and `/health` returned HTTP 200. The runtime had PyTorch
`2.15.0.dev20260921+xpu` and
`vllm-xpu-kernels 0.1.dev387+gd7c35d281`.

The text-to-image request sent
`{"model":"bagel","prompt":"a cat in a hat"}` to
`POST /v1/images/generations`. The image edit sent that PNG, plus the prompt
“Place the cat on the moon, keeping its hat,” as multipart data to
`POST /v1/images/edits`. Both returned HTTP 200 and visually correct
1024x1024 PNGs. Client-observed turnaround was 72.29 seconds for generation
and 76.88 seconds for editing. The edit exercises multiple prefill walks and
the chunked SHM snapshot path.

The upstream fix keeps snapshot descriptors immutable and saves only the
changed tail pages on append. Its focused test set passed 84 tests. After
each XPU request, zero `mstar_kv_*.pt` files remained under `/dev/shm`.
These two timings validate function and cleanup on this revision; they do not
establish a new speedup relative to the August sequential baseline.
