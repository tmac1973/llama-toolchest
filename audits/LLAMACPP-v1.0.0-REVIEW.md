# llama.cpp v1.0.0: what it means for llama-toolchest

Date: 2026-11-13
Release: https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0 (nightly b12359, ggml v1.0.0)
Compared against: v0.8.0 (b11960), the last review ([LLAMACPP-v0.8.0-REVIEW.md](LLAMACPP-v0.8.0-REVIEW.md)). 399 commits.
Code checked: `claude/llama-cpp-1-0-review-65c257` at 114ec63 (v2.58.0)

## Summary

**Nothing has to change for toolchest to build and run v1.0.0.** The 1.0
release is mostly a public C API cleanup (35 deprecated functions removed),
and toolchest only runs the binaries. Every option toolchest writes still
exists, the log lines it parses are still there, and the server's new
security checks do not block toolchest's setup.

One existing feature now reports too little: the memory report misses three
new GPU buffers that DFlash and MTP models allocate. Two new settings are
worth adding: a CPU share for the MoE cache, and FP8 KV caches.

Priorities: **P1** worth doing soon, **P2** useful, **P3** small or optional.

| Section | Item | Priority |
|---|---|---|
| 3.1 | Memory report misses the new layer input, h_nextn and rotation buffers | P1 |
| 3.2 | `--moe-cache-cpu-share` setting (b12191) | P2 |
| 3.3 | FP8 KV cache, `f8_e4m3` (b12274) | P2 |
| 3.4 | `/v1/audio/speech`: works already, docs only (b12238) | P3 |
| 3.5 | `--shard` / `--ple` for models split across files (b12233) | P3 |

All four v0.8.0 follow-ups (#430–#433) are closed.

---

## 1. Checks: nothing breaks

| Area | Result |
|---|---|
| Release tags | `v1.0.0` matches `releaseTagRE["v"]` (`internal/builder/builder.go:1161`). Upstream also pushed an extra tag, `v1.0.0-b12359-b1b770d`, on the same commit as b12359. The regex rejects it, so the ref picker shows only `v1.0.0`. That is what we want. |
| Removed flags | Only `-dt` / `--defrag-thold` was removed. Toolchest never wrote it, and nothing in `internal/` or the curated forks uses it. A user who put `-dt` in Extra Flags will now get a failed load. That is unlikely, and the flag did nothing since 2025. |
| Preset options we write | All still in `common/arg.cpp`: `cache-type-k/v`, `moe-cache-mib`, `spec-*`, `split-mode`, `tensor-split`, `fit`, `mmproj`, `models-max`, `models-memory-margin`, `models-preset`, `reasoning-preserve`, … |
| Binaries | `llama-server`, `llama-bench`, `llama-perplexity` and `llama-cli` are still built. The examples cleanup (#32184) removed only scripts. No CMake options changed apart from a new SYCL one (`GGML_SYCL_SUPPORT_XMX`, ON by default). |
| Log lines we parse | All still there: `spawning server instance with name=… on port …` (serverlog), `cmd_child_to_router:state:`, `srv … model loaded`, and the `… buffer size = … MiB` lines. The `MoE cache size` line gained a suffix (`, N% of the misses on the CPU`); `moeCacheRE` is not anchored at the end, so it still matches. The router's `proxying request to model …` line dropped from INFO to TRACE. Toolchest does not read it. |
| Server security (#32098, #32113, #32115, #32120) | **No effect on toolchest's setup.** (a) The new DNS rebinding and CSRF checks are skipped when `--cors-origins` is `*`, the default; toolchest does not set it. (b) A router child now returns 403 to requests that carry `Origin` or `Sec-Fetch-*` headers, but only for direct requests. The router strips those headers when it proxies, and toolchest only talks to the router. (c) Tools, MCP servers or the MCP proxy on a non-loopback address now need `--api-key` or `--allow-unauthenticated-tools`. Toolchest enables none of them. Children bind to loopback, so per-model `--tools` in Extra Flags still works. A user who adds `--tools` to the *router* through custom env would now see it refuse to start. (d) Image URLs that point to private addresses are refused unless `--media-allow-private-urls` is set. That only matters to a client that sends LAN image URLs. |
| `HF_TOKEN` | Now sent only to huggingface.co hosts (#31730). Toolchest does its own downloads. |

## 2. Fixes in this release that help toolchest directly

- **MoE cache replacement is better** (#31233, ClassLRU): +6–7 % decode upstream on GLM and Qwen3.8-Flash-Next with a warm cache, and 8 % fewer uploads. No setting changes.
- **The MoE cache no longer applies to the draft model** (#31497). Toolchest's VRAM planner never counted it for the draft, so its plans are now closer to what llama.cpp actually does.
- **`fit` accounts for skipped NextN layers** (#31569) when it picks `n_gpu_layers` for MTP models.
- **Router children no longer deep-copy context checkpoints** (#30137), and shared-prefix requests visit parent slots first (#30779). This should help multi-slot models with prompt caching.
- **ROCm: a fixed BLAS workspace** (#31838) replaces the one rocBLAS allocated per call. This is relevant to the R9700 boxes.
- **CUDA: few-row MMQ for batch sizes 2–24** (#31533) and `MMVF` for BF16 at small batches (#31959). Both should help speculative decoding verification on the A4000s.
- **Download progress per file** (#31467): a small draft or mmproj no longer starves the main file's progress updates in router downloads.

## 3. New features and what toolchest should do

### 3.1 Memory report misses three new GPU buffers (P1)

Three new buffers are logged with a name that `lineRE` (`internal/memreport/memreport.go:89`) does not accept. It only knows `model|KV|compute|output`:

```
llama_context:      CUDA0 layer input buffer size =   200.00 MiB (5 layers)   # DFlash, #31506, b12063
llama_context:      CUDA0 h_nextn buffer size =    40.00 MiB                   # MTP, #31927, b12226
llama_kv_cache:      CUDA0 rotation buffer size =     0.50 MiB                 # #31073, b12007
```

The layer input buffer is about `n_batch × n_embd × captured layers × 4 B`, so a few hundred MiB on DFlash models at large batch sizes. h_nextn is about `n_batch × n_embd × 4 B`. The report and anything that learns from it (planner calibration) undercount VRAM on DFlash and MTP models by that much.

**Do:** accept these three names in the buffer regex, file them under a fitting category (most likely "compute" or a new "other" row), and add test lines copied from a real b12359 log. Check against a live DFlash load on compute2. Decide whether the VRAM planner should budget the DFlash layer input buffer as well.

### 3.2 MoE cache: CPU share for misses (P2)

`--moe-cache-cpu-share F` (0–1, default 0; #30966, b12191). Once the cache is full, this share of the misses is computed on the CPU from host memory instead of being uploaded to the GPU. Upstream's numbers show it matters a lot when the cache is small relative to the experts: 24 → 48 t/s at share 0.95 with a 6 GiB cache. It matters little or is negative with a large cache. Upstream ships `scripts/moe-cache-tune.py` to pick both values.

**Do:** add a "CPU share of misses" field next to MoE cache MiB in the model settings, write it as `moe-cache-cpu-share`, and gate it at b12191 in `internal/models/launch.go`, like `versionMoECache`. It does not change VRAM, so the planner is unaffected. Autotune could try share 0 and 0.9 when the MoE cache is on, but add it as a candidate inside the existing runs: autotune runs should not get longer.

### 3.3 FP8 KV cache (P2)

New cache type `f8_e4m3` for `-ctk/-ctv` (#28392, b12274). It takes 1 byte per element, without the block scale of `q8_0` (1.0625 B). It can carry static per-tensor scales from the GGUF (e.g. NVIDIA's NVFP4 checkpoints, which come with FP8 KV scales). The CUDA FlashAttention vec kernels have every K/V pairing with it, and they decode FP8 in software on cards without FP8 hardware, which includes the A4000s (sm86). HIP builds have the same kernel.

**Do:** add `f8_e4m3` to the KV cache and draft KV cache dropdowns (`web/templates/partials/model_config.html:250`, `:555`), return 1.0 for it in `kvBytesPerElem` (`internal/models/vram.go:27`), and gate it at b12274. Benchmark it against `q8_0` on compute2 and the R9700 before letting `fit.go`/autoconfig pick it. Until then, make it something a user chooses, not something toolchest chooses.

### 3.4 `/v1/audio/speech` (P3)

OpenAI-compatible text-to-speech (#31763, b12238): `model`, `input`, `voice`, `response_format` (`wav`, `pcm`, `pcm16`). `/tts` also takes `temperature` now. Toolchest's `/v1/*` catch-all already proxies the route to the router. Only `/audio/speech` without `/v1` is not proxied, and OpenAI clients don't use that path.

**Do:** mention `/v1/audio/speech` wherever the docs list `/tts` (`docs/guides/download-models.md`, `docs/secure.md`, and the speech-model docs). Optionally add the root `/audio/speech` next to `/tts` in `internal/api/server.go:1091`. The new drama-3 TTS model sets `clip.has_gen_audio_encoder`, so `Speaks()` detects it with no change.

### 3.5 Models split across files: `--shard` / `--ple` (P3)

`--shard a.gguf,b.gguf` (alias `--ple`, #31185, b12233) loads extra GGUF shards from anywhere. The main use is a model shipped without its per-layer embeddings (PLE/engrams) next to a separate PLE-only file. Gemma 4, Qwen3.8-Next and GLM-5.3-Flash can be converted this way. The router drops `LLAMA_ARG_SHARD` along with the other model arguments for HF-managed models, so it would have to be a preset key next to `model`.

**Do:** nothing yet. Watch whether Hugging Face repos start publishing PLE-less GGUFs. If they do, the download flow has to fetch the PLE file and the preset has to write `shard = …`.

## 4. Not relevant to toolchest

- Public API: `llama.h` prunes 35 functions, NextN moves into `llama.h`, and `clip.h` is gone (#30565, #31865). Toolchest does not link against llama.cpp.
- New architectures (Kimi-K3.1-Pro, Ripple, xing4_0) and vision projectors (MageVL-Cowork, Kimi-K3.1). `ParseArchNames` reads them from source.
- Web UI redesign (sidebar groups, markdown editor, mobile layout). Toolchest links to the built-in UI but does not embed it.
- Windows unbuffered load, SYCL, Hexagon, OpenCL, WebGPU work.

## 5. Upstream to watch

From the 2026-10-15 audit: #27210 (draft-mtp-adaptive), #26283 (suffix decoding), #28391 (ngram-mod on by default and additive `--spec-type`). All three are still open on 2026-11-13.
