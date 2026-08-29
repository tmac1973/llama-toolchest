# Per-layer embeddings and the VRAM estimate: measured findings

Status: investigation complete. Steps 1 and 3 of the measurement plan are
done and step 4 is answered for one model; steps 2, 5 and 6 are open.
Date: 2026-08-29.
Measured on `compute` — llama-toolchest v2.25.1 (plus the PR #157 backfill fix),
llama.cpp build `b10679-rocm`, 4x AMD Radeon AI PRO R9700 (32 GiB each, gfx1201),
193 GB system RAM, container deployment.

## Summary

The per-layer embedding (PLE) table is **never resident in VRAM**, on either
architecture tested. The "Per-Layer Embeddings" selector added in #149 therefore
does not change VRAM, and the VRAM adjustment added in #153 excludes a table
that was never counted on the GPU in the first place.

The estimate nevertheless comes within ~7 GiB of reality on Qwen3.8-Flash-Next,
because two large errors cancel: it over-counts the table by 26.82 GiB and
under-counts KV plus compute buffers by roughly 20 GiB. **Correcting either half
on its own makes the estimate substantially worse.** That is the main reason this
document exists rather than a patch.

## What was measured

Two models with a PLE table, driven through the toolchest HTTP API — set model
config, restart the router, load via `/v1/chat/completions`, sample
`/api/monitor` before and after. Saved configs were backed up and restored
byte-identically afterwards.

| | gemma-4-E4B-it | Qwen3.8-Flash-Next |
|---|---|---|
| Quant | UD_Q4_K_XL | UD_Q4_K_XL |
| Arch | `gemma4` | `qwen4exp` |
| File size | 4.77 GiB | 103.69 GiB |
| `ple_bytes` | 1.80 GiB | 26.82 GiB |
| Layout | single file | 4 shards |

Three placements were compared:

- **A — resident**: `--tensor-read-lazy off`
- **D — streamed**: `--tensor-read-lazy on`
- **B — system RAM**: `--override-tensor per_layer_token_embd=CPU --load-mode none`

Flag delivery was confirmed from the router log in every case (`--load-mode`,
`--override-tensor`, `--tensor-read-lazy` all appear; no argument errors).

## Results

### gemma-4-E4B (context 4096, mmproj disabled, single GPU)

| Mode | VRAM | Host RAM | Throughput |
|---|---|---|---|
| A resident | +3.42 GiB | +0.30 GiB | 75.7 tok/s |
| D streamed | +3.42 GiB | +0.32 GiB | 75.3 tok/s |
| B system RAM | +3.41 GiB | +2.54 GiB | 75.5 tok/s |

Non-PLE weights are 2.97 GiB and measured VRAM is 3.42 GiB, leaving ~0.45 GiB
for KV and buffers. A GPU-resident 1.80 GiB table would have put this at ~5.2 GiB.
It is not there in any mode.

### Qwen3.8-Flash-Next (saved config: ctx 262144, batch 4096, ubatch 1024, kv q8_0, all 4 GPUs)

| Mode | GPU0 | GPU1 | GPU2 | GPU3 | Total | Host | Throughput |
|---|---|---|---|---|---|---|---|
| A resident | 28.39 | 26.39 | 26.62 | 25.94 | **107.34** | +53.1 | 29.6 tok/s |
| D streamed | — | — | — | — | **107.34** | +53.5 | 29.6 tok/s |
| B system RAM | 25.95 | 23.66 | 23.91 | 23.23 | **96.75** | +31.3 | 27.6 tok/s |
| *A -> B delta* | −2.44 | −2.73 | −2.71 | −2.71 | −10.59 | | −6.8% |

## Conclusions

### 1. The table is not in VRAM, in any mode

The A -> B per-GPU drops are near-uniform (spread 0.29 GiB). A single tensor
lives on exactly one device, so relocating a 26.82 GiB tensor would collapse one
card by ~26.8 GiB and leave the others untouched. An even drop across four cards
is a per-device buffer effect. The table was not on a GPU to begin with.

This matches gemma, where the arithmetic independently rules the table out, and
matches the fact that a 26.82 GiB tensor plus that device's share of layers
(~19 GiB) cannot fit a 32 GiB card at all.

### 2. `--tensor-read-lazy` changes neither VRAM nor speed

A and D are identical on VRAM to two decimal places and within noise on
throughput, on both models. The flag governs **host** residency during load —
whether rows are materialised or read on demand from the mapping. Upstream's own
PR reports an RSS saving (7.37 -> 6.16 GB) and never a VRAM one.

### 3. What variant B actually buys

10.59 GiB of VRAM, spread evenly across devices — compute buffers shrinking
because the embedding lookup moved to the CPU backend. Not the table relocating.
The cost is ~6.8% throughput, `--load-mode none` for the whole model (no mmap),
and +31 GiB of anonymous host RAM.

### 4. `ple_bytes` is correct

Verified against the real tensor table read from HuggingFace by ranged request:
`per_layer_token_embd.weight` has dims `[160, 320001536]`, type `IQ4_NL`
(18 bytes per 32 elements) = 28,800,138,240 bytes exactly, matching the recorded
value. The split-shard scan and the offset-gap size calculation are both sound.

### 5. The estimate is right by cancellation

For Qwen3.8-Flash-Next with its saved `ple_mode: "off"`:

| Term | Estimator | Measured |
|---|---|---|
| Weights | 103.69 (full file, table included) | 76.87 on GPU (table is not) |
| KV + compute + overhead | ~10.57 (implied) | ~30.5 |
| **Total** | **114.26** | **107.34** |

Over by 26.82 on one term, under by ~20 on the other. Fixing only the PLE side
would predict ~87 GiB against an actual 107 — an under-prediction, which is the
dangerous direction for a control users size their hardware against.

## Implications for shipped behaviour

- The **selector** (`model_config.html:135`) presents a choice that does not
  affect VRAM. Its tooltip claims streamed tables are "not counted toward the
  VRAM estimate when streamed", which is true of the estimate and false of the
  hardware.
- **`lazyPLEBytes`** (`vram.go:140`) encodes "resident means resident in VRAM".
  That premise is wrong on both architectures tested.
- **PR #153**'s exclusion is therefore modelling a saving that does not occur.
- **PR #157** (backfill) is unaffected — it fixes delivery of a correctly
  measured value, and that value is confirmed correct above.

## The decomposition, measured

One load of Qwen3.8-Flash-Next at its saved config, verbosity 4, captured
through `/api/service/logs` and parsed by `internal/memreport`.

| Term | GiB | Where |
|---|---|---|
| Model weights | 76.23 | ROCm0-3 |
| Model weights | 27.45 | `CPU_Mapped` — host, of which 26.82 is the PLE table |
| KV cache | 4.38 | ROCm0-3 (two caches: 816 and 306 MiB per card) |
| Recurrent state | 0.44 | ROCm0-3 |
| Compute buffers | 24.40 | ROCm0-3 |
| Compute buffers | 51.04 | `ROCm_Host` |
| Output | 0.004 | `ROCm_Host` |

Weights across all devices total 103.68 GiB against a 103.69 GiB file, so the
report accounts for the model completely. GPU terms total 105.45 GiB against
107.35 GiB measured from the GPU counters, leaving 1.90 GiB of context overhead
the report does not itemise.

Against the estimate:

| | Estimate | Actual | Error |
|---|---|---|---|
| Weights | 103.69 | 76.23 | **−27.46** |
| Everything else | 10.57 | 31.12 | **+20.55** |
| Total | 114.26 | 107.35 | −6.91 |

The two errors are each roughly four times the net, and they have opposite
signs. That is the cancellation, now itemised rather than inferred.

The 51.04 GiB host-side compute buffer also explains the ~53 GiB of host memory
that appears during a load. It was read as page cache earlier in this
investigation; it is a pinned allocation.

## Not recommended: a fourth "system RAM" placement option

The option was conceived to free VRAM by moving the table to host memory. The
table is already in host memory in every mode, so there is nothing to free. The
10.59 GiB that variant B saves is a buffer effect and would be better exposed,
if wanted at all, as what it is rather than as a placement control.

## Measurement plan for the estimator

The estimator cannot be corrected from per-GPU totals alone — they conflate
weights, KV and compute buffers. The work needs a decomposition first.

**Step 1 — get llama.cpp's own buffer report. Done.** The cause was not
filtering in toolchest, which broadcasts every line the router emits
(`process/manager.go:153`). It was log verbosity: llama.cpp's default of 3 passes
warnings from the core library but drops its INFO lines, so `W load:` arrived and
`I load:` did not. The router passes its environment to every model instance it
starts, so raising `LLAMA_ARG_LOG_VERBOSITY` reaches the children that do the
loading. It is now a curated Settings option.

Verbosity 4 is the level to use. Measured on one Qwen3.8-27B load:

| Verbosity | Lines per load | Buffer report |
|---|---|---|
| 3 (default) | 168 | no |
| 4 | 916 | yes |
| 5 | 11,874 | yes, plus per-layer and per-tensor debug |

At 4 the report reads (device names as llama.cpp prints them):

```
load_tensors:         CPU_Mapped model buffer size =  1288.28 MiB
load_tensors:             Meta() model buffer size =  7252.51 MiB
llama_kv_cache:           Meta() KV buffer size    =  4096.00 MiB
llama_kv_cache:  size = 16384.00 MiB (262144 cells, 16 layers, 4/1 seqs),
                 K (f16): 8192.00 MiB, V (f16): 8192.00 MiB
llama_memory_recurrent:   Meta() RS buffer size    =   448.88 MiB
sched_reserve:            Meta() compute buffer size =  384.95 MiB
sched_reserve:         ROCm_Host compute buffer size =  276.02 MiB
llama_context:         ROCm_Host output buffer size  =    3.79 MiB
```

That is the decomposition the estimator needs: weights split by device and by
whether they are mmap-backed, KV with its K/V breakdown, recurrent state, and
compute buffers. Note `CPU_Mapped` appearing for weights on a model with no
per-layer embedding table at all — more than the PLE table can sit off-GPU, which
step 4 should account for.

**Step 3 — decompose one load.** Done for Qwen3.8-Flash-Next; see the table
below. `internal/memreport` parses the report into per-device, per-kind totals.

**Step 2 — build a repeatable harness.** For each (model, context, batch,
ubatch, kv quant, ngl, ple_mode) point: load, record per-GPU VRAM, host RSS, and
the per-buffer breakdown. Prerequisite: add `ple_mode` and `extra_flags` as
benchmark sweep axes (`ConfigOverrides` in `benchmark/job.go:78`, catalogue in
`benchmark/sweep.go:355`) — neither is currently sweepable, which is what forced
this whole investigation to be driven by hand.

**Step 3 — corpus.** Span the axes that matter:
- with PLE: `gemma4` E-series, `qwen4exp`
- without PLE: a dense `qwen35`, a MoE (`Qwen3.6-35B-A3B`), `gpt-oss`
- 3 context sizes each, including one very large (>=128K)
- both single-GPU and 4-GPU placement

**Step 4 — test three specific hypotheses.** Answered for Qwen3.8-Flash-Next by
the decomposition below; still open for every other architecture.
- *H1 — confirmed.* The table is held `CPU_Mapped`, so it is host memory and
  never counts toward VRAM. It should be excluded from the weight term
  unconditionally, not only when streamed. Note the host-mapped weights exceed
  the table by 0.63 GiB, so excluding exactly the table would still be wrong.
- *H2 — confirmed, and it is the dominant term.* Compute buffers take 24.40 GiB
  of VRAM on this load and are not modelled at all.
- *H3 — disproved here.* The KV cache is 4.38 GiB, comfortably inside the
  ~10.57 GiB the estimate allows for everything that is not weights. KV is not
  the problem; compute buffers are.

**Step 5 — validate, with a guardrail.** Accept a revised model only if it
never under-predicts measured VRAM across the corpus. Over-prediction is a
usability cost; under-prediction tells someone a model fits when it does not.

**Step 6 — only then touch the UI.** If H1 holds, the selector stops claiming a
VRAM effect and the tooltip is rewritten around host memory and load behaviour,
which is what the flag actually controls.
