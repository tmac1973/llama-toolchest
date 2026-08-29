# Per-layer embeddings and the VRAM estimate: measured findings

Status: investigation complete. Steps 1 and 3 are done, step 4 is answered
for two models, step 2 is partly done (two of the corpus, swept by hand);
steps 5 and 6 are open.
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

## A second model, swept

Qwen3.8-27B (`qwen35`, dense, 29.30 GiB, no PLE table), split tensor-parallel
across four cards. Context and micro-batch swept independently; every figure
below is a total after scaling the aggregate device.

| ctx | ubatch | weights | KV | RS | compute | predicted | measured | residual |
|---|---|---|---|---|---|---|---|---|
| 8192 | 512 | 27.51 | 0.48 | 0.60 | 0.52 | 29.11 | 31.43 | **2.32** |
| 32768 | 512 | 27.51 | 2.00 | 0.60 | 0.60 | 30.71 | 33.01 | **2.30** |
| 131072 | 512 | 27.51 | 8.00 | 0.60 | 0.96 | 37.07 | 39.38 | **2.31** |
| 262144 | 512 | 27.51 | 16.00 | 0.60 | 1.48 | 45.59 | 47.87 | **2.28** |
| 32768 | 128 | 27.51 | 2.00 | 0.60 | 0.20 | 30.31 | 32.61 | **2.30** |
| 32768 | 2048 | 27.51 | 2.00 | 0.60 | 2.40 | 32.51 | 34.82 | **2.31** |

Two results.

**The unreported remainder is a constant.** Across a 31 to 48 GiB range and both
sweeps it stays within 2.28-2.32 GiB, a spread of 0.04. So the buffer report
plus a fixed per-context overhead accounts for measured VRAM — there is no
missing term that scales. That is the shape a corrected estimate should take.

**Both variable terms are linear, in different things.** KV is linear in context
and independent of micro-batch (0.48 -> 2.00 -> 8.00 -> 16.00 as context
quadruples, unchanged at 128 vs 2048 ubatch). Compute is linear in micro-batch
and grows weakly with context (0.20 -> 0.60 -> 2.40 across a 16x ubatch range).
The current estimate models neither.

Compute buffers are not always small: 1.48 GiB here at 262144 context against
24.40 GiB on Qwen3.8-Flash-Next at the same context. Whatever sets that scale is
architectural and is the next thing worth isolating.

### Aggregate devices report per card

A tensor-parallel split puts every buffer on `Meta()`, llama.cpp's stand-in for
several cards addressed as one, and the sizes reported against it are per card
rather than totals. The 27B reports 7041.71 MiB of `Meta()` weights on a 29.30
GiB model; multiplying by the four cards it spans and adding the host-mapped
remainder reconciles. Reading those figures as totals under-counts by the number
of cards, which is why `memreport.ScaleAggregates` exists and why the card count
has to come from the model's GPU assignment — the log never states it.

## What drives compute buffers

Compute was the term the estimate omits and the one that varies most between
models — 1.48 GiB on the 27B against 24.40 GiB on Flash-Next at the same
262144 context. Two things were tested.

### Flash attention: large, and now measurable

A/B on two layer-split models, KV quantization off so both arms load:

| Model | ctx | ubatch | FA on | FA off | increase |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 32768 | 512 | 0.83 | 5.23 | **+4.40** |
| Qwen3.8-Flash-Next | 32768 | 1024 | 4.87 | 23.87 | **+19.00** |

Flash attention is worth roughly a 5-6x reduction in compute buffers on both.
On Flash-Next that is 19 GiB of VRAM from one toggle.

This only became testable at all once the toggle was fixed — before that,
turning it off wrote nothing and llama.cpp defaulted to auto, so both arms of
any A/B were the same run.

### It is NOT what makes the two models differ

The hypothesis was that `qwen4exp` gets no flash-attention kernel and so pays
for a materialised score matrix. That is **wrong**: flash attention is active on
Flash-Next and saves 19 GiB there. Both models get it.

What actually separates them is how compute scales with context *while flash
attention is on*:

| Model | ctx 32768 | ctx 262144 | growth over 8x context |
|---|---|---|---|
| Qwen3.8-27B | 0.60 | 1.48 | 2.5x |
| Qwen3.8-Flash-Next | 4.87 | 24.40 | 5.0x |

The 27B is nearly flat; Flash-Next grows almost linearly with context.

### Identified: the sparse-attention indexer

`qwen4exp` runs Qwen Sparse Attention. Rather than attending to the whole cache,
each dense-attention layer first *ranks* the cached positions and attends only to
the top-k. That ranking pass is the context-scaling term.

In `src/models/qwen4exp.cpp` at build `50f068ff`, `build_qsa_top_k` produces a
tensor the code names `indexer_score_tokens`:

```
expanded = ggml_get_rows(score, inp->cell_blk);   // F32 [n_kv, n_tokens, n_stream]
expanded = ggml_add(expanded, mask);
top_k    = ggml_top_k(expanded, width);
```

That is a float score for every cache cell for every token in the micro-batch —
`n_kv x ubatch x 4` bytes. It is exactly the shape flash attention exists to
avoid, reintroduced by the indexer, because you cannot pick the top-k of a set
without scoring all of it.

This reconciles the result above rather than overturning it. Flash attention
really does save 19 GiB on Flash-Next: it accelerates the dense attention that
runs *after* selection. It cannot help the selection itself.

The 27B has none of this. `needs_mem_idx` in `llama-model.cpp` is true for
`LLM_ARCH_QWEN4EXP` alone, so no other architecture in this corpus allocates an
`n_kv`-shaped intermediate. That is the factor of sixteen.

The size fits. Per device, Flash-Next's GPU compute against one such tensor:

| n_kv | ubatch | one tensor | measured | ratio |
|---|---|---|---|---|
| 262144 | 1024 | 1.000 | 6.10 | 6.1 |
| 262144 | 256 | 0.250 | 1.67 | 6.7 |
| 32768 | 1024 | 0.125 | 1.22 | 9.7 |

Two independent slopes — one from the micro-batch pair, one from the context
pair — give 5.91 and 5.58. So roughly six of those tensors are live at once,
which the graph makes plausible: the score, its permuted copy, the expanded
per-cell form, an f32 cast of the mask, and the top-k output are each of that
shape or close to it.

### What this means for a formula

Compute is not one term with one shape. Architectures divide into those that
allocate an `n_kv`-sized intermediate and those that do not, and the difference
is a factor of sixteen at the same context. A fit over models without an indexer
would mispredict `qwen4exp` badly; one including it would over-predict
everything else.

So the estimate needs the distinction explicitly, and the GGUF already states
it: `attention.indexer.{head_count,key_length,top_k}` are present only on
architectures that rank, and `qwen4exp.cpp` reads them into `hparams` today. A
model carrying them needs roughly `6 x n_kv x ubatch x 4` bytes per device on
top of the ordinary graph scratch; one that does not needs the ordinary scratch
alone, which is near-flat in context and linear in micro-batch.

The indexer coefficient rests on three points from one model, so it wants a
second indexer-carrying model before it is trusted.

### Two combinations llama.cpp refuses

Both found by loads that failed outright, both now rejected on save:

- `SPLIT_MODE_TENSOR requires flash_attn to be enabled`
- `quantized V cache requires flash_attn to be enabled`

The second is the one a user meets by accident, since a quantized cache is a
common way to save memory.

### Worth acting on now

Flash-Next at its saved config spends 24.40 GiB of VRAM on compute buffers,
and that term is linear in micro-batch: dropping ubatch from 1024 to 256 took
it to 6.66 GiB, freeing about 18 GiB, at some prefill throughput cost. That is
a larger saving than anything the per-layer embedding controls can offer, and
it is available today.

## Not recommended: a fourth "system RAM" placement option

The option was conceived to free VRAM by moving the table to host memory. The
table is already in host memory in every mode, so there is nothing to free. The
10.59 GiB that variant B saves is a buffer effect and would be better exposed,
if wanted at all, as what it is rather than as a placement control.

## The revised estimate

Terms, each grounded in a measured buffer report rather than fitted blind:

| Term | Basis |
|---|---|
| Weights | file size less the per-layer table and the input embedding, both held `CPU_Mapped` on every model measured |
| KV cache | attention layers only; recurrent layers hold no cache (shipped separately) |
| Graph scratch | `0.2911 MiB x ubatch + 0.0009 MiB x ctx` per device |
| Sparse-attention scratch | `5.69 x n_kv x ubatch x 4` per device, only on models carrying indexer keys |
| Indexer key cache | `attn_layers x n_kv x key_length x 4` |
| Per-device overhead | 0.85 GiB |

Scored against the eleven measured points:

| | current | revised |
|---|---|---|
| mean absolute error | 7.74 GiB | **0.99 GiB** |
| worst error | +20.91 | +2.55 |
| under-predicts | 3 of 11 | **0 of 11** |

The guardrail is the direction, not the average: no point may come in under
what the hardware used. An estimate that is high costs someone headroom; one
that is low tells them a model fits when it does not.

### What it rests on

One ROCm machine, four architectures, one and four cards, 3.4 to 99 GiB. The
weight and KV terms are structural and should travel. The three fitted
coefficients are empirical and may not: a different backend can allocate
scratch differently. They are set so the corpus never under-predicts, which
is what makes shipping them defensible on this much evidence.

The corpus lives in `internal/models/vram_corpus_test.go`. Adding a machine
or an architecture means adding points there and re-checking both tests.

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
