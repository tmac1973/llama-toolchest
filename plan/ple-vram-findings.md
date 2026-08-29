# Per-layer embeddings and the VRAM estimate: measured findings

Status: investigation complete, no code changes proposed yet.
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

## Not recommended: a fourth "system RAM" placement option

The option was conceived to free VRAM by moving the table to host memory. The
table is already in host memory in every mode, so there is nothing to free. The
10.59 GiB that variant B saves is a buffer effect and would be better exposed,
if wanted at all, as what it is rather than as a placement control.

## Measurement plan for the estimator

The estimator cannot be corrected from per-GPU totals alone — they conflate
weights, KV and compute buffers. The work needs a decomposition first.

**Step 1 — get llama.cpp's own buffer report.** The router forwards only
`srv`/`slot`/`load`/`cmn` categories, so `load_tensors: ... buffer size` lines
never reach `/api/service/logs`. Either widen what the router captures, or run
`llama-server` directly in the container for measurement runs. Without this every
subsequent step is inference from totals.

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

**Step 4 — test three specific hypotheses.**
- *H1*: the PLE table never counts toward VRAM on any architecture. If it holds,
  the table should be excluded from the weight term unconditionally, not only
  when streamed.
- *H2*: compute buffers scale with ubatch and are not modelled at all today.
  Suspected to be the bulk of the ~20 GiB shortfall at 262K context.
- *H3*: the KV model under-predicts for hybrid attention (SWA, linear attention).
  `kv_full_per_tok` / `kv_swa_per_tok` are captured but the blend may be wrong
  for `qwen4exp`.

**Step 5 — validate, with a guardrail.** Accept a revised model only if it
never under-predicts measured VRAM across the corpus. Over-prediction is a
usability cost; under-prediction tells someone a model fits when it does not.

**Step 6 — only then touch the UI.** If H1 holds, the selector stops claiming a
VRAM effect and the tooltip is rewritten around host memory and load behaviour,
which is what the flag actually controls.
