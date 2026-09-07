# Results — measuring the draft method + n-gram assist combination

Run 2026-09-07 on this machine. Phase 06 of [`overview.md`](overview.md).

## What was run

| | |
|---|---|
| Model | `unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-IQ4_NL`, 5.3 GiB, MTP head baked into the main GGUF (self-speculation) |
| Build | `b10453-rocm-optimized`, ref `b10453`, ROCm profile |
| GPU | AMD Radeon RX 9070 XT, 15.9 GiB, all layers offloaded (`gpu-layers 999`, `device ROCm0`) |
| Context | 131072, flash attention on, KV cache `q8_0` |
| Job | `job-1788821931442-469f5504`, 8 cells = 4 speculative configs × 2 presets |

The four configurations, matching the source report's C0/C1/C5/C7 rows:

- `none`
- `draft-mtp:draft_max=3`
- `ngram-mod:assist_n_max=64,assist_n_min=48,assist_n_match=24`
- `draft-mtp+ngram-mod:` both of the above

The flags reached llama-server. The generated `bench-preset.ini` for the
combined cell carried exactly one `spec-type` line:

```ini
spec-type = draft-mtp,ngram-mod
spec-draft-n-max = 3
spec-ngram-mod-n-max = 64
spec-ngram-mod-n-min = 48
spec-ngram-mod-n-match = 24
```

`prompt_tokens` held steady across repetitions in every cell (1610 for every
`internal-echo` cell, 127/421/1607 for the three `internal-standard` sizes), so
no cell was served from the prompt cache and no throughput figure here is
measuring a cache hit.

## The numbers

Generation tokens/second, mean ± standard deviation over 3 repetitions.

### internal-echo — recall, 1610-token prompt, 512 generated

| config | gen tok/s | vs off | vs MTP alone |
|---|---:|---:|---:|
| off | 78.0 ± 0.4 | 1.00× | — |
| MTP alone | 153.3 ± 1.0 | 1.96× | 1.00× |
| N-gram Mod alone | 436.5 ± 51.4 | 5.59× | 2.85× |
| **MTP + N-gram Mod** | **447.1 ± 8.4** | **5.73×** | **2.92×** |

### internal-standard — new text, by prompt size

| config | 127 tok | 421 tok | 1607 tok |
|---|---:|---:|---:|
| off | 77.5 ± 2.8 | 77.9 ± 0.7 | 77.8 ± 0.3 |
| MTP alone | 133.9 ± 12.6 | 124.5 ± 14.7 | 125.1 ± 7.9 |
| N-gram Mod alone | 92.5 ± 17.4 | 81.4 ± 9.3 | 82.3 ± 6.2 |
| MTP + N-gram Mod | 120.5 ± 15.9 | 118.2 ± 14.7 | 112.4 ± 12.9 |

## What it says

**The recall win is real and larger here than the report predicted.** The
report's table showed 4× on its 27B model; this 9B reaches 5.7× over baseline
and 2.9× over MTP alone. That is the case the project was built for — a file
being rewritten, a structured response, a tool-call loop — and it is the case
neither mode reaches on its own: MTP alone gets 1.96×, and N-gram Mod is where
almost all of the remaining speed comes from.

**The combination is the best configuration on recall, but only just.** 447.1
against N-gram Mod's 436.5 is inside the n-gram cell's own error bar (±51.4).
What the combination buys on this workload is not extra recall speed, it is
that the *same* configuration is also 1.6× on new text where N-gram Mod alone
is 1.06×. One setting covers both workloads; that is the argument for it.

**The combination costs something on new text.** At every prompt size the
combined row sits below MTP alone — 120.5 vs 133.9, 118.2 vs 124.5, 112.4 vs
125.1, about 8-10%. The error bars overlap at each size individually, so no
single comparison is conclusive, but the direction is consistent across all
three sizes and matches both the earlier informal report of roughly 4 tok/s and
the mechanism: llama.cpp gives the draftless method precedence for a contested
step, so a weak n-gram match displaces a good MTP draft. This is exactly what
the help copy in the model config form warns about, and it is measured now
rather than argued about.

**N-gram Mod alone is not baseline on new text.** The report predicted exactly
baseline (its table showed 30.03 against a 30.03 baseline); here it is 85.4
against 77.7, about 1.10×. The cause is the benchmark's own prompt: it repeats
a fixed passage to reach the target length, so even the "analysis" prompt
contains repetition an n-gram method can exploit. Worth knowing before reading
the `internal-standard` n-gram row as a measure of genuinely novel text.

## Practical upshot

For this model on this hardware:

- Leave **MTP alone** on if the work is mostly new prose or reasoning.
- Turn the **N-gram Mod assist** on as well if any meaningful share of the work
  is the model reproducing text it has already seen. Paying ~8-10% on new text
  to gain 2.9× on recall is a good trade for anything doing file rewrites,
  structured output, or tool-call loops; it is a bad trade for pure chat.
- The setting is per model, which is what makes that choice available at all.

## One defect this phase found

The first run measured `internal-echo` at almost exactly `internal-standard`'s
numbers, which looked like the combination doing nothing. It was not a
speculative-decoding result. Qwen3.5 is a reasoning model, and with thinking on
it spent all 512 generated tokens in `reasoning_content` — deliberating about
how to reproduce the passage rather than reproducing it. That is new prose, so
the recall workload was never actually exercised.

Fixed in `f104f24`: the echo prompt style now turns thinking off using whichever
mechanism the model's chat template exposes, which the registry already
detects. Every number in this document is from the corrected run. The lesson
generalises past this preset — any benchmark that means to measure recall has
to make sure the model is not reasoning instead.
