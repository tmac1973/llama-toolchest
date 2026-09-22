# Speculative decoding as a ROCm comparison — method and baseline

22 September 2026. `Qwen3.5-9B-IQ4_NL` (MTP), context 131,072, flash attention
on, all layers on GPU, llama.cpp `v0.4.1`, RX 9070 XT.

The question was whether a speculative-decoding workload would show a difference
between ROCm versions where plain generation showed none. Plain generation on the
4B came out identical to a tenth of a token per second on both lines, which is a
weak result: single-stream decode is one large matmul per token, bandwidth-bound,
and calls the same rocBLAS kernels either way. Speculative decoding exercises
draft passes, batched verification and many more small launches, so it is a
better test in principle.

It is a better test in practice too — but only with a method that controls for
something that turned out to matter far more than the ROCm version.

## The n-gram assist has memory across requests

Measured with an identical prompt, temperature 0, seed 1, `cache_prompt: false`,
eighteen consecutive runs:

```
run  1:  85.6      run  7: 206.5      run 13: 206.3
run  2: 106.9      run  8: 203.5      run 14: 206.6
run  3: 131.5      run  9: 205.9      run 15: 207.3
run  4: 169.3      run 10: 208.2      run 16: 208.0
run  5: 236.1      run 11: 207.1      run 17: 206.5
run  6: 213.2      run 12: 206.1      run 18: 207.3
```

Throughput climbs from 85.6 to a plateau around 207 tok/s and then holds
extremely still: runs 13–18 have a median of 206.9 and a standard deviation of
**0.7**, a spread of ±0.3%. That is tighter than the plain-generation
measurement on the 4B.

The gain is specific to the text being generated. Switching to a different
prompt drops straight back to 86.9, and its second run is 101.4 — the same
climb starting over. `ngram-mod` is accumulating n-gram statistics from what it
has already produced, so the assist drafts better the more it has seen of that
particular output.

**Consequence for any measurement.** A throughput figure for MTP + n-gram is
meaningless without saying where on that curve it was taken. One number can be
85 or 207 for the same configuration on the same hardware. Two figures are
needed, and they answer different questions:

- **cold** — a prompt the assist has not seen. What a user gets on their first
  question of a session.
- **plateau** — the same prompt warmed until the number stops moving, about
  twelve runs here. What a user gets in a repetitive loop, which is the case the
  assist exists for.

## Baseline on ROCm 7.2.4

| configuration | cold | plateau |
|---|---|---|
| MTP + `ngram-mod` (n-max 64, n-min 48, n-match 24) | 86.9 tok/s | **206.9 tok/s** (stdev 0.7) |
| MTP alone (`draft-mtp`, draft-max 6) | 96.9 tok/s | n/a — nothing to warm |

Two things worth noticing beyond the ROCm question.

**The assist costs a little when it is cold**: 86.9 against 96.9 for MTP alone.
It is drafting from statistics it does not have yet, and the failed drafts are
not free. That matches the note Autoconfigure already shows — that Autotune
measures whether the assist helps, rather than assuming it.

**And it is worth 2.1x when warm**: 206.9 against 96.9. On repetitive work the
assist is the single largest speed difference available here, far larger than
anything the ROCm version does.

## Why the earlier three-run comparison was worthless

The first attempt varied the prompt between runs, which varies the thing being
measured: speculative acceptance depends on the content. That produced
80.9–106.9 against 80.7–99.2 for two configurations — overlapping spreads and no
conclusion, which I initially misread as measurement noise. It was not noise; it
was the cache state and the content changing together.

There is a second, smaller effect underneath: the first run after an idle period
is slow on any configuration, including MTP with no assist (80.7 against 99.2),
which is GPU clocks ramping rather than anything in llama.cpp.

## The comparison: ROCm 10 is about 25% faster here

Same model, same config, same llama.cpp ref, same prompt, same protocol.

| | ROCm 10.0.0 | ROCm 7.2.4 | difference |
|---|---|---|---|
| **plateau** | **258.37 tok/s** | 206.90 tok/s | **+24.9%** |
| plateau stdev | 0.52 | 0.70 | |
| plateau spread | 257.39 – 258.78 | 206.30 – 208.00 | no overlap |
| cold (run 1) | 87.80 tok/s | 86.90 tok/s | +1.0% |
| runs to plateau | ~6 | ~12 | |

**This is a real difference and not a measurement artifact.** The spreads are
1.4 and 1.7 tok/s wide and sit 49 tok/s apart. Every one of the six plateau runs
on ROCm 10 is faster than every one of the six on 7.2.4. Where the
plain-generation comparison produced medians inside each other's noise, this
produces two cleanly separated distributions.

Note also that ROCm 10 reaches its plateau in about six runs against twelve, and
plateaus higher — so the assist is both learning faster and paying off more.

**Cold is unchanged.** 87.8 against 86.9 is within run-to-run variation. The
gain appears only once the assist is drafting well, which is consistent with
where it should appear: a warm assist submits many draft tokens per step for
batched verification, and that is the regime the two runtimes differ in. Plain
single-stream decode, one token at a time, was identical on both.

## Caveats, in order of how much they should bother you

**Different compilers.** The two images are gcc 13.3.0 (Ubuntu 24.04) and
15.3.1 (Fedora 43). This is a comparison of two images, and part of the
difference could be the compiler rather than ROCm. Weakly mitigated by the
plain-generation test, which found no difference across the same two images —
but that exercises different code, so it does not settle it. Settling it
properly would need ROCm 7.2.4 and ROCm 10 on the same distribution, which is
not something AMD publishes.

**The generated text was not compared.** At temperature 0 with a fixed seed the
output should be identical, but two ROCm versions could differ in floating-point
detail, and if the text diverged the n-gram acceptance rate would differ too —
which would mean part of the 25% is different content rather than more speed.
The ROCm 10 output is recorded for a future check: 256 tokens, all of them
`reasoning_content`, 1047 characters, sha256 beginning `22779d1725`. The
equivalent was not captured on 7.2.4, which was an oversight — capture it on
both sides next time.

**All 256 tokens were thinking tokens.** Qwen3.5 reasons before answering and
never reached its final answer inside 256 tokens (`finish_reason: length`), so
this measures generation of reasoning text. That is real generation work and
identical in shape on both sides, so the comparison holds — but it is not a
measurement of answering a question end to end.

## What this means in practice

For a repetitive workload with MTP and the n-gram assist warm — editing code,
summarising similar documents, anything that regenerates text resembling what
it has already produced — ROCm 10 is worth about 25% on this hardware. For a
cold prompt, or for generation without speculative decoding, it is worth
nothing measurable.

That is a more useful conclusion than either measurement alone would have given,
and it is the opposite of what the first comparison suggested.
