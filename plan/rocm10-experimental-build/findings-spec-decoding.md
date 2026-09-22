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

## To complete the comparison

The ROCm 10 half needs the container switched back and the same protocol run:
one fresh prompt for the cold figure, then the same prompt repeated until the
number stops moving for the plateau. With a plateau standard deviation of 0.7,
a difference of even 2% between ROCm lines would be clearly visible — which
makes this, properly run, a far sharper instrument than the plain-generation
test that found nothing.
