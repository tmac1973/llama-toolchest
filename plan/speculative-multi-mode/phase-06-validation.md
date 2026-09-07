# Phase 06 — Measure the combination on this hardware

**Depends on:** phases 01, 02, 03, 04, 05 · **Enables:** nothing further. This
is the phase that turns the project's central claim into a local number.

## Goal

Run the C1-vs-C7 comparison from the source report on this machine, using the
benchmark surface phases 04 and 05 built, and write the result into this plan
folder. Two questions get answered: does the n-gram assist cost anything on new
text (the report's unresolved ~4 tokens/second regression), and how much does it
earn on recall (the report's 4×). No code is written in this phase.

## Files touched

- `plan/speculative-multi-mode/results.md` — **new**. The numbers, the exact job
  configuration, and what they mean. This is the only file this phase writes.

## Steps

1. **The model is `unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-UD-Q8_K_XL`.** It is
   the largest model installed on this machine that carries an MTP head
   (12.3 GiB), so generation is the bottleneck — the condition under which
   speculative decoding matters at all. Its MTP head is baked into the main
   GGUF, so `draft-mtp` runs as self-speculation with no separate head file,
   which is the configuration the report's C1 and C7 rows used. Record the build
   ID and the GPU configuration in `results.md` alongside it; a throughput
   number without them is not reusable.

2. **Build the job.** One benchmark job, one model, one build, sweeping two
   axes:

   - **Preset:** `internal-standard` (new text) and `internal-echo` (recall).
   - **spec_type:** four values —
     - `none` (the report's C0 row)
     - `draft-mtp:draft_max=3` (C1)
     - `ngram-mod:assist_n_max=64,assist_n_min=48,assist_n_match=24` (C5)
     - `draft-mtp+ngram-mod:draft_max=3,assist_n_max=64,assist_n_min=48,assist_n_match=24` (C7)

   Eight cells. The `draft_max=3` matches the report's C1/C7 rows rather than
   the toolchest's own MTP default of 6, so the numbers are comparable to the
   table the project was justified on. The C5 row is not optional: n-gram alone
   should sit at baseline under `internal-standard` and near the combined value
   under `internal-echo`, and that is the cheapest available check that the echo
   preset is really producing a recall workload rather than something the model
   is inventing.

3. **Sanity-check before trusting anything.** In the completed job's results,
   confirm for every cell that the recorded prompt-token count — llama.cpp's
   `timings.prompt_n`, stored as `prompt_tokens` on each result — is close to
   the nominal 2048 and does not collapse between repetitions. A collapsed
   count means the prompt cache was hit and the throughput figure is
   meaningless: the failure the nonce machinery exists to prevent, and the one
   to rule out first.

4. **Read the two rows.** The result is one of three, and each says something
   different:

   - **Combined ≈ MTP alone on `internal-standard`, and much faster on
     `internal-echo`.** The report's finding reproduces. Nothing changes.
   - **Combined slower than MTP alone on `internal-standard` beyond the run-to-
     run error bar.** The report predicted this is possible and explained it:
     the draftless method wins a contested step, so a poor n-gram match displaces
     a good MTP draft. Record the size of the cost against the size of the echo
     gain. This does not change the feature — it is a per-workload tuning
     decision, and the point of building it as an independent slot is that a
     user can turn it off for one model and on for another.
   - **Combined no faster on `internal-echo`.** Then either the preset is not
     producing recall (check step 3 and the actual generated text) or the assist
     is not reaching llama-server (check the launched command line in the job
     detail's flag diff for `--spec-type draft-mtp,ngram-mod`). Fix the cause
     before recording anything; this outcome is a bug report, not a result.

5. **Write `results.md`.** Model, quantization, build ID, GPU configuration,
   date. The eight-cell table with generation tokens/second and its error bar
   per cell. Three sentences on which of the three readings above occurred. If the
   combination costs anything on new text, state the number plainly — the
   project's own value section says making it measurable was the goal, so a
   negative number is a result, not a failure.

6. **Leave the help copy alone.** Phase 03's help dialog already tells the user
   that a combined run can be slower on new text and that the sweep is how they
   find out. That statement is true whichever way this measurement lands, so it
   is not edited here — the measured figure goes in `results.md` and nowhere
   else. This phase writes no code and changes no template.

## Build gate

No code changes, so no build gate. The gate is the job itself:

```
# in the UI: Benchmarks → New job, configured as in step 2 (8 cells)
# every cell must reach "completed"; a failed cell means llama-server
# refused the flags, which is a phase 01–04 bug, not a result
```

## Test plan

- All eight cells complete. A failed cell is a defect in an earlier
  phase — read the cell's error, which surfaces llama-server's own startup
  message.
- The `none` cell's generation throughput matches this machine's known baseline
  for that model. If it does not, the measurement environment is wrong and
  nothing else in the table can be trusted.
- The `ngram-mod` alone cell sits at baseline on `internal-standard`. This is
  the strongest available signal that the two presets really are producing
  opposite workloads.
- The compare table labels the combined cell distinguishably from the MTP-alone
  cell — the `specLabel` helper from phase 02, seen in use.

## Commit

```
docs(plan): record the measured cost and benefit of the n-gram assist

Runs the C1-vs-C7 comparison from the design report on this hardware, across
internal-standard (new text) and internal-echo (recall), with speculative
decoding off, MTP alone, n-gram alone, and both.
```

## Rollback

Nothing to roll back — the phase adds one document and runs a benchmark job. If
the numbers turn out to be wrong, delete the job and `results.md` and re-run;
stored benchmark runs are independent of every other phase.
