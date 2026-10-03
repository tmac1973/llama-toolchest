# Phase 05 — An echo-heavy benchmark preset

**Depends on:** nothing (independent of phases 01–04; must land before phase 06)
· **Enables:** phase 06. Without it the validation sweep can only measure the
half of the claim where the combination changes nothing.

## Goal

Every internal benchmark preset builds its prompt with `buildPromptFor`
(`internal/benchmark/runner.go:395`), which repeats a fixed prose passage and
prefixes it with *"Please analyze the following text carefully and provide a
detailed response."* The model then generates novel prose. That is the workload
where an n-gram assist does nothing at all — `ngram-mod` alone measured exactly
baseline (30.03 vs 30.03 tokens/second) in the source report's table.

The whole case for combining modes rests on the other workload: text the model
has already seen and is reproducing, where the same table shows 324 against 82
tokens/second. Nothing in the toolchest currently produces that workload, so the
4× claim cannot be checked on this hardware. This phase adds a prompt style that
asks the model to reproduce its input verbatim, and an `internal-echo` preset
that uses it. It is useful well beyond this project: it makes every future
n-gram change measurable instead of assumed.

## Files touched

- `internal/benchmark/runner.go` — a `PromptStyle` type, an echo prefix
  template, `buildPromptFor` takes the style, and the call chain threads it.
- `internal/benchmark/benchmark.go` — `Preset` gains `PromptStyle`; the
  `internal-echo` preset is added to `Presets()`.
- `internal/api/bench_about.go` — the About modal discloses both prefixes.
- `web/templates/partials/bench_about.html` — render both.
- `internal/benchmark/runner_test.go` — prompt construction tests.

## Steps

1. **Prompt style.** In `runner.go`, next to the existing constants:

   ```go
   // PromptStyle selects the instruction wrapped around the benchmark
   // passage. The two styles produce opposite generation workloads, which
   // is the point: an n-gram speculative method is worth nothing on
   // PromptStyleAnalyze and worth several times baseline on
   // PromptStyleEcho.
   type PromptStyle string

   const (
       PromptStyleAnalyze PromptStyle = ""      // default; existing behaviour
       PromptStyleEcho    PromptStyle = "echo"
   )

   // BenchPromptEchoPrefixTemplate asks the model to reproduce the passage
   // rather than respond to it, so generation is recall of text already in
   // the context. Exposed so the About modal can show the actual template.
   const BenchPromptEchoPrefixTemplate = "This is benchmark repetition number %d. Reproduce the following text exactly, character for character, with no commentary.\n\n"
   ```

2. **`buildPromptFor`** takes a `style PromptStyle` parameter and selects the
   prefix template from it. Everything else is unchanged and must stay
   unchanged: the nonce-and-size line still goes first, before the shared
   boilerplate, so two prompts diverge within the first few tokens. That
   discipline matters more here, not less — an echo prompt is by construction
   the most cacheable thing the benchmark sends, and a cached prefill would
   report a meaningless number as a measurement, which is the exact failure the
   existing comment at line 380 documents.

   Update `buildPrompt` (the nonce-free warmup form) to pass
   `PromptStyleAnalyze`.

3. **Thread the style** through `sendCompletionWithTimings` → `runOneTest` →
   the loop at `runner.go:236`, which already has `cfg.Preset` in hand and
   passes `cfg.Preset.GenTokens`. Pass `cfg.Preset.PromptStyle` the same way.
   `sendCompletion` (warmup) passes `PromptStyleAnalyze`.

4. **`Preset` gains `PromptStyle PromptStyle`.** Every existing preset leaves it
   zero, which is `PromptStyleAnalyze` — no existing preset changes behaviour,
   and no stored result is invalidated.

5. **Add the preset** to `Presets()`, after `internal-standard`:

   ```go
   {
       Name:  "internal-echo",
       Label: "internal-echo — 3 reps × 2048-token prompt, 512 gen, recall workload (~2 min)",
       Description: "Three repetitions of a 2048-token prompt asking the model to reproduce the passage verbatim, with 512 generated tokens. Generation is recall of text already in the context rather than new prose, which is the workload where n-gram speculative decoding pays off: a file being rewritten, a structured response, a tool-call loop. Compare against internal-standard, whose prompts ask for new text, to see what a speculative setting costs and what it earns.",
       Source:       PresetSourceInternal,
       PromptTokens: []int{2048}, GenTokens: 512, Repetitions: 3,
       PromptStyle:  PromptStyleEcho,
   },
   ```

   2048 prompt tokens gives roughly 8000 characters of passage to echo, and 512
   generated tokens is comfortably inside it, so the model is reproducing
   throughout the measured window rather than running out of source text and
   drifting into invention.

6. **About modal.** `AboutBenchmarks.InternalPrompt` currently carries one
   `RepetitionPrefix`. Add `EchoPrefix` beside it, filled from
   `benchmark.BenchPromptEchoPrefixTemplate`, and render both in
   `bench_about.html` under labels "Analysis prefix (most internal presets)" and
   "Recall prefix (internal-echo)". The modal exists so a reader can see exactly
   what was sent; one preset sending something different and the modal not
   saying so would undo that.

## Build gate

```
make build
go test ./internal/benchmark/... ./internal/api/...
go vet ./...
```

## Test plan

- `runner_test.go`: `buildPromptFor("n", 2048, 1, PromptStyleEcho)` contains the
  echo instruction and not the analyze instruction, and vice versa for
  `PromptStyleAnalyze`.
- Two prompts differing only in style, or only in nonce, or only in target size,
  differ within their first 80 characters — the existing cache-defeating
  property, re-pinned now that a second template exists.
- Length is still `targetTokens * BenchPromptCharsPerToken` for both styles.
- Manually: run `internal-echo` against a model with speculative decoding off,
  and confirm from the response that the generation is in fact reproducing the
  passage rather than commenting on it. A model that refuses or paraphrases
  makes the preset useless for its purpose, so check this before phase 06 —
  if it happens, tighten the instruction rather than changing the measurement.
- Confirm `prompt_n` in the results is near 2048 and does not collapse across
  repetitions, which would mean the prompt cache is being hit.

## Commit

```
feat(benchmarks): add internal-echo, a recall-workload preset

Every internal preset asks the model to analyze a passage, so generation is
always new prose -- the workload where n-gram speculative decoding measures
exactly baseline. The workload where it pays off, text the model is
reproducing, had no preset at all, so the setting's main benefit could not be
measured here.

internal-echo asks the model to reproduce the passage verbatim, keeping the
same nonce and sizing machinery so the prompt cache stays defeated. The About
modal now shows both prompt prefixes.
```

## Rollback

Revert the commit, then delete any benchmark run recorded under
`internal-echo`. Those runs keep their preset name, which no longer resolves, so
`GetPreset` falls back to `internal-standard` and would present a recall
measurement under a new-text label — a mislabeled result, which is the failure
this package guards against everywhere else. Deleting them is the fix; do not
try to keep the preset definition around to make them resolve, because it would
then resolve to a preset that no longer runs the workload it names. Safe to leave partially
applied only with the `PromptStyle` field and the preset landing together; the
field defaulting to the analyze style means a half-applied state runs the wrong
workload under the right name.
