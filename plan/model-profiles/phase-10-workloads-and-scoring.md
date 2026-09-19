# Phase 10 — Autotune workloads, spec value encoding and scoring

**Depends on:** 04 (snapshot fields that the scores read), 05 (`CPUMoE`, applied by the moved `ApplySnapshotToConfig`) · **Enables:** 11,
12

## Goal
The measuring pieces autotune needs, all testable without a GPU:
- two benchmark presets that stand for the use cases (chat and code editing),
  including a new code-editing prompt style;
- a way to name a specific draft file in a speculative-decoding sweep value;
- pure functions that score a run for each goal, apply the noise rule, and
  rank settings by simplicity.

## Files touched
- `internal/benchmark/runner.go`:
  - `PromptStyleCode`;
  - a `BenchCodeText` constant (a self-contained Python module of about
    1,400 tokens: a small inventory class with methods and docstrings);
  - `BenchPromptCodePrefixTemplate`: "Rename the method `add_item` to
    `add_stock` everywhere, add type hints to every function, and return the
    complete updated file.";
  - `buildPromptFor` handles the new style, and turns thinking off for it as
    it does for echo.
- `internal/benchmark/benchmark.go` `Presets()`: add
  - `autotune-chat`: sizes {512, 4096}, output 256, 3 reps, style analyze,
    thinking off;
  - `autotune-code`: size {1536}, output 512, 3 reps, style code, thinking
    off;
  - both marked `Hidden: true` (a new `Preset` field);
  - a new `VisiblePresets()` that returns the non-hidden presets.
    `GetPreset` keeps using `Presets()`, so hidden presets still resolve.
- `internal/api/bench_about.go:39`, `internal/api/bench.go:525` and
  `internal/api/bench_jobs.go:804`: use `benchmark.VisiblePresets()` instead
  of `benchmark.Presets()`, so the About page, the quick-run form and the job
  form do not list the autotune presets.
- `internal/benchmark/sweep.go`:
  - `parseSpecValue`/`encodeSpecValue`/`canonicalSpecValue` accept a
    `draft_model` key holding a model registry ID or the absolute path of an
    MTP head file;
  - `applySpecValue` sets `MtpPath` when the value's draft mode is
    `draft-mtp`, and `DraftModelPath` for every other draft mode, resolving
    an ID through the resolver passed in;
  - export `EncodeSpecValue` (the planner in `internal/autotune` needs it)
    and `ApplyOverrides` (renamed from `applyOverrides`).
  - a `split_mode` sweep field (values `layer`, `tensor`), with restart.
- `internal/benchmark/snapshot.go` (from Phase 04): move
  `applySnapshotToConfig` here from `internal/api/jobs_env.go:657` as the
  exported `ApplySnapshotToConfig(cfg *models.ModelConfig, snap ConfigSnapshot)`,
  update `jobs_env.go` to call it, and add
  `ConfigForValues(base models.ModelConfig, values map[string]string, resolve func(id string) (string, error)) (models.ModelConfig, error)`,
  which builds the snapshot, applies `CellOverrides` + `ApplyOverrides`, and
  writes the result back onto a copy of `base`. Phase 12 saves profiles with
  it.
- `internal/autotune/score.go` (new): workloads, goals, scoring, the noise
  rule, and simplicity.
- `internal/autotune/score_test.go`, `internal/benchmark/runner_code_prompt_test.go`,
  `internal/benchmark/sweep_spec_test.go` (additions).

## Steps
1. **Code prompt.**
   - Write `BenchCodeText` in the same `runner.go` constant style.
   - `buildPromptFor(..., PromptStyleCode)` emits the nonce header, the
     prefix, then the code. If the target is longer than the code, repeat the
     code as a second module, "inventory_v2.py".
   - The expected answer repeats most of the input, which is the case n-gram
     assist is built for, without the unrealistic "reproduce exactly" of the
     echo style.
2. **Presets** as listed above. `Hidden` defaults to false, so existing
   presets are unchanged.
3. **`draft_model` in spec values.**
   - Example: `draft+ngram-mod:draft_model=unsloth_Qwen3.5-0.8B-GGUF--Qwen3.5-0.8B-Q8_0.gguf,draft_max=16`.
   - Keys stay sorted in the canonical form.
   - `JobEnv` gains `ResolveModelPath(id string) (string, error)`.
     `jobs_env.go` implements it with `registry.Get(id).FilePath`, and
     `fakeEnv` returns a fixed path.
   - A value that is an absolute path is used as it is.
   - An unknown ID fails the cell with "draft model ID not installed".
4. **`split_mode` sweep field**, following the `flash_attention` string-field
   pattern, with `RestartsRouter: true`.
   - Move `applySnapshotToConfig` to `benchmark.ApplySnapshotToConfig` as
     listed above. At the same time, make it apply `SplitMode`, `MainGPU`,
     `Parallel`, `DraftCtxSize`, `DraftGPULayers` and `DraftKVCacheQuant`
     (the Phase 04 fields). `CPUMoE` is already applied, since Phase 05.
   - **Ordering in `jobs_env.go`:** the existing `resolveGPUAssignment`
     recomputes `SplitMode`/`MainGPU`/`TensorSplit` when `GPUAssign`
     changes. Apply a swept `split_mode` **after** that call, so the swept
     value wins. (Autotune never sweeps `gpu_assign`.)
   - A test fills every `ConfigSnapshot` field and checks that each one with
     a matching `ModelConfig` field arrives, by reflection.
   - Implement `ConfigForValues`.
5. **Workloads and goals in `internal/autotune/score.go`.**
   ```go
   type UseCase string // "chat" | "code" | "mixed"
   type Goal string    // "generation" | "prompt" | "response"
   type Shape struct{ Preset string; PromptTokens, OutputTokens, PPSizeTokens int }
   var Workloads = map[UseCase][]Shape{
       "chat":  {{"autotune-chat", 512, 256, 4096}},
       "code":  {{"autotune-code", 1536, 512, 1536}},
       "mixed": {{"autotune-chat", 512, 256, 4096}, {"autotune-code", 1536, 512, 1536}},
   }
   type Score struct{ Value, Std float64 } // higher is better for all goals (response uses negative seconds)
   func ScoreRuns(uc UseCase, g Goal, runs map[string]*benchmark.BenchmarkRun) (Score, error) // keyed by preset
   ```
   - **generation**: mean `TGMean` over the shapes, read from the
     `SizeSummary` row nearest `PromptTokens`. Std is the root mean square of
     `TGStd`.
   - **prompt**: mean `PPMean` from the row nearest `PPSizeTokens`.
   - **response**: `-Σ (PromptTokens/PP + OutputTokens/TG)` over the shapes,
     with Std propagated as `sqrt(Σ (P/PP²·PPStd)² + (O/TG²·TGStd)²)`.
     Missing rows produce an error; the cell is treated as failed.
6. **Noise rule.**
   `Beats(a, b Score) bool` is true when `a.Value - b.Value > sqrt(a.Std² + b.Std²)`.
7. **Simplicity.** `Complexity(cfg ConfigSnapshot, base ConfigSnapshot) int`
   gives:
   - +1 for each of batch, ubatch, flash attention, split mode and threads
     that differs from base;
   - +2 for an n-gram assist and +3 for a draft method;
   - +1 for each non-default draft or assist parameter.
8. **Candidates and labels.**
   ```go
   type Candidate struct {
       Values map[string]string  // sweep field → canonical value; empty = baseline
       Label  string             // plain language, from Describe
       Config benchmark.ConfigSnapshot
       Scores map[Goal]Score
       Goals  []Goal             // goals this candidate won (set by the runner)
   }
   func Describe(values map[string]string) string
   ```
   `Describe` turns values into plain text, e.g. `ubatch_size=1024` →
   "Prompt batch 1024", `spec_type=draft-mtp+ngram-mod` → "MTP with
   ngram-mod assist", and empty → "Starting profile, unchanged".
9. **Pick.** `Pick(cands []Candidate, g Goal) (winner Candidate, ranked []Candidate)`:
   - find the best score;
   - the winner is the candidate with the lowest `Complexity` among those
     the best does not `Beat` (ties are broken by score);
   - `ranked` is sorted by score, for choosing finalists.

## Build gate
`go build ./... && go vet ./... && go test ./... && make js-test`
(`make js-test` because the job form's preset list changes.)

## Test plan
- Prompt test: a code-style prompt at 1,536 tokens contains the prefix and
  the code, and its length is within 10% of 1536×4 characters.
- The preset list the job form renders does not include `autotune-*`.
- Spec value round trip with `draft_model`; canonical key order; an unknown
  draft ID fails the cell (job runner test with `fakeEnv`).
- `ConfigForValues` with `spec_type=draft-mtp+ngram-mod:draft_max=3` and
  `ubatch_size=1024` returns a config with those fields set and everything
  else equal to the base.
- The About page, quick-run form and job form render without the autotune
  presets.
- A `split_mode` sweep cell produces an INI with `split-mode = tensor`.
- Scoring table tests:
  - response time from known PP/TG;
  - Std propagation;
  - mixed averages over two presets;
  - a missing row is an error.
- `Describe` covers every sweep field autotune uses, and the baseline.
- `Pick` tests:
  - a 1% gain inside the noise keeps the simpler candidate;
  - a 20% gain outside the noise picks the more complex one;
  - MTP alone versus MTP + ngram-mod, where the difference is within the
    noise, picks MTP alone.

## Commit
`feat(benchmark): add use-case workloads and scoring for autotune`

## Rollback
Revert the commit. The hidden presets and the new spec key are unused until
Phase 11. Runs recorded with the new presets stay in history and display
normally.

## As implemented

- **The swept draft file rides on `DraftModelPath`.** The plan added a
  `DraftModel` override field, but two guard tests exist to catch an
  override with no destination and a field the sweep registry does not
  account for, and a second field would have had neither. A spec value's
  `draft_model` therefore writes the unresolved ID into `DraftModelPath`,
  and `runCell` resolves it (`ResolveDraftFile`) before the cell runs and
  before the run records it, moving it to `MtpPath` for `draft-mtp`.
- **`MtpPath` in the snapshot** came forward from Phase 11 to here, since
  that is where the draft file is resolved.
- **`SplitMode` and `MainGPU` are copied only when the snapshot has them.**
  They are derived from the GPU assignment when a config is saved, and runs
  recorded before the snapshot carried them have neither; copying a blank
  would erase the placement a config derived. A swept `split_mode` is
  applied after `resolveGPUAssignment`, so it wins.
- **`split_mode` is `AffectsEval`**: it reaches the evaluation command line
  through the placement flags, like `gpu_assign`.
- **Scoring**: `ScoreRuns` reads the per-size row nearest the size a goal
  asks about, because llama.cpp reports what it actually tokenized. Speeds
  average over a mixed workload's shapes; response times add up, since the
  workload is one of each. Response time is carried as negative seconds so
  that higher is better for every goal.
- **`Pick`** takes the fastest candidate, then replaces it with any plainer
  one the fastest does not `Beat` — the noise rule — and also returns the
  full ranking for choosing finalists.
