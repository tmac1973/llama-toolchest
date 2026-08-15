# Phase 01 — Evaluation engine and binary install

**Depends on:** nothing · **Enables:** dataset plumbing (02) feeds it paths;
the job runner (03) invokes it.

## Goal

The core that runs `llama-perplexity` and understands its output: a new
`internal/evaluate` package with the mode definitions, the
ConfigSnapshot→CLI flag mapping, subprocess execution, and stdout parsing
into typed scores — plus the one-line builder change that installs
`llama-perplexity` next to `llama-server`. After this phase the engine is
fully unit-tested against canned outputs; nothing invokes it yet.

## Files touched

- `internal/builder/builder.go` — the install step (currently copies only
  `llama-server`, `builder.go:511`) also copies `llama-perplexity` from
  the same candidate locations (`bin/llama-perplexity`); missing binary is
  NOT a build failure (old refs, exotic layouts) — capability cells detect
  absence at run time.
- `internal/evaluate/evaluate.go` (new) — modes, spec, flag mapping,
  execution.
- `internal/models/gpu_assign.go` — exported `GPUPlacementFlags` wrapper
  around the unexported `gpuPlacementParams` (step 2); the preset
  emitter's behavior is unchanged.
- `internal/evaluate/parse.go` (new) — stdout parsers per mode.
- `internal/evaluate/evaluate_test.go`, `parse_test.go` (new).
- `internal/benchmark/benchmark.go` — `EvalScores` struct and the additive
  `Eval *EvalScores` field on `BenchmarkRun` (defined here because
  `internal/evaluate` must not import `internal/benchmark`; evaluate
  returns its own result type, the runner copies it over in phase 03).

## Steps

1. Mode registry in `internal/evaluate`:
   ```go
   type Mode string
   const (
       ModePerplexity Mode = "perplexity"
       ModeKLDiv      Mode = "kl-divergence"
       ModeHellaSwag  Mode = "hellaswag"
       ModeWinogrande Mode = "winogrande"
   )
   type Spec struct {
       Binary    string // path to llama-perplexity
       ModelPath string
       Mode      Mode
       DatasetPath string
       Tasks     int    // hellaswag/winogrande task cap; 0 = full
       Chunks    int    // perplexity/KL chunk cap; 0 = full
       KLBasePath string // ModeKLDiv: read base; KL base GENERATION uses GenerateKLBase
       Flags     []string // pre-mapped model/config flags (step 2)
   }
   ```
2. Flag mapping — `MapConfigFlags(snap SnapshotSubset) []string`, the
   single allow-list shared by execution and the recorded snapshot. To
   avoid an import cycle it takes a small struct of plain fields.
   Mapped from those fields: `--n-gpu-layers`, `--threads`,
   `--batch-size`, `--ubatch-size`, `--flash-attn on|off`,
   `--cache-type-k/-v` (KV quant), `--direct-io` (accepted by the tool,
   `common/arg.cpp:2656`; mapping it keeps the excluded list a complete
   statement — every `ConfigSnapshot` field is either mapped or listed).
   GPU placement (`--split-mode`, `--main-gpu`, `--device`, trimmed
   `--tensor-split`) is NOT derivable from the snapshot — it carries
   neither `SplitMode` nor `MainGPU` (`benchmark.go:95-113`), and a
   swept `gpu_assign` value only becomes a tensor-split through the
   merge-and-resolve the performance path already does (the standalone
   package-level functions `applySnapshotToConfig`,
   `internal/api/jobs_env.go:283-313`, and `resolveGPUAssignment`,
   `jobs_env.go:324-342` — directly callable, no extraction needed).
   The api side therefore owns the WHOLE flag construction: phase 03
   adds a `JobEnv` method `EvalFlags(modelID string, snap
   ConfigSnapshot, buildID string) ([]string, error)` whose
   implementation merges the snapshot onto the saved `ModelConfig`,
   runs `resolveGPUAssignment`, fills `SnapshotSubset` (plain fields +
   `PlacementFlags []string`), and calls `evaluate.MapConfigFlags` —
   which remains the single allow-list deciding what reaches the
   command line. The cell loop puts the returned flags into
   `Spec.Flags`; `Spec` itself stays engine-shaped (no model ID, no
   snapshot). `PlacementFlags` come from a new exported wrapper in
   `internal/models`: `GPUPlacementFlags(c *ModelConfig, backend
   string) []string` — a thin wrapper over the unexported
   `gpuPlacementParams` (`internal/models/gpu_assign.go:286`) that
   formats its `specParam` pairs as `--name value` strings. The preset
   emitter and llama-server arg builder keep calling the unexported
   function with their own formatting (INI vs CLI); one implementation,
   the wrapper is just the exported CLI-formatted view.
   Excluded by listing, with a comment naming each: model-config
   context size (replaced by the fixed evaluation context, step 3),
   parallel slots, speculative decoding, sampling, mmproj,
   context-shift.
   Context size: the ctx value IS the perplexity chunk size
   (`n_chunk_max = tokens/n_ctx`, `perplexity.cpp:493`), so it
   determines the score. The tool's own default happens to be 512
   (`perplexity.cpp:2015`; an explicit non-positive value is rejected,
   `perplexity.cpp:2024-2030`), but the engine never relies on an
   upstream default: it always passes a package constant
   `EvalContextSize = 512` — the value llama.cpp's own wikitext-2
   perplexity figures are quoted at — never the model's configured
   context. The constant is part of every invocation in step 3 and
   part of the KL cache key (phase 02).
3. Command assembly per mode (documented against
   `tools/perplexity/perplexity.cpp` in the checkout). Every invocation
   carries `--ctx-size 512` (`EvalContextSize`, step 2) — required by
   the tool and score-determining:
   - perplexity: `-c 512 -f <wikitext> [--chunks N]`
   - hellaswag: `-c 512 --hellaswag -f <hellaswag_val> [--hellaswag-tasks N]`
   - winogrande: `-c 512 --winogrande -f <winogrande.csv> [--winogrande-tasks N]`
   - kl-divergence: `-c 512 --kl-divergence --kl-divergence-base <base.kld> -f <wikitext>`
     — no `--chunks` here: the comparison's chunk count comes solely
     from the base file (`params.n_chunks` is read only by the
     perplexity functions, `perplexity.cpp:344,495`), so the flag would
     be inert. The base file embeds its n_ctx, and the tool does NOT
     reliably guard a mismatch (`perplexity.cpp:1717-1722` logs
     without returning; a smaller embedded n_ctx passes silently and
     wins) — the phase 02 cache key carrying ctx and chunks is the
     ONLY thing keeping base and comparison consistent, which is why
     both are in the key.
   - KL base generation: `GenerateKLBase(spec)` runs
     `-c 512 --kl-divergence-base <out.kld> -f <wikitext> [--chunks N]`
     (no `--kl-divergence` flag → the tool writes the base file,
     `perplexity.cpp:461-470`). Same execution model as `Run` (combined
     buffer, parsed after exit) — NO live output parsing: the per-chunk
     generation output is newline-less comma fragments
     (`perplexity.cpp:633`), so live progress cannot come from the
     stream. Progress during generation is the CALLER's job by polling
     the growing output file's size against the phase 02 estimate
     (phase 03 step 2); the engine only runs the process.
4. Execution: `Run(ctx, Spec) (Result, error)` — `exec.CommandContext`
   with the environment treated the way the codebase already launches
   build binaries: prepend the binary's directory to `LD_LIBRARY_PATH`
   so co-located `libllama.so`/`libggml*.so` resolve (mirroring
   `internal/api/jobs_env.go:72-87` and `process.Manager`; without it
   every eval fails to start). Stdout and stderr are captured into ONE
   combined buffer that is both the parse input and the error-tail
   source (bounded, last 64KB kept): llama.cpp's logger routes plain
   `LOG` lines to stdout but `LOG_INF` to stderr (`common/log.cpp:89-92`),
   and the final Perplexity and Winogrande score lines are `LOG_INF` —
   a stdout-only parser would miss two of the four modes. Context
   cancellation kills the process. A nonzero exit returns the tail of
   the combined output in the error.
5. Parsers (`parse.go`), one per mode, anchored to the source's exact
   format strings (all parse the combined output of step 4):
   - `Final estimate: PPL = <f> +/- <f>` (`perplexity.cpp:654`, stderr)
     → Perplexity, PerplexityErr.
   - `Final Winogrande score(<n> tasks): <f> +/- <f>`
     (`perplexity.cpp:1300`, stderr) → Accuracy plus a symmetric error
     mapped into AccuracyCILow/High (acc−err, acc+err), Tasks.
   - HellaSwag prints one line per task
     (`perplexity.cpp:1006`, stdout):
     `%zu\t%3.8lf%%\t[%3.4lf%%, %3.4lf%%]` — task number, cumulative
     accuracy percent, and an ASYMMETRIC 95% confidence interval in
     brackets. The last such line is the score: Accuracy from column 2,
     AccuracyCILow/High from the bracketed bounds, Tasks from column 1.
     The parser test encodes a captured real output block.
   - KL block (stdout): `Mean    KLD: <f> ± <f>`, `Maximum KLD: <f>`,
     `99.9%   KLD: <f>`, and `Same top p: <f> ± <f> %`
     (`perplexity.cpp:2005`) — parse Mean±err, Max, P99.9, SameTopPct±err
     (exact labels pinned from source lines 1949-2005).
   - Every parser returns a typed error naming the missing line when
     output doesn't match, so a llama.cpp format change fails loudly.
6. `Result` struct mirrors `benchmark.EvalScores` fields: mode, dataset
   name, evaluation context size, requested chunk cap (perplexity/KL —
   copied from `Spec.Chunks`, 0 = full; the final-score lines carry no
   actual chunk count and the plan does not invent one), perplexity±,
   accuracy with CI bounds (`accuracy`, `accuracy_ci_low`,
   `accuracy_ci_high` — asymmetric for HellaSwag, symmetric-derived for
   Winogrande), tasks (parsed, actual), KL stats including
   same-top-token percentage, reference identity (filled by the
   runner). `EvalScores` in
   `internal/benchmark/benchmark.go` carries json tags (`eval` on the
   run, snake_case fields), additive and omitempty like every schema
   addition.

## Build gate

`go build ./... && go vet ./internal/... && go test ./internal/...`

## Test plan

- Parser tests with canned output blocks (captured from a real
  `llama-perplexity` run during implementation) for all four modes, plus
  mismatch cases asserting the named-line errors.
- Flag mapping tests: KV quant reaches `--cache-type-k/-v`, GPU subset
  emits the device list, excluded params provably absent.
- Execution test with a fake binary (shell script echoing a canned block)
  covering success, nonzero exit with tail-of-output error, and context
  cancellation.
- Builder: extend `profiles_test.go`-style coverage asserting the install
  candidate list includes `llama-perplexity` (unit level; a real build is
  manual).
- Manual: rebuild the vulkan build on the dev box; confirm
  `llama-perplexity` lands in the build output dir.

## Commit

`feat(evaluate): llama-perplexity evaluation engine and binary install`

## Rollback

Delete `internal/evaluate/`, revert the two-line install addition and the
additive `EvalScores` field. No stored data or behavior touched.
