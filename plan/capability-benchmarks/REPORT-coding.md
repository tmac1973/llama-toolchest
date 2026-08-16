# Coding report

Unattended coding run for the capability-benchmarks plan (`plan/capability-benchmarks/`).
All 25 steps across 4 phases were implemented, committed, and verified. Working tree is clean
except the plan tracking files. Fresh verification at report time:
`go build ./... && go vet ./internal/... && go test ./...` → all packages `ok` (test cache cleared first, full run ~11s).

## What was built

Commits (one per phase, on top of `e893f1e chore: pre-ralph baseline`):

- `b436d8c` Phase 01 — Evaluation engine and binary install (11 files, +1543)
- `b1a7143` Phase 02 — Dataset download, verification, and the logits cache store (10 files, +1575)
- `64d77c9` Phase 03 — Capability presets, job runner integration, and the job form (17 files, +3042)
- `74a666a` Phase 04 — Results display, export, evaluation-data card, and docs (18 files, +1446)

**Phase 01 — evaluation engine** (`internal/evaluate/evaluate.go`, `parse.go`):
- Mode registry (perplexity, kl-divergence, hellaswag, winogrande) with `EvalContextSize = 512`; engine-shaped `Spec` (binary/model/dataset paths, Tasks/Chunks caps, KLBasePath, pre-mapped Flags).
- `MapConfigFlags(SnapshotSubset)` as the single flag allow-list (`--n-gpu-layers`, `--threads`, `--batch-size`/`--ubatch-size` when >0, `--flash-attn` always pinned, `--cache-type-k/-v`, `--direct-io`, then caller placement flags); documented exclusion list (context size, gpu_assign/tensor_split, speculative decoding, parallel/sampling/mmproj/context-shift).
- Command assembly per mode documented against `tools/perplexity/perplexity.cpp`: every invocation carries `--ctx-size 512`; KL comparison omits `--chunks` (inert — chunk count comes from the base file); `GenerateKLBase` runs the generation form.
- `Run(ctx, Spec) (Result, error)` on `exec.CommandContext`, LD_LIBRARY_PATH prepended (mirrors jobs_env.go/process.Manager), stdout+stderr captured into one mutex-guarded 64KB tail buffer, cancel errors are `errors.Is`-able, output tail returned on nonzero exit. One parser per mode in `parse.go` anchored to the source's exact output format; `Result` mirrors `benchmark.EvalScores`.

**Phase 02 — datasets + logits cache** (`internal/evaluate/datasets.go`, `klcache.go`, `internal/models/gguf.go`):
- Eval-data layout helpers (`EvalDataRoot`/`DatasetsDir`/`LogitsDir`) under a caller-passed root.
- Pinned dataset table: URLs and SHA-256s pinned by actually downloading all three artifacts (wikitext-2 zip `ef7edb56…` with `wiki.test.raw` extraction, hellaswag `d5725393…`, winogrande CSV `6726173e…`); licenses CC BY-SA 3.0 / MIT / Apache-2.0 live in the table as the single source and render in Phase 04's UI.
- `EnsureDataset`: stat short-circuit, temp download → SHA-256 verify → rename, retry for transient 404/5xx/network only, named expected-vs-got mismatch errors with no leftover files, plus `Verify(root)` for UI state.
- `klcache.go`: `KLBaseKey` (chunks+ctx in key) → `~`-separated deterministic filename with lossless round trip, `ListKLBases` (key+size+mtime, ignores `.partial`), `HasKLBase`, idempotent `DeleteKLBase`, `CleanStalePartials` wired into startup next to dir creation. Disk guard uses the plan's estimate formula (verified against perplexity.cpp source, 1.15 factor, 262144 worst-case vocab fallback; refuses when free < estimate + 2 GiB margin; unknown space passes).
- Additive `VocabSize` field in GGUF parse (tokenizer.ggml.tokens array length, header-only read) with registry backfill.

**Phase 03 — presets, runner, job form** (`internal/benchmark/*`, `internal/api/jobs_env.go`, job form):
- 8 capability presets (perplexity / kl-divergence / hellaswag / winogrande × quick / full) with duration-hint labels, descriptions naming the fixed 512-ctx and offline server, `PresetSourceCapability` constant, `EvalMode`/`EvalTasks`/`EvalChunks` fields, and an internal-error guard in runner.go against capability presets reaching the performance path.
- `JobEnv` additions in `jobs_env.go` (with the api server's stoppedForEval flag and ClearEphemeralConfig eval-stop fix): `StopRouterForEval`, `EvalBinary` (rebuild message), `EnsureEvalData`, `ResolveKLReference` (override, else largest repo quant incl. self, else skip case), `EvalFlags` (merge → placement → ValidateBatchSizes → MapConfigFlags, single flag source), `EnsureKLBase` (disk guard, `.kld.partial` → rename, growth-progress ticker, delete-on-failure), `RunEval`, `ModelInfo`.
- Collapse rule via `SweepField.AffectsEval` (9 fields true, deliberately not `RestartsRouter`) in `ExpandCellsWithSweeps`.
- `runCapabilityCell`: first-created run, stop-before-GPU ordering, KL skip path (SkipReason, run deleted, zero runs), terminal state on every exit.
- Router lifecycle: split `CheckBuildRunnable`/`EnsureBuildActive` guards; cache invalidation after capability cells so a following performance cell restarts the router.
- Single-run quick-benchmark form filters capability presets out; job form renders capability presets in the existing preset checkbox list.

**Phase 04 — results, export, data card, docs**:
- Conditional Score column in the run/job list and cell matrix (only when any run/cell has Eval) via shared `evalScoreText` (`internal/api/bench_eval_display.go`): `PPL 6.234 ±0.04 (100 chunks)`, `(full)` for cap 0, `HellaSwag 77.2% [75.9–78.5] (400)`, `Winogrande 74.1% ±2.2 (400)`, `KLD 0.012 ±0.001 · same top token 97.4% (vs ref)`. Capability rows show em-dashes in timing columns and vice versa; KL skip cells render SkipReason muted in the job grid.
- Compare view: same conditional column + Score sort; runs of different modes each render their own metric (no cross-metric math); capability runs (no SizeRows) get a single compare-table row with em-dash timings.
- Export: CSV gains `eval_mode`/`eval_dataset`/`eval_score`/`eval_error`/`eval_tasks_chunks`/`eval_kl_stats`/`eval_reference` (empty for performance runs); JSON carries the run's EvalScores.
- Evaluation Data card under the jobs list (`eval_data.html`/`eval_data.go`): pinned datasets with license/size/state (verified / not downloaded / present-hash-mismatch) and the KL logits list.
- About-benchmarks modal preset table gains the capability rows; help page gets a "Capability benchmarks" subsection; render tests cover run rows with `Eval` set (score + em-dash).

## Verification status

Per-phase, per the TODO file's headings — every phase passed on the first verification, zero repair cycles:

- **Phase 01 — Evaluation engine and binary install**: verified (`go build ./... && go vet ./internal/... && go test ./...` passed fresh, non-cached), repairs: 0. Committed.
- **Phase 02 — Dataset download, verification, and the logits cache store**: verified, repairs: 0. Committed. Dataset hashes were pinned empirically (all three artifacts actually downloaded and hashed during the run), not taken from the plan.
- **Phase 03 — Capability presets, job runner integration, and the job form**: verified cleanly (all packages ok, full run 11s), repairs: 0. Committed.
- **Phase 04 — Results display, export, evaluation-data card, and docs**: verified, repairs: 0. Committed.

Independent re-verification for this report: `go clean -testcache && go build ./... && go vet ./internal/... && go test ./...` — all 12 test-bearing packages `ok` (internal/api 10.1s, internal/benchmark 0.6s, internal/evaluate 0.8s, rest under 0.1s). No test failures, no vet findings.

## Blocked items

None. All 25 steps are marked done, all 4 phase headings record `verify: passed, repairs: 0`, and no item carries the `[!] blocked` mark. No phase was skipped or never reached.

Decisions made unattended (recorded in PROGRESS, worth a human skim — not blockers, but judgment calls):
- **Hash mismatch handling in `EnsureDataset`**: a present-but-wrong-hash file is not silently re-downloaded; it surfaces as a named mismatch error (UI shows "present-hash-mismatch") rather than destroying the user's file.
- **KL reference resolution**: override → largest quantization in the repo (including the model itself) → skip with a reason.
- **Disk-guard formula**: worst-case vocab fallback of 262144 when vocab size is unknown, 1.15 safety factor, 2 GiB margin, "unknown free space" passes.

## Suggested next steps

1. **Manually review the 4 commits** (`git diff e893f1e..74a666a`) before release — especially the unattended decisions above (hash-mismatch behavior, KL reference selection, disk-guard formula) and the flag allow-list in `MapConfigFlags` against real engine behavior.
2. **Run an end-to-end capability cell on real hardware**: the plan's verification is build/vet/test (unit + render tests); nothing here exercises an actual offline perplexity/hellaswag/winogrande run, KL base generation, or the router stop→eval→resume lifecycle with a real binary. A quick-preset job per mode is the cheapest smoke test.
3. **Check the binary install path in the packaging/CI matrix**: Phase 01 adds the evaluation binary install (per the phase title); confirm it's covered by the release build scripts and Docker images (`Dockerfile.cpu/cuda/rocm`) — the commits show changes in `internal/builder` but packaging files were untouched.
4. **Exercise the KL skip and disk-guard paths in the UI** (small free-space scenario, no-reference scenario) — these are test-covered but visually untested (muted SkipReason rendering, refusal message).
5. **Decide on the dataset licenses' UI copy**: CC BY-SA 3.0 (wikitext-2), MIT (hellaswag), Apache-2.0 (winogrande) are now surfaced in the Evaluation Data card — legal/product sign-off if this ships.
6. **Commit or discard the modified plan tracking files** (`plan/capability-benchmarks/TODO-coding.md`, `PROGRESS-coding.md` are the only working-tree changes).
