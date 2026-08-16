# Coding progress

## Phase 01 — Evaluation engine and binary install: build turn finished

Phase 01 (Evaluation engine and binary install) is implemented in full and the verification command `go build ./... && go vet ./internal/... && go test ./...` passes (run fresh, non-cached).

What was built:
- `internal/evaluate/evaluate.go` (new): `Mode` registry (perplexity, kl-divergence, hellaswag, winogrande), `EvalContextSize = 512`, `Spec` (engine-shaped: binary/model/dataset paths, Tasks/Chunks caps, KLBasePath, pre-mapped Flags), `MapConfigFlags(SnapshotSubset)` as the single allow-list (`--n-gpu-layers`, `--threads`, `--batch-size`/`--ubatch-size` when >0, `--flash-attn on|off` always pinned, `--cache-type-k/-v`, `--direct-io`, then caller `PlacementFlags`) with the complete exclusion list documented by name (context size, gpu_assign/tensor_split, speculative decoding, parallel/sampling/mmproj/context-shift). Command assembly per mode documented against tools/perplexity/perplexity.cpp: every invocation carries `--ctx-size 512`; KL comparison omits `--chunks` (inert — chunk count comes from the base file); `GenerateKLBase` runs the generation form (no `--kl-divergence` flag) with the same execution model. `Run(ctx, Spec) (Result, error)` uses exec.CommandContext, prepends the binary dir to LD_LIBRARY_PATH (mirroring jobs_env.go/process.Manager), captures stdout+stderr into ONE mutex-guarded 64KB tail buffer (LOG vs LOG_INF routing), wraps context cancellation (errors.Is-able) and returns the output tail on nonzero exit.
- `internal/evaluate/parse.go` (new): one parse
[… note truncated for the prompt tail]

## Phase 01 — Evaluation engine and binary install: verification PASSED — phase committed

## Phase 02 — Dataset download, verification, and the logits cache store: build turn finished

Phase 02 complete: (1) eval-data layout helpers (EvalDataRoot/DatasetsDir/LogitsDir) in new internal/evaluate/datasets.go; (2) pinned dataset table with URLs and SHA-256s pinned by actually downloading all three artifacts (wikitext-2 zip ef7edb56… with wiki.test.raw extraction, hellaswag d5725393…, winogrande CSV 6726173e…), licenses CC BY-SA 3.0/MIT/Apache-2.0 as the single source; (3) EnsureDataset with stat short-circuit, temp-download→SHA-256-verify→rename, retry for transient 404/5xx/network only, named expected-vs-got mismatch errors with no leftover files, plus Verify(root) for UI state display; (4) new internal/evaluate/klcache.go: KLBaseKey (chunks+ctx in key) → `~`-separated deterministic filename with lossless parse round trip, ListKLBases (key+size+mtime, ignores .partial), HasKLBase, idempotent DeleteKLBase, CleanStalePartials called once at startup in cmd/llama-toolchest/main.go next to dir creation, and the disk guard (plan's exact estimate formula verified against perplexity.cpp source, 1.15 factor, 262144 worst-case vocab fallback, refusal when free < estimate + 2 GiB margin, unknown space passes); (5) additive VocabSize field in internal/models GGUF parse (tokenizer.ggml.tokens array length, header-only read) + Model field + backfill. Tests cover the full phase-02 test plan (fresh/cached/corrupted downloads, round trips, list/delete, partial discipline, disk-guard refusal with faked free space, estimate math). Decisions made unattended: hash mismatch is NOT 
[… note truncated for the prompt tail]
