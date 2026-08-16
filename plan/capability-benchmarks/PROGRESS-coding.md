# Coding progress

## Phase 01 — Evaluation engine and binary install: build turn finished

Phase 01 (Evaluation engine and binary install) is implemented in full and the verification command `go build ./... && go vet ./internal/... && go test ./...` passes (run fresh, non-cached).

What was built:
- `internal/evaluate/evaluate.go` (new): `Mode` registry (perplexity, kl-divergence, hellaswag, winogrande), `EvalContextSize = 512`, `Spec` (engine-shaped: binary/model/dataset paths, Tasks/Chunks caps, KLBasePath, pre-mapped Flags), `MapConfigFlags(SnapshotSubset)` as the single allow-list (`--n-gpu-layers`, `--threads`, `--batch-size`/`--ubatch-size` when >0, `--flash-attn on|off` always pinned, `--cache-type-k/-v`, `--direct-io`, then caller `PlacementFlags`) with the complete exclusion list documented by name (context size, gpu_assign/tensor_split, speculative decoding, parallel/sampling/mmproj/context-shift). Command assembly per mode documented against tools/perplexity/perplexity.cpp: every invocation carries `--ctx-size 512`; KL comparison omits `--chunks` (inert — chunk count comes from the base file); `GenerateKLBase` runs the generation form (no `--kl-divergence` flag) with the same execution model. `Run(ctx, Spec) (Result, error)` uses exec.CommandContext, prepends the binary dir to LD_LIBRARY_PATH (mirroring jobs_env.go/process.Manager), captures stdout+stderr into ONE mutex-guarded 64KB tail buffer (LOG vs LOG_INF routing), wraps context cancellation (errors.Is-able) and returns the output tail on nonzero exit.
- `internal/evaluate/parse.go` (new): one parse
[… note truncated for the prompt tail]
