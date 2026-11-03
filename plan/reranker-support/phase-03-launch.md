# Phase 03 — Launch rerankers with `--reranking`

**Depends on:** 01, 02 · **Enables:** 05 (the endpoint actually answers),
06 (curated downloads serve correctly)

## Goal
Make llama-server serve rerankers: `reranking = true` in their router preset
section and `--reranking` in their effective flags. They get the embedding
launch treatment (context cap, micro-batch raised to the context, one-GPU
placement, VRAM estimated at the launch config) and none of the chat-only
options. After this phase, `/v1/rerank` returns scores on a real build.

## Files touched
- `internal/models/registry.go` — `EffectiveFlagsFor` adds `--reranking`
  for `KindRerank`.
- `internal/models/preset.go` — `writeConfigParams` writes
  `reranking = true` for `KindRerank`.
- `internal/models/embeddings.go` — doc comments on `EmbeddingLaunchConfig`
  and `EmbeddingGPU` say they also cover rerankers, and why: rank pooling
  reads each query and document pair in one micro-batch.
- `internal/api/embedding_gpu.go` — doc comments and log messages say
  "embedding or reranking model". The `!= KindChat` check from phase 01
  already includes rerankers.
- `internal/api/service.go` — `MaxContext` comment covers rerankers (the
  code already uses `kind != KindChat` from phase 01).
- `web/templates/partials/model_config.html` — the context tooltip (line
  94) gets a reranker variant ("The longest query plus document of one
  rerank pair…"), and the GPU tooltip (line 131) says "Embedding and
  reranking models use one GPU when they fit." Both go on `.IsEncoder`.
- `web/templates/help.html` — the "Context Size" and "Embedding models and
  GPUs" entries mention reranking models.
- `internal/models/preset_test.go`, `internal/models/registry_test.go`
  (flags), `internal/models/vram_test.go`, `internal/api/service_handlers_test.go`.

## Steps
1. In `EffectiveFlagsFor`, replace the `--embeddings` branch with a switch:
   `KindEmbedding` → `--embeddings`, `KindRerank` → `--reranking`. Don't
   also pass `--embeddings`, since `--reranking` sets it.
2. In `writeConfigParams`, do the same: `KindEmbedding` →
   `embeddings = true`, `KindRerank` → `reranking = true`. Chat-only
   options remain gated on `kind == KindChat` (from phase 01). Add a comment
   that the router takes the long option name, as with `embeddings`, and
   that `--reranking` is old enough to need no version gate.
3. Confirm, without changing code, that `EmbeddingLaunchConfig`'s
   `oneBatch := m.PoolingType != PoolingLast` is true for rerankers. llama-server
   can split a pooled input over micro-batches only for last-token pooling,
   so a rank-pooled pair must fit in one. Add a test that asserts it. The
   default context cap (`EmbeddingDefaultContext = 8192`) also applies, and
   the trained-context cap keeps bge-reranker-v2-m3 at 8192.
4. Update the doc comments and template tooltips listed above.
5. Check `placeEmbeddingModels` at startup: it runs over every registered
   model, so rerankers downloaded before this change get one-GPU placement
   on the next start without a migration.

## Build gate
```bash
go build ./... && go vet ./... && go test ./...
```

## Test plan
- Preset test: a reranker section contains `reranking = true`, no
  `embeddings = true`, no `flash-attn`, no `lookup-cache-dynamic`, and its
  `ctx-size`/`ubatch-size` match `EmbeddingLaunchConfig`. The embedding
  and chat sections are byte-identical to before.
- Flags test: `EffectiveFlagsFor(KindRerank, …)` contains `--reranking`
  and not `--embeddings`.
- VRAM test: a reranker's breakdown is computed at the launch config
  (raised micro-batch), the same as an embedding model's.
- Placement test: on a two-GPU fake hardware, `placeEmbeddingModel` puts a
  reranker on one GPU.
- Manual, on the current llama.cpp build: register
  `bge-reranker-v2-m3` and `Qwen3-Reranker-0.6B` from a known-good GGUF,
  restart the router, and run
  ```bash
  curl -s localhost:8080/v1/rerank -H 'Content-Type: application/json' -d '{"model":"<public name>","query":"What is a panda?","documents":["hi","The giant panda is a bear native to China."]}'
  ```
  Both return `results` with the panda document ranked first.

## Commit
`feat(models): launch reranking models with --reranking`

## Rollback
Revert the commit. Rerankers stay classified and listed (phase 02) but go
back to failing at `/rerank`, which is the state before this phase. The
router preset is rewritten from config on the next write, so no stale
`reranking = true` lines are left behind.
