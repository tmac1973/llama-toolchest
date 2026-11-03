# Phase 01 — Model kind refactor

**Depends on:** nothing · **Enables:** phase 02 adds a third kind without
having to work through every embedding check again

## Goal
Replace the `IsEmbedding()` checks with a model kind: `ModelKind`, with
`KindChat` and `KindEmbedding` for now. Each call site moves to the check it
actually means: "is an embedding model" (`== KindEmbedding`) or "is not a
chat model" (`!= KindChat`). There are only two kinds in this phase, so the
two checks are equivalent and behaviour is unchanged. This is a refactor
only. All existing tests pass without edits, apart from signature updates.

## Files touched
- `internal/models/registry.go` — add `ModelKind`, its constants,
  `Model.Kind()` and `KindFromName()`. `IsEmbedding()` becomes
  `Kind() == KindEmbedding`. `Capabilities()` switches on the kind. The
  draft-candidate skip (~line 1771) moves to `!= KindChat`.
  `EffectiveFlagsFor` takes a `ModelKind`.
- `internal/models/preset.go` — `isEmbed` becomes `kind := m.Kind()`.
  `EmbeddingLaunchConfig` and the lookup-cache skip use `kind != KindChat`.
  `writeConfigParams` takes `kind ModelKind`: `embeddings = true` when
  `kind == KindEmbedding`, and the block now under `if !isEmbedding` (flash
  attention onward) under `kind == KindChat`, with its contents unchanged.
- `internal/models/vram.go` — the `EmbeddingLaunchConfig` branch (~line 410)
  uses `m.Kind() != KindChat`.
- `internal/models/projector.go` — `CanChat` uses `m.Kind() == KindChat`.
- `internal/api/models.go` — `filterModels(all, kind ModelKind)` and
  `filterPending(all, kind ModelKind)` (the latter via `KindFromName`).
  `CanAutoconfig` uses `m.Kind() == KindChat`.
- `internal/api/service.go` — `isEmbedding` becomes `kind := model.Kind()`.
  The template data gets `IsEncoder` (`kind != KindChat`) next to
  `IsEmbedding` (`kind == KindEmbedding`). `effectiveFlags` takes the kind.
- `web/templates/partials/model_config.html` — the `{{if not .IsEmbedding}}`
  around the chat-only fields (line 186) becomes `{{if not .IsEncoder}}`.
  The two embedding-specific tooltips (lines 94, 131) stay on `.IsEmbedding`.
- `internal/api/server.go` — `servedChatModels` skips `m.Kind() != KindChat`.
- `internal/api/presets.go`, `internal/api/reasoning_levels.go` — skip on
  `m.Kind() != KindChat`.
- `internal/api/embedding_gpu.go` — `placeEmbeddingModel` returns early on
  `m.Kind() == KindChat`.
- `internal/api/capabilities.go` — `"embedding": m.Kind() == KindEmbedding`.
- Tests that call `EffectiveFlagsFor`, `writeConfigParams`, `filterModels`
  or `filterPending` — pass the kind in place of the bool.

## Steps
1. In `registry.go`, next to `IsEmbedding`, add:
   ```go
   // ModelKind is what a model is served for: chat (and the other
   // generative uses), embeddings, or reranking.
   type ModelKind int

   const (
       KindChat ModelKind = iota
       KindEmbedding
   )
   ```
   Add `func (m *Model) Kind() ModelKind`, returning `KindEmbedding` when
   `m.Embedding || IsEmbeddingModel(m.ModelID) || IsEmbeddingModel(m.ID)`,
   else `KindChat`. Add `func KindFromName(name string) ModelKind` for
   downloads that have no file yet. Change `IsEmbedding()` to
   `return m.Kind() == KindEmbedding`, keeping its doc comment and noting
   that it means embedding only.
2. Update each call site listed above. A site uses `== KindEmbedding` only
   when it is about embeddings specifically: the `--embeddings` flag, the
   `"embedding"` capability, the embedding list, embedding tooltips. Every
   other site uses `!= KindChat` or `== KindChat`.
3. Change the `EffectiveFlagsFor(isEmbedding bool, …)` and
   `writeConfigParams(…, isEmbedding bool, …)` signatures to take
   `kind ModelKind`, and update their callers and tests.
4. Change the `filterModels`/`filterPending` signatures to take a kind.
   `handleListModels` passes `KindChat` and `handleListEmbeddingModels`
   passes `KindEmbedding`.
5. Run `grep -rn 'IsEmbedding()' internal` and confirm that the only
   remaining uses are the method itself and sites that mean embedding
   specifically. Convert those to `Kind() == KindEmbedding` as well, so that
   later phases have a single pattern to follow.

## Build gate
```bash
go build ./... && go vet ./... && go test ./...
```

## Test plan
- The existing suites pass unchanged: `internal/models` (preset, flags,
  VRAM, embeddings) and `internal/api` (model lists, capabilities,
  embedding presets, service handlers).
- New table test `TestModelKind` in `internal/models/registry_test.go`: `Embedding: true` gives
  embedding; a name-matched record (`Qwen/Qwen3-Embedding-0.6B-GGUF`, no
  flag) gives embedding; a plain chat record gives chat; `KindFromName`
  agrees with `IsEmbeddingModel` on the same names.
- Preset golden check: for one chat and one embedding model, the preset
  text before and after is byte-identical. Add this assertion to the
  preset test if one doesn't exist already.

## Commit
`refactor(models): classify models by kind instead of an embedding flag`

## Rollback
Revert the single commit. It changes no saved data and no on-disk format,
so a partial revert is never needed.
