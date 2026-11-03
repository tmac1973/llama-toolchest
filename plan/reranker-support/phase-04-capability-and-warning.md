# Phase 04 — Rerank capability and missing-pooling warning

**Depends on:** 01, 02, 03 · **Enables:** 05 (docs describe the
capability), clients that pick models by capability

## Goal
Report rerankers as rerankers through the APIs: a `"rerank"` capability in
`Model.Capabilities()` (which feeds `/v1/models`) and a `"rerank"` field in
the capabilities API. Flag a model that is a reranker only by name, when its
parsed file has no rank pooling key. Those files load but may fail or score
badly. The warning shows on the config page and as a badge on the model's
row in "Reranking Models".

## Files touched
- `internal/models/registry.go` — `Capabilities()` returns `"rerank"` for
  `KindRerank`. Add `Model.RerankPoolingMissing()`.
- `internal/api/capabilities.go` — add `"rerank": m.Kind() == KindRerank`.
- `internal/api/service.go` — `RerankPoolingWarning string` in the config
  view data, filled from `RerankPoolingMissing()`.
- `web/templates/partials/model_config.html` — render
  `RerankPoolingWarning` in the same markup and place as `GPUAssignWarning`.
- `internal/api/models.go` — a `RerankPoolingWarning string` field on
  `modelCardView`, set where `HasVision`/`Speaks` are set (~line 470).
- `web/templates/partials/model_card.html` — the `check file` badge.
- `internal/api/capabilities_test.go`, `internal/api/service_handlers_test.go`,
  `internal/api/models_handlers_test.go`, `internal/models/registry_test.go`.

## Steps
1. In `Capabilities()`, add a `KindRerank` case that appends `"rerank"` and
   not `"chat"`. Update the doc comment's list of kinds.
2. Add:
   ```go
   // RerankPoolingMissing reports whether m is a reranker only by its name:
   // its file was read by a parser that records the pooling type, and has
   // no rank pooling. llama-server still starts it with --reranking, but
   // older conversions without the classifier head fail or score badly.
   func (m *Model) RerankPoolingMissing() bool {
       return m.Kind() == KindRerank && m.GGUFMetaVersion >= ggufMetaVersionPooling && m.PoolingType != PoolingRank
   }
   ```
   Version 8 is the parser generation that started recording
   `PoolingType`. Declare it as
   `const ggufMetaVersionPooling = 8` next to `GGUFMetaVersion` in
   `registry.go`, and use the constant in place of the bare 8.
3. In `service.go`, declare the text once:
   `const rerankPoolingWarning = "This file has no rank pooling key, so it may not rerank correctly. Use a conversion made with a recent llama.cpp."`
   Set the config view's `RerankPoolingWarning` to it when
   `RerankPoolingMissing()`. The config page and the list badge both read
   this constant, so they always say the same thing. Phase 06 adds a pointer to the Download
   Models card once that card exists.
4. Render the warning in `model_config.html` beside `GPUAssignWarning`, with
   its markup: `<small><mark>&#9888; {{.RerankPoolingWarning}}</mark></small>`.
5. In the `modelCardView` literal, set `RerankPoolingWarning` to
   `rerankPoolingWarning` when `m.RerankPoolingMissing()`, and leave it empty
   otherwise. In `model_card.html`, add the badge after the `tools` mark, in
   the existing mark style:
   `{{if .RerankPoolingWarning}} <mark style="padding:0 0.3rem;font-size:0.75rem;" title="{{.RerankPoolingWarning}}">check file</mark>{{end}}`.

## Build gate
```bash
go build ./... && go vet ./... && go test ./... && make js-test
```

## Test plan
- `Capabilities()` for a reranker returns `["rerank"]`. For embedding and
  chat models the output is unchanged.
- `buildCapabilities` reports `rerank: true, embedding: false, chat: false`
  for a reranker. Test it directly, and through the loaded-models listing
  that embeds it (`internal/api/service.go:284`).
- `/v1/models` lists the reranker with the `rerank` capability.
- `RerankPoolingMissing`: true for a name-matched `qwen3` record at
  `GGUFMetaVersion: 9, PoolingType: 0`. False with `PoolingType: PoolingRank`.
  False at `GGUFMetaVersion: 7` (not yet known). False for embedding and
  chat models.
- Config page render test: the warning appears for the missing-pooling
  record and not for a good one. List render test: the badge appears only
  on that row.

## Commit
`feat(models): report the rerank capability and warn on files without rank pooling`

## Rollback
Revert the commit. It is display and API output only. Clients that started
to filter on `"rerank"` would no longer see it, but nothing is saved.
