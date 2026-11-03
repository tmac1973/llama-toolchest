# Phase 02 — Classify rerankers and list them

**Depends on:** 01 · **Enables:** 03 (launching by kind), 04 (capability
and warning), 06 (card links into the new section)

## Goal
Add `KindRerank`. A model is a reranker when its file has rank pooling or
its name says `rerank` or `cross-encoder`, and that check runs before the embedding checks, so
bge-reranker no longer falls into the embedding name pattern. Rerankers
leave the chat and embedding lists and get their own "Reranking models"
section on the Models page, fed by a new list endpoint. After this phase
rerankers are classified and visible but not yet launched correctly (phase
03).

## Files touched
- `internal/models/registry.go` — `KindRerank`, `rerankPattern`,
  `IsRerankModel(name)`, the rerank branch in `Kind()` and `KindFromName()`.
- `internal/models/registry_test.go` — reranker cases in `TestModelKind`
  (from phase 01).
- `internal/api/models.go` — `handleListRerankModels`.
- `internal/api/server.go` — route `r.Get("/rerankers", s.handleListRerankModels)`
  under `/api/models`, before `/{id}`.
- `web/templates/models.html` — a "Reranking Models" section after
  "Embedding Models", with `id="rerank-list"`. Add `'rerank-list'` to
  `reRenderedLists`.
- `internal/api/models_handlers_test.go` — list tests.

## Steps
1. Add `KindRerank` to the `ModelKind` constants.
2. Add
   `var rerankPattern = regexp.MustCompile(`(?i)(rerank|cross-encoder)`)`
   and `func IsRerankModel(name string) bool`, documented like
   `IsEmbeddingModel`: the only test for a download with no file yet, also
   used for records parsed before `PoolingType` was recorded.
3. In `Model.Kind()`, return `KindRerank` first when
   `m.PoolingType == PoolingRank || IsRerankModel(m.ModelID) || IsRerankModel(m.ID)`,
   then fall through to the embedding check and then chat. In
   `KindFromName`, check `IsRerankModel` before `IsEmbeddingModel`. The doc
   comment on `Kind()` explains the order: bge rerankers match the
   embedding name pattern, and a reranker on a chat architecture (Qwen3)
   matches nothing else.
4. In `internal/api/models.go`, add `handleListRerankModels`, a copy of
   `handleListEmbeddingModels` that uses `KindRerank`, with the same
   empty-list HTMX behaviour (no heading when there is nothing to show) and
   the same JSON shape. Pull the two handlers' shared body into one helper,
   `listModelsOfKind(w, r, kind)`, so the two handlers stay one-liners.
5. Register the route in `server.go` next to `/embeddings`.
6. In `models.html`, add the section header `<h4>Reranking Models</h4>` and
   `<div id="rerank-list" hx-get="/api/models/rerankers" hx-trigger="load, modelsChanged from:body" hx-swap="innerHTML">`.
   Mirror the embedding block's markup. Leave out the "Get reranking
   models" button; phase 06 adds it with the card.
7. Add `'rerank-list'` to `reRenderedLists` (line ~167).

## Build gate
```bash
go build ./... && go vet ./... && go test ./... && make js-test
```

## Test plan
- `TestModelKind` gets these cases: `PoolingType: PoolingRank` on a
  `qwen3` record gives rerank; `BAAI/bge-reranker-v2-m3` by name only gives
  rerank (not embedding); `Qwen/Qwen3-Reranker-0.6B` by name only gives
  rerank; an embedding model is unchanged; `KindFromName("BAAI/bge-reranker-v2-m3")`
  gives rerank.
- Handler tests: with one chat, one embedding and one reranker registered,
  `/api/models` lists only the chat model, `/api/models/embeddings` only the
  embedding model, and `/api/models/rerankers` only the reranker (JSON and
  HTMX). A pending config named like a reranker ghosts only in the reranker
  list.
- `servedChatModels` leaves the reranker out, and so do generated client
  configs, which read it (`internal/api/agentconfigs.go:73`). `/v1/models`
  still lists it, since `advertisedToClients` doesn't look at the kind. It
  has an empty capability list until phase 04. The draft-model candidate
  list skips it.
- Manual: the Models page shows the new section only when a reranker is
  registered.

## Commit
`feat(models): recognise reranking models and list them separately`

## Rollback
Revert the commit. Kinds are computed when read and never saved, so
reverting puts bge rerankers back under embeddings and Qwen3 rerankers back
under chat with no data to clean up.
