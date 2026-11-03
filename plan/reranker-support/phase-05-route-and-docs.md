# Phase 05 — Root `/rerank` route and docs

**Depends on:** 03, 04 · **Enables:** RAG clients that call the root path,
and users who find the endpoint from the docs

## Goal
Forward llama-server's root `/rerank` path through the proxy, as `/tts` is,
so clients configured with a TEI- or Jina-style base URL work. Document
`/v1/rerank` (and the root alias) in the OpenAPI spec, the API docs page and
the help page, and add a test script beside `test-embeddings.sh`. Usage
counting already matches `/rerank` (`internal/api/engine_notes.go:256`) and
needs no change.

## Files touched
- `internal/api/server.go` — `r.With(s.apiKeyAuth).Post("/rerank", llamaProxy.ServeHTTP)`
  next to `/tts`.
- `internal/api/proxy_rerank_test.go` (new) — a root-route test modelled on
  `internal/api/proxy_tts_test.go`.
- `web/apidocs/openapi.yaml` — a `/v1/rerank` path: request (`model`,
  `query`, `documents`, optional `top_n`) and response (`results[]` with
  `index` and `relevance_score`, `usage`). Also a root `/rerank` path that
  refers to the same request and response schemas, as `/tts` has its own
  root entry. `TestOpenAPISpecMatchesRouter` checks only the root paths
  `publicRoute` (`internal/api/apidocs_test.go`) names, so `/rerank` is
  added to that list, and the test then requires the entry. The Scalar page at
  `/docs/api` renders the spec as is, so `internal/api/apidocs.go` doesn't
  change.
- `web/templates/help.html` — a "Reranking models" entry: what a reranker
  is for, that it is served at `/v1/rerank`, a curl example, and the
  missing-head warning.
- `scripts/test-rerank.sh` — a new script modelled on
  `scripts/test-embeddings.sh`.
- `scripts/lib-test.sh` — `pick_model rerank` lists models with the
  `rerank` capability, and `pick_model chat` no longer offers them.
- `README.md` — add reranking to the list of served endpoints/model kinds.

## Steps
1. Register the root route in `server.go` with a comment in the `/tts`
   style: llama-server's rerank route at the root, as llama-server has it,
   proxied like `/v1` with the same API key.
2. Write both OpenAPI entries, matching the style of the existing
   `/v1/embeddings` block (line ~260) and the root `/tts` block (line ~289).
3. Add the help entry next to the embedding model entries (lines ~203–204).
4. Write `scripts/test-rerank.sh`: take a base URL and model name (the same
   argument style as `test-embeddings.sh`), POST a query with three
   documents to `/v1/rerank` and then `/rerank`, and check that each
   response's top result is the relevant document. It exits non-zero on
   failure.
5. Add the README line.

## Build gate
```bash
go build ./... && go vet ./... && go test ./... && bash -n scripts/test-rerank.sh
```

## Test plan
- Route test: a POST to `/rerank` reaches the proxy handler (assert it on a
  stubbed upstream) and is rejected without the API key when one is set.
- `TestOpenAPISpecMatchesRouter` passes with both paths, and `/docs/api`
  shows them.
- Manual: `scripts/test-rerank.sh http://localhost:8080 <bge public name>` and
  the same for Qwen3-Reranker-0.6B both pass on a real build. The Server
  tab's usage counts the requests.

## Commit
`feat(proxy): forward the root /rerank route and document reranking`

## Rollback
Revert the commit. `/v1/rerank` keeps working through the catch-all. Only
the root alias, docs and script go away.
