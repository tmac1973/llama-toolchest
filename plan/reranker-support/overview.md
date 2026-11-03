# Reranker Support — Project Overview

## Problem
llama-toolchest cannot serve a reranking model today. It parses the GGUF
pooling type, and `GGUFMeta.IsEmbedding()` already excludes rank pooling
(`PoolingRank = 4`, `internal/models/embeddings.go`), but nothing past that
knows what a reranker is. A reranker is misclassified one of two ways:

- **BAAI/bge-reranker-v2-m3** matches the embedding name pattern
  (`[-/]bge[-/]`, `internal/models/registry.go`), and `Model.IsEmbedding()`
  ORs the name check in. It is listed as an embedding model and launched
  with `embeddings = true`.
- **Qwen3-Reranker** is a `qwen3` architecture that no name pattern matches.
  It is treated as a chat model: listed to chat clients, given chat flags,
  sampling presets and reasoning levels, and sized for VRAM like a chat model.

Neither gets `--reranking`, which sets both `params.embedding = true` and
`params.pooling_type = LLAMA_POOLING_TYPE_RANK`. llama-server answers
`/rerank` only when both are set, so every request fails with "This server
does not support reranking. Start it with `--reranking`". People running a
RAG pipeline against toolchest cannot use it for the reranking step.

## Goals
- Recognise a reranking model from its file (rank pooling) or, before the
  file exists, from its name (`rerank` or `cross-encoder`), and keep it apart from chat and
  embedding models.
- Launch rerankers with `--reranking` (and `reranking = true` in the router
  preset), with the context, micro-batch and one-GPU placement that embedding
  models already get.
- Serve `/v1/rerank` and the root `/rerank` through the proxy, with usage
  counted.
- Keep rerankers out of everything that is only for chat models: chat model
  lists, sampling presets, reasoning levels, chat-only config fields.
- Report a `"rerank"` capability for them in `/v1/models` and the model API.
- Give rerankers their own "Reranking models" section on the Models page.
- Warn on a model named like a reranker whose file has no rank pooling key.
- Offer a curated "Get reranking models" download card with known-good
  conversions.
- Document `/v1/rerank` in the API docs and help page.

## Non-goals
- Toolchest using a reranker internally (as a helper model or for its own
  features).
- Wiring rerankers into generated client configs (`internal/agentconfig`).
  Embedding models aren't wired there either.
- An in-UI form for testing a reranker.
- Benchmarks or evals built for rerankers. The benchmark forms list every
  enabled model, embedding models included, and rerankers are listed the
  same way. Nothing reranker-specific is added.
- Changing the request/response format. llama-server's Jina- and
  TEI-compatible formats pass through unchanged.
- Rerankers in generation backends or fork instances.

## Users & primary flow
A self-hoster running a RAG pipeline (Open WebUI, LangChain, a custom script)
against toolchest:

1. Opens the Models page and downloads a reranker, either from the curated
   "Get reranking models" card or from a HuggingFace search.
2. The model appears under "Reranking models", placed on one GPU, with
   chat-only fields hidden from its config form.
3. Toolchest writes it to the router preset with `reranking = true`.
4. The RAG client points its reranker setting at
   `http://<toolchest>/v1/rerank` with `"model": "<public name>"`.
5. The proxy routes the request to the reranker, llama-server returns
   scores, and toolchest records the request's usage.

## Constraints
- Go backend, HTMX templates under `web/templates`, and the existing
  `chi` routes in `internal/api/server.go`.
- llama-server router mode: one preset section per model in `preset.ini`,
  written by `internal/models/preset.go`. The router reads it only at
  startup.
- `--reranking` is present in every llama.cpp build toolchest supports, so
  it needs no `version*` gate in `internal/models/launch.go`.
- Model records saved before this change have to classify correctly without
  a rescan. The name fallback covers those.
- Embedding model behaviour must not change: the same flags, presets,
  placement and lists.
- Commits follow Conventional Commits with no AI attribution, on a feature
  branch, never `main`.

## Success criteria
- bge-reranker-v2-m3 and Qwen3-Reranker-0.6B (known-good GGUFs) each appear
  only under "Reranking models", never in chat or embedding lists, and
  `/v1/models` reports `"rerank"` for them.
- Their preset sections contain `reranking = true` and no `embeddings = true`
  or chat-only options. Their effective flags show `--reranking`.
- `curl /v1/rerank` and `curl /rerank` with a query and documents return
  ranked scores from both models on a real build.
- Embedding models produce the same preset, flags and lists as before
  (existing tests pass unchanged).
- A file named like a reranker without rank pooling is served as a reranker
  and shows the warning.
- The curated card lists the presets and downloads them, and they serve
  correctly.
- `go build ./...`, `go vet ./...`, `go test ./...` and `make js-test` pass.

## Decisions
- **Main use case** → A RAG client calls `/v1/rerank` on toolchest. Toolchest
  doesn't use rerankers itself.
- **Models page placement** → Rerankers get their own "Reranking models"
  section with its own list endpoint, separate from embeddings.
- **Curated presets** → In scope, as a late phase after serving works.
- **Client configs (agentconfig)** → Non-goal. The docs show how to point a
  client at `/v1/rerank`.
- **File named like a reranker but no rank pooling key** → Serve it as a
  reranker and show a warning on the model.
- **Code shape** → A `Model.Kind()` enum (chat / embedding / rerank), brought
  in by a refactor-only phase before any reranker behaviour.
- **In-UI test form** → No. API docs and the help page show the endpoint.
- **Warning placement** → On the model's config page, plus a `check file`
  badge on its row.
- **Curated card models** → bge-reranker-v2-m3, Qwen3-Reranker-0.6B,
  Qwen3-Reranker-4B and jina-reranker-v2-base-multilingual, each verified
  before it goes in.
- **Landing** → One feature branch and PR, one commit per phase. Each commit
  builds and passes tests.
