# Phase 06 — Curated "Get reranking models" card

**Depends on:** 02, 03, 04, 05 · **Enables:** one-click setup of a known-good
reranker

## Goal
Offer a curated list of reranker GGUFs on the Download Models tab, the same
way the embedding card does, and link to it from the "Reranking Models"
section. Every entry is checked on HuggingFace before it is added: the file
name, size, context, and that the header has `{arch}.pooling_type = 4`. The
card offers only conversions that serve correctly without the
missing-pooling warning.

## Files touched
- `internal/models/rerankers.go` (new) — `RerankModelPreset` and
  `CuratedRerankModels()`.
- `internal/models/rerankers_test.go` (new) — list sanity tests.
- `internal/api/models.go` — `handleRerankPresets` and
  `handleDownloadRerankPreset`.
- `internal/api/server.go` — routes `/api/models/rerank-presets` (GET) and
  `/api/models/rerank-presets/download` (POST), next to the embedding preset
  routes.
- `web/templates/partials/rerank_presets.html` (new) — the card, modelled
  on `partials/embedding_presets.html`.
- `web/templates/models_browse.html` — `<div id="rerank-presets"></div>`,
  the `:not(:empty)` margin rule, the `rerank` entry in the `?open=` map
  (line ~105), and the comment at line ~35 listing `?open=rerank`.
- `web/templates/models.html` — a "Get reranking models" button in the
  Reranking Models header, linking to `/models/browse?open=rerank`.
- `internal/api/rerank_presets_test.go` (new) — card and download tests.
- `internal/api/service.go` — the `rerankPoolingWarning` constant from
  phase 04 gets the pointer to the new card.

## Steps
1. Before writing code, check each candidate on HuggingFace. For each
   repo, choose the file by this rule: the Q8_0 GGUF, else the largest
   quantisation below Q8_0, else the only GGUF in the repo. Download it,
   and read its header with llama.cpp's `gguf-py` dump script
   (`python -I gguf-py/gguf/scripts/gguf_dump.py --no-tensors <file>`) for
   the architecture, `pooling_type`, context length and size. Check the
   repos in the order listed. The first repo that passes is the one used,
   and later repos are not checked. Candidates:
   - bge-reranker-v2-m3: `ggml-org/bge-reranker-v2-m3-GGUF`, then
     `gpustack/bge-reranker-v2-m3-GGUF`.
   - Qwen3-Reranker-0.6B and Qwen3-Reranker-4B: `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`
     and `ggml-org/Qwen3-Reranker-4B-Q8_0-GGUF`, then the official
     `Qwen/Qwen3-Reranker-0.6B-GGUF` and `Qwen/Qwen3-Reranker-4B-GGUF`.
   - jina-reranker-v2-base-multilingual: `gpustack/jina-reranker-v2-base-multilingual-GGUF`.
   An entry goes in only if its file has `pooling_type = 4` and it serves a
   correct ranking with `scripts/test-rerank.sh`. If no repo for a
   candidate passes, leave it out and record why in the doc comment above
   `CuratedRerankModels()`, as the embedding list records its omissions
   (issue #421 style).
2. In `rerankers.go`, define `RerankModelPreset` with the embedding preset's
   fields except `Dims` (`Name`, `Repo`, `Filename`, `SizeMB`, `Context`,
   `License`, `Desc`) and a `SizeLabel()`. Factor `SizeLabel` into a shared
   `sizeLabelMB(int) string` that both preset types call. Fill
   `CuratedRerankModels()` smallest first within each family, with the
   licence shown, including jina's CC-BY-NC-4.0, so non-commercial use is
   visible on the card.
3. Implement `handleRerankPresets` and `handleDownloadRerankPreset` by
   factoring the embedding pair's body into helpers that take the preset
   list. The download path then goes through the same HF download as
   embedding presets, which already calls `placeEmbeddingModel` after
   download (`internal/api/hf.go:609`). After phase 03 that places
   rerankers too.
4. Write the card template, the browse-tab wiring and the Models page
   button.
5. In `service.go`, append ` The Download Models tab has known-good
   reranking models.` to the `rerankPoolingWarning` constant. The config
   page and the list badge both pick it up. Update their render tests.

## Build gate
```bash
go build ./... && go vet ./... && go test ./... && make js-test
```

## Test plan
- `rerankers_test.go`: every entry has a repo, a `.gguf` filename, a
  non-zero size and context, and a licence. Names are unique.
- `rerank_presets_test.go`: the card lists every preset with its context,
  licence and size, and marks one already registered as downloaded (mirror
  `TestEmbeddingPresetsCard`). A download request for an unknown preset is
  rejected, and a known one starts a download.
- `/models/browse?open=rerank` opens the card.
- Manual: download each preset from the card on a real machine. It lands in
  Reranking Models on one GPU with no warning badge, and
  `scripts/test-rerank.sh` passes against it.

## Commit
`feat(models): curated reranking model downloads`

## Rollback
Revert the commit. Downloaded rerankers stay registered and keep serving,
since serving doesn't depend on this phase. Only the card, the routes and
the button go away.
