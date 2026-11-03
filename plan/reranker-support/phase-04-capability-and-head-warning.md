# Phase 04 — Rerank capability and missing-head warning

**Depends on:** 01, 02, 03 · **Enables:** 05 (docs describe the
capability and the warning), 06 (the card's file check uses the same
field), clients that pick models by capability

## Goal
Report rerankers as rerankers through the APIs: a `"rerank"` capability in
`Model.Capabilities()` (which feeds `/v1/models`) and a `"rerank"` field in
`buildCapabilities`. Warn about a reranker whose file has no classifier
head: no `cls.weight` and no `cls.output.weight` tensor. With `--reranking`,
llama.cpp's rank pooling applies only the head tensors that are present
(`src/llama-graph.cpp`, `LLAMA_POOLING_TYPE_RANK`). With neither present,
it returns the raw hidden state as the score: no error, just meaningless
numbers. The warning shows on the config page and as a badge on the model's
row in "Reranking Models".

The rank pooling key is not the test. The phase 03 real-build check found
that `gpustack/bge-reranker-v2-m3-GGUF` has no `pooling_type` key but does
have `cls.weight`, `cls.bias` and `cls.output.*`, and it ranks correctly.
`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` has `pooling_type = 4` and
`cls.output.weight`. What the two have in common is the head.

## Files touched
- `internal/models/gguf.go` — the tensor scan records the head:
  `classifierTensorNames`, `HasClassifier` on the scan result,
  `HasClassifierHead` on `GGUFMeta`, sharded files included. `ApplyTo` copies
  it to the model record and sets `ClassifierChecked`.
- `internal/models/registry.go` — `HasClassifierHead` and
  `ClassifierChecked` fields on `Model`. `GGUFMetaVersion` goes to 10, with
  a history line. `Capabilities()` returns `"rerank"` for `KindRerank`. Add
  `Model.RerankHeadMissing()`.
- `internal/api/capabilities.go` — add `"rerank": m.Kind() == models.KindRerank`.
- `internal/api/service.go` — the `rerankHeadWarning` constant, and a
  `RerankHeadWarning string` field in `modelConfigPanelData`.
- `web/templates/partials/model_config.html` — render `RerankHeadWarning`
  under the panel header.
- `internal/api/models.go` — a `RerankHeadWarning string` field on
  `modelCardView`, set where `HasVision`/`Speaks` are set (~line 470).
- `web/templates/partials/model_card.html` — the `check file` badge.
- Tests: `internal/models/gguf_classifier_test.go` (new),
  `internal/models/registry_test.go`, `internal/api/capabilities_test.go`,
  `internal/api/service_handlers_test.go`, `internal/api/models_handlers_test.go`.

## Steps
1. In `gguf.go`, next to `outputTensorName` and `markovHeadTensorName`, add
   ```go
   // classifierTensorNames are the classification head tensors llama.cpp's
   // rank pooling applies (cls is the hidden projection, cls.output the
   // projection to the score). A reranker without either scores with its
   // raw hidden state, which is meaningless.
   var classifierTensorNames = map[string]bool{"cls.weight": true, "cls.output.weight": true}
   ```
   In the tensor scan loop (~line 1289), set `scan.HasClassifier = true`
   for a name in that set. Add `HasClassifier bool` to the scan result
   struct (~line 1174), with a comment in the style of its neighbours. Copy
   it in the full parse (~line 712: `meta.HasClassifierHead = scan.HasClassifier`).
   OR it across shards where `HasOutput` is ORed (~lines 295 and 1437).
2. On `GGUFMeta`, add `HasClassifierHead bool` with a comment, next to
   `HasOutputTensor`.
3. On `Model` in `registry.go`, add
   ```go
   // HasClassifierHead records a classification head (cls.weight or
   // cls.output.weight) in the file's tensor table, and ClassifierChecked
   // that the table was read, so a record parsed before version 10, or
   // from the header alone, is not taken for a file without one.
   HasClassifierHead bool `json:"has_classifier_head,omitempty"`
   ClassifierChecked bool `json:"classifier_checked,omitempty"`
   ```
   In `ApplyTo`, set `m.HasClassifierHead = meta.HasClassifierHead` and
   `m.ClassifierChecked = !meta.MetaOnly`.
4. Raise `GGUFMetaVersion` to 10 and add the history line:
   `10 — whether the file has a classification head (HasClassifierHead), so a reranker that would return meaningless scores is flagged.`
   The startup backfill then re-reads every record once, including
   rerankers downloaded during phases 02–03.
5. In `Capabilities()`, add a `KindRerank` case that appends `"rerank"`.
   Update the doc comment's list.
6. Add:
   ```go
   // RerankHeadMissing reports whether m is a reranker whose file has no
   // classification head. llama-server still serves it with --reranking,
   // but scores it with the raw hidden state, so its scores mean nothing.
   func (m *Model) RerankHeadMissing() bool {
       return m.Kind() == KindRerank && m.ClassifierChecked && !m.HasClassifierHead
   }
   ```
7. In `service.go`, declare the text once:
   `const rerankHeadWarning = "This file has no classification head, so its rerank scores are meaningless. Use a reranker conversion made with a recent llama.cpp."`
   Set `modelConfigPanelData.RerankHeadWarning` to it when
   `model.RerankHeadMissing()`. The config page and the list badge both read
   this constant, so they always say the same thing. Phase 06 appends a
   pointer to the Download Models card.
8. Render the warning in `model_config.html` under the panel header, in the
   `ReadOnlyReason` alert's markup:
   `<p role="alert" style="margin:0.5rem 0;"><mark>&#9888; {{.RerankHeadWarning}}</mark></p>`.
   (Beside `GPUAssignWarning` was the first choice, but that block renders
   only when the machine has GPUs, so a CPU-only machine would never show
   it.)
10. A reranker is never offered for tools. Qwen3-Reranker keeps Qwen3's
   chat template, which handles tools, so `SupportsTools` is set. Add
   `Model.CallsTools()` (`SupportsTools && Kind() != KindRerank`) and use it
   for the `tools` capability (`Capabilities()`, `buildCapabilities`) and
   the card's `tools` badge (a `CallsTools` field on `modelCardView`).
   `Capabilities()` returns only `["rerank"]` for a reranker.
9. In the `modelCardView` literal, set `RerankHeadWarning` to
   `rerankHeadWarning` when `m.RerankHeadMissing()`, and leave it empty
   otherwise. In `model_card.html`, add the badge after the `tools` mark, in
   the existing mark style:
   `{{if .RerankHeadWarning}} <mark style="padding:0 0.3rem;font-size:0.75rem;" title="{{.RerankHeadWarning}}">check file</mark>{{end}}`.

## Build gate
```bash
go build ./... && go vet ./... && go test ./... && make js-test
```

## Test plan
- `gguf_classifier_test.go`: build small GGUF files with named tensors
  using `writeHeadGGUF(t, arch, keys, tensorNames)` from
  `internal/models/gguf_draft_head_test.go`, which the draft-head tests use
  for the same kind of tensor-name check. One has `cls.weight`, one has only
  `cls.output.weight`, and one has neither. `ParseGGUFMeta` reports
  `HasClassifierHead` true, true and false. `ApplyTo` sets
  `ClassifierChecked`. A metadata-only parse (`ParseGGUFMetaOnly`) leaves
  `ClassifierChecked` false.
- `RerankHeadMissing`:
  - True for a reranker with `ClassifierChecked: true, HasClassifierHead: false`.
  - False with the head.
  - False when not checked (an old record).
  - False for embedding and chat models without a head.
- `Capabilities()` for a reranker returns `["rerank"]`. For embedding and
  chat models the output is unchanged.
- `buildCapabilities` reports `rerank: true, embedding: false, chat: false`
  for a reranker. Test it directly.
- `/v1/models` lists the reranker with the `rerank` capability. Extend
  `TestMdlListRerankModels`.
- Config page render test: the warning appears for a headless reranker and
  not for one with a head. List render test: the badge appears only on that
  row.
- Manual: with the real files from phase 03 (in the session scratchpad,
  `rr-models/`), `ParseGGUFMeta` gives `HasClassifierHead` true for both
  gpustack bge and ggml-org Qwen3.

## Commit
`feat(models): report the rerank capability and warn on rerankers without a classification head`

## Rollback
Revert the commit. The version bump has already caused one re-read at
startup, and the extra JSON fields are ignored by the older code
(`omitempty`, unknown fields dropped on load). Clients that started to
filter on `"rerank"` would no longer see it.
