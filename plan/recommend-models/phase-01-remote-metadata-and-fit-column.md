# Phase 01 — Remote GGUF metadata and a better Fit column

**Depends on:** — · **Enables:** 03 (the engine plans every finalist from
this metadata), 05 (the same planner fills in the config)

## Goal
Read the model description (layers, heads, KV heads, sliding window,
experts, trained context) from a GGUF file on HuggingFace with one small
ranged read, without downloading the tokenizer or the tensor table. Use it
to replace the browse tab's rough Fit column with the real planner, so each
file says what context it holds on this machine.

This phase is useful by itself, and phase 03 builds directly on it.

## Background
- `rangeReader` (`internal/modelsource/remote.go`) is an `io.ReadSeeker`
  over HTTP Range requests. Its window starts at 64 KiB and grows ×4 up to
  4 MiB.
- `models.ParseGGUFMetaFrom` (`internal/models/gguf.go:246`) parses all the
  metadata, then walks on to the tensor table (`scanTensorBlock`,
  `gguf.go:933`). In an unsplit file the tensor table sits behind the
  tokenizer, about 12 MB away for a large model. That is why `ProbePLE`
  only runs on split files of 8 GiB or more (`ple_probe.go:20-47`).
- In practice the `{arch}.*` keys come before `tokenizer.*`, so the fields
  the fit needs are in the first few hundred KB. The GGUF format does not
  promise this order, so the reader must not depend on it.
- These fields come only from the tensor table today
  (`gguf.go:510-516`): `PLEBytes`, `TokenEmbdBytes`, `ExpertBytes`,
  `ExpertLayerFirst`, `ExpertLayers`, `HasBlockTensors`, `HasTrunkBlock0`.
  `PlanFit` needs `ExpertLayerFirst` and `ExpertLayers` for MoE expert
  offload. The VRAM estimate uses the byte sizes.
- The Fit column today is `vramFit` (`internal/api/server.go:461-471`): file
  size × 1.1 against `GPU[0]` total memory.

## Files touched
- `internal/models/gguf.go`: a metadata-only parse mode.
- `internal/models/gguf_derive.go` (new): fills in the tensor-table fields
  from the metadata and the parameter count.
- `internal/modelsource/remote_meta.go` (new): `ProbeMeta`, the remote
  metadata read.
- `internal/modelsource/meta_cache.go` (new): disk cache of probe results.
- `internal/api/hf.go`: the estimates handler probes one file per repo and
  plans every file.
- `internal/api/server.go`: remove `vramFit`; add a template function for
  the new fit cell.
- `web/templates/partials/hf_files.html`: the new Fit cell and its tooltip.
- Tests next to each file.

## Steps
1. **Metadata-only parse.**
   - Add `ParseGGUFMetaOnly(r io.ReadSeeker) (*GGUFMeta, error)`. It shares
     the key loop with `ParseGGUFMetaFrom`, but:
     - for `tokenizer.ggml.tokens` it reads the array's type and count
       (this gives `VocabSize`), then skips the entries without keeping
       them;
     - it skips `tokenizer.ggml.scores`, `token_type` and `merges` the same
       way;
     - it stops after the last metadata key, before the tensor table.
   - Skipping a string array still reads every length prefix, so the bytes
     are downloaded. To keep the read small, the parse also stops early once
     it reaches the first `tokenizer.*` key **and** every key the fit needs
     has been seen. If a needed key comes after the tokenizer, it reads on
     to the end of the metadata.
   - The fields the fit needs: `general.architecture`, `{arch}.block_count`,
     `embedding_length`, `attention.head_count`, `attention.head_count_kv`,
     `attention.key_length`/`value_length` (and `_swa`), `sliding_window`
     and its pattern, `full_attention_interval`, `context_length`,
     `expert_count`, `expert_used_count`, `expert_feed_forward_length`,
     `leading_dense_block_count`, `nextn_predict_layers`,
     `embedding_length_per_layer_input`. Missing optional keys are fine.
   - Mark the result `MetaOnly: true` so callers know the tensor fields are
     derived (step 2), not measured.
2. **Derive the tensor-table fields** (`DeriveSizes(meta, fileBytes,
   paramCount int64)`). `paramCount` is HuggingFace's `gguf.total` for the
   repo (phase 03), or 0 when not known.
   - **Expert layer span.** `ExpertLayerFirst = leading_dense_block_count`
     (0 when absent). `ExpertLayers = block_count − first − nextn layers`.
     Only when `expert_count > 0`.
   - **Expert bytes.** Expert parameters are
     `ExpertLayers × expert_count × 3 × n_embd × expert_feed_forward_length`.
     `ExpertBytes = fileBytes × expertParams / paramCount`. Without a
     parameter count, assume experts are 90% of the file. This only affects
     how much system memory an offload plan reports.
   - **Token embedding bytes.** `vocab × n_embd` parameters, scaled the
     same way. When unknown, leave it at 0. That makes the VRAM estimate a
     little high, which errs on the safe side.
   - **PLE bytes.** Only for architectures with
     `embedding_length_per_layer_input`:
     `vocab × block_count × per_layer_dim` parameters, scaled the same way.
     The existing split-file `ProbePLE` result replaces this when present.
3. **Check the derivation against real files.** Add a test helper and a
   one-off check (`go test -run TestDeriveMatchesFullParse -tags local`)
   that parses local GGUFs both ways and compares the VRAM estimate at 32K.
   - Run it on compute2's model folder. It should cover a dense model, a
     sliding-window model (Gemma), an MoE model (Qwen3 30B-A3B), an MoE
     model with leading dense layers (GLM or DeepSeek) and a PLE model.
   - Target: the estimate from derived fields is within 3% of the full
     parse, and never lower by more than 0.5 GiB. Record the results in
     this file's "As built" section.
4. **Remote probe.**
   `ProbeMeta(ctx, client, token, url string) (*models.GGUFMeta, error)`
   runs `ParseGGUFMetaOnly` over a `rangeReader` with a 4 MiB budget. For a
   split file it reads the first shard, which holds the metadata.
   - Measure the cost on 10 popular repos and record it. The expectation is
     one or two requests and under 512 KB.
5. **Disk cache.**
   - Store at `<DataDir>/cache/gguf-meta/<safe repo id>/<file oid>.json`.
     The oid is the LFS `oid` from the tree listing, which changes whenever
     the file changes. If the listing has no oid, use the file name plus
     its size.
   - Entries never expire. A new upload has a new oid.
   - Write with `internal/atomicfile`. A cache that cannot be read or
     written is ignored, never an error.
6. **One probe per repo.** All quants in a repo describe the same model, so
   one metadata read is enough.
   - Probe the smallest main-model file. Skip mmproj files and MTP heads.
   - Check `general.architecture` and `block_count` across the probed file
     and the listing names. If a repo mixes models (for example a base and
     a draft model in one repo), only files whose name shares the probed
     file's stem use its metadata. Other files keep the old estimate.
7. **Plan every file.** In `handleHFModelEstimates` (`hf.go:575`):
   - Build a `models.Model` per file from the shared metadata, the file's
     size and `DeriveSizes`. Use `meta.ApplyTo(m)` as `onDownloadComplete`
     does (`hf.go:471`).
   - Run `models.PlanFit(m, defaultConfig, s.hardware(), class)` for each
     context class.
   - Record per file: the largest class it holds fully on the GPUs, and
     whether it needs expert offload or partial layers at the 32K class.
   - `defaultConfig` is the same default `Registry.Add` uses
     (`internal/models/registry.go:513-521`). Move it to one exported
     function so both use it.
8. **The Fit cell.** Replace the `vramFit` cell with:
   - "Up to 128K", "Up to 32K", "Up to 8K": fully on the GPUs at that
     context.
   - "Experts in RAM": an MoE model that fits only with experts in system
     memory. Tooltip: "Some expert layers are kept in system memory. It
     runs, but generation is slower."
   - "Partly on CPU": a dense model with layers on the CPU. Tooltip: "Only
     part of the model fits on the GPUs. The rest runs on the CPU, which is
     much slower."
   - "Too large": nothing fits, even with offload.
   - The tooltip of every cell lists each class with its plan, e.g.
     "8K: all on GPU, f16 KV · 32K: all on GPU, f16 KV · 128K: all on
     GPU, 8-bit KV · 256K: does not fit". It ends with "Estimated from the
     file's description. Real use can differ by a few percent."
   - Files with no probe result (the probe failed or timed out) keep the
     old size-based label, with a tooltip that says so.
9. **VRAM Est. column.** Show the plan's estimate at 32K (or the model's
   maximum, if smaller) instead of `size × 1.1`. The column header tooltip
   says which context it is for.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- `ParseGGUFMetaOnly` on test fixtures:
  - same metadata fields as `ParseGGUFMetaFrom`;
  - stops before the tensor table, measured with a counting reader;
  - stops at the tokenizer when every needed key was already seen;
  - reads past the tokenizer when a needed key comes after it.
- `DeriveSizes` table tests: dense, MoE, MoE with leading dense layers,
  sliding window, PLE, no parameter count.
- `ProbeMeta` against an `httptest` server that serves a fixture with Range
  support. Check the byte count and the number of requests.
- The cache: a hit makes no HTTP request; a new oid misses; an unwritable
  directory is ignored.
- Render test for `hf_files` with each Fit label and its tooltip.
- `TestDeriveMatchesFullParse` (local only, step 3).

## Commit
`feat: the Download Models Fit column says what context each file holds`

## Rollback
Revert the commit. The disk cache folder can be deleted; nothing else reads
it until phase 03.
