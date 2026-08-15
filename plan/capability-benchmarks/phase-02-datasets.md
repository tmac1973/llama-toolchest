# Phase 02 — Dataset download, verification, and the logits cache store

**Depends on:** 01 (the engine defines the modes datasets serve) ·
**Enables:** the runner (03) calls `EnsureDataset`/logits-cache functions;
the UI (04) lists and deletes what this phase stores.

## Goal

Every file an evaluation needs, obtained and managed: the three pinned
datasets downloaded on first use with SHA-256 verification, and the KL
reference logits cache with its naming scheme, disk-space guard, and
deletion. All filesystem layout decisions land here.

## Files touched

- `internal/evaluate/datasets.go` (new) — pinned dataset table,
  `EnsureDataset`, cache paths.
- `internal/models` GGUF metadata parse — additive `VocabSize` field
  (step 4's estimate needs it and nothing in the codebase records it
  today).
- `cmd/llama-toolchest/main.go` — the one-line `CleanStalePartials`
  call at startup (step 4).
- `internal/evaluate/klcache.go` (new) — logits cache key/paths, listing,
  delete, disk guard.
- `internal/evaluate/datasets_test.go`, `klcache_test.go` (new).

## Steps

1. Layout under the data dir (constant root passed in by callers —
   `<dataDir>/eval-data/`):
   ```
   eval-data/
   ├── datasets/wikitext-2-raw-test.txt
   ├── datasets/hellaswag_val_full.txt
   ├── datasets/winogrande-debiased-eval.csv
   └── logits/<SafeModelID>~<quant>~<dataset>~c<chunks>~ctx<n>.kld
   ```
   (`SafeModelID` lives in `internal/huggingface/names.go:9`;
   `internal/evaluate` importing `internal/huggingface` is cycle-free —
   huggingface imports neither benchmark nor evaluate. The field
   separator is `~`, NOT `--`: `SafeModelID` maps `/` to `--`, so a
   `--` separator would make reverse-parsing the filename ambiguous;
   `~` is filesystem-safe and appears in neither `SafeModelID` output
   nor quant names, keeping the key→filename→key round trip lossless.)
2. Pinned dataset table — one struct per dataset: name, download URL,
   SHA-256, license string, approximate size. Sources (implementer
   verifies URLs and pins the hashes at implementation time by
   downloading and hashing once):
   - wikitext-2 raw test set (CC BY-SA 3.0) — the file llama.cpp's own
     perplexity documentation uses.
   - `hellaswag_val_full.txt` (MIT) — the preprocessed file the
     `--hellaswag` mode consumes (the format `perplexity.cpp` parses; the
     llama.cpp wiki names its canonical HF source).
   - Winogrande debiased eval CSV (Apache-2.0) — the format
     `load_winogrande_from_csv` parses.
3. `EnsureDataset(ctx, root, name) (path, error)`: present + hash
   matches → return; absent → download to a temp file (plain
   `net/http` — these are static files, not HF-API objects; the HF
   client's auth/resume machinery isn't needed), verify SHA-256, rename
   into place. Hash mismatch → delete temp, return an error naming
   expected vs got. No re-verification on every call (hash once at
   download; a `Verify(root)` helper exists for the UI's state display).
4. Logits cache (`klcache.go`):
   - `KLBaseKey{ModelID, Quant, Dataset string, Chunks, Ctx int}` →
     deterministic filename as in step 1. Chunks is in the key because
     a base generated at 100 chunks cannot serve a full-run comparison
     (quick and full cache separately). Ctx is in the key because the
     base file embeds its n_ctx and the embedded value silently wins in
     the comparison — `llama-perplexity` only logs on a mismatch, it
     does not reject (`perplexity.cpp:1717-1722`), so this key is the
     ONLY consistency guard. Today ctx is always
     `evaluate.EvalContextSize` (512); keying it means a future
     constant change invalidates cleanly instead of mis-serving.
   - `KLBasePath(root, key)`, `ListKLBases(root) []KLBaseInfo` (key
     parsed back from filename + size + mtime), `DeleteKLBase`,
     `HasKLBase`.
   - Interruption safety — same temp-then-rename discipline as the
     datasets, because the stakes are higher, not lower: generation
     writes a valid-looking header FIRST (magic + n_ctx,
     `perplexity.cpp:465-470`, then n_vocab/n_chunk/tokens at
     `:523-525`) and streams log-probs for minutes to hours, so a
     killed or canceled generation leaves a truncated file whose
     header claims the full chunk count; the comparison reader then
     fails mid-loop with an unrelated-looking error, and a
     presence-keyed cache would serve that corpse forever. Rule:
     generation writes to `<final-name>.partial`; `HasKLBase` and
     `ListKLBases` ignore `.partial` files; clean exit renames into
     place; failure/cancel deletes the partial. Stale partials (a
     crash skipped the delete) are removed by
     `CleanStalePartials(root)` in `klcache.go`, called once at
     process startup from `cmd/llama-toolchest/main.go` next to the
     existing startup directory creation (`main.go:99`). Phase 02
     test: `HasKLBase`/`ListKLBases` ignore a `.partial` file and
     `CleanStalePartials` removes it. (That an interrupted generation
     leaves no cache entry and a later `EnsureKLBase` regenerates is
     phase 03's test.)
   - Disk guard: `CheckKLBaseSpace(root, estimate)` — from the tool's
     own math (writer at `perplexity.cpp:461-470`, `:523-525`, `:619`;
     per-token layout confirmed on the reader side at `:1752-1760`):
     per chunk the tool stores
     `n_ctx − 1 − n_ctx/2` evaluated tokens, each carrying
     `2×((n_vocab+1)/2) + 4` uint16 values, so
     `estimate = chunks × (n_ctx − 1 − n_ctx/2) × (2×((n_vocab+1)/2) + 4) × 2 bytes`
     with a 1.15 safety factor for headers/rounding. Cross-check the
     formula at implementation against the perplexity README's own
     scale figures (≈11 GiB for a full LLaMA-2 wikitext base, ≈37 GiB
     for LLaMA-3 — vocab-size driven).
     `n_vocab` source: the codebase records no vocab size anywhere, so
     the GGUF metadata parse in `internal/models` gains an additive
     `VocabSize` field read from the `tokenizer.ggml.tokens` array
     LENGTH in the header (array lengths sit in the header; the
     contents are not read). Models registered before the field exists
     fall back to a documented worst-case constant of 262144 (covers
     current large vocabularies, e.g. Qwen's ~248K) — a conservative
     overestimate that can only refuse early, never under-reserve.
     Callers refuse generation when
     free space minus the 2 GiB safety margin (same constant the
     downloader uses, `internal/huggingface/disk.go:9`) is
     insufficient, with an error naming the estimate.
5. License strings live in the dataset table and render in phase 04's UI
   and help; the table is the single source.

## Build gate

`go build ./... && go vet ./internal/... && go test ./internal/...`

## Test plan

- `EnsureDataset` against `httptest.Server`: fresh download + hash verify
  + rename; cached short-circuit (server hit count stays 0 on second
  call); corrupted body → named hash-mismatch error, no file left behind.
- KL cache: key→filename→key round trip (including quants with dots like
  `Q4_K_M` and model IDs with slashes through `SafeModelID`), list/delete,
  disk-guard refusal with a too-small free-space fake.
- Manual: none needed beyond phase 04's end-to-end.

## Commit

`feat(evaluate): pinned evaluation datasets and KL logits cache`

## Rollback

Delete the two files; `eval-data/` on disk is inert without them.
