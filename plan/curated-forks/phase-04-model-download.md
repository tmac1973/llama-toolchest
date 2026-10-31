# Phase 04 — Fork model sets: one-click download

**Depends on:** 01 (the setting), 02 (fork model files and the store) · **Enables:** Phase 05 (an instance needs a complete model set)

## Goal

With the setting on, the Models page shows a "Fork model sets" section. Each curated fork has one row with its model name, size, license and a Download button. The button downloads every file of the set from Hugging Face with the existing downloader (resume included) into `<models dir>/fork-models/`. The set is marked complete when every file is on disk at its listed size. The LLM model scan does not pick up these files.

## Files touched

- `internal/models/registry.go` — new `ForkModelsDirName = "fork-models"` and skip it in the model walk, next to `GenModelsDirName`.
- `internal/api/fork_models.go` — new: `forkModelsDir()`, the section handler, download, delete and file-done callback.
- `internal/api/server.go` — routes under `/api/forks/models`, the forks store loaded at startup (`forks.Load(dataDir)`), and `ForksEnabled` in the Models page data.
- `web/templates/models.html` — the "Fork model sets" section, shown only while forks are on.
- `web/templates/partials/fork_models.html` — new: the section partial.
- `internal/api/fork_models_test.go` — new tests.

## Steps

1. In `registry.go`, add `ForkModelsDirName` and rename `isGenModelsDir` (line 1250) to `isAppModelsDir`, which returns true for a folder directly under the walk root named `GenModelsDirName` or `ForkModelsDirName`. Update its callers (`ScanModels`, `OrphanParts`).
2. Add `forks *forks.Store` to `Server`, loaded at startup next to the generation store. A load error is logged and the forks feature reports "fork store unreadable" on its pages; it does not stop the server.
3. `forkModelsDir()` returns `filepath.Join(cfg.ModelsPath(), models.ForkModelsDirName)`. A set's directory is `huggingface.FilePath(root, f.ModelRepo, "")`, so every file of the set, including `MTP/...` and `moe-cache-ranking.bin`, lands under one folder.
4. Routes, each returning 404 while forks are off (a `forkStore(w)` helper like `genStore(w)`):
   - `GET /api/forks/models` — the section partial.
   - `POST /api/forks/models/{fork}/download` — start or resume.
   - `DELETE /api/forks/models/{fork}` — refuse while any stored fork instance has this `ForkID` ("Delete the fork instances that use this model set first."); else delete the folder and the store entry. Instances can only be added by tests in this phase; Phase 05 adds the API.
5. Download: create or load the `ModelSet{ForkID, Dir}`. For each `ModelFile`, a file already on disk at its listed size counts as done; the others go to `s.downloader.StartTo(modelsource.SourceHuggingFace, f.ModelRepo, file.Path, root, file.Bytes, onDone)`. `onDone` re-checks all files and sets `Complete` when every one is present at its size. Before starting, check free disk space against the bytes still missing plus 1 GiB; when short, answer 507 with "Not enough free disk space: X GiB needed, Y GiB free."
6. The partial shows, per fork: model name, total size, license, state (not downloaded / downloading with progress from the downloader / complete), and Download, Resume or Delete. Progress uses the same polling the generation catalog uses.
7. Tooltips: Download — "Download all files of this model set. Downloads continue after a restart."; Delete — "Delete the downloaded files of this model set."

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `fork_models_test.go`, with a fake downloader: download starts one job per missing file and none for files already present at their size; the set becomes complete only after the last `onDone`; a size mismatch keeps it incomplete; delete is refused while a stored instance has the fork's ID; all routes 404 with forks off; low disk space returns 507.
- Registry test: a GGUF under `fork-models/` is not listed as an LLM.
- Manual on asmodean: download the Avalon set (~89.7 GiB), stop the toolchest halfway, start it again, press Resume, and check that the set completes and the Models list shows no Avalon shards as LLMs.

## Commit

`feat(forks): download a fork's model set from the Models page`

## Rollback

Revert the commit. Files already downloaded stay under `fork-models/`; with the skip rule reverted, the LLM scan would list the Avalon shards, so also delete that folder by hand when rolling back.
