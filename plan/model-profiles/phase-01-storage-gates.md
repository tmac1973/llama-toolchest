# Phase 01 — Schema gates and atomic writes for models.json and benchmarks.json

**Depends on:** nothing · **Enables:** Phase 02 (profiles can be added to
`models.json` without an older build or a parse error deleting them); Phase 11
(autotune records stored next to benchmark history)

## Goal
Stop the two JSON stores from overwriting themselves after a failed read.
Today `Registry.load()` (`internal/models/registry.go:1446`) logs a parse
error and carries on with empty maps, and the next `save()` writes those empty
maps over the file. `Store.load()` (`internal/benchmark/benchmark.go:1130`)
returns with nothing loaded when neither format parses, with the same result.
This phase adds a schema version that is actually read, a read-only mode with
a visible reason, and write-then-rename. It adds no feature, so it can be
reverted on its own.

## Files touched
- `internal/models/registry.go`:
  - add `SchemaVersion int` to the registry file struct;
  - read it in `load()`;
  - add a `readOnly string` reason and a `writableLocked() error` check;
  - make `save()` refuse when read-only and write via a temp file + `os.Rename`.
- `internal/models/schema_test.go` (new): tests for a corrupt file, a newer
  version, a missing version, and the atomic write.
- `internal/benchmark/benchmark.go`: the same gate on `Store` (`readOnly`
  reason, refusing writes, atomic write); compare `schemaVersion` (currently 4)
  with the file's `Version`.
- `internal/benchmark/schema_gate_test.go` (new): the same cases for
  `benchmarks.json`.
- `internal/api/service.go` (`handleGetModelConfig`) and
  `web/templates/partials/model_config.html`: show "Model settings are
  read-only: <reason>" at the top of the config panel when the registry is
  read-only.
- `internal/api/server.go` and `web/templates/benchmarks.html`: the same
  banner on the Benchmarks page when the store is read-only.

## Steps
1. **Registry versions.**
   - Add `RegistrySchemaVersion = 1` in `registry.go` and write it as
     `"schema_version"` on the envelope.
   - An absent version (0) is read as current, because every existing file has
     no version.
2. **What `load()` treats as fatal.** Set `r.readOnly` (keeping the models
   that did parse) when:
   - the file exists but cannot be read (any error other than
     `os.ErrNotExist`);
   - `json.Unmarshal` fails;
   - `schema_version` is greater than `RegistrySchemaVersion`.
   The reason text is plain language, e.g. "models.json was written by a
   newer version of llama-toolchest (schema 3; this build reads up to 1).
   Nothing will be saved until it is fixed or this build is upgraded."
3. **Mutators check first.**
   - Every exported method that changes state calls `writableLocked()` before
     changing anything in memory, and returns the error: `Add`, `Remove`,
     `Delete`, `SetConfig`, `SetSamplingPresets`, `BackfillGGUFMeta`,
     `DeduplicateModels`, `ScanModels`, `AutoDetectMMProj`,
     `BackfillSpecAssist`, `AutoDetectMTP`, and the pending-config methods in
     `pending.go`.
   - The background ones (Scan/Backfill/AutoDetect) log and return 0 instead
     of an error.
   - Checking before the change matters: a change made in memory that then
     fails to save would launch a config the panel reported as not saved.
4. **`save()`**: return an error (change the signature to `save() error` and
   update callers) when read-only. Write to `models.json.tmp`, `fsync`, then
   `os.Rename`.
5. Add `func (r *Registry) ReadOnlyReason() string`.
6. **`benchmarks.json`**: apply steps 1–5 to `benchmark.Store`.
   - Keep the v1 bare-array fallback as it is.
   - Only a file that fails *both* parses, or whose `Version` is greater than
     `schemaVersion`, sets `readOnly`.
   - Add `func (s *Store) ReadOnlyReason() string`.
   - `SaveJob`/`Save` return the error; the job queue logs it and marks the
     cell failed with that reason.
7. **Banners.**
   - Pass `ReadOnlyReason` into the config panel and Benchmarks page data.
   - Render it with the existing warning style: `role="alert"`, plain text,
     and a tooltip explaining that the file on disk has not been changed.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- **Unit tests**, one set per store:
  - Write a corrupt file, then call `load()`. `ReadOnlyReason() != ""`,
    `SetConfig` returns an error, and the file bytes on disk are unchanged.
  - A file with `schema_version: 99` behaves the same way.
  - A file with no version loads and saves normally; the saved file has the
    current version.
  - A save leaves no `.tmp` file behind.
- **Manual:**
  - Copy a real `models.json`, add `"schema_version": 99`, and start the
    server. The Models page lists the models, the config panel shows the
    banner, and the file is unchanged after trying to edit a field.
  - Do the same with `benchmarks.json`.

## Commit
`fix(storage): refuse to overwrite models.json or benchmarks.json this build could not read`

## Rollback
Revert the commit. The files written by this phase are ordinary JSON with one
extra `schema_version` key that older builds ignore, so reverting is safe at
any point.
