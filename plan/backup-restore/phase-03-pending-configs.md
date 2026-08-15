# Phase 03 — Pending configs and auto-claim

**Depends on:** 02 (restore engine produces Missing entries) · **Enables:**
the Models-page ghost rows and download wiring (phase 04).

## Goal

Configs for models the target doesn't have stop being dead ends: the restore
stores them as pending in the registry, and whenever a matching model
arrives — downloaded through any path or found by a scan — its pending
config attaches automatically. After this phase the full cross-server story
works headlessly: restore, download the model later, config applies itself.

## Files touched

- `internal/models/registry.go` — `PendingConfig` type, `registryData`
  field, `SetPendingConfig` / `PendingConfigs` / `DiscardPendingConfig`,
  claim inside `Add`.
- `internal/models/pending_test.go` (new) — persistence, claim, discard.
- `internal/backup/restore.go` — Apply now calls the `Deps.SavePending`
  hook (the field exists since phase 02, nil until now) for Missing
  entries, replacing the "model not installed" skip;
  `MissingModel.Pending` set true on success.
- `internal/backup/restore_test.go` — pending path assertions.
- `web/templates/partials/restore_report.html` — Missing rows render
  "held as pending — will apply when the model is installed" when
  `.Pending` is true (the `Pending` field exists since phase 02).
- `internal/api/backup.go` — wire `SavePending` closure; add
  `POST /api/backup/pending/discard` handler.
- `internal/api/server.go` — the discard route.

## Steps

1. Registry storage (additive, downgrade-safe — older binaries ignore the
   field):
   ```go
   type PendingConfig struct {
       ModelID  string             `json:"model_id"`
       Quant    string             `json:"quant"`
       Filename string             `json:"filename"` // required — Parse gate guarantees it; drives the download offer
       Config   ModelConfig        `json:"config"`
       SavedAt  time.Time          `json:"saved_at"`
   }
   // registryData gains: PendingConfigs []PendingConfig `json:"pending_configs,omitempty"`
   ```
   A slice, not a map: no composite-key encoding, deterministic order
   (sorted by ModelID+Quant on save). Upsert semantics on
   `SetPendingConfig` — re-importing a backup refreshes the pending entry.
2. Claim hook in `Registry.Add`: after the model registers, scan
   `PendingConfigs` for `ModelID == m.ModelID && Quant == m.Quant`; on
   match: resolve the stored config's path fields with phase 02's shared
   in-package helper `ResolveConfigPaths(cfg, r.modelsDir)` (it lives in
   `internal/models` precisely so the registry can call it without an
   import cycle) — existence is meaningful now
   that the model's files (including any sibling mmproj) have arrived, and
   any resulting blank is logged — then set the config for the new ID,
   remove the pending entry, persist, and `slog.Info("pending config
   claimed", ...)`. `ScanModels` already routes every registration through
   `Add` (verified: `registry.go:932` calls `r.Add(m)`), so the hook in
   `Add` alone covers downloads and scans.
   GPU placement is **not** re-resolved at claim: pending entries store the
   config already topology-normalized by phase 02's normalize-then-match
   Apply flow, valid for this machine as of import. Stated honestly: if
   the GPU topology shrinks between import and claim (the "weeks later"
   case), nothing re-checks it — there is no topology-mismatch safety net
   in the codebase today (`migrateGPUAssign` handles legacy values and the
   iGPU "all" case, not removed GPUs; `gpuAssignWarning` covers only
   iGPU-without-kernels), so a claimed config after a GPU removal has
   exactly the same exposure as any *saved* config after a GPU removal.
   That parity is the design decision; guarding hardware shrinkage
   generally (for all saved configs, e.g. via phase 02's new
   `AssignGPUsOutOfRange` in the page-load migration) is explicitly out of
   scope here.
   Dirty-marking is unnecessary on claim: a just-arrived model has never
   been loaded, so there is no running config to diverge from; the preset
   picks it up on the next regeneration.
3. `DiscardPendingConfig(modelID, quant string) bool` + list accessor
   `PendingConfigs() []PendingConfig` (copy, sorted).
4. Restore integration: `Apply` calls `Deps.SavePending(missing)` for each
   Missing entry when ModelConfigs is selected (the entry already carries
   the normalized config from phase 02); on success set
   `missing.Pending = true` and drop the Skipped("model not installed")
   line; on error keep a Skipped entry with the save error. The report
   partial's Missing rows render the pending annotation via the `.Pending`
   flag.
5. API: `POST /api/backup/pending/discard` (form: `model_id`, `quant`) →
   discards, responds 204 with an `HX-Trigger: modelsChanged` header. That
   is the endpoint's final contract, not a placeholder: the models page's
   listing container (`models.html`: `<div id="model-list" hx-get="/api/models"
   hx-trigger="load, modelsChanged from:body">`) already re-fetches itself
   on that event, so a discard refreshes the whole listing — the ghost
   card simply disappears from the flat card list — through existing
   machinery, and phase 04 adds no handler changes.

## Build gate

`go build ./... && go vet ./internal/... && go test ./internal/...`

## Test plan

- `TestPendingPersistAndClaimOnAdd`: SetPendingConfig, reload registry from
  disk (new Registry on same dataDir), `Add` a matching model → config
  attached, pending gone, persisted.
- `TestClaimOnScan`: pending entry + GGUF placed in the models dir →
  `ScanModels` claims it.
- `TestClaimExactQuantOnly`: pending for `UD-Q8_K_XL` does not claim a
  `Q4_K_M` download of the same repo.
- `TestDiscard`: discard removes and persists; discarding a nonexistent
  entry reports false.
- Restore-engine test: a file with one installed and one missing model →
  one applied, one pending via the Deps recorder; re-import upserts rather
  than duplicates.
- Downgrade safety: a models.json containing `pending_configs` loads
  cleanly under a struct without the field (json ignores unknowns) — noted,
  not tested (can't compile the old struct here).

## Commit

`feat(backup): pending model configs with auto-claim on download and scan`

## Rollback

Remove the field, methods, hook, and route. A models.json already carrying
`pending_configs` still loads after rollback (unknown JSON fields are
ignored); pending entries are simply forgotten.
