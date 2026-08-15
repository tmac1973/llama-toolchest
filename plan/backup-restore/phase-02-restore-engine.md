# Phase 02 — Restore engine and endpoint

**Depends on:** 01 (schema, `Assemble` for round-trip tests) · **Enables:**
pending-config integration (phase 03) and the restore UI (phase 04).

## Goal

Validate, filter, and apply a backup file with merge semantics and an
itemized report. After this phase, `curl -F file=@b.json /api/restore` works
end to end for settings, runtime env, flag presets, and configs of models
that are installed; configs for missing models are reported as skipped (they
become pending in phase 03).

## Files touched

- `internal/backup/restore.go` (new) — `Parse`, `Selections`, `Apply`,
  `Report`.
- `internal/backup/restore_test.go` (new) — engine tests.
- `internal/models/gpu_assign.go` (+ test) — new exported helper
  `AssignGPUsOutOfRange` (step 4a).
- `internal/models/config_paths.go` (+ test, new) — exported
  `ResolveConfigPaths(cfg *ModelConfig, modelsDir string) (warnings []string)`
  (step 4c). It lives in `internal/models` — not the backup package —
  because phase 03's claim hook calls it from inside the registry, and
  `internal/backup` already imports `internal/models` (the reverse import
  would cycle).
- `internal/api/backup.go` — add `handleRestore`.
- `internal/api/server.go` — route `r.Post("/restore", s.handleRestore)`.

## Steps

1. `Parse(data []byte) (*File, error)` — structural gate, defined by an
   exhaustive rule: **shape errors reject the whole file; value errors are
   per-item skips (step 4).** Shape errors are exactly: not valid JSON;
   `version != 1` ("backup version N is not supported by this build"); a
   `model_configs` entry with empty `model_id`, `quant`, or `filename`; a
   `flag_presets` entry with empty `name` or `profile`. Everything else —
   invalid preset name characters, unknown env variable, malformed env
   value, unparseable GPU assign, nonexistent paths — is a value error
   handled per item during Apply. All shape errors are collected into one
   rejection message; nothing is applied if Parse fails.
2. `type Selections struct { Settings, RuntimeEnv, FlagPresets, ModelConfigs bool }`
   — server-side filter, authoritative regardless of what the client preview
   showed.
3. `type Report struct`:
   - `Applied []string` (human lines: "settings: models_max, auto_start",
     "model config: unsloth/Qwen3.8-27B-GGUF UD-Q8_K_XL")
   - `AppliedModelConfigs int` — machine-readable counter the handler keys
     preset regeneration on (never parse the human lines)
   - `Notes []string` — informational lines that are neither successes nor
     problems; the restart reminder ("these changes take effect on the next
     server restart", added whenever settings or runtime env applied
     anything) lives here
   - `Skipped []SkippedItem{Item, Reason string}`
   - `NotSelected []string` (sections present in the file but unticked)
   - `Warnings []string`
   - `Missing []MissingModel` where
     `MissingModel{ModelID, Quant, Filename string, Config models.ModelConfig, Pending bool}`
     — the normalized config rides along so phase 03 can persist it;
     `Pending` is false in this phase (rendered "install this model to apply
     its config") and set true by phase 03. The report partial receives
     ModelID/Quant/Filename per row — the data contract phase 04's download
     buttons rely on.
4. `Apply(f *File, sel Selections, deps Deps) Report` where `Deps` is a small
   struct of injected collaborators so the engine stays testable without the
   api package:
   ```go
   type Deps struct {
       ApplySettings    func(Settings) (changed []string, err error) // api closure: mutate cfg under cfgMu + saveConfig
       CurrentEnv       func() RuntimeEnv                            // target's live env — the engine needs it to build the merge
       ApplyEnv         func(RuntimeEnv) error                       // stores the engine-built, engine-validated merged set
       SaveFlagPreset   func(builder.FlagPreset) error
       InstalledModels  func(modelID, quant string) []string // all matching registry IDs; empty = not installed
       ApplyModelConfig func(registryID string, cfg models.ModelConfig) error
       SavePending      func(MissingModel) error // nil in this phase; wired by phase 03
       NumGPUs          int
       ModelsDir        string
   }
   A nil `SavePending` (this phase) leaves Missing entries with
   `Pending: false` and the "model not installed" skip — exactly the
   behavior phase 03 upgrades.
   ```
   Per-item semantics:
   - Settings: single item; pointer merge — each non-nil field in the file
     is applied (a present zero like `auto_start: false` is a real value),
     nil fields are untouched — with one defensive exception: a present but
     **empty** `hf_token`/`api_key` is skipped with a Warning ("empty
     secret ignored") rather than blanking the target's credential
     (`Assemble` never emits empty secrets, so this only fires on
     hand-edited files). `ApplySettings` returns `(changed []string, err
     error)` so the report names the fields that actually changed value.
     When settings or runtime env applied anything, the fixed reminder
     line "these changes take effect on the next server restart" is
     appended to `Report.Notes` (there is no global restart flag to set;
     per-model dirty marks below cover model configs).
   - RuntimeEnv: **per-key merge, honoring never-deletes** — the engine
     fetches the target's live set via `Deps.CurrentEnv()` and builds the
     merge: target curated map overlaid with the file's curated entries
     (file wins per key; target-only keys survive), and the target's
     `Extra` replaced only when the file's `extra` is non-empty (an
     absent/empty `extra` leaves the target's untouched). The engine
     validates the **merged** EnvSet (`config.EnvSet.Validate` — it's what
     will actually apply); on error → one Skipped entry with the validation
     message, nothing stored; on success `Deps.ApplyEnv(merged)` stores it
     and the engine surfaces `EnvSet.Warnings()` (known-risky variables) into
     `Report.Warnings`.
   - FlagPresets: one item each via `SaveFlagPreset` (its name/profile
     validation produces the skip reason).
   - ModelConfigs: one item each, **normalize-then-match** — the topology
     normalization runs before the install check so Missing entries carry a
     config already valid for this machine (phase 03 stores exactly that as
     pending, and claim becomes a plain attach):
     a. GPU re-resolution on the config copy: if `Config.GPUAssign` is set
        and not "custom", detect out-of-range references with a new small
        exported helper `models.AssignGPUsOutOfRange(assign string,
        numGPUs int) bool`. Implementation note that matters: for the
        legacy "tensor-N" form it must use the **unclamped**
        `parseTensorAssign` (N > numGPUs → out of range) —
        `tensorAssignGPUs` clamps N to numGPUs and would report
        "tensor-4" on a 2-GPU box as in-range; explicit-set forms
        ("tensor:0,2", "0-1", "1,2") check max referenced index ≥ numGPUs
        via the existing unclamped parsers.
        Out of range → fall back to `"all"` **and re-derive the split
        fields for it** (`ResolveGPUAssign("all", NumGPUs)` → clears to
        `("", "layer", 0)`; never carry the source box's TensorSplit or
        SplitMode into the fallback) + Warning naming the original.
        In range → re-derive
        `TensorSplit/SplitMode/MainGPU` with
        `models.ResolveGPUAssign(assign, NumGPUs)` — producing the
        zero-padded split documented on that function (e.g. "0-1" with
        NumGPUs=4 → TensorSplit "1,1,0,0"). `custom` imports verbatim +
        Warning ("verify tensor split against local GPUs"). NumGPUs == 0 →
        verbatim + Warning.
     b. Identity match via `InstalledModels(ModelID, Quant) []string` —
        plural: duplicate registrations of the same quant file are possible
        (DeduplicateModels exists because of them), and the config applies
        to **every** match, each reported. Empty → append to Missing (with
        the normalized config) and a Skipped entry ("model not installed")
        — phase 03 replaces the skip with pending. Path fields stay in
        exported form inside Missing: existence can only be checked once
        the model's files arrive.
     c. Matched only — path resolution via the shared exported helper
        `models.ResolveConfigPaths(cfg, modelsDir)` (see Files touched;
        phase 03 reuses it in-package at claim time): a relative path joins
        onto the models dir; an absolute path (exported verbatim because it
        was outside the source's models dir) is used as-is; either way, if
        the resulting file doesn't exist, blank the field and return a
        warning, which Apply surfaces into `Report.Warnings`. The helper
        covers all **three** machine-local GGUF path fields —
        `MmprojPath`, `MtpPath`, and `DraftModelPath` (the disabled flags
        `MmprojDisabled`/`MtpDisabled` import untouched; a blanked path
        with its disabled flag set is harmless since the flag already
        suppresses emission).
     d. `ApplyModelConfig` (api closure: `registry.SetConfig`);
        `AppliedModelConfigs++`.
5. `handleRestore` in `internal/api/backup.go`:
   - Refuse with 409 while a benchmark job is running (`routerBusyWithJob`),
     mirroring the router start/restart guards — a restore mid-job would
     change configs a running cell reports as fixed.
   - Parse multipart form: `file` (limit 10 MB), checkboxes `sec_settings`,
     `sec_env`, `sec_flags`, `sec_models` → Selections. All four false →
     400 "select at least one section to restore" before touching anything
     (phase 04's UI also disables Restore in that state).
   - Build Deps closures: settings under `cfgMu` + `saveConfigLocked`;
     `InstalledModels` scans `registry.List()` comparing ModelID+Quant,
     returning every match; `ApplyModelConfig` calls `SetConfig` then
     `s.markDirty(id)`.
   - After Apply: if `report.AppliedModelConfigs > 0`,
     `registry.WritePresetINI(s.activeBackend())`.
   - Response contract (failures must be visible — htmx doesn't swap
     non-2xx): for htmx requests the handler **always returns 200 with the
     report partial**; refusals (structural rejection, "benchmark job
     running", "select at least one section") render as a report carrying
     only an `Error string` field (add it to `Report`), styled as an error
     banner. Non-htmx clients get real status codes — 400 for structural /
     empty-selection, 409 for job-busy — with a JSON `{"error": ...}` body,
     and `respondJSON(report)` with 200 on success. The report partial lives in
     `web/templates/partials/restore_report.html`, created here; phases 03
     and 04 extend it (pending annotation, download buttons) — its row data
     (ModelID/Quant/Filename per Missing entry) is designed for that from
     the start.
6. Report partial: sections rendered in order Applied / Notes / Warnings /
   Skipped / NotSelected / Missing; Missing rows show identity + "install
   this model to apply its config" when `.Pending` is false. The partial
   also renders the failure shape (step 5): a report whose only content is
   `Error` text.

## Build gate

`go build ./... && go vet ./internal/... && go test ./internal/...`

## Test plan

- Round trip: `Assemble` → marshal → `Parse` → `Apply` with all sections
  selected against a fake Deps recording calls; everything lands, report
  shows all Applied.
- `TestParseRejectsWholeFile`: truncated JSON, version 2, and a config entry
  without ModelID each reject with nothing applied (Deps records zero calls).
- `TestApplyPerItemFailure`: one flag preset with an invalid name among two
  valid ones → two applied, one Skipped with the validation message.
- `TestSelectionsFilter`: file with all four sections, only FlagPresets
  selected → other sections in NotSelected, zero non-preset Deps calls.
- `TestGPUReResolution`: "tensor-4" config applied with NumGPUs=2 → falls
  back to local "all" + warning; "0-1" with NumGPUs=4 → re-resolved padded
  split for 4; "custom" passes verbatim + warning.
- `TestPathResolution`: relative mmproj that exists resolves absolute;
  missing one blanks + warns.
- `TestMergeNeverDeletes`: a file containing one model config leaves a
  second installed model's config untouched.
- Manual: export on the dev box, `curl -F file=@b.json -F sec_flags=on
  localhost:3000/api/restore`.

## Commit

`feat(backup): restore engine with selective merge and itemized report`

## Rollback

Delete `restore.go`, the handler, route, and report partial. Phase 01 export
still works alone. No migration to unwind — restore writes only through
existing setters.
