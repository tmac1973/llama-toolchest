# Configuration Backup & Restore — Project Overview

## Problem

The toolchest now persists a meaningful amount of hand-crafted configuration —
server settings and runtime environment (`llama-toolchest.yaml`), per-model
launch configs (`models.json`), and saved build flag sets
(`build-flag-presets.json`) — with no way to capture or transfer it. Losing the
data dir means re-toggling everything from memory, and standing up a second
server means manually re-creating every model config and flag set. The state
audit showed the valuable part is small (kilobytes of intent) while everything
bulky (build binaries, GGUF files, generated presets) is rebuildable or
re-downloadable, so a configuration-only export/import is both cheap and
sufficient.

The hard part is model configs: they are stored keyed by registry ID with
machine-local file paths, and a target server is not guaranteed to have the
same models installed or the same GPU topology.

## Goals

- A single versioned JSON backup file (`"version": 1`) with sections:
  `settings`, `runtime_env`, `flag_presets`, `model_configs` — downloadable
  from a new Backup & Restore card on the Settings page, and fetchable as
  `GET /api/backup` for scripting.
- Secrets (HF token, API key) included only behind an explicit
  "include secrets" checkbox at export, default off, with a plaintext warning.
- Model configs exported keyed by stable HF identity `(ModelID, Quant)` —
  never registry ID or file path — carrying only user intent (launch config,
  aliases), not registry/GGUF-derived metadata.
- Restore is a merge: settings overwrite the preference fields, model configs
  upsert by identity, flag presets upsert by name; nothing is deleted by a
  restore. Applied model configs are marked pending-reload via the existing
  per-model dirty mechanism, and the report carries a fixed reminder line
  whenever settings or runtime env were applied: "these changes take effect
  on the next server restart" (there is no global restart flag in the
  codebase to set; the per-model marks plus the report line are the whole
  mechanism).
- Restore of settings applies preferences only: `ModelsMax`, `AutoStart`,
  `LogLevel`, runtime env (curated + extra), and — when the file includes
  secrets — `HFToken` and `APIKey`. Deployment-identity fields (`ListenAddr`,
  `LlamaPort`, `ExternalURL`, `DataDir`, `ModelsDir`, `ActiveBuild`) are
  exported for reference but never applied.
- Restore is selective by category: four checkboxes mapping 1:1 to the file's
  sections (settings, runtime env, flag presets, model configs), all checked
  by default. Picking a file parses it client-side (no upload yet) and
  annotates each checkbox: item counts for the list-shaped sections
  ("Model configs (12)", "Flag presets (3)", "Runtime env (5)") and a
  "contains secrets" badge on Settings (a scalar bundle — badge instead of
  count) when a non-empty token or key is present; sections absent from the
  file render disabled. One POST submits file + selections together.
- Import produces a report, not silent success: applied items, skipped items
  with reasons, deselected sections noted as skipped-by-choice, warnings, and
  missing models.
- Configs referencing models not installed on the target become **pending
  configs**: persisted alongside the registry and attached automatically when
  a matching model is later downloaded or scanned (full auto-claim). The
  import report offers one-click download of missing models through the
  existing download pipeline.
- GPU placement survives hardware differences: on import, non-`custom`
  `GPUAssign` values are re-resolved against the local GPU count/topology via
  the existing `ResolveGPUAssign` machinery; `custom` tensor splits import
  verbatim with a warning. Config paths (`MmprojPath`, `MtpPath`,
  `DraftModelPath`) are
  exported relative to the models dir and re-joined onto the target's models
  dir at apply time (for installed models) or claim time (for pending ones);
  a path that resolves to a nonexistent file is blanked with a warning, and
  a path that was outside the source's models dir travels verbatim and is
  kept only if it exists on the target.
- Failure handling: structural problems (not JSON, unknown version, malformed
  sections) reject the whole file before anything is touched; past that gate,
  each item applies independently and failures appear in the report as
  skipped-with-reason.

## Non-goals

- No backup of benchmark history — results data, already covered by the
  benchmark export; the backup restores behavior, not memories.
- No backup of build binaries, the llama.cpp checkout, GGUF model files, the
  HF cache, `builds.json`, or generated presets (`preset.ini`,
  `bench-preset.ini`) — all rebuildable, re-downloadable, or derived.
- No scheduled/automatic backups in v1 — export is manual; the `GET
  /api/backup` endpoint makes external scheduling (cron + curl) possible
  without in-app machinery.
- No "replace all" restore mode — merge only; deleting local state is never a
  restore side effect.
- No restore-time application of deployment-identity settings, even
  optionally.
- No encryption of the backup file — secrets inclusion is opt-in and the
  warning makes the file's sensitivity explicit; protecting the file is the
  user's job.

## Users & primary flow

Self-hosters running one or more toolchest servers.

**Backup:** Settings → Backup & Restore card → optionally tick "include
secrets" → Export. The browser downloads
`llama-toolchest-backup-<date>.json`. Scripted alternative:
`curl host/api/backup > backup.json` (same-origin UI route, no API key —
network exposure is governed by the deployment, as with all `/api` routes).

**Restore / migrate:** on the target server, Settings → Backup & Restore →
choose file. The card immediately previews the file's contents: a checkbox
per category with item counts and a secrets badge, sections not present
disabled. Untick anything unwanted (e.g. take flag sets and runtime env,
skip model configs), then Restore. The card renders a report: settings fields applied;
flag presets upserted; model configs applied to installed models (GPU
assignments re-resolved for the local topology, with warnings where the
source topology didn't fit); configs for missing models held as pending, with
a one-click "download missing models" action. Models downloaded later — via
that button or manually, even weeks after — automatically pick up their
pending config. Applied model configs show the existing pending-reload
badges, and the report reminds that settings/env changes take effect on the
next restart.

## Constraints

- Go backend, htmx + Pico server-rendered UI; the report and card follow the
  existing Settings-page patterns (article cards, tooltips via `title`,
  status areas as htmx swap targets).
- `/api/backup` and `/api/restore` are same-origin UI routes with no
  API-key check, exactly like every other `/api` route (the `apiKeyAuth`
  middleware guards `/v1` only — verified in `server.go`). A secrets-bearing
  export therefore has the same exposure as the existing settings API;
  network-level protection is the deployment's job (the secure-caddy setup),
  and the help text says so. This is also what lets the export button be a
  plain browser download link.
- Restore reuses existing single-source machinery rather than duplicating it:
  `config.EnvSet.Validate()` for runtime env, `builder.SaveFlagPreset` for
  flag presets (name/profile validation included), `models.ResolveGPUAssign`
  + the migration/warning helpers for GPU placement, and the registry's
  `SetConfig`/`WritePresetINI` path so the preset regenerates.
- Pending configs persist in `models.json`'s registry data as a new
  `pending_configs` list (entries keyed by ModelID+Quant, upsert on
  re-import, sorted for determinism) so they survive restarts; the claim
  hook lives in `Registry.Add`, which both downloads and `ScanModels`
  route through (verified: `registry.go:932`). The field is additive —
  older binaries ignore it (downgrade-safe).
- Identity matching uses `(Model.ModelID, Model.Quant)` exactly as stored;
  the export never contains registry IDs or absolute paths except the
  reference-only settings section.
- Export must be deterministic (sorted keys/sections) so two backups of
  identical state are diffable.
- The backup schema carries `"version": 1`; restore rejects unknown versions
  with a clear message rather than guessing.
- The client-side file preview is a small inline script (FileReader +
  JSON.parse) following the existing plain-JS precedent (e.g. the settings
  page's `filterEnvRows`); the server never trusts it — selection filtering
  and all validation happen server-side on the single restore POST.

## Success criteria

- Export from a configured server, wipe its config dir, restore: model
  configs, flag presets, runtime env, and preference settings all return;
  effective preset.ini after restart is byte-identical for installed models.
- Export from server A (4 GPUs, model X installed), import on server B
  (2 GPUs, model X absent): report shows X pending with a download offer;
  after downloading X, its config is attached automatically with GPU
  assignment re-resolved to B's topology **as of import time** (topology
  changes between import and claim are not re-checked — the same exposure
  any saved config has after a hardware change) and a warning recorded for
  anything that couldn't map.
- A backup without secrets contains no HF token or API key anywhere in the
  file; one with secrets was produced only after an explicit checkbox, and
  the restore preview visibly flags such a file before import.
- Restoring with only "flag presets" selected changes nothing outside flag
  presets, and the report says the other sections were skipped by choice.
- A file with one invalid model config restores everything else and lists the
  failure with a reason; a truncated/wrong-version file changes nothing and
  says why.
- Restore never deletes: models, configs, presets, and settings absent from
  the file are untouched.
- `go build ./... && go test ./internal/...` passes; new tests cover the
  export shape, merge semantics, identity matching, pending claim, GPU
  re-resolution on import, and the no-secrets guarantee.

## Decisions

Settled in the pre-planning audit discussion:

- **Scope** → Back up intent, not artifacts: settings, runtime env, flag
  presets, model configs. Exclude builds.json, binaries, models, caches,
  generated presets.
- **Format** → Single versioned JSON file; Settings-card UI plus curl-able
  endpoints.
- **Secrets** → Opt-in checkbox at export, default off, plaintext warning.
- **Model config identity** → Keyed by `(ModelID, Quant)`; no paths or
  registry IDs.
- **Missing models** → Pending configs + one-click download offer.
- **Hardware differences** → Re-resolve GPU assignments on import; warn on
  custom splits; remap or blank absolute paths inside configs.
- **Restore semantics** → Merge/upsert; never delete; per-model dirty marks
  plus a report reminder line for restart-requiring changes.

From the planning interview:

- **Benchmark history** → Excluded; results data with its own export.
- **Settings breadth on restore** → Preferences only (ModelsMax, AutoStart,
  LogLevel, runtime env, opt-in secrets); deployment-identity fields exported
  for reference, never applied.
- **Pending-config auto-claim** → Full auto-claim in v1: pending configs
  persist and attach whenever a matching model arrives, however it arrives.
- **Partial-failure handling** → Validate the file structurally up front
  (reject wholly), then apply per item with skipped-with-reason entries in
  the report.
- **Selective restore** → Per-category checkboxes (settings, runtime env,
  flag presets, model configs), all checked by default; runtime env separate
  from settings; server-side filtering is authoritative.
- **Restore preview UX** → Client-side file preview on selection: checkbox
  counts per section, secrets badge, absent sections disabled; single POST
  carries file + selections.
