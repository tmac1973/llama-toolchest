# Phase 01 — Backup schema and export engine

**Depends on:** nothing · **Enables:** the restore engine (phase 02) consumes
this schema; the UI (phase 04) triggers this endpoint.

## Goal

Define the versioned backup file format and produce it: a new
`internal/backup` package that assembles a deterministic JSON export from the
live config, builder, and registry, plus a `GET /api/backup` endpoint that
serves it as a download. After this phase, `curl` backup works end to end;
nothing consumes the file yet.

## Files touched

- `internal/backup/backup.go` (new) — schema types and `Assemble`.
- `internal/backup/backup_test.go` (new) — shape, determinism, no-secrets
  guarantee.
- `internal/api/backup.go` (new) — `handleBackupExport`.
- `internal/api/server.go` — route `r.Get("/backup", s.handleBackupExport)`
  inside the existing `/api` group (same-origin UI route with no API-key
  check, like all `/api` routes — see step 6).

## Steps

1. Define the schema in `internal/backup/backup.go`. **Every field carries an
   explicit snake_case JSON tag** — this is a versioned wire format and the
   client-side preview (phase 04) reads these exact keys. Settings fields
   are **pointers** so restore can distinguish "absent from file" (don't
   touch) from "present with zero value" (apply) — the merge-never-deletes
   guarantee depends on this:
   ```go
   type File struct {
       Version      int                  `json:"version"` // 1
       ExportedAt   time.Time            `json:"exported_at"`
       Source       SourceInfo           `json:"source"` // reference only, never applied
       Settings     *Settings            `json:"settings,omitempty"`
       RuntimeEnv   *RuntimeEnv          `json:"runtime_env,omitempty"`
       FlagPresets  []builder.FlagPreset `json:"flag_presets,omitempty"` // FlagPreset already has snake_case tags
       ModelConfigs []ModelConfigExport  `json:"model_configs,omitempty"`
   }
   type SourceInfo struct { // documents the origin server; restore ignores it
       ListenAddr  string   `json:"listen_addr,omitempty"`
       LlamaPort   int      `json:"llama_port,omitempty"`
       ExternalURL string   `json:"external_url,omitempty"`
       DataDir     string   `json:"data_dir,omitempty"`
       ModelsDir   string   `json:"models_dir,omitempty"`
       ActiveBuild string   `json:"active_build,omitempty"`
       NumGPUs     int      `json:"num_gpus"`
       GPUs        []string `json:"gpus,omitempty"` // marketing names, for topology warnings
   }
   type Settings struct { // all pointers: absent = untouched on restore
       ModelsMax *int    `json:"models_max,omitempty"`
       AutoStart *bool   `json:"auto_start,omitempty"`
       LogLevel  *string `json:"log_level,omitempty"`
       HFToken   *string `json:"hf_token,omitempty"` // only when includeSecrets
       APIKey    *string `json:"api_key,omitempty"`  // only when includeSecrets
   }
   type RuntimeEnv struct {
       Curated map[string]string `json:"curated,omitempty"`
       Extra   string            `json:"extra,omitempty"`
   }
   type ModelConfigExport struct {
       ModelID  string             `json:"model_id"` // HF org/repo — required, restore's Parse gate enforces
       Quant    string             `json:"quant"`    // required
       Filename string             `json:"filename"` // required: download aid + claim precision; Parse gate enforces
       Config   models.ModelConfig `json:"config"`   // full launch config (path fields relativized, step 3)
   }
   ```
   `Assemble` pointer rules, exactly: `ModelsMax`, `AutoStart`, `LogLevel`
   are always set (a fresh export is complete). `HFToken`/`APIKey` are set
   **only when `includeSecrets` is true AND the value is non-empty**;
   otherwise the pointers stay nil and `omitempty` drops the keys — so a
   secrets-on export from a server with no token contains no `hf_token` key
   at all, an empty string is never emitted, and restore (which skips
   present-but-empty secrets defensively, phase 02) can never blank a
   target's credentials. The preview's secrets badge (phase 04) keys on a
   present, non-empty value.
2. `Assemble(cfg *config.Config, b *builder.Builder, reg *models.Registry, gpus []monitor.GPUInfo, includeSecrets bool) File`:
   - Settings from cfg preference fields; HFToken/APIKey only when
     `includeSecrets`.
   - RuntimeEnv from `cfg.RuntimeEnv` + `cfg.RuntimeEnvExtra`.
   - FlagPresets from `b.FlagPresets("")` (already name-sorted).
   - ModelConfigs: for every registry model with a config, emit
     `{m.ModelID, m.Quant, m.Filename, *cfg}` — skip models with no config.
     **Dedupe by identity**: duplicate registrations of the same
     (ModelID, Quant) exist in practice (DeduplicateModels exists because
     of them) and `Registry.List` order is not stable across calls, so
     emit exactly one entry per identity, chosen deterministically as the
     duplicate with the lexicographically smallest registry ID. Then sort
     entries by (ModelID, Quant) — determinism now holds even with
     duplicates present.
   - SourceInfo from cfg + gpus.
3. Path relativization: in the exported copy of each ModelConfig, rewrite
   all three machine-local GGUF path fields — `MmprojPath`, `MtpPath`, and
   `DraftModelPath` — via `filepath.Rel(cfg.ModelsPath(), path)`; a path counts as **inside** the models dir only when `Rel`
   succeeds AND the result neither is `..` nor starts with `../` — anything
   else (Rel error, or a `..`-escaping result) exports the original
   absolute path verbatim. The export therefore emits only two shapes:
   clean relatives (inside) and absolutes (outside), which is exactly the
   dispatch phase 02's `models.ResolveConfigPaths` switches on with
   `filepath.IsAbs`. The import-side rule is defined once there (in
   `internal/models`, so the registry can reuse it at claim time, phase 03).
   Never mutate the registry's structs — copy.
4. Marshal with `json.MarshalIndent`; determinism comes from sorted slices
   (step 2) and Go's stable struct-field order. Two exports of unchanged
   state must be byte-identical except `exported_at` — keep `exported_at` as
   the sole timestamp, at the top.
5. `handleBackupExport` in `internal/api/backup.go`: reads `?secrets=1`,
   locks `cfgMu` for the config read, calls `Assemble`, writes with
   `Content-Type: application/json` and `Content-Disposition: attachment;
   filename=llama-toolchest-backup-<YYYY-MM-DD>.json`.
6. Register the route in the `/api` group. Note on auth: `/api` routes are
   same-origin UI routes with no API-key check (`apiKeyAuth` guards `/v1`
   only), which is what makes the plain-link browser download in phase 04
   work; a secrets-bearing export has the same exposure as the existing
   settings API, and the phase 04 help text states this.

## Build gate

`go build ./... && go vet ./internal/... && go test ./internal/...`

## Test plan

- `TestAssembleShape`: a populated fake config/builder/registry produces all
  four sections with correct identities and no registry IDs or absolute
  model paths in `model_configs`.
- `TestAssembleNoSecretsByDefault`: marshal without `includeSecrets`, assert
  the byte output contains neither the token nor the key strings; with the
  flag, both appear under `settings`.
- `TestAssembleDeterministic`: two assembles of the same state marshal
  byte-identical after normalizing `exported_at`.
- `TestAssemblePathRelativization`: an MmprojPath under the models dir
  exports relative; one outside exports verbatim.
- Manual: `curl localhost:3000/api/backup > b.json` on the dev box; inspect.

## Commit

`feat(backup): versioned configuration export engine and GET /api/backup`

## Rollback

Delete `internal/backup/`, `internal/api/backup.go`, and the route line.
No stored state or existing behavior is touched; fully safe to revert alone.
