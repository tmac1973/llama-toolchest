# Phase 02 — Profile storage and registry API

**Depends on:** 01 · **Enables:** 03 (profile bar UI), 04 (benchmarks record
the profile), 09 (autoconfigure saves a profile), 12 (autotune saves profiles)

## Goal
Store named snapshots of a model's config (profiles) and give the registry
methods to save, list, restore and delete them. This ports vllm-toolchest's
`internal/models/profiles.go`, adapted to llama-toolchest's `ModelConfig`,
which has pointer and slice fields and so cannot be compared with `==`. The
phase also adds provenance fields that autoconfigure and autotune fill in
later.

## Files touched
- `internal/models/profiles.go` (new): the `ConfigProfile` type, name rules,
  the registry methods, and the drift comparison.
- `internal/models/profiles_test.go` (new).
- `internal/models/registry.go`:
  - add `Profiles []ConfigProfile` (`json:"config_profiles"`) to the envelope;
  - add `ActiveProfile string` (`json:"active_profile,omitempty"`) to
    `ModelConfig`;
  - bump `RegistrySchemaVersion` to 2.
- `internal/backup/backup.go`:
  - export profiles as `Profiles []ProfileExport` on the backup file;
  - restore them by merging (a profile with the same model identity and
    folded name is replaced; others are added).
- `internal/backup/backup_test.go`: a profile survives an export → restore
  round trip.

## Steps
1. **The profile type.**
   ```go
   type ConfigProfile struct {
       ModelID string      `json:"model_id"`   // registry ID (<safeRepo>--<safeFilename>)
       Name    string      `json:"name"`
       Config  ModelConfig `json:"config"`     // Enabled and Aliases always stored zeroed
       SavedAt time.Time   `json:"saved_at"`
       BuildID string      `json:"build_id,omitempty"` // active llama.cpp build when saved
       Source  string      `json:"source"`     // "user" | "autoconfig" | "autotune"
       Notes   []ProfileNote `json:"notes,omitempty"`     // per-setting reasons (autoconfig) — filled in Phase 09
       Measured *ProfileMeasurement `json:"measured,omitempty"` // autotune results — filled in Phase 12
   }
   type ProfileNote struct{ Field, Reason, Origin string }
   // Origin: "hardware fit" | "model card" | "publisher preset" | "model file" | "default" | "autotune"
   type ProfileMeasurement struct {
       Workload string; PPTokPerSec, TGTokPerSec, ResponseSec float64
       BaselinePP, BaselineTG, BaselineResponseSec float64
       Goals []string; AutotuneID string
   }
   ```
2. **Names.**
   - `NormalizeProfileName` trims the name and collapses runs of whitespace.
   - `profileKey` also lowercases it; two names with the same key are the
     same profile.
   - `MaxProfileNameLen = 64`. The errors `ErrProfileName` and
     `ErrProfileNotFound` are copied from vllm-toolchest.
3. **`profileFields(cfg ModelConfig) ModelConfig`** returns a deep copy with
   `Enabled=false`, `Aliases=nil` and `ActiveProfile=""`. Deep means the
   `Aliases` slice, the `ReasoningOverride` pointer and the sampling pointers
   are copied, not shared.
4. **`ProfileEqual(a, b ModelConfig) bool`** compares `profileFields(a)` and
   `profileFields(b)` with `reflect.DeepEqual`.
   - A test builds two configs that differ in a single field, for every field
     of `ModelConfig` via reflection, and asserts `ProfileEqual` is false.
   - Another test asserts that differences only in `Enabled` or `Aliases`
     compare equal. This keeps the check correct when fields are added later.
5. **Registry methods.** Each takes `r.mu` and calls `writableLocked()` first.
   - `SaveProfile(modelID, name, source string, buildID string) (replaced bool, err error)`
     copies the live config through `profileFields` and sets
     `cfg.ActiveProfile = name`.
   - `SaveProfileFrom(modelID, name string, p ConfigProfile) (replaced bool, err error)`
     saves a profile built elsewhere (autoconfigure/autotune) without
     touching the live config.
   - `Profiles(modelID string) []ConfigProfile`, sorted by folded name.
   - `GetProfile(modelID, name string) (ConfigProfile, error)`.
   - `ApplyProfile(modelID, name string) error`:
     - copies the profile's fields over the live config;
     - keeps the live `Enabled` and `Aliases`;
     - sets `ActiveProfile`;
     - runs `NormalizeSpec`.
   - `DeleteProfile(modelID, name string) error`. If the deleted profile was
     active, clear the label and leave the config as it is.
   - `ActiveProfileState(modelID string) (name string, edited bool)`:
     `edited` is `!ProfileEqual(live, profile.Config)`. If the named profile
     no longer exists, return `""`.
6. **Where profiles live.** Profiles sit on the envelope, not on `Model`.
   They are not pruned when `Delete`/`Remove` removes a model, because a
   re-download produces the same registry ID and should find them again.
7. **`SetConfig` keeps the label.**
   - `SetConfig` must keep `ActiveProfile` when the incoming config leaves it
     empty. The config form never posts it, so it would otherwise be erased
     on every autosave.
   - Copy the existing value across in `SetConfig` before storing.
8. **Backup.**
   - Add `Profiles` to the backup file.
   - Identity on restore is `(ModelID, folded name)`, and the profile's
     `ModelID` is mapped through the same `(repo, quant)` resolution that
     `assembleModelConfigs` uses.
   - A profile whose model is not installed becomes pending, and is kept on
     the envelope as it is. It is harmless, because profiles are never pruned.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- Save, list, get, apply and delete round trip, with a temp-dir registry
  (`models.NewRegistry(t.TempDir(), "/models")`).
- Saving "Long Ctx" over "long  ctx" returns `replaced=true` and leaves one
  entry.
- An autosave via `SetConfig` with an empty `ActiveProfile` keeps the label.
- `ActiveProfileState` reports `edited=true` after one field changes.
- Applying a profile keeps the live `Aliases` and `Enabled`.
- Deleting the model keeps its profiles. Re-adding a model with the same ID
  lists them again.
- A read-only registry refuses `SaveProfile`.
- Backup round trip, including a profile for a model that is not installed.
- An old `models.json` (schema 1, no profiles) loads and is saved as schema 2.

## Commit
`feat(models): store named config profiles per model`

## Rollback
Revert the commit. A build with Phase 01 but not this phase refuses to write
a schema-2 file (the gate works as intended). To go back fully, also remove
`config_profiles` from `models.json` and set `schema_version` back to 1.
