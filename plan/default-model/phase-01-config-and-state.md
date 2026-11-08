# Phase 01 — Config fields and the last-used state file

**Depends on:** nothing · **Enables:** Phase 02 (recording the last-used model),
Phase 03 (resolution), Phase 05 (Settings card)

## Goal

Add the saved settings for the virtual model name (on/off, name, pin) and a
small state file that remembers the chat model used last across restarts. No
request behavior changes in this phase. It ships the data the later phases read
and write, with backup support, so each later phase only adds behavior.

## Files touched

- `internal/config/config.go` — new `DefaultModel DefaultModelConfig` field,
  the `DefaultModelConfig` type, the `DefaultModelName`, `DefaultModelOn` and
  `ValidDefaultModelName` helpers, and the `DefaultModelForkPrefix` constant.
- `internal/config/config_test.go` — tests for the helpers and the YAML round
  trip.
- `internal/api/default_model_state.go` (new) — the `defaultModelState` type
  that loads and saves `<DataDir>/default-model.json`.
- `internal/api/default_model_state_test.go` (new) — load/save tests.
- `internal/api/server.go` — a `defaultState *defaultModelState` field on
  `Server`, loaded once at server start.
- `internal/backup/backup.go`, `internal/backup/restore.go` — export and apply
  the three settings.
- `internal/api/backup.go` — `ApplySettings` sets the new fields under
  `cfgMu`, as it does for `eviction`.
- `internal/backup/backup_test.go`, `internal/backup/restore_test.go` — cover
  the new fields.

## Steps

1. In `internal/config/config.go`, add to `Config`:

   ```go
   // DefaultModel is the virtual model name clients can send instead of a
   // real one; see DefaultModelConfig.
   DefaultModel DefaultModelConfig `yaml:"default_model,omitempty"`
   ```

   and the type:

   ```go
   // DefaultModelConfig sets the virtual model name. It is on unless Off
   // is set, so a config from before the setting has it on.
   type DefaultModelConfig struct {
       Off  bool   `yaml:"off,omitempty"`
       // Name is what clients send; "" means DefaultModelNameDefault.
       Name string `yaml:"name,omitempty"`
       // Pin is the model the name always resolves to: a registry ID, or
       // DefaultModelForkPrefix + a fork instance ID. "" means no pin.
       Pin  string `yaml:"pin,omitempty"`
   }
   ```

   Constants: `DefaultModelNameDefault = "default"` and
   `DefaultModelForkPrefix = "fork:"`. All fields are values, so `Clone`
   needs no change; add a line to the Clone test that the struct is copied.
2. Add `func (c *Config) DefaultModelName() string` (trimmed `Name`, or
   `DefaultModelNameDefault` when empty) and `func (c *Config) DefaultModelOn()
   bool` (`!c.DefaultModel.Off`).
3. Add `func ValidDefaultModelName(name string) error`: after trimming, the
   name must be 1–64 characters of letters, digits and `. _ - : /`. It returns
   an error whose text says what is allowed. The clash check with real names
   needs the registry, so it lives in Phase 03, not here.
4. Create `internal/api/default_model_state.go`:

   ```go
   // defaultModelState remembers the chat model used last, so the virtual
   // model name has a target after a restart with nothing loaded. It is
   // kept in <DataDir>/default-model.json.
   type defaultModelState struct {
       path string
       mu   sync.Mutex
       last lastChatModel
   }
   type lastChatModel struct {
       ID     string    `json:"id"`     // registry ID or fork instance ID
       Fork   bool      `json:"fork,omitempty"`
       UsedAt time.Time `json:"used_at"`
   }
   ```

   Methods:
   - `loadDefaultModelState(path string) *defaultModelState` — reads the file;
     a missing or unreadable file gives an empty state and a `slog.Warn` for an
     unreadable one (never an error that stops the server).
   - `get() lastChatModel`.
   - `note(id string, fork bool, at time.Time)` — updates `last`. It writes the
     file only when `id` or `fork` differs from the saved value, so a run of
     requests to one model writes nothing. The write goes to a temp file in the
     same directory, then `os.Rename`, with mode `0o600`. A write error is
     logged with `slog.Warn` and the in-memory value is kept.
5. In `internal/api/server.go`, set `s.defaultState =
   loadDefaultModelState(filepath.Join(cfg.DataDir, "default-model.json"))`
   where the server is built.
6. Backup: add `DefaultModelOff *bool`, `DefaultModelName *string` and
   `DefaultModelPin *string` (JSON `default_model_off`, `default_model_name`,
   `default_model_pin`) to `backup.Settings`, fill them on export, and apply
   them in `restoreDeps().ApplySettings` with the same "changed" reporting as
   `eviction`. A restored name that fails `ValidDefaultModelName` is skipped
   with a warning in the restore report. The state file is runtime state and
   is not in the backup.

## Build gate

```bash
go build ./... && go vet ./... && go test ./internal/config/... ./internal/backup/... ./internal/api/...
```

## Test plan

- Config: a YAML file with no `default_model` key gives `DefaultModelOn() ==
  true` and `DefaultModelName() == "default"`; `name: "  "` also gives
  `"default"`; a saved config round-trips all three fields.
- `ValidDefaultModelName`: accepts `default`, `my-model:latest`, `team/chat`;
  rejects empty, a space, 65 characters, `ä`.
- State file: `note` twice with the same ID writes once (check the file's
  mod time or count writes through a test hook); a different ID writes again;
  `loadDefaultModelState` reads back what was written; a corrupt file gives an
  empty state.
- Backup: export includes the three fields; restore applies them and lists
  them as changed; an invalid name is skipped with a warning.

## Commit

`feat(config): add default model settings and last-used state file`

## Rollback

Revert the commit. Nothing reads the new fields yet, so a revert is safe. A
config written by this build keeps a `default_model` key, which an older build
ignores (unknown YAML keys are dropped on its next save). The
`default-model.json` file can stay; nothing older reads it.
