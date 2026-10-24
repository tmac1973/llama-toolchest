# Phase 02 — Feature Switch, Settings and Generation Store

**Depends on:** nothing · **Enables:** Phases 03–10 (every later phase reads
the switch, the ports and the store)

## Goal

Add the settings that turn generation backends on, and the port range for their
web UIs. Add a separate JSON store that will hold model sets and backend
instances.

With the switch off, which is the default, nothing visible changes and no new
files are created.

Generation data lives in its own file and its own model directory, for two
reasons:
- The LLM registry stays untouched.
- `Registry.ScanModels` never registers diffusion or audio GGUFs as LLMs.

## Files touched

- `internal/config/config.go`: new config fields and their defaults in `Load`,
  and an environment override.
- `internal/api/settings.go`: handle the new fields in `handleUpdateSettings`
  (JSON and form branches) and in the settings GET response.
- `internal/api/server.go`: pass the new fields to the settings page data in
  `handleSettingsPage` (L906-951).
- `web/templates/settings.html`: a "Generation backends" section with the
  switch, the port range start and the port count. Each has a tooltip.
- `packaging/config/llama-toolchest.yaml.example`: document the new keys.
- `internal/generation/store.go` (new): the `Store` type, persisted to
  `<data>/config/generation.json`.
- `internal/generation/types.go` (new): the `ModelSet`, `Component`, `Instance`
  and `BackendKind` types.
- `internal/generation/store_test.go` (new).
- `internal/api/server.go`: create the store in the server constructor and add
  it as `Server.gen`.

## Steps

1. **Add four fields to `Config`:**

   | Field | yaml key | Default | Meaning |
   |---|---|---|---|
   | `GenerationEnabled bool` | `generation_enabled` | `false` | |
   | `GenPortStart int` | `gen_port_start` | `8081` | Port the backends listen on |
   | `GenPortCount int` | `gen_port_count` | `8` | |
   | `GenPublicPortStart int` | `gen_public_port_start` | `0` | First port as seen from outside. `0` means the same as `GenPortStart`. Phase 09 builds web UI links from it, so a container whose host ports differ still gets working links. |

   Set the defaults in `Load`. In `applyEnvOverrides`, next to
   `LLAMA_TOOLCHEST_EXTERNAL_URL`, add one environment override:
   `LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START` sets `GenPublicPortStart`.
   There is no environment override for `GenPortStart`.

2. **Validate the port range in `handleUpdateSettings`:**
   - The start must be between 1024 and 65535.
   - The count must be between 1 and 32.
   - The range must not include `LlamaPort` or the listen port.

   **In a container** (`builder.RunningInContainer()`), `gen_port_start` and
   `gen_port_count` are shown read-only at 8081 and 8, and updates to them are
   rejected. Those are the ports the image publishes (Phase 10).

   Use the `*_touched` pattern for the checkbox, as `auto_start_touched` does.
   Return the same validation messages through the existing error partial.

3. **Define the types in `internal/generation/types.go`.**

   **`BackendKind`** is a string constant with two values:
   - `"sd"` (stable-diffusion.cpp)
   - `"ace"` (acestep.cpp)

   **`Component`** fields:

   | Field | Type | Notes |
   |---|---|---|
   | `Slot` | `string` | sd recipe slot ID such as `diffusion_model`, `llm`, `tokenizer`, `vae`, `model`; for ace, `lm`, `dit`, `embedding`, `vae` |
   | `Repo` | `string` | HF repo, empty for custom sets |
   | `File` | `string` | Path within the repo |
   | `Path` | `string` | Absolute local path once downloaded |
   | `Bytes` | `int64` | |

   **`ModelSet`** fields:

   | Field | Type | Notes |
   |---|---|---|
   | `ID` | `string` | Slug, unique |
   | `Name` | `string` | |
   | `Kind` | `BackendKind` | |
   | `CatalogID` | `string` | Empty for custom sets |
   | `Recipe` | `string` | sd-server recipe ID, e.g. `ming-image`; empty for ace |
   | `Components` | `[]Component` | |
   | `LaunchArgs` | `[]string` | Extra backend flags, e.g. `--sigmas qwen-image-2.1-turbo` |
   | `Capabilities` | `[]string` | e.g. `"alpha"`, `"edit"` |
   | `License` | `string` | |
   | `DefaultWidth` | `int` | Image size used by the measured test run; 0 for ace |
   | `DefaultHeight` | `int` | Image size used by the measured test run; 0 for ace |
   | `Complete` | `bool` | True once every component file is present; catalog downloads set it in Phase 04 |
   | `Created` | `time.Time` | |

   **Model set IDs:**
   - A catalog set's ID equals its catalog ID, e.g. `ming-image`.
   - A custom set's ID is the lowercase kebab-case slug of its name.
   - `PutModelSet` rejects an ID that is already taken.

   **`Instance`** fields:

   | Field | Type | Notes |
   |---|---|---|
   | `ID` | `string` | Lowercase kebab-case slug of `Name`; unique; rejected by `PutInstance` if taken |
   | `Name` | `string` | Shown in the UI; also accepted as the API `model` value through the ID |
   | `ModelSetID` | `string` | |
   | `BuildID` | `string` | Empty means the newest successful build for the kind |
   | `GPU` | `int` | Monitor index |
   | `Port` | `int` | 0 means "assign from the range" |
   | `ReservedMiB` | `int` | 0 means "not measured yet" |
   | `MeasuredPeakMiB` | `int` | |
   | `MeasuredAt` | `time.Time` | |
   | `MeasureEstimated` | `bool` | True when the peak came from whole-GPU usage, because there was no per-process data (Vulkan) |
   | `ExtraArgs` | `[]string` | Extra backend flags for this instance, added after the set's `LaunchArgs` |
   | `Enabled` | `bool` | |
   | `AutoStart` | `bool` | |

   **`Enabled` vs `AutoStart`:**
   - **`Enabled`** is true by default when an instance is created. An enabled
     instance's reservation always counts against its GPU, even while the
     instance is stopped, so starting it later can never fail for lack of VRAM.
     A disabled instance cannot run and holds no reservation.
   - **`AutoStart`** is the instance's own setting. It is separate from the
     router's `Config.AutoStart`.

4. **Write `Store` in `internal/generation/store.go`:**
   - Methods: `Load(dataDir)`, `Save()` (atomic write through
     `atomicfile.WriteMode(path, 0600)`), `ModelSets()`, `ModelSet(id)`,
     `PutModelSet`, `DeleteModelSet` (refused while an instance uses the set),
     `Instances()`, `Instance(id)`, `PutInstance`, `DeleteInstance`, and
     `AssignPort(start, count)`.
   - `AssignPort` returns the lowest port in the range that no instance uses,
     or an error when every port is taken.
   - File format: `{"schema_version":1,"model_sets":[…],"instances":[…]}`.
     A missing file means an empty store.
   - A `sync.RWMutex` guards all access.

5. **Add `generation.ModelsDir(cfg)`.** It returns
   `<ModelsPath()>/../gen-models` when `ModelsDir` is set, and
   `<data>/gen-models` otherwise. It creates the directory only when it is
   first needed. Document this path in the yaml example.

6. **Create the store only when it is needed.** `Server.gen` is built at
   startup. `Load` only reads `generation.json`. Nothing is written until the
   first `Put*`, so installs with the switch off gain no files.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/config/... ./internal/generation/... ./internal/api/...`

## Test plan

- **`store_test.go`:**
  - A save/load round trip.
  - `AssignPort` skips ports already in use and errors when the range is full.
  - `DeleteModelSet` is refused while an instance references the set.
  - A missing file loads as empty.
  - A file with an unknown `schema_version` gives a clear error.
- **Settings tests,** next to the existing settings tests in `internal/api`:
  - The switch saves and reloads.
  - A port range that overlaps `LlamaPort` is rejected.
  - Without the new keys, an existing config loads with the defaults.
- **Manual:** with the switch off, the settings page shows the section
  collapsed with the switch off, and `<data>/config/generation.json` does not
  exist.
- **Container:** the port fields are read-only.
- **Store:** `PutModelSet` with an existing ID is rejected.

## Commit

`feat(config): add generation backend switch, port range and store`

## Rollback

Revert the commit. Old configs never contained the new yaml keys, so a reverted
binary that reads a newer config just ignores them, because the yaml loader
skips unknown fields. `generation.json` is left on disk unused.
