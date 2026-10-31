# Phase 02 — The forks package: curated list, recipes and store

**Depends on:** nothing · **Enables:** Phase 03 (builder projects from the curated list), Phase 04 (catalog download into the store), Phase 05 (instances launched from recipes)

## Goal

Create `internal/forks`, a package whose only toolchest import is `internal/atomicfile`. It must never import `internal/builder`, `internal/generation` (which imports the builder) or `internal/api`, because the builder imports it in Phase 03. Splitting the extra-arguments text is done in the API layer (Phase 05). It holds the curated fork list with Avalon as its only entry, the type for a fork's launch recipe and paired model files, and a store in `<data>/config/forks.json` for downloaded model sets and fork instances. Nothing uses it yet, so the phase ships with tests only.

## Files touched

- `internal/forks/types.go` — new: `Fork`, `Recipe`, `Needs`, `ModelFile`, `ModelSet`, `Instance`.
- `internal/forks/curated.go` — new: `Curated() []Fork` and `Find(id string) (Fork, bool)`, with the Avalon entry.
- `internal/forks/store.go` — new: `Store` with load/save and model set and instance methods, modelled on `internal/generation/store.go`.
- `internal/forks/launch.go` — new: `Args(f Fork, set ModelSet, inst Instance) []string` and `Env(f Fork, inst Instance, backend string, device int) []string`; `device` is the GPU as the backend numbers it (`models.BackendDeviceIndex`, computed by the caller).
- `internal/forks/needs.go` — new: `CheckNeeds(n Needs, host HostInfo) []string` that returns warning sentences.
- `internal/forks/*_test.go` — new tests for each file.

## Steps

1. `types.go`:
   - `Fork{ID, Name, Repo, PinnedSHA, PinnedDate string; Backends []string; Recipe Recipe; ModelFiles []ModelFile; ModelRepo, ModelName, License, Notes string; Needs Needs; SamplingDefaults map[string]any}`. `ID` is a lowercase slug ("avalon") and is also the builder project ID and the build ID prefix (`avalon-`).
   - `Recipe{Args []string; ModelFileArg string; DraftFileArg string; Env []string}`. `Args` are the fixed llama-server options. `ModelFileArg` is the path of the first GGUF shard inside the set, passed as `-m`; `DraftFileArg` is the draft model path, passed as `-md`, empty when there is none.
   - `Needs{MinRAMGiB int; AVX2 bool; ROCmArchs []string; CUDA bool; MinDiskGiB int}`.
   - `ModelFile{Path string; Bytes int64}`. `Path` may contain a subfolder (`MTP/...`).
   - `ModelSet{ForkID, Dir string; Complete bool}`: one per fork, keyed by fork ID. `Dir` is the folder that holds the set's files; Phase 04 sets it to `huggingface.FilePath(<models dir>/fork-models, ModelRepo, "")`.
   - `Instance{ID, ForkID, BuildID string; GPU, Port int; ExtraArgs []string; ExtraEnv []string; AutoStart bool}`. `ID` is both the display name and the API model name; there is no separate name field. Empty `BuildID` means the newest successful build of the fork. `ExtraArgs` is stored already split, like generation instances.
2. `curated.go`, the Avalon entry:
   - `Repo: "https://github.com/freriadoc/Avalon"`, `PinnedSHA` set to Avalon `main` HEAD on the day this phase is written (at planning time `ad2c6e750c7e…`, 2026-10-31; use the full 40-character SHA). Phase 08 confirms it on asmodean or moves it.
   - `Backends: {"rocm", "cuda"}`.
   - `ModelRepo: "freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF"`; `ModelFiles`: the 33 shards `Qwen3.8-Flash-Next-Avalon-IQ3KS-000NN-of-00033.gguf`, `MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf` (1,484,237,088 bytes) and `moe-cache-ranking.bin` (177,192 bytes). Take every shard size from the Hugging Face tree API once and write it in; a test checks that the sum is about 89.7 GiB.
   - `Recipe.Args`: `-c 262144 -b 9536 -ub 9536 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 --spec-type draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0`. `ModelFileArg` is the first shard; `DraftFileArg` is the MTP file.
   - `Needs{MinRAMGiB: 32, AVX2: true, ROCmArchs: {"gfx1100", "gfx1101", "gfx1102"}, CUDA: true, MinDiskGiB: 98}`.
   - `SamplingDefaults`: `temperature 1.0, top_p 0.95, top_k 20, min_p 0.0`, the thinking-mode values on the Qwen/Qwen3.8-Flash-Next model card (cite the card in a comment).
   - `License: "MIT (Avalon); Qwen Community License 1.0 (model)"`, and `Notes` with one sentence on what Avalon is for.
3. `store.go`: `Load(dataDir)`, `ModelSets()`, `ModelSet(forkID)`, `PutModelSet`, `DeleteModelSet`, `Instances()`, `Instance(id)`, `AddInstance`, `UpdateInstance`, `DeleteInstance`. One mutex, save on every change, same file layout rules as the generation store. `AddInstance` refuses an ID that is empty, has characters other than `[a-z0-9._-]`, or is already used.
4. `launch.go`:
   - `Args` returns `--host 127.0.0.1 --port <port> -m <dir>/<ModelFileArg>`, then `-md <dir>/<DraftFileArg>` when set, then `Recipe.Args`, then `ExtraArgs`. The API layer (Phase 05) splits the form's text into `ExtraArgs` with `generation.ValidateExtraArgs`, the same call generation instances use.
   - `Env` returns the GPU variable for the backend (`HIP_VISIBLE_DEVICES` for rocm, `CUDA_VISIBLE_DEVICES` for cuda) set to `device`, then `Recipe.Env`, then `ExtraEnv`. A later entry with the same name wins. `ExtraEnv` lines must be `NAME=value` with `NAME` matching `[A-Z_][A-Z0-9_]*`; others are refused when the instance is saved (validated in `store.go`).
5. `needs.go`: `HostInfo{RAMGiB int; AVX2Known, AVX2 bool; GPUArch, Backend string; FreeDiskGiB int}` (zero means unknown and gives no warning; callers leave `FreeDiskGiB` zero once the set is downloaded). `CheckNeeds` returns sentences such as "This host has 24 GiB of RAM. Avalon needs 32 GiB or more." One sentence per need that is not met. An empty list means all are met.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- `curated_test.go`: every curated fork has an ID matching `[a-z0-9-]+`, a 40-character `PinnedSHA`, at least one backend, at least one model file, and `ModelFileArg`/`DraftFileArg` that are in `ModelFiles`. Avalon's file sizes add up to between 89 and 90.5 GiB.
- `store_test.go`: round trip through `forks.json`; duplicate and bad instance IDs refused; bad `ExtraEnv` lines refused.
- `launch_test.go`: `Args` for Avalon has `-m` and `-md` with full paths, the recipe options, and extra arguments at the end; `Env` sets `HIP_VISIBLE_DEVICES=1` for rocm GPU 1, and an `ExtraEnv` entry overrides a recipe entry of the same name.
- `needs_test.go`: each failing need gives its sentence; a host that meets all gives none.

## Commit

`feat(forks): add the curated forks package with Avalon`

## Rollback

Revert the commit. The package is new and nothing imports it yet.
