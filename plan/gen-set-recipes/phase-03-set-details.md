# Phase 03 — Set details and the command preview

**Depends on:** Phase 02 (flag reasons, slot labels) · **Enables:** Phase 04 (the Details dialog and routes render this data)

## Goal

Build, in `internal/generation`, one function that describes how a set runs: its recipe, each slot with its label and file, each flag with its reason, and the command line. It works for a set on this machine and for a catalog entry that is not downloaded. The command comes from the same code that `LaunchSpec` uses, so the preview cannot drift from what really runs. No routes or UI yet.

## Files touched

- `internal/generation/launch.go` — move the argument building out of `LaunchSpec` into a pure function that both `LaunchSpec` and the preview call.
- `internal/generation/details.go` (new) — `SetDetails` type and the `DetailsForSet` and `DetailsForCatalog` functions.
- `internal/generation/details_test.go` (new) — tests.
- `internal/generation/launch_test.go` — keep the existing `LaunchSpec` tests passing; add one that compares `LaunchSpec` args with the preview args.

## Steps

1. In `launch.go`, add `type argPaths struct { Host, Port, MaxVRAM, LoraDir, UpscalerDir, AceModelsDir string }` and `func buildArgs(set ModelSet, extra []string, p argPaths) (args []string, fixed [][2]string, err error)`. `fixed` lists each flag and value that toolchest adds itself (`--listen-ip`, `--listen-port`, `--lora-model-dir`, `--max-vram`, `--hires-upscalers-dir`, `--host`, `--port`, `--models`, `--max-batch`), in the order they appear in `args`; slot flags are not in it. It holds the per-kind `switch` from `LaunchSpec` that builds `args`: slot flags, `--lora-model-dir`, `RunArgs()`, `extra`, `--max-vram` when `p.MaxVRAM != ""`, the upscaler and ace fixed flags. It has no side effects: no `MkdirAll`, no links. It still returns the errors `LaunchSpec` returns today for an unknown slot or kind.
2. `LaunchSpec` computes the real values (host, port, `maxVRAMGiB(...)`, `LoraDir(set)`, the instance's `upscalers` and `models` folders), makes the folders and links as it does today, and then calls `buildArgs`. Its result must be the same as before for every kind.
3. In `details.go`, add:

   ```go
   type SlotDetail struct {
       Slot, Label string
       Required    bool
       Path        string // the file on this machine, or where a catalog file will be saved
       Repo, File  string // Hugging Face source; empty for a custom set
       Bytes       int64
       Present     bool   // the file is on disk now
   }
   type FlagDetail struct {
       Flag, Value, Reason string
       Fixed               bool // toolchest adds it; not part of the set's flags
   }
   type SetDetails struct {
       ID, Name, Kind, Recipe string
       CatalogID              string
       Custom                 bool
       Program                string       // "sd-server" or "ace-server"
       Slots                  []SlotDetail
       Flags                  []FlagDetail // the set's flags: RunArgs or the entry's LaunchArgs
       FixedFlags             []FlagDetail // the flags toolchest adds for each instance
       Command                string       // one line, shell-quoted
       Notes, License         string
   }
   ```
4. `DetailsForSet(set ModelSet) SetDetails` and `DetailsForCatalog(e CatalogEntry, modelsDir string) SetDetails`. For a catalog entry, each slot's `Path` is `huggingface.FilePath(modelsDir, c.Repo, c.File)`, the same path the downloader uses, and `Present` comes from `os.Stat`. `Required` comes from the recipe's `RecipeSlotSet` (or `AceSlots`, or the upscaler slot); for a `OneOf` group, mark each member not required and add the label suffix "(one of: <slots>)", built from the group's slot names joined with ", " (today every group is `vae, tae`).
5. Build the command with `buildArgs` and these markers in `argPaths`: `Host: "<listen address>"`, `Port: "<port>"`, `MaxVRAM: "<vram limit>"`, `UpscalerDir: "<instance folder>/upscalers"`, `AceModelsDir: "<instance folder>/models"`. `LoraDir` is the real folder from `LoraDir(set)`. Prefix the program name. Quote words that hold spaces or shell characters, and leave the `<...>` markers unquoted. No instance extra args are added: the dialog notes that each instance can add its own.
6. Fill `Flags` from `SplitFlags(set.RunArgs())` (or `e.LaunchArgs`) with `FlagReason`, and `FixedFlags` from the `fixed` pairs that `buildArgs` returns, with their reasons from `FlagReasons`. Slot flags (`--diffusion-model` and so on) are not listed as flags: the slot table already shows them.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/generation/...`

## Test plan

- `LaunchSpec` gives the same args as before this phase for an sd set, a video set with LoRAs, an upscaler set and an ace set (golden slices in the test).
- For a set and instance, the `LaunchSpec` args equal the preview args after the markers are replaced with the instance's real host, port, VRAM limit and folders.
- `DetailsForCatalog` for `flux3-image-klein` with a temp models folder: three slots with labels, paths under `gen-models/`, `Present` false; every flag has a reason; the command starts with `sd-server --listen-ip <listen address> --listen-port <port> --diffusion-model ` and contains `--max-vram <vram limit>`.
- `DetailsForSet` for a custom `wan22-i2v` set with LoRAs: `--lora-model-dir` shows the real folder and is a fixed flag; the two LoRA slots are listed with labels.
- `DetailsForCatalog` for `ace-step-xl-sft`: program `ace-server`, the four slots, fixed flags `--max-batch 1` and `--models <instance folder>/models`.
- A path with a space is quoted in the command.

## Commit

`feat(gen): describe how a set runs, from the same code that launches it`

## Rollback

Revert the commit. The `LaunchSpec` refactor is the only change to running code; the golden-args tests guard it. Nothing is stored.
