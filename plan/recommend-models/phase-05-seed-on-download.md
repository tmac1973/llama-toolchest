# Phase 05 — Fill in the config on download

**Depends on:** 03 (`PlanFit` inputs and `Suggested`), 04 (downloads from
the feed carry the chosen context)

## Goal
When a model downloaded from the feed finishes, set its starting config
from the same plan the card showed, instead of the default 8K context. Show
the user where the settings came from. This matches vllm-toolchest's phase
19, including the fixes it needed afterwards (its PR #81).

## Background
- `onDownloadComplete` (`internal/api/hf.go:435`) parses the local GGUF,
  then calls `registry.Add`. That creates the default config
  (`internal/models/registry.go:512-521`) unless a config restored from a
  backup is waiting for this model (`claimPendingLocked`).
- Downloads made for Autoconfigure's helper model are claimed by
  `claimDownloadedHelper`, and must not be seeded.
- `PlanFit` returns a complete config with notes (`FitResult.Config`,
  `FitResult.Notes`, origin "hardware fit").

## Files touched
- `internal/api/hf.go`: carry `ctx` and `from=recommend` from the download
  request; seed in `onDownloadComplete`.
- `internal/huggingface/downloader.go`: store the two values on the
  in-memory download record. Downloads are not kept across a restart
  today, so neither is this.
- `internal/models/registry.go`: a `ConfigSource` field on `Model`, and a
  way to tell whether a config is still the default.
- `internal/api/` config update and profile apply paths: clear the mark.
- `web/templates/partials/model_config.html`: the note.
- Tests.

## Steps
1. **Carry the class with the download.** `handleHFDownload`
   (`hf.go:136`) accepts `ctx` and `from=recommend`, and stores them on the
   download record.
   - Only files started from a feed-opened Details panel carry them. A
     download from a plain search is never seeded.
2. **Decide whether to seed.** In `onDownloadComplete`, after
   `registry.Add`, seed only when all of these hold:
   - the download carries `from=recommend` and a class;
   - the model is a main model file, not an mmproj file or an MTP head
     (these already return early);
   - the model is not a helper download;
   - the config is still exactly the default one that `Add` created. A
     config restored from a backup must never be overwritten.
3. **Plan from the real file.** Run
   `models.PlanFit(m, current config, s.hardware(), class)` on the
   registered model. The full local parse is now available, so this is more
   exact than the feed's plan.
   - If the plan does not reach the class's context fully on the GPUs (or
     with experts in RAM for an MoE model), seed anyway with what it gives,
     and say so in the note (step 5). The hardware may have changed since
     the feed was built.
   - If `Fits` is false, do not seed. Leave the default config, and log
     the reason.
4. **What is copied.** From `FitResult.Config`: `ContextSize`,
   `KVCacheQuant`, `GPULayers`, `CPUMoE`, `GPUAssign`, `TensorSplit`,
   `SplitMode`, `MainGPU`, `Threads` and `FlashAttention`. Nothing else
   changes. Sampling values still come from the presets
   (`enrichModelPresets`). `MmprojPath` and `MtpPath` are still set by the
   auto-association code, which runs after this step.
   - Order matters. The mmproj and MTP association in `onDownloadComplete`
     edit the config, so seeding must run before them, and those edits must
     not clear the mark (step 6).
5. **Mark and note.**
   - Set `Model.ConfigSource = "recommended"` and record the class and
     time.
   - The model's config panel shows: "Settings suggested for this machine
     when this model was downloaded from the recommendations (32K
     context). Autoconfigure and Autotune can refine them." The plan's
     notes are shown under it, in the same style as Autoconfigure's review.
   - When step 3 had to give less than the class: "The suggested 128K
     context did not fit any more, so 64K was set."
6. **Clear the mark** on the first change the user makes:
   - a config update through the form or the API;
   - applying a saved profile, or an Autoconfigure or Autotune result;
   - restoring from a backup.

   The automatic mmproj and MTP association does not clear it.
   vllm-toolchest's PR #81 found that applying a profile had been missed.
7. **Nothing else changes for downloads not from the feed.**

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- A download with `from=recommend&ctx=32k` that completes: the config has
  the planned context, KV type and placement, and the mark is set.
- A download without the parameters: the default config, no mark.
- A config restored from a backup and waiting for the model: not
  overwritten, no mark.
- A helper download: not seeded.
- An MoE model planned with experts in RAM: `CPUMoE` is set and threads
  follow `ThreadsFor`.
- A plan that falls short of the class: seeded with the shorter context, and
  the note says so.
- A plan that does not fit at all: the default config, no mark.
- The mark is cleared by a config update and by applying a profile. It is
  not cleared by mmproj association.
- Render test for the note on the config panel.

## Commit
`feat: models downloaded from the recommendations start with planned settings`

## Rollback
Revert the commit. Existing marks in `models.json` are ignored by older
code. Configs already seeded stay as they are, as ordinary configs.
