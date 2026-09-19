# Model Profiles, Autoconfigure and Autotune — Project Overview

## Problem
llama-server has dozens of launch settings, and the ones that matter most for
speed are the hardest to understand: batch and ubatch sizes, flash attention,
split mode, and speculative decoding in its many forms (a draft model, MTP,
EAGLE3, DFlash, five n-gram modes, and combinations such as MTP together with
an n-gram assist). A user who is not an LLM expert today has to learn all of
this, try settings by hand, and remember what worked. llama-toolchest keeps
only one live config per model (`models.json` → `configs[id]`), autosaved on
every change, so a good config is lost the moment it is edited.

The benchmark sweeps can already measure most of these settings, but nothing
turns a result back into a config, and nothing reads a model's documentation
to suggest a starting point. There is also a data-loss bug underneath: if
`models.json` fails to parse, `Registry.load()` logs the error and the next
`save()` writes an empty registry over the file
(`internal/models/registry.go:1446-1490`).

## Goals
- **Saved profiles.** A user can save a model's current config under a name,
  restore it later, and delete it. Behavior matches vllm-toolchest's config
  profiles (`vllm-toolchest/plan/phase-11-config-profiles.md`), so both apps
  work the same way.
- **Safe storage.** `models.json` gets a schema-version gate and an atomic
  write, so a corrupt file or a file from a newer build is never overwritten.
- **Autoconfigure.** One button produces a sensible starting profile for a
  model:
  - code rules make the model fit the hardware;
  - a small helper LLM reads the model card for model-specific advice.
  The user reviews each setting, with a plain-language reason, before saving.
- **Autotune.** One job measures the lossless speed settings from a starting
  profile the user picks, and saves up to three profiles: fastest generation,
  fastest prompt processing, and fastest full response.
- **Less obvious combinations are tried.** Both features consider speculative
  decoding setups a beginner would not think of, in particular a draft method
  together with an n-gram assist (e.g. MTP + ngram-mod).
- **Plain language.** Every setting and result is explained in plain language
  that non-native English speakers can follow, with tooltips and visible
  "how to read this" notes.

## Non-goals
- Profiles shared across models, or a global profile. Profiles belong to one
  model.
- Autotune changing any setting that changes the model's output. Context
  size, KV cache type, sampling values, the main model file and the mmproj
  file come from the starting profile and are never altered. Autotune may add
  an already installed draft model or MTP head, because speculative decoding
  does not change the answers.
- Running autoconfigure without being asked. Nothing loads the helper LLM
  unless the user clicks.
- Downloading draft models or heads without the user's approval.
- Tuning across builds. Autotune uses the active llama.cpp build.
- Tuning several models in one autotune job.
- Quality benchmarks (perplexity, KL-divergence) inside autotune.
- Changing vllm-toolchest.

## Users & primary flow
Users: people running llama-toolchest on their own GPU machine, many of them
not LLM experts. The same controls also save time for experts.

1. The user downloads a model. The model card shows a hint: "Autoconfigure
   can suggest settings for this model."
2. The user clicks **Autoconfigure**. If no helper model is set, they are
   offered a one-click download of Qwen3.5-4B (Q4_K_M, about 3 GB). Settings
   lets them pick any installed model as the helper instead.
3. Autoconfigure:
   - reads the GGUF metadata and the hardware, and computes the settings that
     must fit (context size, GPU layers, MoE expert offload, KV cache type)
     with the existing VRAM estimator. Batch sizes stay at llama.cpp's
     defaults, because Autotune measures them;
   - fetches the model card (the GGUF repo's README and the base model's
     README) and asks the helper LLM for model-specific advice as structured
     JSON (sampling, reasoning, speculative method, special flags);
   - checks every value against the allowed values.
4. A review screen lists each proposed setting beside the current value, with
   a one-line reason and its source (model card, hardware fit, default).
   Draft files named in the card but not installed are listed with a size and
   a Download button.
5. The user clicks **Save as profile** (saved as "Autoconfig") or **Save and
   apply** (also becomes the live config).
6. Later the user starts **Autotune** and picks:
   - a starting profile (Autoconfig or any profile of their own);
   - a use case: General chat, Coding / editing, or Mixed.
7. Autotune runs as an ordinary benchmark job, in stages:
   - (1) batch/ubatch and flash attention, plus split mode when more than one
     GPU is assigned and threads when layers run on the CPU;
   - (2) speculative decoding: each installed draft method alone, each n-gram
     assist alone, and draft + n-gram pairs;
   - (3) draft length and n-gram parameters for the finalists;
   - (4) a confirmation grid that measures side by side the top 2 stage-1
     settings × the finalists of stages 2–3 (whose stage-2 choice is already
     included), plus the starting profile.
8. The results screen shows each winning profile's speed-up against the
   starting profile. Autotune saves up to three profiles ("Autotune – fastest
   generation", "– fastest prompt", "– fastest response"). A profile that wins
   more than one goal is saved once, and its name lists every goal it won.
9. The user restores whichever profile suits them from the profile bar.

## Constraints
- **Stack.**
  - Go with the chi router;
  - `html/template` + htmx + Pico CSS on the web side, with no JS framework;
  - JSON files under `<data_dir>/config/` for storage, with no database.
  New code follows these patterns.
- **Profile contents.** A profile holds every `ModelConfig` field except
  `Enabled` and `Aliases`. `ModelConfig` has pointer and slice fields, so
  checking whether the live config has drifted from its profile needs a
  field-aware comparison, not `==`.
- **Profiles live on the `models.json` envelope,** not on `Model`, so that a
  re-download keeps them. They are not deleted with the model.
- **The profile bar sits outside the autosaving config form** in
  `web/templates/partials/model_config.html`.
- **Backup/restore** (`internal/backup/backup.go`) must include profiles.
- **Autotune reuses the existing machinery:** the benchmark job queue, sweep
  axes (`internal/benchmark/sweep.go`) and ephemeral configs
  (`internal/api/jobs_env.go`). The job queue runs one job at a time and
  restarts the router for each cell, so the server is unavailable for normal
  use while autotune runs. The UI must say so before the job starts.
- **The helper LLM runs through the same llama-server router** as the other
  models.
- **Noise rule.** A candidate only replaces the current best if its gain is
  larger than the measured run-to-run spread (standard deviation) of both.
  Otherwise the simpler setting is kept (e.g. MTP alone over MTP + n-gram).
- **Wording.** Only plain technical English in the UI and docs, with no slang.

## Success criteria
- **Profiles:**
  - A profile can be saved, restored and deleted from the model config panel.
  - The panel shows which profile is active and whether it has been edited
    since.
  - Profiles survive deleting and re-downloading the model, and survive a
    backup and restore.
- **Storage:** a corrupt or newer-version `models.json` puts the registry in
  read-only mode with a visible reason; the file is never overwritten.
- **Autoconfigure:**
  - On a freshly downloaded model it produces a review screen within a few
    minutes, and every proposed value passes the same validation as a manual
    config save.
  - A profile saved from that screen launches successfully on the machine it
    was produced on.
- **Autotune:**
  - Run on the Autoconfig profile of a model with an MTP head and the Coding
    use case, it tests MTP, the n-gram assists and MTP + n-gram pairs.
  - It saves profiles whose measured speeds match what the benchmark history
    shows for those settings.
  - A typical single-GPU model finishes in under about 1.5 hours.
- **Speculative decoding files:** a draft method whose file is not installed
  is reported as skipped, not silently left out.
- **Explanations:** every autoconfigure setting and every autotune result has
  a plain-language explanation visible in the UI.

## Decisions
- **Plan scope** → One plan, phased: profiles first, then autoconfigure, then
  autotune.
- **Profile design** → Port vllm-toolchest's design unchanged (named profiles
  per model, the live config keeps autosaving, explicit Save-as / Restore /
  Delete, models.json schema gate).
- **Profile contents** → Launch settings + sampling; everything except
  `Enabled` and `Aliases`.
- **Autoconfigure split** → Hybrid: code rules handle hardware fit (plus
  `cpu_moe` offload for MoE models that do not fit), the LLM
  reads the model card for model-specific advice, and code validates the
  result.
- **Helper model** → Qwen3.5-4B offered as a one-click download by default;
  any installed model can be chosen in Settings.
- **Autoconfigure trigger** → On demand, with a hint on the model card after a
  download.
- **Autoconfigure result** → A review screen with reasons and sources, then
  Save as profile, or Save and apply.
- **Autotune search** → Staged search, then a confirmation stage that
  measures side by side the top 2 stage-1 settings × the stage 2–3
  finalists, plus the starting profile.
- **Autotune workload** → The user picks a use case (General chat, Coding /
  editing, Mixed), each mapped to a fixed prompt set.
- **Autotune scope** → Lossless settings only. Autotune asks for a starting
  profile (Autoconfig or user-defined), and that profile's context size and KV
  cache type stay fixed.
- **Combined score** → Fastest full response: total time per request on the
  chosen workload.
- **Autotune goals** → One run saves up to three profiles (fastest generation,
  fastest prompt, fastest response); a winner shared by several goals is saved
  once.
- **Missing draft files** → Autoconfigure suggests them and the user approves
  each download; autotune tests only installed files and reports what it
  skipped.
- **Multi-GPU and threads** → Tuned only when relevant: split mode with more
  than one GPU assigned, threads when layers are offloaded to the CPU.
- **Autotune orchestration** → One benchmark job for each stage, linked by
  an autotune record; resumable from the last finished stage.
- **Context size in autoconfigure** → The user picks a size class (Short 8K,
  Medium 32K, Long 128K, Maximum). Code fits it, lowering it or using q8_0 KV
  cache if needed.
- **Model too large for VRAM** → Add a `cpu_moe` setting and offload MoE
  experts to the CPU; lower GPU layers for dense models.
- **Shipping** → One PR per feature: phases 01–04 (profiles), 05–09
  (autoconfigure), 10–13 (autotune).
- **Noise rule** (set during drafting, open to change at review) → Keep the
  simpler setting unless the gain is larger than the run-to-run spread.
