# Generation Recipes From stable-diffusion.cpp — Project Overview

## Problem

Toolchest keeps a hand-made copy of stable-diffusion.cpp's server recipes in Go. `RecipeSlots` and `VideoRecipes` in `internal/generation/recipes.go` copy each recipe's slots, the video size multiple and a capability list from upstream at one commit (`master-1000-95ff762`, with some parts from `master-1015-e41fbcf`). `RecipeSlotLabels` copies upstream's slot labels in toolchest's words. Upstream now has 108 recipe files (`examples/server/recipes/*.json` at `master-1047-984eb06`). Toolchest knows 25 of them. The other 83 (Ernie-Image, Krea2, LongCat, PixArt, SD3.x, the FLUX.1 control and fill variants, HiDream and more) cannot be used, even in a custom set, because the custom set form offers only the recipes toolchest knows. A new upstream recipe needs a toolchest release before anyone can use it.

Upstream's files already hold more than toolchest copies: slot labels, `required_any`, `size_multiple`, `video_constraints`, defaults (steps, CFG, size, sampler, scheduler, flow shift), `features` (`ref_images`, `init_image`, `end_image`, …), `disabled_features` and a description. Toolchest never passes a recipe to sd-server: it passes one flag for each slot (`sdSlotFlags` in `internal/generation/launch.go`). So the recipes only drive the custom set form, set validation, capabilities, slot labels, Details and the video size multiple.

Tracked in issue #528. Left out of #515 on purpose.

## Goals

- When toolchest builds stable-diffusion.cpp, it copies `examples/server/recipes/*.json` from the checkout into the build's folder (`builds/<id>/recipes/`). Each build keeps the recipes it was built with.
- Toolchest ships a snapshot of upstream's recipe files inside the binary (`go:embed`). It is used when no sd build has a `recipes` folder: a fresh install, or builds made before this change.
- One parser reads both the build folders and the snapshot. The hand-written slot tables `RecipeSlots` and `VideoRecipes` are removed.
- The active recipe list is the newest successful sd build's recipes, or the snapshot when that build has none. The custom set form, set validation, copy, edit and Details all use it.
- A recipe is a video recipe when its file has `video_constraints`; otherwise it is an image recipe.
- A small toolchest overlay, in Go, sits on top of the upstream data. It holds:
  - the edit-image limits (`RecipeMaxEditImages`),
  - the rule that Qwen-Image recipes edit only with an `llm_vision` file,
  - the video recipes that also start from text alone, and Ming-Image's `alpha`,
  - recipes left out on purpose, with a reason (`wan21-vace`, `wan22-s2v`),
  - toolchest's reworded slot labels.
- Capabilities come from upstream's `features`, then the overlay: `ref_images` gives `edit`, `init_image` gives `start_image`, `end_image` gives `end_image`.
- Slot labels come from the toolchest overlay first, then upstream's label, then the generic `SlotLabels` name.
- The custom set form lists every recipe. A recipe toolchest cannot run shows as not available, with a short reason: a required slot that has no sd-server flag in `sdSlotFlags`, an input that toolchest does not send (control images, audio, video, VACE), or a recipe `context` setting other than `vae_format` and `model_args`. A recipe whose only unknown slot is optional stays available, without that slot.
- When the user picks a recipe in the custom set form, the form fills the launch-argument field with sd-server flags from upstream's `defaults`. The user can change or clear them.
- A saved set whose recipe the active list does not know stays in place. Details shows a warning that names the recipe. Copy and edit still open it. Starting an instance on a build whose recipes do not know the recipe is refused with that reason.
- Docs (`docs/guides/generation.md`) and the Help page explain where the recipes come from.

## Non-goals

- LoRAs for image sets (issue #527, set aside).
- New sd-server flag mappings for slots such as `clip_g`, `control_net`, `ip_adapter` or `photo_maker`. Recipes that need them show as not available.
- Sending control images, audio or video to sd-server.
- Passing a recipe to sd-server, or using sd-server's own recipe loading.
- Changing existing sets' flags. Upstream defaults only prefill the form for a new set.
- Catalog changes. Catalog entries keep their own flags and recipe names.
- A choice of build in the custom set form.
- Reading recipes from the shared checkout at run time.

## Users & primary flow

1. A user builds a new stable-diffusion.cpp on the Builds page. The build copies the checkout's recipe files into its folder.
2. The user opens "Add a custom set" on the generation page and picks image or video. The recipe list now holds every recipe of the newest sd build. Recipes that toolchest cannot run are greyed, each with its reason.
3. The user picks a recipe, for example `ernie-image`. The form shows one field for each slot with its label, and fills the launch arguments from the recipe's defaults.
4. The user fills the files, saves, measures and starts the set, as today.
5. Later the user starts that set on an older build that does not have the recipe. Toolchest refuses the start and says that build does not know the recipe.

## Constraints

- Go only on the toolchest side. Upstream's files are JSON with `schema_version: 1`. A file with another schema version is skipped with a log line.
- Toolchest runs in a container and in host mode. The build folder is under the data folder in both cases, so the copy and the read work the same way.
- Builds made before this change have no `recipes` folder. They use the embedded snapshot.
- Existing custom sets and catalog sets must keep working without changes. Every catalog recipe must be in the embedded snapshot (a test checks this).
- The snapshot is refreshed by a script from a checkout, not by hand.
- Text in the UI, tooltips and Help follows the house style: short, about 80% ASD-STE100, no jargon.
- The docs page and Help change in the same pull request as the behaviour they describe.
- Work on a branch, Conventional Commits, no AI attribution.

## Success criteria

- A new sd build has `builds/<id>/recipes/*.json`, the same files as the checkout at that build's commit.
- With that build as the newest, the custom set form offers every upstream image and video recipe that toolchest can run. The rest show as not available, each with a reason.
- On a fresh install with no build, the form offers the recipes of the embedded snapshot.
- A custom set made from a new recipe (one toolchest did not know before) can be saved, measured and started, and makes an image.
- All existing catalog sets and custom sets validate and start as before, with the same capabilities and labels. The only allowed differences are ones where upstream's file at the snapshot commit says otherwise; the user approves that list in the phase 01 pull request.
- A set whose recipe the build does not know is refused at start with a clear reason, and Details warns about it.
- Tests cover parsing real upstream files (copied fixtures), the overlay, the active-list choice, the fallback to the snapshot, the not-available reasons, and the defaults-to-flags mapping.
- `RecipeSlots` and `VideoRecipes` are gone from the code.

## Decisions

- **Where recipes are read from** → copied into each build's folder at build time (`builds/<id>/recipes/`).
- **Which list the form and validation use** → the newest sd build's list. An instance start checks the set against the recipes of the build it runs on.
- **Fallback when no build has recipes** → an embedded snapshot of upstream's JSON files (`go:embed`), read by the same parser. The Go slot tables are removed.
- **Recipes toolchest cannot run** → shown greyed, with a reason. A recipe whose only unknown slot is optional stays usable without it.
- **Slot labels** → toolchest's reworded labels first, then upstream's, then the generic name.
- **Capabilities** → derived from upstream `features` (`ref_images` → edit, `init_image` → start image, `end_image` → end image), with an overlay for text-only video starts, Ming's alpha and the Qwen `llm_vision` rule.
- **Upstream defaults** → prefill the launch-argument field of the custom set form for a new set.
- **A saved set whose recipe is unknown** → keep it, warn in Details, refuse to start it on a build that does not know the recipe. Nothing is deleted or changed.
