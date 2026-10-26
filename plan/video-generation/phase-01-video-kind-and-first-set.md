# Phase 01 — Video Kind, Video Recipes and the Wan2.2 TI2V 5B Set

**Depends on:** nothing · **Enables:** Phase 02 (measuring video sets), and
every later phase, which all need the video kind and its sets

## Goal

Toolchest knows that a model set can be a video set. A video set runs
sd-server like an image set, but it is never picked for image jobs, and the UI
labels it "video". The Models page catalog offers Wan2.2 TI2V 5B, which
downloads in one click and starts as an instance on the Server page. The
custom-set form accepts sd.cpp's video recipes and the new file slots they
need. There is no video API yet, and Measure does not handle video sets yet, so
in this phase a video instance gets its reservation by hand.

## Files touched

- `internal/generation/types.go`: add `KindVideo BackendKind = "video"` and
  `DefaultFrames`, `DefaultFPS`, `MaxFrames`, `MaxWidth`, `MaxHeight` on
  `ModelSet`.
- `internal/generation/catalog.go`: the same five fields on `CatalogEntry`,
  copied into downloaded sets; the Wan2.2 TI2V 5B entry; a shared `umt5Q8`
  component for later Wan sets.
- `internal/generation/recipes.go`: a `VideoRecipes` table (slots and
  capabilities of each video recipe) and `VideoRecipeIDs()`;
  `ValidateComponents`, `describeSet` and `RecipeCapabilities` handle
  `KindVideo`.
- `internal/generation/launch.go`: `sdSlotFlags` gains
  `high_noise_diffusion_model`, `embeddings_connectors`, `audio_vae`,
  `clip_vision`; `LaunchSpec` launches `KindVideo` exactly as `KindSD`.
- `internal/generation/manager.go`: the supervisor label is "sd-server" for
  both `KindSD` and `KindVideo`.
- `internal/builder/project.go`: a helper `ProjectForKind(kind string)` that
  maps "video" to the stable-diffusion.cpp project.
- `internal/api/build.go`, `internal/api/gen_instances.go`:
  `resolveBackendBuild`, `kindProject` and the instance form's build choices
  use `ProjectForKind`, so a video set runs on sd builds.
- `internal/api/gen_models.go`: the custom-set handler accepts kind `video`
  (recipe required; test width, height and frames read from the form;
  capabilities from `RecipeCapabilities`); catalog downloads copy the new
  fields.
- `internal/api/gen_images.go`: `pickInstance` callers for images keep passing
  `KindSD`, so video sets are never picked for image jobs (no code change
  beyond a test).
- `internal/api/gen_vram.go`: `runMeasureGeneration` returns
  "measuring video sets is not supported in this version; set the reservation
  by hand" for `KindVideo`, instead of treating it as music.
- `internal/api/v1models.go`: `genV1Model` sets `meta.kind` `"video"` for
  video sets and gives them no `api` value yet.
- `web/templates/partials/gen_model_sets.html`,
  `web/templates/partials/gen_instances.html`: labels become Image / Music /
  Video by kind; the custom-set kind select gains Video and shows the frames
  and frame-rate fields for it; capability badges for `text`, `start_image`
  and `end_image` have tooltips.
- `internal/api/build.go` (where the page templates' function map is built):
  template functions `kindLabel` and `kindTitle`.
- Tests: `internal/generation/catalog_test.go` (catalog and recipe tests,
  including a new allowed-repos test), `internal/generation/launch_test.go`,
  `internal/api/gen_models_test.go`, `internal/api/gen_images_test.go`,
  `internal/api/v1models_test.go`.

## Steps

1. **Kind.** In `types.go`, add `KindVideo BackendKind = "video"` and the
   comment "runs sd-server with a video model". Add to `ModelSet`:
   `DefaultFrames int` (frames of the measured test video and of a request
   that gives no length), `DefaultFPS int`, `MaxFrames int`, `MaxWidth int`,
   `MaxHeight int` (the ceiling; JSON names `default_frames`, `default_fps`,
   `max_frames`, `max_width`, `max_height`, all `omitempty`). For video sets,
   the existing `DefaultWidth`/`DefaultHeight` are the test size.
2. **Catalog fields.** Add the same five fields to `CatalogEntry`, and copy
   all of them in `handleGenCatalogDownload` where the set is created.
3. **Slots and flags.** In `launch.go`, add to `sdSlotFlags`:
   `high_noise_diffusion_model` → `--high-noise-diffusion-model`,
   `embeddings_connectors` → `--embeddings-connectors`, `audio_vae` →
   `--audio-vae`, `clip_vision` → `--clip_vision`. In `LaunchSpec`, change
   `case KindSD:` to `case KindSD, KindVideo:`. In `manager.go`, label
   anything other than `KindAce` "sd-server", as today.
4. **Video recipes.** In `recipes.go`, add
   `type VideoRecipe struct { Slots RecipeSlotSet; Caps []string }` and a map
   `VideoRecipes map[string]VideoRecipe`, copied from stable-diffusion.cpp
   `examples/server/recipes/*.json` at `master-1000-95ff762`, for:
   `wan22-ti2v-5b`, `wan22-i2v`, `wan22-t2v`, `wan21-t2v`, `wan21-i2v`,
   `wan21-flf2v`, `ltx23-distilled`, `ltx23-dev`, `ltx25-dev`,
   `hunyuan-video-15-t2v`, `hunyuan-video-15-i2v`, `lingbot-video-dense`,
   `lingbot-video-moe`. Copy each recipe's `slots` (required/optional) and
   `required_any` exactly. Leave out `wan21-vace` and `wan22-s2v`, whose inputs
   (control videos, audio) are out of scope. `Slots` copies each recipe's
   `slots` and `required_any`; `Caps` is filled as step 6 says. The comment
   names the sd.cpp version, as `RecipeSlots` does. These recipes use nine slots:
   `diffusion_model`, `high_noise_diffusion_model`, `t5xxl`, `llm`,
   `clip_vision`, `embeddings_connectors`, `vae`, `tae` and `audio_vae`. All
   of them have a flag after step 3, and a test checks that this stays true.
5. **Validation.** `ValidateComponents` takes slots from
   `VideoRecipes[recipe].Slots` for `KindVideo`, and refuses a video recipe for
   `KindSD` and an image recipe for `KindVideo` ("recipe wan22-ti2v-5b is a
   video recipe"). `describeSet` reads "video recipe X" for `KindVideo`.
   `Recipes()` returns image recipes only, as now. A new `VideoRecipeIDs()`
   returns the video ones the same way `Recipes()` orders image recipes: the
   recipes of `KindVideo` catalog entries first, in catalog order (read from
   `Catalog()`, so later phases' sets join automatically), then the rest
   alphabetically.
6. **Capabilities.** Video sets use three capability names: `text`
   (text-to-video), `start_image` and `end_image`. Each `VideoRecipe.Caps` is
   written into the table from the recipe's JSON `features` (`init_image` gives
   `start_image`, `end_image` gives `end_image`) plus `text` for recipes that
   can work from a prompt alone:
   - `wan22-ti2v-5b`, `lingbot-video-dense`, `lingbot-video-moe`: `text`,
     `start_image`
   - `wan22-t2v`, `wan21-t2v`, `hunyuan-video-15-t2v`: `text`
   - `wan22-i2v`, `wan21-flf2v`: `start_image`, `end_image`
   - `wan21-i2v`, `hunyuan-video-15-i2v`: `start_image`
   - `ltx23-distilled`, `ltx23-dev`, `ltx25-dev`: `text`, `start_image`,
     `end_image`

   `RecipeCapabilities` returns `VideoRecipes[recipe].Caps` for a video
   recipe.
7. **Builds for the video kind.** In `internal/builder/project.go`, add
   `ProjectForKind(kind string) (Project, bool)`, which maps "video" to the
   project with Kind "sd" and any other kind to the project with that Kind. Use
   it in `resolveBackendBuild` (`internal/api/build.go`), `kindProject` and
   `genFormChoices` (`internal/api/gen_instances.go`). Extend
   `genFormChoices`' kind list to `{KindSD, KindVideo, KindAce}`.
8. **Wan2.2 TI2V 5B catalog entry.** Add, after the image sets and before the
   ACE-Step sets:
   - ID `wan22-ti2v-5b`, Name "Wan2.2 TI2V 5B", Kind `KindVideo`, Recipe
     `wan22-ti2v-5b`.
   - Components:
     - `diffusion_model`: `QuantStack/Wan2.2-TI2V-5B-GGUF`,
       `Wan2.2-TI2V-5B-Q8_0.gguf`, 5400179040 bytes.
     - `t5xxl`: the shared `umt5Q8` = `city96/umt5-xxl-encoder-gguf`,
       `umt5-xxl-encoder-Q8_0.gguf`, 6043068256 bytes.
     - `vae`: `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`,
       `split_files/vae/wan2.2_vae.safetensors`, 1409400960 bytes.
   - Capabilities `text`, `start_image`. License "Apache-2.0".
   - LaunchArgs: `--steps 20 --cfg-scale 5 --flow-shift 5
     --sampling-method euler --diffusion-fa --vae-tiling`.
   - DefaultWidth 832, DefaultHeight 480, DefaultFrames 33, DefaultFPS 24,
     MaxFrames 121, MaxWidth 1280, MaxHeight 704 (the recipe's native size;
     121 frames is 5 s at 24 fps).
   - Notes: "Text-to-video and image-to-video in one small model; 20 steps.
     Up to 5 seconds at 1280×704." Phase 04 adds the measured time and memory.
9. **Custom sets.** In `handleGenCreateSet`, accept kind `video`: a recipe from
   `VideoRecipeIDs()` is required; test width and height default to 832×480,
   frames to 33 and frame rate to 24 when the form leaves them empty; the
   ceiling fields are left at zero for custom sets, which means no ceiling. In
   `gen_model_sets.html`, the kind select gains "Video", which switches the
   recipe list to the video recipes and shows two fields:
   - "Test frames", tooltip "Length of the test video Measure makes, in
     frames. sd-server rounds it down to 4n+1";
   - "Frame rate", tooltip "Frames per second of the test video, and of
     requests that give none".
10. **Labels.** Everywhere a template now says
    `{{if eq .Kind "sd"}}image{{else}}music{{end}}` (the catalog, the "On this
    machine" table, the instances table, the instance form), use one of two
    new template functions: `kindLabel`, which returns "image", "video" or
    "music", and `kindTitle`, which returns "Image", "Video" or "Music" where
    the template shows a capitalised word. The reservation
    tooltip's sd branch applies to video too ("sd-server is held to this
    amount"). Add capability badges with tooltips: `text` "Makes video from a
    prompt alone", `start_image` "Can start from an image you give",
    `end_image` "Can end on an image you give".
11. **Measure and /v1/models for now.** In `runMeasureGeneration`, return the
    error "measuring video sets is not supported in this version; set the
    reservation by hand" for `KindVideo`. In `genV1Model`, set `meta.kind` to
    `"video"` for `KindVideo` and leave `api` empty.
12. **Allowed repos.** In `catalog_test.go`, add `TestCatalogReposAllowed`
    with a fixed list of the Hugging Face repos the catalog may use, each
    checked to be ungated, and check every catalog component against it. The
    list is the repos the catalog uses today:
    `black-forest-labs/FLUX.2-small-decoder`,
    `city96/t5-v1_1-xxl-encoder-gguf`, `Comfy-Org/Ming-Image`,
    `Comfy-Org/Qwen-Image-2.1`, `Comfy-Org/Qwen-Image_ComfyUI`,
    `Comfy-Org/z_image_turbo`, `inclusionAI/Ming-Image-0.1-Design`,
    `leejet/FLUX.2-klein-4B-GGUF`, `leejet/flux-3-image-klein-GGUF`,
    `leejet/Z-Image-Turbo-GGUF`, `mradermacher/Qwen2.5-VL-7B-Instruct-GGUF`,
    `Qwen/Qwen3-VL-4B-Instruct-GGUF`, `Qwen/Qwen3-VL-8B-Instruct-GGUF`,
    `Serveurperso/ACE-Step-1.5-GGUF`, `silveroxides/Chroma1-Flash-GGUF`,
    `silveroxides/Chroma-GGUF`, `stabilityai/stable-diffusion-xl-base-1.0`,
    `unsloth/Qwen3-4B-GGUF`, `unsloth/Qwen3-4B-Instruct-2507-GGUF`,
    `unsloth/Qwen-Image-2.1-Turbo-GGUF`, `unsloth/Qwen-Image-Edit-2511-GGUF`,
    `voltaire321/Ming-Image-0.1-Design-GGUF`; plus this phase's
    `QuantStack/Wan2.2-TI2V-5B-GGUF`, `city96/umt5-xxl-encoder-gguf` and
    `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`. Later phases add their repos to the
    list. A gated repo then cannot slip into the catalog unnoticed.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/generation/ ./internal/builder/ ./internal/api/`
- `gofmt -l internal/` prints nothing.

## Test plan

- `TestCatalogEntriesAreValid` covers the new entry: valid slots, positive
  sizes, a test size, and for video sets positive `DefaultFrames`,
  `DefaultFPS`, and a ceiling at least as large as the test video.
- New test: every slot of every `VideoRecipes` recipe has an `sdSlotFlags`
  entry.
- `TestCatalogReposAllowed` passes.
- New test: `ValidateComponents` refuses a video recipe for `KindSD` and an
  image recipe for `KindVideo`.
- New test: `RecipeCapabilities` gives `wan22-ti2v-5b` `text, start_image`,
  `wan22-i2v` `start_image, end_image` (no `text`), and `ltx23-distilled`
  `text, start_image, end_image`.
- New test: `LaunchSpec` for a `KindVideo` set builds sd-server arguments with
  `--high-noise-diffusion-model` for a set with that slot.
- New test: `ProjectForKind("video")` is the stable-diffusion.cpp project.
- New test: `pickInstance(model, KindSD)` never returns a running video
  instance, even when it is the only one.
- New test: `/v1/models` lists a video instance with `meta.kind` `"video"` and
  no `api`.
- Manual, on asmodean: download Wan2.2 TI2V 5B from the Models page; create
  an instance; set a reservation of 20 GiB by hand; start it; open its web UI
  and make a video there.

## Commit

`feat(generation): add the video kind, video recipes and the Wan2.2 TI2V 5B set`

## Rollback

Revert the commit. Sets and instances of kind `video` already saved in
`generation.json` then fail validation at launch with "unknown backend kind";
delete them on the Models and Server pages before or after reverting. Nothing
else is stored differently.
