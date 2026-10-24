# Phase 04 — Model Sets: Catalog, Downloads and Custom Sets

**Depends on:** Phase 02 (store, types, `gen-models` directory) · **Enables:**
Phase 05 (instances run a model set)

## Goal

Users get image and music models as **model sets**: named groups of component
files.
- A built-in catalog lists known-good sets that download in one click.
- A form defines a custom set from files already on disk.

Downloads reuse the existing Hugging Face downloader, with progress and resume.
Files land in the `gen-models` directory, which keeps them out of the LLM
registry scan.

## Files touched

- `internal/generation/catalog.go` (new): the embedded catalog.
- `internal/generation/recipes.go` (new): `RecipeSlots`, the production table
  of slots for each allowed sd recipe (see step 3). Custom-set validation and
  the catalog test both use it.
- `internal/generation/catalog_test.go` (new).
- `internal/huggingface/downloader.go`:
  - `Start` gets a `StartFile(ctx, source, repo, file, destDir, expectedBytes)`
    variant that downloads any file type to a given directory.
  - The existing `Start` calls it with the LLM models dir.
  - The `.gguf` filter in `client.go` L132 stays as it is for the LLM browser.
- `internal/api/gen_models.go` (new): handlers for listing the catalog and the
  installed sets, downloading a catalog set, creating and deleting a custom
  set, and set download progress over SSE.
- `internal/api/server.go`: register `/api/gen/models/*` routes. They are
  active only when `GenerationEnabled` is on; otherwise they return 404.
- `web/templates/models.html` and new `web/templates/partials/gen_model_sets.html`:
  - A "Generation models" section, shown only when the switch is on.
  - It has a catalog table, installed sets, per-set download progress and a
    "Custom set" form.

## Steps

1. **Embed the catalog as Go data in `catalog.go`.** Each entry carries:

   | Field | Notes |
   |---|---|
   | `ID` | |
   | `Name` | |
   | `Kind` | |
   | `Recipe` | sd-server recipe ID; empty for ace |
   | `Components` | `[]Component` with `Slot`, `Repo`, `File`, `Bytes` |
   | `LaunchArgs` | |
   | `Capabilities` | |
   | `License` | |
   | `Notes` | Shown as a tooltip |

   The entries are:

   - **`ming-image`** — "Ming-Image 0.1 Design". This is the default image
     entry and is listed first. The note says "Used by Haruspex; outputs PNG
     with an alpha channel".
     - Kind `sd`, recipe `ming-image`. Licence MIT.
     - Components:

       | Slot | Repo | File | Size |
       |---|---|---|---|
       | `diffusion_model` | `voltaire321/Ming-Image-0.1-Design-GGUF` | `ming_image_0.1_design-Q8_0.gguf` | 6.54 GB |
       | `llm` | `voltaire321/Ming-Image-0.1-Design-GGUF` | `ming_ling_mini_2.0-Q4_K.gguf` | 10.52 GB |
       | `vae` | `Comfy-Org/Ming-Image` | `vae/ming_image_vae_bf16.safetensors` | 0.25 GB |
       | `tokenizer` | `inclusionAI/Ming-Image-0.1-Design` | `mllm/tokenizer.json` | |

     - LaunchArgs: `--steps 12 --cfg-scale 1 --sampling-method euler
       --diffusion-fa --vae-tiling`.
     - Capabilities: `alpha`.
     - Default size: 2048 × 2048, the recipe default.

   - **`qwen-image-2.1-turbo`** — "Qwen-Image 2.1 Turbo (Q4)".
     - Kind `sd`, recipe `qwen-image-21-turbo`.
     - Licence: "Qwen-Image 2.1 licence (see model card)". Hugging Face lists it
       as `other`. The tooltip links to `https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo`.
     - Default size: 1024 × 1024, the size measured on asmodean.
     - Components:

       | Slot | Repo | File | Size |
       |---|---|---|---|
       | `diffusion_model` | `unsloth/Qwen-Image-2.1-Turbo-GGUF` | `qwen-image-2.1-turbo-Q4_K_M.gguf` | 4.20 GB |
       | `llm` | `Qwen/Qwen3-VL-8B-Instruct-GGUF` | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | |
       | `vae` | `Comfy-Org/Qwen-Image-2.1` | `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |

     - LaunchArgs: `--sigmas qwen-image-2.1-turbo --cfg-scale 1
       --sampling-method euler --diffusion-fa --vae-tiling`. These are the
       settings measured on asmodean.
     - Capabilities: none.

   - **`flux2-klein-4b`** — "FLUX.2 klein 4B".
     - Kind `sd`, recipe `flux2-klein-4b`. Licence Apache-2.0.
     - Components:

       | Slot | Repo | File | Size |
       |---|---|---|---|
       | `diffusion_model` | `leejet/FLUX.2-klein-4B-GGUF` | `flux-2-klein-4b-Q8_0.gguf` | 4.30 GB |
       | `llm` | `unsloth/Qwen3-4B-GGUF` | `Qwen3-4B-Q8_0.gguf` | 4.28 GB |
       | `vae` | `black-forest-labs/FLUX.2-small-decoder` | `full_encoder_small_decoder.safetensors` | 0.25 GB |

       The VAE comes from the small-decoder repo because it is ungated and
       Apache-licensed. The FLUX.2-dev VAE is gated.
     - LaunchArgs: `--steps 4 --cfg-scale 1 --sampling-method euler
       --diffusion-fa`.
     - Capabilities: `edit`.
     - Default size: 1024 × 1024.

   - **`sdxl-base`** — "Stable Diffusion XL base 1.0".
     - Kind `sd`, recipe `sdxl-checkpoint`. Licence openrail++.
     - Components:

       | Slot | Repo | File | Size |
       |---|---|---|---|
       | `model` | `stabilityai/stable-diffusion-xl-base-1.0` | `sd_xl_base_1.0.safetensors` | 6.94 GB |

     - LaunchArgs: `--steps 20 --cfg-scale 5 --vae-tiling`.
     - Capabilities: none.
     - Default size: 1024 × 1024.

   - **`ace-step-turbo`** — "ACE-Step 1.5 turbo".
     - Kind `ace`. Repo `Serveurperso/ACE-Step-1.5-GGUF`. Licence MIT.
     - Components:

       | Slot | File |
       |---|---|
       | `lm` | `acestep-5Hz-lm-4B-Q8_0.gguf` |
       | `dit` | `acestep-v15-turbo-Q8_0.gguf` |
       | `embedding` | `Qwen3-Embedding-0.6B-BF16.gguf` |
       | `vae` | `vae-BF16.gguf` |

   - **`ace-step-xl-sft`** — "ACE-Step 1.5 XL SFT". This is the set measured
     on asmodean.
     - Kind `ace`. Same repo, licence and components as `ace-step-turbo`,
       except that `dit` is `acestep-v15-xl-sft-Q8_0.gguf`.

   `Bytes` for each component is filled in from the Hugging Face tree API when
   the catalog is rendered. The figures above are the expected sizes for
   display before that request returns.

2. **Download a catalog set.** `POST /api/gen/models/catalog/{id}/download`:
   - starts one `StartFile` per component into
     `<gen-models>/<repo org--name>/<file>`
   - creates the `ModelSet` in the store. Its ID is the catalog ID, and its
     `DefaultWidth`/`DefaultHeight` come from the entry. `Path` is filled in
     once each file completes.
   - is refused when a set with that ID already exists
   - skips any component whose file already exists at the target path with the
     expected size. The two ACE-Step sets share their `lm`, `embedding` and
     `vae` files, so the second download only fetches its own `dit` file.
   - sets `ModelSet.Complete = true` when every component is present. Phase 05
     refuses to create an instance from an incomplete set.
   - leaves the set marked incomplete until every component is present
   
   Progress SSE: `/api/gen/models/sets/{id}/progress` merges the per-file
   download broadcasters.

3. **Create a custom set.** `POST /api/gen/models/sets` takes:
   - name
   - kind
   - for `sd`, a recipe ID chosen from a fixed list: the four `sd` recipes above
     (`ming-image`, `qwen-image-21-turbo`, `flux2-klein-4b`,
     `sdxl-checkpoint`), plus `chroma`, `flux1-dev`, `flux1-schnell`,
     `qwen-image-21`, `z-image`
   - for `sd`, a default width and height, each defaulting to 1024
   - one absolute path per slot
   - optional extra launch args

   The set's ID is the slug of its name (Phase 02). A custom set is
   `Complete = true` as soon as validation passes.

   A custom set's `Capabilities` are worked out from its recipe when it is
   created, and stored on the set:
   - `alpha` for `ming-image`
   - `edit` for `qwen-image-21`, `qwen-image-21-turbo` and `flux2-klein-4b`,
     the recipes whose sd.cpp docs describe editing
   - none for the others

   `/v1/models` (Phase 07) and the edits route read the stored value.

   **`RecipeSlots`** is copied from sd.cpp `examples/server/recipes/*.json` at
   `master-1000-95ff762` (2026-10-24). "One of" means at least one of the
   listed slots must be filled.

   | Recipe | Required | Optional | One of |
   |---|---|---|---|
   | `ming-image` | `diffusion_model`, `llm`, `tokenizer` | `vae`, `tae` | `vae`/`tae` |
   | `qwen-image-21-turbo` | `diffusion_model`, `llm` | `llm_vision`, `vae`, `tae` | `vae`/`tae` |
   | `qwen-image-21` | `diffusion_model`, `llm` | `llm_vision`, `vae`, `tae` | `vae`/`tae` |
   | `flux2-klein-4b` | `diffusion_model`, `llm` | `vae`, `tae` | `vae`/`tae` |
   | `z-image` | `diffusion_model`, `llm` | `vae`, `tae` | `vae`/`tae` |
   | `chroma` | `diffusion_model`, `t5xxl` | `vae`, `tae` | `vae`/`tae` |
   | `flux1-dev` | `diffusion_model`, `clip_l`, `t5xxl` | `vae`, `tae` | `vae`/`tae` |
   | `flux1-schnell` | `diffusion_model`, `clip_l`, `t5xxl` | `vae`, `tae` | `vae`/`tae` |
   | `sdxl-checkpoint` | `model` | `vae`, `tae` | none |

   For `ace` sets the four slots `lm`, `dit`, `embedding` and `vae` are all
   required.

   **Shared argument check.** Write `generation.ValidateExtraArgs(args)` in
   this phase. It does the splitting and deny-list check described under
   Validation below. Phase 05 uses the same function for an instance's
   `ExtraArgs`.

   **Validation:**
   - Every path exists and is a regular file.
   - Every required slot has a file.
   - For `ace`, all four slots are filled.

   Launch args are split with a shell-word splitter and checked against a deny
   list. The deny list blocks `--listen-ip`, `--listen-port`, `--host`,
   `--port`, `--max-vram` and `--models`, because toolchest sets those.

4. **Delete a set.** `DELETE /api/gen/models/sets/{id}`:
   - Refused while an instance uses the set.
   - Catalog sets also get their files deleted.
   - A catalog set's files are deleted only if no other set uses the same
     `Path`, which keeps the shared ACE-Step files.
   - For custom sets only the record is removed, never the user's files.

5. **Show a disk-space check.** The UI shows free disk space next to the
   download button, using the same helper as the LLM downloader
   (`AvailableForDownload`).

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/generation/... ./internal/huggingface/... ./internal/api/...`

## Test plan

- **`catalog_test.go`:**
  - Every entry has a unique ID.
  - Every `sd` entry fills the required and "one of" slots of its recipe,
    checked against the production `RecipeSlots` table.
  - The `ming-image` entry exists, is the first `sd` entry, and has the `alpha`
    capability.
- **Downloader:** a `StartFile` test against an `httptest` server checks that
  a non-GGUF file downloads to the given directory and that resume works.
- **API tests:**
  - A custom set with a missing file is rejected.
  - Denied launch args are rejected.
  - Deleting a set that an instance uses is refused.
  - With the switch off, the routes return 404.
- **Manual on asmodean:**
  - Download `ming-image` and `ace-step-xl-sft` from the UI and confirm that
    progress, the files under `gen-models`, and the set records are right.
  - Create a custom set named `qwen-turbo-dev` from the Qwen-Image 2.1 Turbo
    files in `~/models/gen-test`.
    - Recipe: `qwen-image-21-turbo`.
    - Extra launch arguments: `--sigmas qwen-image-2.1-turbo --cfg-scale 1
      --sampling-method euler --diffusion-fa --vae-tiling`.
    - Phases 05 and 06 reuse this set. Phase 11 creates its own
      `qwen-turbo-local` set on a clean install.
  - Confirm the LLM model list does not show any of these files.

## Commit

`feat(generation): add model set catalog, downloads and custom sets`

## Rollback

Revert the commit. Downloaded files stay in `gen-models`, and records stay in
`generation.json`. Neither is read by the reverted binary. Delete them by hand
if they are not wanted.
