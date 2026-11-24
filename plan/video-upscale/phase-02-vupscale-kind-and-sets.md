# Phase 02 — The vupscale kind, its slots, catalog sets and custom sets

**Depends on:** nothing · **Enables:** Phase 03 (launch), Phase 04 (scripts pick by `Recipe`), Phase 05 (routes find vupscale instances), Phase 06 (Measure)

## Goal

Add the set kind `vupscale` (`generation.KindVUpscale`). A vupscale set has two fixed slots, `dit` and `vae`, a script (`relayvsr` or `seedvr2`, kept in the set's `Recipe` field) and a scale. The catalog gets three sets: RelayVSR ×2, RelayVSR ×4 and SeedVR2 3B. Users can download them, see them on the Models page, build custom vupscale sets in the form, and add instances on the Server page. The instances cannot start yet, because Phase 03 adds the launch. A start attempt fails with "video upscaler sets are not supported yet". This phase is shippable because the catalog sets only download files.

## Files touched

- `internal/generation/types.go` — `KindVUpscale BackendKind = "vupscale"`, with a comment; extend the `Scale` comment to cover vupscale sets.
- `internal/generation/recipes.go`:
  - `VUpscaleScripts = []string{"relayvsr", "seedvr2"}` and `VUpscaleSlots = []string{"dit", "vae"}`.
  - `ValidateComponents`: a `KindVUpscale` case (fixed slots, and `recipe` must be one of `VUpscaleScripts`).
- `internal/generation/details.go` — `KindSlots`: `KindVUpscale` returns `RecipeSlotSet{Required: VUpscaleSlots}`.
- `internal/generation/catalog.go` — three entries (step 4); `UpscaleScale()` also answers for `KindVUpscale`.
- `internal/generation/launch.go` — `LaunchSpec`: a `KindVUpscale` case that returns the error "video upscaler sets are not supported yet". Phase 03 replaces it.
- `internal/builder/project.go` — `ProjectForKind`: map `"vupscale"` to `"sd"`; update the comment at line 169.
- `internal/api/gen_models.go` — `catalogTags`: vupscale sets get the tag "video".
- `internal/api/gen_set_form.go` — `setFromForm` and `genSetFormFrom`: the vupscale kind (step 5).
- `web/templates/partials/gen_set_form.html` — the kind option, the Script select and the scale select.
- `web/templates/partials/gen_model_sets.html`, `gen_set_details.html`, `gen_catalog.html` — show "Video upscaler" as the kind label, and the scale.
- `internal/api/gen_instances.go` — the per-kind tooltip and label maps get vupscale entries (copy the video wording, replacing "makes videos" with "makes videos larger").
- `internal/api/v1models.go` — leave vupscale out of `/v1/models` in this phase: `genV1Model` skips `KindVUpscale`. Phase 05 adds it.
- `docs/guides/generation.md` — under custom sets, one line: the Video upscaler kind, its Script select and its two slots (Phase 07 writes the full section).
- Tests: `internal/generation/recipes_test.go`, `details_test.go`, `catalog_test.go`, `internal/builder/project_test.go`, `internal/api/gen_set_form_test.go`, `gen_models_test.go`, `gen_instances_test.go`.

## Steps

1. **Find every kind switch.** Run `grep -rn "KindUpscale\|KindVideo\|\"upscale\"\|\"video\"" internal web --include='*.go' --include='*.html' --include='*.js'`. Give each switch and map a vupscale case or entry, with this rule: vupscale goes with `KindUpscale` (no recipe book, fixed slots, no image or video request fields) everywhere except where this plan says otherwise (launch in Phase 03, routes and `/v1/models` in Phase 05, Measure in Phase 06). Where a switch has no upscale case and its `default` would run recipe or video code, add an explicit vupscale case that does what the upscale path does. List the places in the PR description.
2. **Kind and slots.**
   - Add `KindVUpscale` with the comment: "KindVUpscale also runs sd-server, with no diffusion model at start: a DGL script loads a video upscaler (RelayVSR or SeedVR2) for each job, for /v1/videos/upscale/jobs. The set's Recipe names the script."
   - `ValidateComponents` for `KindVUpscale`:
     - `recipe` not in `VUpscaleScripts` → `unknown video upscaler script %q: use relayvsr or seedvr2`.
     - Otherwise the slots are `VUpscaleSlots`, both required, through the same missing/unknown/repeated checks as the other kinds.
3. **Labels.** Add `func vupscaleSlotLabel(script, slot string) string`:
   - relayvsr: `dit` → "RelayVSR diffusion model", `vae` → "RelayVSR VAE (same scale)".
   - seedvr2: `dit` → "SeedVR2 diffusion model", `vae` → "SeedVR2 VAE".
   - `SlotLabel` calls it first when the recipe is one of `VUpscaleScripts`.
4. **Catalog entries.** Add after `real-esrgan-x4plus`, all `Kind: KindVUpscale`, `License: "Apache-2.0"`, not gated:

   | ID | Name | Recipe | Scale | dit (Repo, File, Bytes) | vae (Repo, File, Bytes) |
   | --- | --- | --- | --- | --- | --- |
   | `relayvsr-x2` | RelayVSR ×2 | relayvsr | 2 | Comfy-Org/RelayVSR, `diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors`, 5163295248 | Comfy-Org/RelayVSR, `vae/relayvsr_flash_vae_x2_fp32.safetensors`, 127759464 |
   | `relayvsr-x4` | RelayVSR ×4 | relayvsr | 4 | Comfy-Org/RelayVSR, `diffusion_models/relayvsr_x4_int8_convrot.safetensors`, 5164179952 | Comfy-Org/RelayVSR, `vae/relayvsr_flash_vae_x4_fp32.safetensors`, 127759784 |
   | `seedvr2-3b` | SeedVR2 3B | seedvr2 | 2 | Comfy-Org/SeedVR2, `diffusion_models/seedvr2_3b_int8_convrot.safetensors`, 3458259704 | Comfy-Org/SeedVR2, `vae/seedvr2_ema_vae_fp16.safetensors`, 501324814 |

   Notes text, one per set:
   - RelayVSR ×2: "Makes a video 2 times larger and sharper, and keeps it steady from frame to frame. Works on windows of 16 frames by default. Fastest on NVIDIA cards (fused CUDA kernel); other cards use a slower tiled path. Needs stable-diffusion.cpp master-1116 or newer. Takes AVI, WebM (VP8) or animated WebP; no MP4."
   - RelayVSR ×4: the same with "4 times".
   - SeedVR2 3B: "Makes a video 2 times larger and restores detail. Each frame is treated on its own by default; a larger window gives smoother motion but needs more VRAM. Needs stable-diffusion.cpp master-1116 or newer. Takes AVI, WebM (VP8) or animated WebP; no MP4."
   - SeedVR2's scale is a script setting, not a property of the weights. The catalog fixes it at 2. A custom SeedVR2 set may choose 2, 3 or 4.
5. **Custom set form.**
   - The kind select gets "Video upscaler" (`vupscale`).
   - For vupscale, the form shows a **Script** select (RelayVSR, SeedVR2) in place of the Recipe select, using the same `hx-get` reload as the Recipe select.
   - It shows two file slots, `slot_dit` and `slot_vae`, with the labels from step 3.
   - **Scale:** for relayvsr, a select of 2 or 4 with the hint "Must match the RelayVSR files: ×2 files give 2, ×4 files give 4." For seedvr2, a select of 2, 3 or 4.
   - `setFromForm` saves `Recipe` = the script, `Scale` from the select, and refuses any other value with a form error.
   - `setFromForm` also checks a relayvsr set's `dit` file name against the regular expression `(^|[_.-])x([24])([_.-]|$)`. The first match gives the files' scale. (In `relayvsr_x2_lightx2v_int8_convrot.safetensors` it matches `_x2_`, not `lightx2v`.) When the found scale differs from the chosen one, the form refuses with "the RelayVSR files are for ×N; set the scale to N". A name with no match passes.
   - The video and image fields (test size, frames, fps, max fields) are hidden for vupscale.
6. **Instances.** Adding a vupscale instance on the Server page works like an upscale instance (GPU, port, build). The instance list shows the kind label "Video upscaler". The Measure button stays hidden for vupscale until Phase 06: add `KindVUpscale` to the kinds that hide it.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `ValidateComponents`: a good relayvsr set and a good seedvr2 set pass; a missing `vae`, an extra `upscaler` slot and recipe `"wan"` each fail with the expected text.
- `KindSlots(KindVUpscale, "relayvsr")` returns `dit`, `vae`.
- `ProjectForKind("vupscale")` returns the stable-diffusion.cpp project.
- Catalog test: the three entries validate. Their files have `Bytes`. `UpscaleScale()` gives 2, 4 and 2.
- Form tests:
  - A relayvsr custom set with scale 4 and ×4 files saves.
  - The same set with scale 2 is refused with the expected message.
  - A seedvr2 set with scale 3 saves.
  - The rendered form for vupscale has the Script select and no frames field.
- `LaunchSpec` for a vupscale instance returns "video upscaler sets are not supported yet".
- `genV1Model` output has no vupscale entries.
- Manual: on a local install, open Download Models, see the three sets with their notes, and download RelayVSR ×2. Add an instance, set a hand reservation of 16 GiB (no instance of any kind starts without one), and press Start: the "not supported yet" message shows.

## Commit

`feat(generation): add the vupscale set kind with RelayVSR and SeedVR2 catalog sets`

## Rollback

Revert the commit. Saved vupscale custom sets and instances would then have a kind the app does not know. Before reverting on a machine that has them, delete those sets and instances on the Models and Server pages, so no record with an unknown kind is left.
