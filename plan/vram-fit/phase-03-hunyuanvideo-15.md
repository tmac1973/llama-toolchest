# Phase 03 — HunyuanVideo 1.5 Catalog Sets

**Depends on:** Phase 01 (measuring eight sets quickly), Phase 02 (the
placement rules these sets follow) · **Enables:** nothing

## Goal

Every HunyuanVideo 1.5 version that runs on consumer cards and that sd.cpp
master-1015 supports downloads in one click from the catalog.

HunyuanVideo 1.5 (Tencent, November 2025) is an 8.3B video model. Tencent
says it needs at least 14 GB of VRAM with model parts kept in system RAM.
sd.cpp runs it with two recipes, `hunyuan-video-15-t2v` and
`hunyuan-video-15-i2v`, which toolchest already lists for custom sets
(`recipes.go`). This phase adds catalog entries; no new code is needed for
the model itself.

## The sets

All eight share three files, downloaded once:

| Slot | File | Size |
|---|---|---|
| `llm` | `mradermacher/Qwen2.5-VL-7B-Instruct-GGUF` `Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf` | 4.68 GB, **already used by Qwen-Image Edit 2511** |
| `t5xxl` | `Comfy-Org/HunyuanVideo_1.5_repackaged` `split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors` | 0.44 GB |
| `vae` | `Comfy-Org/HunyuanVideo_1.5_repackaged` `split_files/vae/hunyuanvideo15_vae_fp16.safetensors` | 2.52 GB |

Each set adds its own `diffusion_model`:

| ID | Recipe | Diffusion model | Steps, CFG, flow shift (Tencent's) |
|---|---|---|---|
| `hunyuan15-t2v-480p` | t2v | `jayn7/HunyuanVideo-1.5_T2V_480p-GGUF` `480p/hunyuanvideo1.5_480p_t2v-Q8_0.gguf`, 9.00 GB | 50, 6, 5 |
| `hunyuan15-t2v-480p-cfg-distilled` | t2v | same repo, `480p_distilled/hunyuanvideo1.5_480p_t2v_cfg_distilled-Q8_0.gguf`, 9.00 GB | 50, 1, 5 |
| `hunyuan15-t2v-720p` | t2v | `jayn7/HunyuanVideo-1.5_T2V_720p-GGUF` `720p/hunyuanvideo1.5_720p_t2v-Q8_0.gguf`, 9.00 GB | 50, 6, 9 |
| `hunyuan15-i2v-480p` | i2v | `jayn7/HunyuanVideo-1.5_I2V_480p-GGUF` `480p/hunyuanvideo1.5_480p_i2v-Q8_0.gguf`, 9.00 GB | 50, 6, 5 |
| `hunyuan15-i2v-480p-cfg-distilled` | i2v | same repo, `480p_distilled/hunyuanvideo1.5_480p_i2v_cfg_distilled-Q8_0.gguf`, 9.00 GB | 50, 1, 5 |
| `hunyuan15-i2v-480p-step-distilled` | i2v | `Comfy-Org/HunyuanVideo_1.5_repackaged` `split_files/diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors`, 16.66 GB (see step 2) | 8, 1, 7 (Tencent: 8 or 12) |
| `hunyuan15-i2v-720p` | i2v | `jayn7/HunyuanVideo-1.5_I2V_720p-GGUF` `720p/hunyuanvideo1.5_720p_i2v-Q8_0.gguf`, 9.00 GB | 50, 6, 7 |
| `hunyuan15-i2v-720p-cfg-distilled` | i2v | same repo, `720p_distilled/hunyuanvideo1.5_720p_i2v_cfg_distilled-Q8_0.gguf`, 9.00 GB | 50, 1, 7 |

Left out:
- **720p I2V sparse CFG-distilled:** needs NVIDIA H-series cards.
- **720p T2V CFG-distilled:** not released ("coming soon" on Tencent's page).
- **480p→720p and 720p→1080p upscalers:** sd.cpp does not support them.

Q8_0, as TI2V uses: it is close to the original and fits a 32 GB card with
room for the video. Phase 02's placement rules decide what goes on the CPU.

**Licence:** Tencent Hunyuan Community License. It does not apply in the
European Union, the United Kingdom or South Korea, and the licence column
says so. (MiniMax-H3 was left out of the catalog because its licence also
excludes the US; this one does not.)

## Files touched

- `internal/generation/catalog.go`: three shared `Component` values
  (`hunyuanQwenVL`, which is the Qwen-Image Edit 2511 component moved to a
  shared value; `hunyuanByT5`; `hunyuanVAE`) and eight entries, after
  LTX-2.3. Each has `VideoSizes`, `FullMeasureTime`, `SizeMultiple: 16`
  (from the recipe), `DefaultFPS: 24`, `MaxFrames: 121`, and notes with the
  measured times and reservations.
- `internal/generation/catalog_test.go`: the eight entries pass the existing
  catalog checks (slots match the recipe, sizes within limits, shared files
  identical).
- `web/templates/help.html`: the video section lists the HunyuanVideo sets
  and how they differ.
- `plan/vram-fit/verification.md`: a Phase 03 section.

## Steps

1. **Shared components.** Move Qwen-Image Edit 2511's `llm` component to a
   shared value, used there and by the eight sets, so that the same file is
   named the same way. Add `hunyuanByT5` and `hunyuanVAE`.

2. **The step-distilled file.** No GGUF of it exists. On asmodean, try in
   this order and keep the first that loads and makes a good video:
   1. the fp16 file with `--type q8_0` in its launch arguments. sd.cpp then
      converts it to Q8_0 as it loads: 16.66 GB on disk, about 9 GB in
      memory. Note how much longer the start takes;
   2. the fp8 file (`..._fp8_scaled.safetensors`, 8.34 GB). sd.cpp converts
      fp8 weights to f16 when it loads them, so check that the scaled version
      gives a correct video and how much memory it holds;
   3. the fp16 file as it is.

   Write which one was chosen, and why, in a comment on the entry.

3. **Sizes.** For each set:
   - Short is 33 frames at 848×480 (480p sets) or 1280×720 (720p sets);
   - Medium is 65 frames;
   - Full is 121 frames (5 s at 24 fps).

   `MaxWidth`/`MaxHeight` is 848×480 for the 480p sets and 1280×720 for the
   720p sets, the sizes the versions were trained at. Measure each set at
   Short on asmodean with Phase 01's Measure, then at Medium and Full. Record
   the peaks, the reservations, the time estimate and a real job's time.

4. **Launch arguments.** `--steps`, `--cfg-scale`, `--flow-shift` and
   `--sampling-method euler` from the table, `--diffusion-fa`, and Phase 02's
   placement and decode flags. Check the decode with temporal tiling and with
   spatial tiling (`--vae-tiling`, which sd.cpp's example uses). On ROCm,
   spatial tiling hung the LTX decode and crashed the Wan2.2 decode, so
   prefer temporal tiling if both work.

5. **Capabilities.** T2V sets: `text`. I2V sets: `start_image` (the recipe
   has no end image). These match `VideoRecipes` in `recipes.go`.

6. **Notes.** Each set's notes say, in one or two sentences:
   - what it is for: text or start picture, 480p or 720p;
   - what distilled means for it. "CFG-distilled: one pass per step instead of
     two, so about twice as fast, same 50 steps". "Step-distilled: 8 steps,
     the fastest; Tencent suggests 12 for a little more quality";
   - the Short time and reservation on the R9700.

7. **A4000 check.** On compute2, download one 480p set and one 720p set.
   Measure each at Short and make a video, to show the sets work on a 16 GB
   card.

8. **Help.** List the sets in `help.html`, grouped by text or picture, with a
   short line on which to pick: "step-distilled 480p I2V for speed, 720p I2V
   for detail".

## Done when

- All eight sets download from the catalog. Downloading a second set fetches
  only its diffusion model, and fetches nothing at all for the Qwen2.5-VL file
  when Qwen-Image Edit 2511 is already downloaded.
- Deleting one set keeps the shared files while another set uses them.
- Each set is measured at Short on the R9700 and makes a correct video. Its
  times and reservation are in its notes.
- One 480p set and one 720p set work on the A4000.
- `go test ./...` and the JS tests pass.
