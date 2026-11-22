# Phase 04 — The "Wan2.2 S2V 14B (Q4)" catalog set

**Depends on:** Phase 03 (recipe, `audio_encoder` slot, caps, Measure) ·
**Enables:** Phase 05 (the field test downloads this set)

## Goal

A catalog set that downloads in one click from the Models page and runs
speech-to-video with sd.cpp's recipe defaults. It shares the UMT5 text model
and Wan2.1 VAE with the A14B sets, so a machine that has those downloads only
the new model and the audio encoder. It shows the "start image" and "audio"
badges.

## Files touched

- `internal/generation/catalog.go`:
  - a shared `wav2vec2Large` component beside `wan21VAE`
  - the new entry after `wan22-i2v-a14b-lightning`
- `internal/generation/video_test.go` — `TestCatalogReposAllowed`: add
  `"QuantStack/Wan2.2-S2V-14B-GGUF"` under "Video sets", after the I2V A14B
  repo. `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` is already allowed.
- `internal/api/gen_models_test.go` — `TestGenCatalogTags`: the new ID's tags.
- `docs/reference/measured-reservations.md` — a `{ #table }` row for the set,
  with "not measured yet" in the measured columns. Phase 05 fills it.
- `docs/guides/generation.md` — the catalog list: the new set, one line.

## Steps

1. Add the shared component, with the byte size read from Hugging Face on
   2026-11-22:
   `wav2vec2Large = Component{Slot: "audio_encoder", Repo:
   "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", File:
   "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors",
   Bytes: 630997322}`.
2. Before writing the entry, read the exact byte size of
   `QuantStack/Wan2.2-S2V-14B-GGUF` → `Wan2.2-S2V-14B-Q4_K_M.gguf` from the
   Hugging Face API. It is 12.91 GiB; `Bytes` must be the exact value.
3. Add the entry:
   - `ID: "wan22-s2v-14b"`, `Name: "Wan2.2 S2V 14B (Q4)"`,
     `Kind: KindVideo`, `Recipe: "wan22-s2v"`.
   - `Components`: the Q4_K_M `diffusion_model`, `umt5Q8`, `wav2vec2Large`,
     `wan21VAE`.
   - `LaunchArgs`: the recipe defaults (`--steps 20 --cfg-scale 4.5
     --flow-shift 3 --sampling-method euler`), plus `--diffusion-fa`,
     `--temporal-tiling` and `--backend te=cpu`, as on the A14B sets.
   - `FlagNotes`: `--temporal-tiling` and `--backend te=cpu` reuse the A14B
     wording (the same Wan2.1 VAE; the ROCm hang with `--vae-tiling`).
   - `Capabilities: []string{CapStartImage, CapAudio}`;
     `License: "Apache-2.0"`.
   - `Notes`: "Speech-to-video: makes the person in a start image speak or sing
     along with an audio track (WAV, MP3 or FLAC). Needs both. With no length
     given, the video is as long as the audio, up to 5 seconds at 1280×720 and
     16 fps. Newer sd-server builds put the audio in the WebM; older ones make
     a silent video. Shares its text model and VAE with Wan2.2 I2V A14B."
   - `DefaultWidth: 832, DefaultHeight: 480, DefaultFrames: 33,
     DefaultFPS: 16, MaxFrames: 81, MaxWidth: 1280, MaxHeight: 720`.
   - `VideoSizes`: Short 33 × 832×480, Medium 49 × 1280×720, Full 81 ×
     1280×720.
   - `FullMeasureTime`: "Not measured yet; expect a few minutes for Short, as
     for Wan2.2 I2V A14B." Phase 05 replaces it with the measured time.
   - `SizeMultiple: 16`.
4. `catalogTags` (Phase 02) gives the badges "start image" and "audio". Add
   the row to `TestGenCatalogTags`.
5. Docs: add the reservations table row and the catalog line. The Help page
   lists no sets, so it does not change.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make`
- JS tests
- `make docs`

## Test plan

These existing tests cover the entry:
- `TestCatalogEntriesAreValid`
- `TestCatalogVideoSets`: the caps equal the recipe caps
  `[start_image, audio]`, `SizeMultiple` > 0, and the ceiling ≥ the test size
- `TestCatalogTextEncodersOnCPU`: `te=cpu`
- `TestVideoRecipeSizeMultiples`: 16 = the recipe's 16
- `TestCatalogReposAllowed`
- `TestHasHighNoiseModel`: unchanged, because S2V has no high-noise model

Manual check on this machine's dev server: the Models page catalog shows the
set with both badges and the right download size. Do not download it here;
Phase 05 does that on the test machines.

## Commit

`feat(video): add the Wan2.2 S2V 14B catalog set`

## Rollback

Revert the commit. The set leaves the catalog. Copies that were already
downloaded stay as sets and keep working as custom-like sets, because
Phase 03's recipe stays in place.
