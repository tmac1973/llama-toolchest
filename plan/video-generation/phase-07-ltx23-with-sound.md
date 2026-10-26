# Phase 07 — LTX-2.3 Distilled 1.1 Set, with Sound

**Depends on:** Phase 01 (the `embeddings_connectors` and `audio_vae` slots and
the `ltx23-distilled` recipe), Phase 02 (Measure for video), Phase 03 (the job
API), Phase 04 (the proven pipeline and `verification.md`), Phase 05 (measure
presets and measured limits), Phase 06 (its set comes before this one in the
catalog, and its results are part of the success check) · **Enables:** the release: after this phase the overview's success
criteria can all be checked

## Goal

The catalog offers LTX-2.3 distilled 1.1:
- text-to-video, image-to-video and start-and-end videos;
- 8 steps at 24 fps;
- a sound track in WebM output.

This is the first set with a Gemma text model, embeddings connectors and an
audio VAE, and the first under a licence that is not fully free (the LTX-2
Community Licence, plus Google's Gemma terms for its text model). The set is
measured and checked on asmodean, tried on a 16 GB A4000, and the project's
results are summed up for the release notes.

## Files touched

- `internal/generation/catalog.go`: the LTX-2.3 distilled 1.1 entry.
- `internal/generation/catalog_test.go`: its licence names both licences;
  `unsloth/LTX-2.3-GGUF` and `unsloth/gemma-3-12b-it-GGUF` join the
  allowed-repos list.
- `plan/video-generation/verification.md`: an LTX-2.3 section, the summary
  against the overview's success criteria, and a release-notes paragraph.
- `web/templates/help.html`: a reservations-table row; the video jobs part
  says which sets make sound and that WebP is silent.

## Steps

1. **Catalog entry.** Add after Wan2.2 I2V A14B:
   - ID `ltx23-distilled`, Name "LTX-2.3 distilled 1.1 (Q4)", Kind
     `KindVideo`, Recipe `ltx23-distilled`.
   - Components:
     - `diffusion_model`: `unsloth/LTX-2.3-GGUF`,
       `distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf`, 14194441248
       bytes.
     - `llm`: `unsloth/gemma-3-12b-it-GGUF`, `gemma-3-12b-it-Q4_K_M.gguf`,
       7300778336 bytes.
     - `embeddings_connectors`: `unsloth/LTX-2.3-GGUF`,
       `text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors`,
       2312144712 bytes.
     - `vae`: `unsloth/LTX-2.3-GGUF`,
       `vae/ltx-2.3-22b-dev_video_vae.safetensors`, 1452256522 bytes.
     - `audio_vae`: `unsloth/LTX-2.3-GGUF`,
       `vae/ltx-2.3-22b-dev_audio_vae.safetensors`, 364853140 bytes.
   - Capabilities `text`, `start_image`, `end_image`.
   - License "LTX-2 Community Licence (free under US$10M yearly revenue);
     text model: Gemma Terms of Use".
   - LaunchArgs: `--steps 8 --cfg-scale 1 --sampling-method euler
     --diffusion-fa --vae-tiling`. These are the `ltx23-distilled` recipe's
     defaults.
   - DefaultWidth 768, DefaultHeight 512, DefaultFrames 33, DefaultFPS 24,
     MaxFrames 121, MaxWidth 1280, MaxHeight 720 (121 frames is 5 s at 24 fps).
   - VideoSizes (Phase 05): Short 33 frames at 768×512; Medium 65 frames at
     1280×720; Full 121 frames at 1280×720.
   - Notes: "Text-to-video, image-to-video and start-and-end videos with sound;
     8 steps. WebM videos have a sound track; animated WebP is silent. Up to
     5 seconds at 1280×720. Licence: free unless the organisation makes
     US$10 million a year or more; terms at
     https://huggingface.co/Lightricks/LTX-2.3. Its Gemma 3 text model is under
     Google's Gemma Terms of Use." The measured figures are added in step 5.
2. **Test.** In `catalog_test.go`, check that the LTX entry's licence names
   both licences, and add `unsloth/LTX-2.3-GGUF` and
   `unsloth/gemma-3-12b-it-GGUF` (both checked to be ungated) to the
   allowed-repos list from Phase 01.
3. **asmodean.** Deploy, download the set, and measure it alone on GPU 0 at
   the Short preset. If sd-server crashes in the VAE decode with
   `--vae-tiling`, as the Wan2.2 VAE did in Phase 01, remove `--vae-tiling`
   from the entry's LaunchArgs with a comment and measure again. Run through
   the job API:
   1. Text-to-video at the defaults, WebM.
   2. Image-to-video, WebM.
   3. Start-and-end, WebM.
   4. Text-to-video as animated WebP.

   Then measure at the Full preset, which sets the reservation and the
   instance's limit to Full, and run:
   5. 1280×720 at 121 frames, watching that the card stays within the
      reservation.

   Check:
   - that jobs 1–3 and 5 play with sound in Firefox and Chromium;
   - that `ffprobe` shows a video and an audio stream in the WebM;
   - that the WebP has no sound and plays in an image viewer.

   Record times, peaks, reservations and file sizes, and set the entry's
   `FullMeasureTime` to the Full measuring time, in the form "about N minutes
   on a Radeon AI PRO R9700".
4. **compute2 (16 GB A4000).** Measure the set at Short and run job 1 on one
   A4000.
   Record the peak and time, or the error and where it failed. A failure here
   is written down, not fixed.
5. **Write it down.** Add an LTX-2.3 section to `verification.md`. Then go
   through every success criterion in `overview.md`, marking each met or not
   met with a pointer to the result. Under "Not run", record that video was
   not tested on a Vulkan build in this project. Add a short release-notes paragraph:
   - the three sets;
   - the video job API and waiting route;
   - formats;
   - the licences.

   Add the measured figures to the catalog notes and the help table (the Short reservation in the table row, Full in the note under it, as Phase 05 sets out); in the
   help page's video jobs part, say that LTX-2.3 WebMs carry sound and WebP
   output is silent.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/...`
- `gofmt -l internal/` prints nothing.

## Test plan

- `TestCatalogEntriesAreValid`, the licence test and the allowed-repos test
  pass.
- The asmodean runs in step 3 succeed, with sound in WebM and none in WebP.
- The compute2 result is recorded either way.
- Every success criterion in `overview.md` is marked in `verification.md`.

## Commit

`feat(generation): add the LTX-2.3 distilled video set with sound`

## Rollback

Revert the commit. Downloaded copies of the set stay as a set without a
catalog entry and keep working with their saved launch settings until deleted
on the Models page. Nothing else changes.
