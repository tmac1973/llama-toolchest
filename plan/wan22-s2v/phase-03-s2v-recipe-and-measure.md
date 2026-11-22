# Phase 03 — The wan22-s2v recipe, audio_encoder slot and Measure

**Depends on:** Phase 01 (WAV writer for the test track), Phase 02 (`CapAudio`,
`ModelSet.FixedFPS()`, the audio input) · **Enables:** Phase 04 (the catalog
set uses the recipe and slot)

## Goal

Make the `wan22-s2v` recipe available. sd-server builds whose recipe book has
it then offer it in the custom set form, with an "Audio encoder" slot passed
as `--audio-encoder`, the capabilities `start_image` + `audio`, and 16 fps.
Measure can test such a set, because it sends a generated audio track. After
this phase a user can build a working S2V custom set by hand.

## Files touched

- `internal/generation/launch.go` — `sdSlotFlags`: add
  `"audio_encoder": "--audio-encoder"` in the video group.
- `internal/generation/recipes.go`:
  - `SlotLabels`: move the existing `"audio_encoder": "audio encoder"`
    entry out of the "no sd-server flag" group into the video group, as
    `"audio_encoder": "Audio encoder"`. Do not add a second key.
  - `RecipeCapabilities`: add `CapAudio` for audio recipes.
- `internal/generation/recipe_overlay.go`:
  - remove `wan22-s2v` from `recipeLeftOut`
  - add `recipeAudioInput`
  - add a `recipeLabels` entry for `wan22-s2v`
- `internal/generation/recipebook.go` — `unavailableReason`: skip the
  `ref_audios` feature when `recipeAudioInput[r.ID]`.
- `internal/api/gen_vram.go` — `runMeasureVideo`: send a test track for
  `CapAudio` sets, and use `set.FixedFPS()` when it is set.
- `internal/api/gen_video.go` — `measureAudio(seconds float64) []byte` beside
  `measurePicture`.
- `internal/api/gen_set_form.go`:
  - add `FixedFPS int` to the form data struct that `genSetFormFrom` fills,
    set from `ActiveRecipes().Get(recipe)`
  - a video custom set whose recipe has a fixed frame rate saves
    `DefaultFPS = FixedFPS`, whatever the form's fps field says
- `web/templates/partials/gen_set_form.html` — under the fps field,
  `{{if .FixedFPS}}This model always makes {{.FixedFPS}} fps.{{end}}`. The
  form already reloads when the recipe changes.
- `docs/guides/generation.md` — custom sets: the Audio encoder slot, and that
  S2V sets need a start image and audio.
- `web/templates/help.html` — `#generation`: one sentence saying video sets
  can also take a speech track. Keep the anchors.
- Tests: `internal/generation/video_test.go`,
  `internal/generation/flag_reasons_test.go`,
  `internal/generation/recipebook_test.go`,
  `internal/api/gen_video_test.go`, `internal/api/gen_set_form_test.go`.

## Steps

1. Add the `audio_encoder` flag, and move its label as above. `parseRecipeFile` keeps the slot
   only when `sdSlotFlags` has it. Without the flag the required
   `audio_encoder` makes the recipe unavailable even after step 2.
2. In `recipe_overlay.go`:
   - Add `var recipeAudioInput = map[string]bool{"wan22-s2v": true}`, with a
     comment saying these are the recipes whose `ref_audios` toolchest sends
     (one track, the job's `audio`).
   - Delete the `wan22-s2v` line from `recipeLeftOut`.
   - Keep `"ref_audios"` in `recipeNeedsInput`, so any other recipe that
     needs audio (now or in a later sd.cpp) stays unavailable until toolchest
     supports it. (`minimax-h3-ref2va` is unavailable for its `ref_videos`
     feature, and `minimax-h3-fl2va` has no `ref_audios`; neither changes.)
   - Add the labels `{"t5xxl": "UMT5-XXL text encoder", "vae": "Wan2.1 VAE",
     "audio_encoder": "wav2vec2 audio encoder (Large English)"}`.
3. In `unavailableReason`, skip the feature `ref_audios` when
   `recipeAudioInput[r.ID]` is true. Every other feature in
   `recipeNeedsInput` still makes the recipe unavailable.
4. In `RecipeCapabilities`, video branch: after `CapEndImage`, add `CapAudio`
   when `r.HasFeature("ref_audios") && recipeAudioInput[r.ID]`.
   - `wan22-s2v` has `init_image` and is not in `recipeTextStart`, so its caps
     are `[start_image, audio]`. It has no `end_image` feature.
5. **Measure.**
   - `measureAudio` returns `audiofile.WriteWAV` of a 220 Hz sine at 16 kHz
     mono, amplitude 0.3, of the given length.
   - In `runMeasureVideo`, set `fps := set.FixedFPS()`, or
     `max(set.DefaultFPS, 1)` when it is 0, and use it for `req.FPS`.
   - When the caps include `CapAudio`, set `req.RefAudios` to the base64 of
     `measureAudio(float64(frames)/float64(fps) + 0.5)`, so the track covers
     every frame.
   - The start image is already sent, because S2V has `CapStartImage`.
6. **Custom set form.** In the video branch of the form save
   (`gen_set_form.go`, where `DefaultFPS` is read), look up the recipe with
   `ActiveRecipes().Get`. If its `FixedFPS > 0`, store that value. The
   template note reads the new `.FixedFPS` form-data field.
7. **Docs.**
   - `docs/guides/generation.md`: add the slot to the custom-set slot list,
     and add a short "Speech-to-video" paragraph: start image plus audio;
     length from the audio; 16 fps. It links to `api.md#generation-api`.
   - Help: one sentence plus the existing "More in the docs" link.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make`
- JS tests
- `make docs`

## Test plan

- `TestVideoRecipeSlotsHaveFlags` passes with `wan22-s2v` included: every slot
  has a flag, and its caps are not empty.
- New in `video_test.go`: `RecipeCapabilities("wan22-s2v", nil)` is
  `[start_image, audio]`.
- New in `recipebook_test.go`: in the snapshot, `wan22-s2v` has an empty
  `Unavailable`, its required slots are `diffusion_model`, `t5xxl` and
  `audio_encoder`, and it has `OneOf [vae tae]`.
- Update `TestRecipeAvailability` (`recipebook_test.go`): remove the
  `"wan22-s2v": "needs an audio input"` row, and add a new check in the same
  test that the snapshot's `wan22-s2v` has `Unavailable == ""`. The `minimax-h3-ref2va` row
  ("needs a reference video input") stays as it is.
- `TestSlotLabels` passes with the new label and the `recipeLabels` entry.
- Extend `TestMeasureVideoUsesHeaviestMode` with a fake `wan22-s2v` set. The
  fake sd-server receives `init_image`, `ref_audios` with one WAV (it starts
  with `RIFF`, and its length is at least frames/16 s), `fps` 16, and no
  `end_image`.
- `gen_set_form_test.go`: saving a `wan22-s2v` custom set with fps 24 stores
  `DefaultFPS` 16. Saving a `wan22-i2v` set with fps 24 stores 24.
- Manual check on this machine's dev server (`run` skill): open the custom
  set form, pick `wan22-s2v`, and see the Audio encoder slot and the 16 fps
  note.

## Commit

`feat(video): offer the Wan2.2 S2V recipe with an audio encoder slot`

## Rollback

Revert the commit. `wan22-s2v` goes back to unavailable. Custom S2V sets that
users made would then fail to start (an unknown `audio_encoder` slot). They
stay listed and can be deleted. Phases 01 and 02 stay in place and are
harmless on their own.
