# Phase 02 — Audio input on video jobs

**Depends on:** Phase 01 · **Enables:** Phase 03 (recipes and sets that have
the `audio` capability can then be used)

## Goal

The video job API takes an `audio` input and sends it to sd-server as
`ref_audios`. A new capability, `audio`, says that a set takes audio and must
have it. With no length in the request, the length comes from the audio. A
fixed frame rate, if the set has one, is used before frames are worked out.
No set has the `audio` capability yet, so real sets only gain a new refusal
("cannot take audio"). Tests use a fake set that has it.

One set kind that can run today changes: custom sets on the
`minimax-h3-fl2va` recipe (available, `fixed_fps: 24`) now always get 24 fps.
sd-server already forces 24, so only the shown request, the frames worked out
from `seconds`, and `/v1/models` change, and they become right.

## Files touched

- `internal/generation/recipes.go` — add `CapAudio = "audio"` beside
  `CapText`, `CapStartImage` and `CapEndImage`.
- `internal/generation/recipebook.go`:
  - add `FixedFPS int` to `Recipe`
  - read `video_constraints.fixed_fps` into it in `parseRecipeFile` (0 when
    the recipe has none)
- `internal/generation/catalog.go` — add `func (m ModelSet) FixedFPS() int`
  beside `MaxEditImages`. It returns the set's recipe's `FixedFPS` from
  `ActiveRecipes()` (the active build's recipe book, as `RecipeCapabilities`
  uses), or 0. It is worked out at run time, not stored, so it is
  right for catalog and custom sets alike.
- `internal/generation/recipebook_test.go` — test the new field.
- `internal/api/gen_video_jobs.go`:
  - `videoInput`: add `Audio string`.
  - `readVideoInput`: read the `audio` file part, and strip a data URL in JSON.
  - `videoReservedExtra`: add `ref_audios`.
  - `buildVideoRequest`: the checks, the length and fixed fps rules.
  - `videoJobRequest`: add `Audio bool` and `AudioSeconds float64`.
  - `videoInputs`: add the audio wording.
- `internal/api/gen_video.go` — `vidGenRequest`: add
  `RefAudios []string \`json:"ref_audios,omitempty"\``.
- `internal/api/v1models.go` — `meta.video.fixed_fps` when `FixedFPS > 0`.
- `internal/api/gen_models.go` — `catalogTags`: an "audio" tag ("Needs a
  speech or music track you give.").
- `web/apidocs/openapi.yaml`:
  - an `audio` field in `requestBodies.VideoRequest` and `schemas.VideoRequest`
  - `audio` in the `meta.capabilities` description
  - `fixed_fps` under `meta.video`
  - `audio`/`audio_seconds` in `VideoJob.request`
  - the new error codes in `responses.VideoBadRequest`
- `docs/api.md` — the video field list, the `audio` capability, the length
  rule and the sound note.
- Tests: `internal/api/gen_video_jobs_test.go`,
  `internal/api/gen_models_test.go`, `internal/api/gen_video_test.go`.

## Steps

1. **Reading.**
   - Multipart: add `"audio": &in.Audio` to the file-part loop. It is read
     whole and base64-encoded like the images.
   - JSON: `in.Audio = stripDataURL(in.Audio)`.
   - The request limit stays `maxImageRequestBytes` (128 MiB).
2. **Capability checks**, added to the existing switch in
   `buildVideoRequest` after the image checks:
   - `in.Audio != ""` and the set lacks `CapAudio` → `input_not_supported`,
     "%s cannot take audio".
   - `in.Audio == ""` and the set has `CapAudio` → new code `audio_required`,
     "%s needs an audio track (WAV, MP3 or FLAC)".
   - The existing `start_image_required` check already refuses a missing
     start image on a set without `CapText`. S2V sets have no `CapText`
     (Phase 03), so they need both inputs.
3. **Decode.** When `in.Audio` is set, base64-decode it; bad base64 gives
   `bad_value`, "audio is not valid base64". Then call `audiofile.Decode`.
   - `Decode` returns only errors that wrap `audiofile.ErrFormat` or are
     `audiofile.ErrTooLong` (Phase 01). Both → new code `bad_audio`, with the
     error text as the message.
   - Keep `a.WAV`, base64-encoded, for `RefAudios`, and keep `a.Seconds`.
4. **fps.** Work it out before frames:
   - If `set.FixedFPS() > 0`, then `fps = set.FixedFPS()`, whatever the request
     says. A sent fps that differs is not an error.
   - Otherwise keep today's rule (the request, else `set.DefaultFPS`).
5. **Frames**, in this order:
   1. `frames` sent → use it.
   2. `seconds` sent → `round(seconds × fps)`.
   3. Audio present → `n = ceil(a.Seconds × fps)`, then rounded up to the
      next 4n+1: `frames = ((n − 1 + 3) / 4) × 4 + 1` (integer division; for
      n ≤ 1, frames = 1). The S2V recipe rounds a frame count down to 4n+1
      (`frame_rounding: "floor"`), so sending 4n+1 keeps the end of the audio.
      A 3.0 s track gives n = 48 and frames = 49.
   4. Otherwise `set.DefaultFrames`.
   - New: when the frames came from the audio and pass `set.MaxFrames`, refuse
     with `over_limit`: "the audio is %.1f s; %s makes at most %.1f s. Send a
     shorter track, or send frames or seconds to cut it". The measured-size
     check (`checkMeasuredSize`) applies as for any length.
   - When the client sent a length, a longer audio is fine: sd-server cuts it,
     and a shorter one is padded with silence.
6. **The over_limit message** uses the `fps` from step 4, not
   `set.DefaultFPS`, so the seconds it shows are right for fixed-fps sets.
7. **Forwarding.** Set `vid.RefAudios = []string{wavBase64}` when audio is
   present.
8. **Shown request.** Set `shown.Audio = true` and
   `shown.AudioSeconds = a.Seconds`, rounded to 0.1. The JSON names are
   `audio` and `audio_seconds`.
9. **`videoInputs`.** Add the audio cases first:
   - `RefAudios` and `InitImage` → "a start image and audio"
   - `RefAudios` alone → "audio"
10. **`/v1/models`.** When `set.FixedFPS() > 0`, add
    `video["fixed_fps"] = set.FixedFPS()`.
11. **Catalog tag.** In `catalogTags`, add "audio" when the set's
    capabilities include `CapAudio`.
12. **Docs.**
    - `docs/api.md`: the field is `audio` (multipart file, or base64 / data
      URL in JSON); WAV, MP3 and FLAC; one track; the length rule;
      `fixed_fps`.
    - Sound: sd-server builds from 6cd0ef1 put the audio in a WebM; older
      builds and WebP give a silent video, and the client adds the track (an
      ffmpeg example).
    - Mirror this in `openapi.yaml`.
    - Help needs no change: it lists no fields.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make` (the repo build target), and the JS tests as CI runs them

## Test plan

Unit tests in `internal/api/gen_video_jobs_test.go`, using a fake video set
with recipe `wan22-s2v` (so `FixedFPS()` is 16 from the snapshot), caps
`start_image` + `audio` and `MaxFrames: 81`, and the Phase 01 test tones:

- **Refusals.** Extend `TestVideoJobChecks` with these new rows:
  - audio on a set without `audio` → `input_not_supported`
  - no audio on the fake set → `audio_required`
  - no start image on the fake set → `start_image_required`
  - an OGG file → `bad_audio`
  - bad base64 → `bad_value`
  - `extra.ref_audios` → `bad_value`
- **Fixtures.** Tests make WAVs of any length with `audiofile.WriteWAV`
  (16 kHz mono sine), and read the MP3 and FLAC tones (1 s each) with
  `os.ReadFile("../audiofile/testdata/tone.mp3")` and `tone.flac`. The OGG
  refusal reads `../audiofile/testdata/tone.ogg`.
- **Length from audio:**
  - a 3.0 s WAV with no length gives 49 frames and fps 16, even when the
    request sends fps 24
  - a 1 s MP3 tone with no length gives 17 frames
  - `seconds: 2` gives 32 frames
  - `frames: 33` gives 33
  - a 6 s WAV with no length gives `over_limit`, and its message says "6.0 s"
    and "5.1 s"
  - the same WAV with `frames: 81` passes
- **Formats.** MP3 and FLAC are sent to sd-server as a WAV: decode the
  `ref_audios[0]` that the fake sd-server receives, and check that it starts
  with `RIFF`.
- **Multipart.** Extend `TestVideoJobMultipart` with an `audio` file part.
- **Inputs wording.** Extend `TestVideoInputs` with the two new strings.
- **`/v1/models`.** Extend `TestV1ModelsVideoMeta`: `fixed_fps` is present for
  the fake set and missing for a `wan22-i2v` set.
- **Recipe book.** In `recipebook_test.go`, the snapshot's `wan22-s2v` has
  `FixedFPS` 16, `minimax-h3-fl2va` has 24 and `wan22-i2v` has 0.
- **MiniMax fl2va.** A fake `minimax-h3-fl2va` set that gets `fps: 30` and
  `seconds: 2` gives fps 24 and 48 frames.
- **Catalog tags.** `TestGenCatalogTags` stays green; no catalog set has the
  `audio` capability yet.

## Commit

`feat(video): accept an audio track on video jobs`

## Rollback

Revert the commit. No set has `CapAudio`, so the only behaviour that goes
away is the new refusals and the 24 fps rule for `minimax-h3-fl2va` custom
sets, which sd-server applies anyway. Stored jobs gain only
the optional `audio`/`audio_seconds` fields, which older code ignores.
