# Phase 05 — Measured Video Size and Measuring Inside the Card's Budget

**Depends on:** Phase 01 (video sets and their ceiling fields), Phase 02
(Measure for video sets), Phase 03 (the video job API and its checks),
Phase 04 (the hardware findings this phase answers) · **Enables:** Phases 06
and 07, whose sets get measured sizes from the start

## Goal

A video instance can only make videos it has been measured for. Phase 04
found two problems:
- **Larger videos go over the reservation.** The reservation came from a short
  test video, but requests up to the model's ceiling were allowed. At the
  ceiling, sd-server went about 4 GiB over its reservation.
- **Smaller cards can't be measured.** On a 16 GB card, Measure saw sd-server
  fill the card and suggested more than the card can give.

After this phase:
- **Measure lets the user choose the video size** for a video set: a preset
  (Short, Medium, Full) or a custom size up to the model's limit.
- **The measured size becomes the instance's limit.** A request bigger than
  what the instance was measured for is refused, with a message that says why
  and how to allow it.
- **The ceiling is the model's limit.** The catalog's maximum stays as the
  most the model can make at all, and Measure cannot go past it.
- **Measure runs inside what the card can give** for sd-server instances
  (image and video), and never suggests a reservation the card cannot hold.

## Files touched

- `internal/generation/types.go`:
  - `VideoSize` type;
  - `Instance` gains `MeasuredFrames`, `MeasuredWidth`, `MeasuredHeight` and
    `MeasuredSuggestedMiB`.
- `internal/generation/catalog.go`: `CatalogEntry.VideoSizes` (the measure
  presets) and `CatalogEntry.FullMeasureTime`; TI2V 5B's three presets and its
  measuring time.
- `internal/generation/video_limit.go` (new):
  - `VideoPresets(set)`;
  - `VideoLimit(inst, set)`, the size an instance may make;
  - `PixelFrames(frames, w, h)`.
- `internal/generation/launch.go`: `LaunchOptions.MaxVRAMMiB` replaces
  `OmitMaxVRAM`.
- `internal/api/gen_vram.go`:
  - `handleGenMeasure` reads the chosen size;
  - `measureInstance` runs inside the card's budget, caps its suggestion, saves
    the measured size, and has a 2-hour limit for video;
  - `runMeasureVideo` uses the chosen size.
- `internal/api/gen_video_jobs.go`: `buildVideoRequest` checks the instance's
  measured limit after the model's.
- `internal/api/v1models.go`: `meta.video` adds the measured limit.
- `internal/api/gen_instances.go`: `genInstanceRow` gains the video fields the
  row and the Measure dialog need (step 12); editing an instance clears the
  measured size with the reservation.
- `web/templates/partials/gen_instances.html`: the Measure button opens a size
  dialog for video instances; the row shows the measured size.
- `web/templates/server.html`:
  - the new `gen-measure-video-dialog`;
  - `openGenMeasureStart` offers the same size choice for video instances.
- `web/templates/help.html`: the Measure step, the reservations notes and the
  video jobs section explain measured sizes.
- Tests:
  - `internal/generation/video_test.go`;
  - `internal/api/gen_video_jobs_test.go`;
  - `internal/api/gen_video_test.go`;
  - `internal/api/gen_vram_test.go`.
- `plan/video-generation/verification.md`: a Phase 05 section.

## Steps

1. **Types.** In `types.go`:
   - Add `type VideoSize struct { Label string; Frames, Width, Height int }`
     (JSON `label`, `frames`, `width`, `height`).
   - Add `MeasuredFrames`, `MeasuredWidth` and `MeasuredHeight` to `Instance`
     (JSON `measured_frames` etc., `omitempty`): the video size of the
     measurement that set `MeasuredPeakMiB`. They stay zero for image and music
     instances.
   - Add `MeasuredSuggestedMiB` (JSON `measured_suggested_mib`, `omitempty`):
     the reservation the last measurement suggested, after the cap in step 6.
     It is stored for every sd-server measurement (image and video), because
     the cap depends on what else was on the card at the time. Music
     measurements leave it zero.
   - `Instance.clone` needs no change (no new slices).
2. **Presets in the catalog.** Add `VideoSizes []VideoSize` to `CatalogEntry`
   and copy it into `ModelSet` (new field `VideoSizes`, JSON `video_sizes`) in
   `handleGenCatalogDownload`. The first preset is always the set's test size
   (`DefaultFrames`, `DefaultWidth`, `DefaultHeight`); the last is at most the
   model's limit (`MaxFrames`, `MaxWidth`, `MaxHeight`). For Wan2.2 TI2V 5B:
   - Short: 33 frames, 832×480;
   - Medium: 81 frames, 1280×704;
   - Full: 121 frames, 1280×704.

   Every frame count is 4n+1, so sd-server does not round it.

   Also add `FullMeasureTime string` to `CatalogEntry`: a short phrase saying
   how long measuring the largest preset took, for the Measure dialog. For
   TI2V 5B it is "about 21 minutes on a Radeon AI PRO R9700". It is not copied
   into the set; the dialog reads it from the current catalog entry.
3. **Presets and limits.** New `internal/generation/video_limit.go`:
   - `PixelFrames(frames, w, h int) int64` returns `frames × w × h`: how much
     video there is, which is what drives memory.
   - `VideoPresets(set ModelSet) []VideoSize` returns, in this order:
     1. the current catalog entry's `VideoSizes`, when the set has a
        `CatalogID` that is still in the catalog (as `RunArgs` does, so sets
        downloaded before this phase get presets too);
     2. otherwise the set's own copy, `set.VideoSizes` (a catalog set whose
        entry was later removed);
     3. otherwise one preset, "Short", at the set's test size (custom sets).
   - `VideoLimit(inst Instance, set ModelSet) (size VideoSize, measured bool)`
     returns:
     - **the measured size, with `measured` true,** when the instance has one
       and its reservation is at least `MeasuredSuggestedMiB`;
     - **otherwise the set's test size, with `measured` false.** This covers
       an instance whose reservation was set by hand below what its
       measurement suggested, or one never measured.

     `size.Label` is the name of the preset whose frames, width and height
     match exactly (for example "Short"), or "custom size" when none does.

     `suggestReservation` moves from `internal/api/gen_vram.go` to this file,
     exported as `SuggestReservation`, so both packages use one function.
4. **Launch options.** In `launch.go`, replace `OmitMaxVRAM bool` with
   `MaxVRAMMiB int`:
   - 0 means the instance's reservation, as now;
   - a positive value is used as `--max-vram` instead.

   ace-server takes no memory limit, so `MaxVRAMMiB` does nothing for ace.
5. **Measure takes a size.** `handleGenMeasure` reads `frames`, `width` and
   `height` from the form.
   - **Video instances:** all three are required, positive, within the model's
     limit (in either orientation, as in Phase 03), sizes in multiples of 8.
     Anything else is a 400 with one of these messages:
     - a field missing or not a positive number: "Choose a video size to
       measure: frames, width and height";
     - over the model's limit: "Wan2.2 TI2V 5B makes at most 121 frames and
       1280×704; choose a size within that" (built from the set's name and
       limit);
     - not a multiple of 8: "Width and height must be multiples of 8".

     The Start-then-measure flow (`start=1`) passes the size the same way.
   - **Image and music instances:** the fields are ignored.

   The size goes to `measureInstance` and on to `runMeasureVideo`, which uses
   it instead of the set's test size. The frame rate stays `DefaultFPS`.
6. **Measure inside the card's budget (image and video).** In
   `measureInstance`, for `KindSD` and `KindVideo`:
   1. Before launching, work out the instance's room: the card's budget
      (`models.CardBudgetMiB`) minus loaded LLMs and the other instances'
      reservations, as `genFitCheck` counts them. When that is less than
      1 GiB, refuse as today.
   2. Launch with `LaunchOptions{MaxVRAMMiB: room}`. sd-server then plans
      inside what the card can give, instead of filling it.
   3. After the test run, the suggestion is
      `min(SuggestReservation(peak), room)`, stored in `MeasuredSuggestedMiB`.
      - **Video instances:** the suggestion is always applied: the user chose
        the size on purpose, so measuring larger (or smaller) sets the
        reservation, and with it the instance's limit, to that size.
      - **Image instances:** applied as Measure does today, automatically when
        the instance has no reservation yet; otherwise the reservation stays
        and the row offers "Use suggested".

      Because the suggestion never exceeds the room, it always fits, so the
      branch that disables the instance when the suggestion does not fit is no
      longer reached for these kinds.
   4. **A video size over the room is not granted.** Added after the hardware
      check: sd-server went past `--max-vram` decoding large videos, because
      tiled decoding crashes for the Wan2.2 VAE (on the R9700, Medium peaked at
      30.6 GiB held to 28.2). So for a video measurement whose peak is more
      than the room plus 1 GiB (`measureOverheadMiB`, for the GPU runtime's own
      memory, which the whole-card reading includes and `--max-vram` does not
      count), Measure refuses the size: "81 frames at 1280x704 needs about
      30.6 GiB, but this card can give the instance 28.2 GiB; measure a smaller
      size". The instance keeps its previous size and reservation. Image sets
      are still capped and granted.

   Music instances keep measuring with no limit, as now (`MaxVRAMMiB` 0, and
   the existing fit check). The measurement note says when the suggestion was
   capped: "Peak 15.3 GiB; reservation capped at 14.5 GiB, all this card can
   give it."
7. **Save the measured size.** After a video measurement, store `frames`,
   `width` and `height` in `MeasuredFrames`, `MeasuredWidth` and
   `MeasuredHeight`, next to `MeasuredPeakMiB` and `MeasuredSuggestedMiB`. For
   every instance, the row's suggested value and its "Use suggested" button use
   `MeasuredSuggestedMiB` when it is set, and the old peak-plus-10% figure
   otherwise (music instances, and instances measured before this phase).
   A video measurement's note reads "Measured for 81 frames at 1280×704
   (Medium): peak 27.0 GiB. Reservation set to 29.5 GiB." When the suggestion
   was capped (step 6), the last sentence reads instead "Reservation capped at
   14.5 GiB, all this card can give it." The note names the size:
   "Measured for 81 frames at 1280×704: peak 27.0 GiB …".
8. **2-hour limit for video.** `measureTimeout(KindVideo)` becomes 2 hours.
   The Full preset took 21 minutes on the R9700, and a slower card working
   inside a smaller budget can take several times as long. Image and music
   keep 10 minutes.
9. **Clearing the measured size.** Wherever an instance's reservation is
   cleared (editing its set, build or GPU in `handleGenEditInstance`), clear
   `MeasuredFrames`, `MeasuredWidth`, `MeasuredHeight` and
   `MeasuredSuggestedMiB` too. Setting the
   reservation by hand (`handleGenSetReservation`) does not clear them;
   `VideoLimit` decides whether they still apply.
10. **Requests check the measured limit.** In `buildVideoRequest` (which gains
    the instance as an argument), after the model-limit check (`over_limit`)
    and before the size-multiple check:
    1. Take `VideoLimit(inst, set)`.
    2. If the request's `PixelFrames(frames, w, h)` is more than the limit's,
       refuse with 400 and code `over_measured_size`. The message:

       "ti2v is measured for videos up to 33 frames at 832×480 (Short).
       This request is 121 frames at 1280×704, which needs more GPU memory
       than the instance's reservation covers. To allow it, measure the
       instance at a larger size on the Server page (Measure, then Full)."

       The part in brackets is `size.Label`. When `VideoLimit` returned
       `measured` false, the first sentence reads instead "ti2v is only
       allowed its test size, 33 frames at 832×480, because it has not been
       measured at a larger size (or its reservation was set below what that
       measurement needs)." The last sentence names the smallest preset whose
       pixel-frames are at least the request's (the same measure the check
       uses), or "a custom size" when none is.

    Comparing pixel-frames means any shape within the measured amount is
    allowed: 5 seconds at 832×480 and 2 seconds at 1280×704 are both fine
    after measuring 81 frames at 1280×704. The model's limit (frames, and each
    side in either orientation) still applies on its own.
11. **/v1/models.** In `genV1Model`, `meta.video` keeps `max_frames` and
    `max_size` (the model's limit, as Phase 03 shipped them). It adds:
    - `measured_frames` and `measured_size`: the instance's current
      `VideoLimit`;
    - `measured_pixel_frames`.

    A client can then tell, before sending, what this instance will take.
12. **Server page.**
    - **Measure button.** For a video instance, it opens
      `gen-measure-video-dialog` instead of posting at once. The dialog
      offers:
      - one radio button per preset, showing its frames, seconds at the
        set's frame rate, and size (for example "Medium — 81 frames (3.4 s)
        at 1280×704");
      - "Custom", with frames, width and height fields. Frames is limited to
        the model's `MaxFrames`; width and height each to the larger of
        `MaxWidth` and `MaxHeight`, since either orientation is allowed, and
        the server checks the pair as in step 5.

      The dialog gets its data from the instance row: `genInstanceRow` gains
      `VideoPresets []VideoSize` (from `VideoPresets`), `VideoFPS`,
      `FullMeasureTime` (from the catalog entry), `ModelMaxFrames`,
      `ModelMaxSide`, `LimitFrames`/`LimitWidth`/`LimitHeight`/`LimitLabel`
      and `LimitMeasured` (from `VideoLimit`). The row's Measure button carries
      them as a JSON `data-video` attribute, which `openGenMeasureVideo` reads
      to fill the dialog.

      Text under the choices: "Longer and larger videos need more GPU memory
      and take much longer to measure. Measuring the largest size took about
      21 minutes on a Radeon AI PRO R9700. Requests larger than the size you
      measure are refused." The middle sentence is built from the catalog
      entry's `FullMeasureTime` and left out when it is empty (custom sets).
    - **Start-then-measure.** `openGenMeasureStart` shows the same choices
      for a video instance.
    - **Instance row.** The VRAM reserved cell of a video instance adds
      "measured for up to 81 frames at 1280×704 (Medium)" when `VideoLimit`
      returned `measured` true, or "test size only: 33 frames at 832×480" when
      it returned false. Its tooltip explains that
      requests larger than this are refused.

    All new controls have tooltips.
13. **Help.** Update `help.html`:
    - **The Measure step:** for video sets, Measure asks which size to
      measure, and the instance will only make videos up to that size.
      Measuring runs inside the memory the card can give.
    - **The reservations notes:** replace Phase 04's "uses about 4 GiB more
      than the reservation" warning and the A4000 hand-set footnote. Measure
      now handles both; give the measured figures from the hardware checks in
      the Test plan instead. A video set's row in the reservations table shows
      the reservation for its Short preset on each card, and a note under the
      table gives the Full preset's reservation and measuring time.
    - **The video jobs section:** describe `over_measured_size` and the new
      `meta.video` fields.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `go test -race ./internal/api/ ./internal/generation/`
- `gofmt -l internal/` prints nothing.

## Test plan

Automated:
- **`TestVideoPresets`:** a catalog set gets its three presets, the first
  being the test size; a custom set gets one "Short" preset; a set downloaded
  before this phase gets the catalog's current presets.
- **`TestVideoLimit`:**
  - measured at Medium with a matching reservation → Medium, `measured`
    true, label "Medium";
  - a custom measured size → label "custom size";
  - the same, with the reservation set by hand below the suggestion → the
    test size, `measured` false;
  - never measured → the test size, `measured` false.
- **`TestVideoMeasureAlwaysApplies`:** a video instance with a Short
  reservation measured at Full gets the Full suggestion as its reservation,
  and `VideoLimit` returns Full; measuring at Short again lowers both.
- **`TestUseSuggestedCapped`:** an image instance measured with its peak
  above the room has `MeasuredSuggestedMiB` equal to the room, and "Use
  suggested" sets that value.
- **`TestCatalogVideoSets` (extended):** every video entry's presets start at
  the test size, end at or under the model's limit, and use 4n+1 frame counts.
- **`TestBuildVideoRequestMeasuredLimit`:**
  - over the measured pixel-frames → 400 `over_measured_size`, with the
    message naming both sizes;
  - a different shape with fewer pixel-frames → accepted;
  - over the model's limit → still `over_limit`, checked first.
- **`TestMeasureVideoUsesChosenSize`** (fake sd-server): the vid_gen body has
  the chosen frames and size, and the instance stores them afterwards.
- **`TestMeasureRefusesBadVideoSize`:** missing fields, a size over the
  model's limit, and a size that is not a multiple of 8 each give 400.
- **`TestMeasureCapsAtRoom`:**
  - a fake instance whose peak is above the room gets a reservation equal to
    the room and stays enabled;
  - the launch arguments carry `--max-vram` at the room;
  - a music instance keeps the old behaviour.
- **`TestEditClearsMeasuredSize`.**
- **`TestV1ModelsVideoMeasuredMeta`.**
- The template render tests cover the dialog and the row text.

On hardware (recorded in `verification.md`):
- **asmodean (R9700), TI2V 5B:**
  1. Measure at Short. The reservation is set to its suggestion, and a Full
     request is then refused with `over_measured_size`.
  2. Measure at Medium, then at Full. Each sets the reservation and the
     instance's limit without pressing anything. Record time, peak and
     reservation.
  3. After the Full measurement, make the Full video. The card's peak stays
     within the reservation (watched with `rocm-smi`).
- **compute2 (A4000 16 GB), TI2V 5B:** Measure at Short with the
  card-budget limit.
  - The reservation is at most the room (about 14.7 GiB), and the instance
    stays enabled.
  - Text-to-video then works through the job API.
  - Measure at Medium too, and record whether it fits or is capped.
- Put both machines back as they were.

## Commit

`feat(generation): measure video sets at a chosen size and refuse larger requests`

## Rollback

Revert the commit.
- **Data:** instances keep their `measured_frames`/`measured_width`/
  `measured_height`/`measured_suggested_mib` fields in `generation.json`; the
  older code ignores them.
- **Behaviour:** requests go back to being checked against the model's limit
  only, and Measure goes back to running with no limit for sd-server.
- **Reservations:** those set by this phase's Measure stay valid.
