# Phase 06 — Measure for vupscale instances, and the measured ceiling

**Depends on:** Phase 01 (`WriteRGBAVI`), Phase 02 (`KindVUpscale`, the hidden Measure button), Phase 03 (launch with `--max-vram`), Phase 04 (`RenderVUpscale`, `runDGLJob`), Phase 05 (the job routes and their checks) · **Enables:** Phase 07 (docs describe Measure for vupscale), Phase 08 (field test)

## Goal

Measure works for vupscale instances. The admin enters a test clip size (width, height, frames) and a window size. The app writes a test clip, runs one real upscale alone on the GPU, records the VRAM peak and sets the reservation. After that, the job route refuses a clip larger than the measured one (frame size, frame count or window size), as video sets do. Like every generation instance, a vupscale instance cannot start, and gets no jobs, until it has a reservation (from Measure or set by hand). That rule already exists for all kinds (`gen_instances.go:702`, `gen_images.go:64`). This phase makes its Start button open the Measure dialog when it has no reservation, as the video branch at `gen_instances.html:91` does.

## Files touched

- `internal/generation/types.go` — `Instance.MeasuredWindow int` (`json:"measured_window,omitempty"`), beside `MeasuredFrames`/`MeasuredWidth`/`MeasuredHeight`.
- `internal/api/gen_vram.go`:
  - `handleGenMeasure` (551): read `frames`, `width`, `height` and `window_size` for vupscale (`vupscaleMeasureSize`).
  - `measureInstance` (649): treat `KindVUpscale` like `KindVideo` for the `--max-vram` room (705, 923), the abort (`measureAbort`, 766), the over-room check (910) and always-apply (927, 938).
  - `runMeasureGeneration` (1199): a `KindVUpscale` case calling `runMeasureVUpscale`.
  - `measureTimeout` (1283): 2 h for vupscale.
  - `measureBudget` (1320): accept vupscale.
- `internal/api/gen_vupscale_measure.go` — new: `runMeasureVUpscale` and `measureClip`.
- `internal/api/gen_vupscale_jobs.go` — the ceiling check (step 5), as check 9 after the Phase 05 checks.
- `internal/api/v1models.go` — `meta.vupscale` gets `measured_size {width,height}`, `measured_frames` and `measured_window`, when measured.
- `internal/api/gen_instances.go` — a new `VUpscaleJSON` field on the instance view gives the dialog its defaults (script, scale, the last measured values or the step 1 defaults). `VideoJSON` stays video-only.
- `web/templates/partials/gen_instances.html` — a vupscale branch beside the video one at line 91: the Start button of an instance with no reservation, and the Measure button, call `openGenMeasureVUpscale`, with tooltips (step 6). Remove `KindVUpscale` from the kinds that hide Measure (added in Phase 02).
- `web/templates/server.html` — new `openGenMeasureVUpscale(btn)` dialog beside `openGenMeasureVideo` (569).
- `internal/api/js_vupscale_measure_test.go` — new `TestVUpscaleMeasureJS`, which extracts `vupscaleClipLimitText` from `server.html` (like `TestMeasureNeedsJS`); add it to the `js-test` list in `Makefile`.
- `Makefile` — add `TestVUpscaleMeasureJS` to the `js-test` run pattern.
- `docs/guides/generation.md` — under GPU memory, one line: Measure for a video upscaler asks for a clip size and window, and clips up to that size are then allowed.
- Tests: `internal/api/gen_vupscale_measure_test.go` (new), `gen_vram_test.go`, `gen_vupscale_jobs_test.go`, `v1models_test.go`.

## Steps

1. **Dialog values.** Width and height are the input clip's size, before scaling. Frames and window size are also inputs.
   - Defaults: 832 × 480, 81 frames (5 s of Wan at 16 fps), window = `DefaultWindowSize(script)`.
   - Width and height: 64–2048, multiples of 16. Frames: 2–1024. Window: checked with `WindowSizeOK`.
   - The dialog also runs the Phase 05 limit checks 4–7 in a JS function `vupscaleClipLimitText(script, scale, w, h, frames, window)` (same formulas). It shows the first failure under the field. The server checks them again.
   - The server returns 400 for bad values with the same texts.
2. **Test clip.** `measureClip(w, h, frames int) []byte` uses `videofile.WriteRGBAVI` at 16 fps with a moving diagonal gradient plus a 16-pixel checker block that moves 4 pixels per frame. This gives the model real edges to work on. Call `VUpscaleWorkDirs` first, then write the clip to `<instance dir>/input/measure.avi` before the instance starts.
3. **`runMeasureVUpscale`:**
   - Render the script with `OutputFormat: "webm"`, `InputPath: "input/measure.avi"`, `OutputPath: "output/measure.webm"`, a fixed seed of 1 and the chosen window.
   - Run `runDGLJob`, then delete both files.
   - A DGL failure fails the Measure with the sd.cpp text. An sd.cpp error that holds `out of memory` or `failed to allocate` (case-insensitive) counts as out of memory and gets the same "did not fit in the room" handling the video path has.
4. **Apply.** Same as video: the suggestion is the peak plus 10%, rounded up to 512 MiB and capped by the room. It always applies. Record `MeasuredWidth`, `MeasuredHeight`, `MeasuredFrames` and `MeasuredWindow`, and add a `MeasurePoint`. The existing `measuredStale` logic marks the measurement stale when the set, build, GPU or extra args change. The code that clears the measured fields at that point (`gen_instances.go:565`) also clears `MeasuredWindow`.
   - **Abort.** When `measureAbort` stops a vupscale Measure (stuck, or over the room), it stops the instance. sd-server cannot cancel a DGL job (Phase 04), so this is the only way to free the GPU.
5. **Ceiling in the job route** (only when the instance has a measurement). Refuse with 400 when any of these hold:
   - input `width` > `MeasuredWidth`, or input `height` > `MeasuredHeight` (each side on its own, so a portrait clip needs its own Measure);
   - `frames` > `MeasuredFrames`;
   - `window_size` > `MeasuredWindow`.

   The message: "this instance was measured with <W>×<H>, <F> frames and window <N>; this clip is <w>×<h>, <f> frames, window <n>. Measure it again at the larger size on the Server page, or send a smaller clip". With no measurement, there is no ceiling, as for video sets.
6. **UI wording** (ASD-STE100, short):
   - Measure tooltip: "Upscales one test clip at a size you choose, alone on this GPU, and reserves its peak VRAM. Clips up to that size are then allowed."
   - Start-first tooltip: "No VRAM reservation yet. Toolchest measures first, at a clip size you choose. It asks before it starts."
   - Dialog title: "Measure <name>". Field labels: "Clip width", "Clip height", "Frames", "Window size". Hint under Window size, for relayvsr: "Frames per window (2–32). More frames use more VRAM." For seedvr2: "1 treats each frame alone. 5, 9, 13 … give smoother motion and use more VRAM."
7. **Reservation tooltip** in `gen_instances.html` (line 30 area): add vupscale with the video wording, replacing "video" with "clip".

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `measureClip(64, 48, 5)` probes as 64×48, 5 frames, 16 fps, no audio.
- Measure handler test with the fake sd-server:
  - The fake receives a script whose `input_path` is `input/measure.avi` and whose `window_size` is the chosen value.
  - After the fake completes, the instance has the four measured fields and a reservation.
- Bad dialog values (width 100, window 4 on seedvr2, a clip over 256 MiB) → 400 with the texts.
- Ceiling tests (each over-limit field alone → 400; equal values pass; an unmeasured instance → no ceiling).
- Changing the instance's build clears `MeasuredWindow` with the other measured fields.
- A Measure that `measureAbort` stops leaves the instance stopped.
- `TestVUpscaleMeasureJS`: `vupscaleClipLimitText` matches the Go check for six sample sizes (both scripts, inside and over each limit).
- Manual on asmodean: Measure RelayVSR ×2 at 832×480, 81 frames, window 16. The peak and reservation appear on the card. A 97-frame clip is then refused with the step 5 message.

## Commit

`feat(generation): measure video upscaler instances and refuse clips above the measured size`

## Rollback

Revert the commit. Measure for vupscale goes back to hidden, and the ceiling check goes away. `measured_window` stays in saved instance JSON and is ignored. Reservations already set stay set.
