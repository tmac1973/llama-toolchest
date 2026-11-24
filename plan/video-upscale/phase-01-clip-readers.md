# Phase 01 — Clip readers and a test-clip writer

**Depends on:** nothing · **Enables:** Phase 05 (request checks), Phase 06 (Measure test clip)

## Goal

Add a small Go package that reads the size, frame count, frame rate and audio presence of a clip in the formats sd.cpp's `video::load` reads, without decoding any frames. The package refuses other formats (MP4, VP9 WebM, interframe VP8, OpenDML AVI) by their content, with a clear message. It also writes an uncompressed RGB AVI, which Measure uses as its test clip. Nothing uses the package yet. It is shippable and fully tested on its own.

## Files touched

- `internal/videofile/probe.go` — new: `Info` type, `Probe(r io.ReaderAt, size int64) (Info, error)`, the format errors.
- `internal/videofile/avi.go` — new: RIFF/AVI reader (`avih`, `strh`, `strf`, `movi` index check) and `WriteRGBAVI`.
- `internal/videofile/webp.go` — new: WebP reader (`VP8 `, `VP8L`, `VP8X` + `ANIM`/`ANMF`).
- `internal/videofile/webm.go` — new: minimal EBML reader for WebM (Segment → Tracks → Video size and CodecID; Clusters → SimpleBlock count and keyframe flags; audio track CodecID).
- `internal/videofile/probe_test.go`, `avi_test.go`, `webp_test.go`, `webm_test.go` — new tests.
- `internal/videofile/testdata/` — new fixtures (see Test plan), each under 200 KiB.
- `internal/videofile/probe_field_test.go` — new, build tag `fieldtest`: `TestProbeField` probes every file named in `VIDEOFILE_PROBE` (comma-separated paths) and logs its `Info`. Later phases use it to check real clips: `VIDEOFILE_PROBE=a.webm,b.avi go test -tags fieldtest -run TestProbeField -v ./internal/videofile/`.

## Steps

1. **`Info`.** Define:
   ```go
   type Info struct {
       Format   string // "avi", "webp" or "webm"
       Width    int
       Height   int
       Frames   int
       FPSNum   int    // 0 when the file gives no rate (still WebP)
       FPSDen   int
       HasAudio bool
   }
   // DecodedBytes is what video::load holds: 3 × width × height × frames.
   func (i Info) DecodedBytes() int64
   ```
2. **`Probe`.** Read the first 16 bytes and pick the reader:
   - `RIFF....AVI ` → AVI.
   - `RIFF....WEBP` → WebP.
   - `1A 45 DF A3` (EBML) with DocType `webm` → WebM.
   - `....ftyp` at offset 4 → `ErrMP4`: "MP4 is not supported: send AVI (MJPEG or RGB), WebM (VP8) or animated WebP".
   - Anything else → `ErrUnknownFormat` with the same list of formats.
3. **AVI reader.**
   - Read `hdrl/avih` for width, height and total frames.
   - Find the video `strh` (`vids`). Its `fccHandler`/`strf` biCompression must be `0` (BI_RGB with 24 bits) or `MJPG`. Anything else → `ErrCodec` naming the FourCC.
   - Use `dwRate/dwScale` for the frame rate.
   - An `auds` stream sets `HasAudio` when its `wFormatTag` is 1 (PCM) or 3 (IEEE float). Other audio → `ErrCodec` ("audio codec 0x.. is not supported in AVI; send PCM audio or no audio").
   - A `RIFF....AVIX` chunk after the first RIFF (OpenDML) → `ErrCodec` ("segmented OpenDML AVI is not supported").
4. **WebP reader.**
   - `VP8 `/`VP8L` → one frame with its size.
   - `VP8X` → canvas size; with the animation flag, count the `ANMF` chunks and set `Frames`.
   - Use the first `ANMF` duration (ms) for the rate, as `FPSNum = 1000`, `FPSDen = duration`, or 0/0 when the duration is 0.
   - WebP never has audio.
5. **WebM reader.**
   - Parse only the elements needed: EBML header DocType, Segment, Info (TimecodeScale), Tracks/TrackEntry (TrackNumber, TrackType, CodecID, Video/PixelWidth, PixelHeight, DefaultDuration) and Cluster/SimpleBlock (track number and keyframe flag).
   - Skip every other element by its size. An unknown-size element other than Segment or Cluster → `ErrCodec` ("this WebM uses unknown-size elements; it is not supported").
   - The video CodecID must be `V_VP8`. Every video SimpleBlock must have the keyframe flag set. Otherwise → `ErrCodec` ("WebM must use independent VP8 frames, as written by stable-diffusion.cpp").
   - BlockGroups and laced blocks → `ErrCodec` with the same message.
   - An audio track sets `HasAudio`; its CodecID must be `A_PCM/INT/LIT` (PCM16). Other audio → `ErrCodec` ("WebM audio must be PCM16").
   - Frames = the count of video SimpleBlocks. The rate comes from DefaultDuration (ns per frame), as `FPSNum = 1e9`, `FPSDen = DefaultDuration`, reduced by their GCD; 0/0 when the file has none.
6. **`WriteRGBAVI(w io.Writer, width, height, frames, fps int, frame func(i int, rgb []byte)) error`**.
   - It writes an AVI 1.0 file with one BI_RGB 24-bit video stream, bottom-up rows as AVI expects, an `idx1` index and no audio.
   - `frame` fills each frame's top-down RGB buffer.
   - Phase 06 draws a moving gradient with it.
7. **Scan limits.** Block payloads are skipped with a seek and never read, and skipped bytes do not count toward any limit, so a 512 MiB upload probes in a few milliseconds. To stop a hostile file, every reader stops with `ErrCodec` ("file has too many chunks to check") after 64 MiB of bytes actually read, or after 2,000,000 chunks or elements. A clip at the 256 MiB decoded limit has at most about 40,000 frames, so real clips never get near these numbers.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./internal/videofile/...`

## Test plan

- Fixtures (make them once with ffmpeg on the dev machine and commit them; the app itself never runs ffmpeg):
  - `rgb-64x48-5f.avi`: written by `WriteRGBAVI` in the test itself (no fixture).
  - `mjpeg-pcm-64x48-8f.avi`: MJPEG with PCM16 audio.
  - `mjpeg-aac.avi`: MJPEG with AAC audio.
  - `opendml.avi`: has an `AVIX` chunk.
  - `anim-64x48-6f.webp` and `still-64x48.webp`.
  - `vp8-key-64x48-9f.webm`: VP8 with every frame a keyframe (`ffmpeg … -c:v libvpx -g 1 -auto-alt-ref 0`), the same layout sd.cpp writes. Phase 08 also probes a real sd.cpp output on asmodean.
  - `vp8-interframe.webm` and `vp9.webm`.
  - `tiny.mp4`.
- Table test of `Probe`: each fixture → the expected `Info`, or the expected error and message text.
- `WriteRGBAVI` round trip: write 5 frames, then `Probe` reads 64×48, 5 frames, the given rate, no audio.
- `DecodedBytes` for 1280×720×120 = 331,776,000.
- Truncated copies of each fixture (cut at 10 points) return an error and never panic.
- Fuzz test (`FuzzProbe`) seeded with the fixtures, run for 30 s locally; CI runs the seed corpus only.

## Commit

`feat(generation): read clip size and frame count for video upscaling`

## Rollback

Revert the commit. The package has no callers yet, so nothing else changes.
