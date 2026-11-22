# Phase 01 — Audio decoding package

**Depends on:** nothing · **Enables:** Phase 02 (the video job audio input
uses it to check, decode and time audio), Phase 03 (Measure uses its WAV
writer for a test track)

## Goal

A small pure-Go package that takes the bytes a client sent and returns a WAV
that sd-server can read, plus its length in seconds. It finds the format from
the first bytes (not a file name), passes a WAV through unchanged after it
checks it, and decodes MP3 and FLAC to 16-bit PCM WAV. It can also write a
WAV from samples, for Measure's test track. Nothing calls it yet, so this
phase changes nothing a user sees.

## Files touched

- `go.mod`, `go.sum` — add `github.com/hajimehoshi/go-mp3` and
  `github.com/mewkiz/flac` (latest tagged releases).
- `internal/audiofile/audiofile.go` (new) — `Decode`, `Format`, `ErrFormat`,
  `WriteWAV`.
- `internal/audiofile/wav.go` (new) — WAV header reader and writer.
- `internal/audiofile/audiofile_test.go` (new) — tests.
- `internal/audiofile/testdata/` (new) — `tone.wav`, `tone.mp3`, `tone.flac`
  (1 s, 440 Hz, 44.1 kHz stereo, each under 200 KB), `broken.mp3` (the first
  2 KB of `tone.mp3` followed by zero bytes), `tone.ogg` (a format we refuse),
  and `README.md` saying how the files were made (the `ffmpeg` commands).

## Steps

1. Add the two modules with `go get`, then `go mod tidy`.
2. In `audiofile.go`, define:
   - `type Format string` with `WAV = "wav"`, `MP3 = "mp3"`, `FLAC = "flac"`.
   - `var ErrFormat = errors.New("audio must be WAV, MP3 or FLAC")`.
   - `var ErrTooLong = errors.New("audio is longer than 10 minutes")`.
   - Every other error `Decode` returns wraps `ErrFormat` as
     `fmt.Errorf("%w: <detail>", ErrFormat)`, so its text always starts
     "audio must be WAV, MP3 or FLAC: ".
   - `type Audio struct { WAV []byte; Format Format; Seconds float64 }`.
   - `func Decode(b []byte) (Audio, error)`.
   - `func WriteWAV(samples []int16, rate, channels int) []byte`.
3. Detection in `Decode`, from the first bytes only:
   - `RIFF....WAVE` → WAV.
   - `fLaC` → FLAC.
   - `ID3` (an ID3v2 tag), or a first byte `0xFF` with the top three bits of
     the second byte set (an MPEG frame sync) → MP3.
   - Anything else → `ErrFormat`.
4. WAV path (`wav.go`): walk the RIFF chunks and read `fmt ` and `data`.
   - Accept the encodings sd-server's `load_wav_from_memory` reads: format 1
     (PCM) at 8, 16, 24 or 32 bits, format 3 (float) at 32 or 64 bits, and
     format `0xFFFE` (extensible) whose sub-format is one of these two.
   - Refuse 0 channels, a 0 sample rate, or a missing or empty `data` chunk,
     for example "audio must be WAV, MP3 or FLAC: the WAV has no data".
   - `Seconds = dataBytes / (rate × channels × bytesPerSample)`.
   - Return the input bytes unchanged as `WAV`.
5. MP3 path: decode with `mp3.NewDecoder`. It gives 16-bit stereo at the
   source rate. Read all samples, then `WriteWAV(samples, d.SampleRate(), 2)`.
   `Seconds` = samples per channel / rate. A decode error, or zero samples,
   gives "audio must be WAV, MP3 or FLAC: the MP3 could not be decoded: …".
6. FLAC path: decode with `flac.New` and loop `ParseNext` over the frames.
   - Convert each sample to int16: shift right by `bitsPerSample − 16` for
     depths over 16, and shift left by `16 − bitsPerSample` for depths under
     16.
   - Interleave the channels, keeping the source channel count and rate.
   - Errors wrap `ErrFormat` in the same way as for MP3.
7. `WriteWAV` writes a 44-byte canonical header (PCM, 16-bit) and the
   little-endian samples.
8. Limits: `Decode` returns `ErrTooLong` for audio over 10 minutes. For WAV
   it checks the header's length; for MP3 and FLAC it stops decoding once it
   passes 10 minutes. This guards memory; the video limits in Phase 02 are
   far lower.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/audiofile/...`
- `govulncheck ./...` (CI runs it; also run it locally if it is installed)

## Test plan

- Table test over `testdata`:
  - WAV, MP3 and FLAC each give the right `Format` and `Seconds` within 0.05.
  - The WAV bytes are unchanged.
  - The MP3 and FLAC output parses again with `Decode` as a WAV.
- `tone.ogg`, an empty input and random bytes give `ErrFormat`
  (`errors.Is`).
- `broken.mp3` gives an error that wraps `ErrFormat`, and its text starts
  "audio must be WAV, MP3 or FLAC: ".
- WAV header cases (hand-built in the test): a float32 WAV passes; a 12-bit
  PCM WAV, a WAV with no `data` chunk and a WAV with 0 channels are refused.
- `WriteWAV` round trip: write 16000 samples mono at 16 kHz; `Decode` gives
  1.0 s.
- An input over 10 minutes, made with `WriteWAV` at 8 kHz mono, gives
  `ErrTooLong`.

## Commit

`feat(audio): decode WAV, MP3 and FLAC uploads to WAV`

## Rollback

Revert the commit. Nothing imports the package yet, so reverting is safe and
touches only the new folder and `go.mod`/`go.sum`.
