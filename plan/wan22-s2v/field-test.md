# Wan2.2 S2V — field test (Phase 05)

Date: 2026-11-22. Binary: `2.63.0-s2v682-test` (branch `feat/wan22-s2v`, PR #686,
commit b507b14), swapped into the containers on both machines. sd.cpp:
`master-1116-bf31b9c`, which is newer than 6cd0ef1, built fresh on each machine
(ROCm 10.1 on asmodean, CUDA 12.8 with graphs on compute2).

Inputs: a portrait made with Ming-Image on asmodean (832×480), and a speech
clip made with Qwen3-TTS on asmodean's `/tts`, cut to 2.9 s and saved as WAV,
MP3 (128k) and FLAC. There was also a 7 s WAV (the speech twice, then padded),
an OGG, and the `broken.mp3` test file.

## Machines

| | asmodean | compute2 |
| --- | --- | --- |
| GPU | Radeon AI PRO R9700, 32 GB | RTX A4000, 16 GB (GPU 2) |
| Download | 14.5 GB new (UMT5 and VAE already there) | same |

## Measure

| Size | asmodean peak | compute2 peak |
| --- | --- | --- |
| Short: 33 × 832×480 | 27.6 GiB (whole decode), reservation 27.7 GiB | 14.1 GiB (decode in pieces), reservation 14.0 GiB (whole card) |
| 49 × 832×480 | – | 12.6 GiB (in pieces), reservation 13.5 GiB |
| Medium: 49 × 1280×720 | 17.6 GiB (in pieces) | – |
| Full: 81 × 1280×720 | 22.0 GiB (in pieces), reservation 24.5 GiB | – |

Short took about 2 minutes on both machines. The Wan2.1 VAE decode did not
crash on the A4000. On the older sd.cpp it crashes there for the A14B sets.

## Jobs

All jobs used the portrait and had no length unless noted.

| Job | asmodean | compute2 |
| --- | --- | --- |
| WAV 2.9 s | 49 frames, 16 fps, 3.0 s WebM with the speech as sound (PCM in WebM), 11 min | 49 frames, 18 min |
| MP3 2.9 s (decodes as 3.0 s: encoder padding) | 49 frames, 11 min | 49 frames, 18 min |
| FLAC 2.9 s | 49 frames, 11 min | 49 frames, 18 min |
| 7 s WAV with `frames: 81` | 81 frames, 5.0 s, sound cut to 5 s, 20 min | – |
| WebP, `frames: 17` | – | animated WebP, 17 frames, no sound |

- The first request on each machine, measured only for Short, was refused
  with `over_measured_size` (49 > 33 frames), as designed.
- The lips move with the speech: the mouth opens, shows teeth and closes. The
  motion is small. The user has the asmodean video to judge.
- `/v1/models` shows `capabilities: [start_image, audio]` and
  `meta.video.fixed_fps: 16`. The instance's last video says "a start image and
  audio".

## Refusals (compute2, and asmodean for the TI2V case)

| Request | Answer |
| --- | --- |
| No audio | `audio_required` |
| No start image | `start_image_required` |
| Audio on Wan2.2 TI2V 5B | `input_not_supported` ("cannot take audio") |
| OGG | `bad_audio`: "audio must be WAV, MP3 or FLAC: the file is none of these" |
| Broken MP3 | `bad_audio`: "audio must be WAV, MP3 or FLAC: the MP3 could not be decoded: EOF" |
| 7 s, no length | `over_limit`: "the audio is 7.0 s; … makes at most 5.1 s …" |

## Other video sets on master-1116

- **Wan2.2 TI2V 5B:** works. asmodean: Short in 70 s, 17.6 GiB measured.
  compute2: 33 × 1280×704 measured at 11.3 GiB, and a 33-frame video in 87 s.
- **Wan2.2 I2V A14B Lightning (and A14B):** do not start. sd-server
  master-1116 refuses `--high-noise-cfg-scale` ("unknown argument").
  `--high-noise-sampling-method` is gone too, and `--moe-boundary` now works
  only with `--high-noise-steps -1`. Both A14B catalog sets pass these flags.
  This is outside S2V and needs its own fix.

## Not covered

- S2V on an sd.cpp build older than 6cd0ef1 (silent output) was not run.
  e41fbcf has the `wan22-s2v` recipe and the same `ref_audios` field.
- compute2 was not measured at Medium or Full.
