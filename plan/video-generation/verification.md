# Video Generation — Hardware Verification

Results of the hardware checks: Phase 04 (Wan2.2 TI2V 5B), Phase 05 (measured
video size), Phase 06 (Wan2.2 I2V A14B), Phase 08 (Wan2.2 I2V A14B
Lightning) and Phase 07 (LTX-2.3), then the overview's success criteria and
the release notes. Each entry gives the machine, the date and the build refs.

## Machines

| Name | GPUs | Software |
|---|---|---|
| asmodean | 1× Radeon AI PRO R9700, 32 GB | Fedora 44 host; toolchest's ROCm container image (ROCm 10.1), Podman |
| compute2 | 3× RTX A4000, 16 GB each | toolchest's CUDA container image, Docker |

Both ran llama-toolchest built from `main` at `4ee04a1` (Phases 01–03), copied
into the existing containers. stable-diffusion.cpp was built by toolchest's
Builds tab at `master-1015-e41fbcf`, the newest master tag on 2026-10-26:
`sd-master-1015-e41fbcf-rocm` on asmodean, `sd-master-1015-e41fbcf-cuda` on
compute2. Its video recipes are the same as at `master-1000-95ff762`, where
Phase 01 copied them, so `VideoRecipes` needed no change. Both builds include
libwebm: every WebM below came from them.

## Phase 04 — Wan2.2 TI2V 5B

All on 2026-10-26.

### asmodean (R9700, ROCm)

The other instances on the GPU (Ming-Image, klein) were disabled for Measure
and the jobs, except where a step says otherwise.

| Check | Result |
|---|---|
| Download from the catalog | 12.9 GB, no token needed |
| Measure | 80 s. Peak 23.6 GiB (estimated from the whole GPU), reservation 26.0 GiB |
| 1. Text-to-video, defaults (832×480, 33 frames), WebM | 70 s; 33 frames at 24 fps, 1.6 MB, VP8; all frames decode; progress showed all 20 steps |
| 2. Image-to-video, photo sent as a multipart form, WebM | 66 s; 33 frames, 4.8 MB; starts from the photo (cropped to 832×480 by sd-server) |
| 3. The same as animated WebP | 66 s; 33 frames, 4.5 MB, `image/webp`; Pillow reads it as animated and decodes every frame |
| 4. Ceiling: 1280×704, 121 frames (5 s), WebM | 1,239 s (20.6 minutes); sampling 51.8 s per step; 121 frames, 5.3 MB; all frames decode. Whole card peaked at 31.1 GiB |
| Preview cost | Sent straight to sd-server: 66 s with `proj` previews and 66 s with none, twice each (2 s timing resolution). No measurable cost, so toolchest keeps a preview at every step |
| Refusals | 122 frames → 400 `over_limit` ("Wan2.2 TI2V 5B makes at most 121 frames (5 s at 24 fps) and 1280x704"); an end image → 400 `input_not_supported`; `avi` → 400 `bad_value`. None reached the GPU queue |
| Waiting route | 200 after 66 s; the same JSON as `GET` on the job |
| Client leaves the waiting route | Left after 25 s; the job was cancelled in sd-server and the GPU queue was free about 2 s later |
| `DELETE` on a running job | 204; the job showed `running` with `cancel_requested`, then `cancelled` 1.5 s later |
| Eviction | With eviction on, Ming-Image (10.5 GiB, unlocked) and klein (11.5 GiB) running, and TI2V stopped with its reservation set to 12 GiB: a video job stopped klein only, started TI2V and finished in 75 s. Ming-Image kept running |
| Locking | With Ming-Image locked and TI2V's reservation set to 20 GiB: the video job failed with "not enough VRAM on GPU 0 (31.9 GiB): ti2v needs 20.0 GiB, but only 18.8 GiB is left after … ming-image 10.5 GiB and the safety margin". Ming-Image kept running. The job is refused when its backend is started, so it comes back as a failed job (HTTP 500 from the waiting route), as image jobs do |
| Players | Not opened in Firefox or Chromium (no desktop on asmodean). Every WebM was checked with `ffprobe -count_frames` and a full `ffmpeg` decode; ffmpeg 8.1.3 cannot decode animated WebP, so the WebP was checked with Pillow |

### compute2 (RTX A4000 16 GB, CUDA)

| Check | Result |
|---|---|
| Download | 12.9 GB |
| Measure | The test video worked (81 s), but sd-server filled the card: peak 15.3 GiB. The suggested reservation (17.0 GiB) does not fit the card's 14.7 GiB budget, so Measure disabled the instance (see finding 2) |
| Reservation set by hand to 14.5 GiB (`--max-vram 14.5`) | Starts and runs |
| 1. Text-to-video, defaults, WebM | 72 s; 33 frames, 1.6 MB |
| 2. Image-to-video, multipart, WebM | 76 s; 33 frames, 4.8 MB |
| Card peak during the jobs | 14.9 GB (`nvidia-smi`, including the CUDA context) |
| Tiled VAE decode (`extra: {"vae_tiling_params": {"enabled": true}}`) | sd-server crashed at tile 5 of 10: `ggml_cuda_compute_forward: CONCAT failed`, `CUDA error: unspecified launch failure` (finding 1) |

### Findings

1. **Tiled Wan2.2 VAE decode crashes in sd.cpp on both ROCm and CUDA.** Found
   in Phase 01 on ROCm (master-1000 and master-1006: a HIP graph assertion and
   a segfault), and confirmed here on CUDA with master-1015. Sampling is not
   affected; the crash comes when the tiled decode starts. The TI2V set runs
   without `--vae-tiling` (Phase 01). This is an upstream bug; it has not been
   reported to stable-diffusion.cpp yet.
2. **Measure cannot reserve a set that fills a smaller card.** With no memory
   limit, sd-server takes what the card has, so on the 16 GB A4000 Measure saw
   15.3 GiB and suggested 17.0 GiB, more than the card's budget, and disabled
   the instance. Held to 14.5 GiB by hand, TI2V runs at about the same speed
   (72–76 s against 70 s on the R9700). Not changed in this phase: the user
   decides whether Measure should cap its suggestion at the card's budget, or
   measure within a limit.
3. **At the ceiling, sd-server goes over its reservation.** With a 26 GiB
   reservation (`--max-vram 26.0`), the 1280×704, 121-frame video peaked at
   31.1 GiB for the whole card, about 30 GiB for sd-server. The untiled decode
   of 121 large frames is the likely cause (finding 1). Next to another
   instance this could run the GPU out of memory. Not changed in this phase:
   the user decides whether to lower the ceiling.
4. **The ceiling is slow.** 20.6 minutes for 5 seconds at 1280×704 on the
   R9700, against 70 s for the 33-frame 832×480 test video.
5. **Error text after a backend crash.** When sd-server crashed mid-job, the
   job's error read "generating: Get "http://127.0.0.1:8084/…": connection
   refused". The supervisor had already restarted the backend, so the job did
   not report "backend stopped". Cosmetic; not changed.

## Phase 05 — measured video size

All on 2026-10-26, with this branch's toolchest build and the same
`sd-master-1015-e41fbcf` builds. asmodean had been rebooted before the check
and its toolchest container restarted for it.

### compute2 (RTX A4000 16 GB, CUDA)

| Check | Result |
|---|---|
| Measure at Short (33 frames, 832×480), with the instance's old hand-set reservation | 91 s. Peak 15.3 GiB (whole card). "Reservation capped at 14.7 GiB, all this card can give it." The instance stayed enabled and its limit became Short — in Phase 04 Measure disabled it |
| Text-to-video at Short through the job API | 76 s, 33 frames, 1.6 MB |
| A Medium-size request (81 frames, 1280×704) | 400 `over_measured_size`: "ti2v is measured for videos up to 33 frames at 832x480 (Short). … To allow it, measure the instance at a larger size on the Server page (Measure, then Medium)." |
| `meta.video` | `measured_frames` 33, `measured_size` 832x480, `measured_pixel_frames` 13178880, alongside `max_frames` 121 and `max_size` 1280x704 |
| Measure at Medium | Sampling finished (20 steps, 433 s); the decode did not fit in 14.7 GiB, sd-server retried with temporal, then spatial and temporal tiling, and crashed (`CUDA error: unspecified launch failure`, finding 1). Measure failed after 461 s and the instance kept its Short size and reservation. On a 16 GB card Medium cannot be made until sd.cpp's tiled decode works |

### asmodean (R9700 32 GB, ROCm)

| Check | Result |
|---|---|
| Measure at Short | 80 s. Peak 23.5 GiB, reservation 26.0 GiB, limit Short |
| Measure at Medium | 674 s. Peak 30.6 GiB while held to the 28.2 GiB room: sd-server went about 2.4 GiB past `--max-vram` in the untiled decode. The build at that time capped the reservation at 28.2 GiB and granted Medium |
| Measure at Full | 1,247 s. Peak 31.5 GiB, more than the card's budget. The capped suggestion no longer fitted because other programs had grown during the run, and the instance dropped to its test size: both bugs, fixed below |

### Fixed after the check

- **Video sizes over the room are refused** (decided with the user): a video
  measurement whose peak is more than the room plus 1 GiB is not granted, and
  the instance keeps its previous size. By the measurements above this grants
  Short on both cards and refuses Medium and Full on the R9700. Re-running the
  R9700 Medium measurement with this rule was not done; the rule is checked by
  `TestApplyMeasurementRefusesVideoOverRoom` with the measured figures.
- **A measurement that cannot apply keeps the previous size and reservation**
  (it overwrote the size before).
- **The room is worked out again after the test run**, not only before it.
- **A backend that dies mid-run** is reported as "the backend stopped …",
  pointing to its log, in Measure and in video jobs (finding 5 of Phase 04).

### Finding

6. **sd-server does not stay within `--max-vram` when decoding large videos
   without tiles.** Until tiled decoding works for the Wan2.2 VAE (finding 1),
   TI2V 5B is limited to about the Short size on a 32 GB card and to Short on
   a 16 GB card, within the reservation promise.

### Not run

- Opening the videos in Firefox and Chromium (no desktop on the test
  machines); the files were checked with ffprobe, ffmpeg and Pillow instead.
- Video on a Vulkan build.
- Measure for LTX-2.3 (Phase 07).

## Phase 06 — Wan2.2 I2V A14B

2026-10-26 and 27, with this branch's toolchest build and the same
`sd-master-1015-e41fbcf` builds. Every A14B video here is 33 frames at
832×480 (the Short size), 20 steps on each expert.

### asmodean (R9700 32 GB, ROCm)

| Check | Result |
|---|---|
| Download | 19.6 GB (the Wan2.1 VAE is new; the text model is shared with TI2V 5B) |
| Launch | The `wan22-i2v` recipe has `high_noise_flow_shift`, but sd-server has no `--high-noise-flow-shift`: the first launch failed with "unknown argument". `--flow-shift` covers both experts. `--moe-boundary 0.9` sets the recipe's boundary (sd-server's default is 0.875) |
| With `--vae-tiling` | Hung encoding the start picture: one CPU core at 100%, the GPU at 3–4%, for 27 minutes until stopped, twice. The first time ComfyUI was also using the card, the second time nothing else was. Without tiling the encode takes 10 s (finding 7) |
| Without tiling | The video worked (1,645 s: 810 s per expert, 15 s decode), but the decode took the whole card from 28.9 to 32.3 GB, over the 28.5 GiB room, so Measure refused the size and disabled the instance |
| With `--temporal-tiling` (the catalog's choice), straight to sd-server | 1,643 s; whole card peaked at 27.9 GB, decode included; 33 frames, the fox from the start picture looks up and smiles, no seams. `--temporal-tiling` tiles only the decode |
| Measure at Short, instance running when Measure was pressed | sd-server was given 16.5 GiB, not the 28.5 GiB room: Measure read the card 2 s after stopping the instance, before the monitor's next reading, so the stopped backend's 12 GB counted as another program's. Sampling still fitted; the decode did not, sd-server fell back to spatial tiling and hung as above. Fixed (see below) |
| Measure at Short, with the fix, instance running when Measure was pressed | 1,658 s. sd-server was given the full 28.5 GiB room. Peak 29.0 GiB (per process), within the room plus the 1 GiB allowance; whole card 28.2 GiB (`rocm-smi`). Reservation 29.0 GiB, limit Short. Restarting the instance afterwards failed with "instance a14b is being measured" (fixed, see below) |
| Medium and Full | Not measured. Short takes about 28 minutes; Medium has about 3 times and Full about 5 times the video to sample, and attention grows faster than that, so both are expected to pass Measure's 2-hour limit. The Measure dialog says so |
| ComfyUI on the same card | During the first `--vae-tiling` run, ComfyUI (Haruspex) ran Qwen-Image jobs on the card with about 27 GB loaded and hit a HIP out-of-memory error. Toolchest does not know about ComfyUI's memory (finding 9) |

### compute2 (RTX A4000 16 GB, CUDA)

| Check | Result |
|---|---|
| Download | 19.6 GB |
| Measure at Short | 1,503 s. Sampling 711 + 722 s (a little faster than the R9700). The decode did not fit in the room; sd-server fell back to spatial and temporal tiling, which worked this time (42 s). Peak 14.9 GiB (whole card); reservation capped at 14.7 GiB, limit Short |
| Refusals | No start image → 400 `start_image_required`; 49 frames at 1280×720 → 400 `over_measured_size`, naming the measured Short size; 97 frames → 400 `over_limit` ("at most 81 frames"). None reached the GPU |
| TI2V 5B at Short, for the restart check | sd-server crashed in the decode with the same `CONCAT failed` error after sampling (49 s); in Phase 05 the same measurement worked |
| Start-and-end job (the fox photo, and the last frame of the asmodean video as the end) | Failed twice. First: sd-server crashed after 4 s encoding the pictures (`ggml_cuda_compute_forward: CONCAT failed`, `CUDA error: unspecified launch failure`). Second: sampling finished (24 minutes), then the tiled decode crashed with the same error at tile 1 of 9 |

So on a 16 GB card the A14B pair samples at about the R9700's speed, but
the decode needs tiles there, and sd.cpp's tiled Wan decode on CUDA crashes
more often than not (finding 1). The set is not usable on a 16 GB card until
that is fixed upstream.

### Fixed in this phase

- **Measure on a running instance** waits, after stopping it, until two
  monitor readings taken after the stop show the card no longer giving
  memory back (at most 20 s), before it reads the baseline and the room.
- **The start after a measurement** (the instance was running, or
  *Measure, then start*) always failed with "instance … is being measured",
  because Measure cleared that mark only after it had returned. It is now
  cleared just before the start. This bug dates from the first reservations
  (#313).
- **Measure's error after a stop**: when the user stops the instance during
  the test video, Measure says so, not "probably ran out of GPU memory". A
  backend that crashed still gets the out-of-memory message: Measure reads
  the state before it stops the backend itself.
- Checked on compute2 after both fixes: FLUX.2 klein running on GPU 1,
  Measure pressed, "… Restarted." and the instance kept running.

### Findings

7. **The Wan2.1 VAE hangs encoding with `--vae-tiling` on ROCm**
   (master-1015). sd-server's own fallback to spatial tiling after a decode
   runs out of memory hangs the same way. `--temporal-tiling` does not touch
   the encode and works. Not yet reported upstream.
8. **A14B is slow.** About 27 minutes for 2 seconds at 832×480 on the R9700
   or the A4000, against 70 s for TI2V 5B: two 14B experts × 20 steps, and
   with `--cfg-scale 3.5` each step runs the model twice. The Lightning
   4-step LoRAs cut that to about 3 minutes (Phase 08).
9. **ComfyUI and toolchest share asmodean's GPU without knowing about each
   other.** Toolchest's reservations do not count ComfyUI's memory, so either
   can run the card out of memory, or slow the other to a crawl.
10. **Measure has no stall check.** On a backend that hangs, it waits out its
    2-hour limit. Not changed.

## Phase 08 — Wan2.2 I2V A14B Lightning

2026-10-26 and 27, with this branch's toolchest build and the same
`sd-master-1015-e41fbcf` builds. The set is the 20-step A14B pair plus
Kijai's fp16 copies of the Lightning 4-step LoRAs, with 4 steps on each
expert at CFG 1.

### Before the code (straight to sd-server)

lightx2v's own Lightning I2V files (`lightx2v/Wan2.2-Lightning`, F32 with
I64 `alpha` values) made sd-server abort as the LoRAs loaded
("terminate called recursively" from the loader threads). Kijai's fp16
copies of the same weights load. On asmodean, Short size, the fox picture,
seed 42, `--max-vram 28.5`:

| Run | Time | Whole-card peak | Look |
|---|---|---|---|
| 20 steps, CFG 3.5, no LoRA (Phase 06) | 1,643 s | 27.9 GB | sharp |
| 2 + 2 steps, CFG 1 | 110 s | 29.5 GB | softer; the face smears in the last frames |
| 4 + 4 steps, CFG 1 (the catalog's choice) | 200 s | 31.3 GB | about as sharp as 20 steps |

Per step the LoRAs cost nothing measurable: about 20 s per step at CFG 1,
against about 40 s at CFG 3.5, which runs the model twice.

### asmodean (R9700 32 GB, ROCm)

The generation port range was full, so the check used the disabled
`qwen-turbo-local` instance, pointed at the Lightning set and put back
afterwards. Ming-Image and klein were disabled for Measure: with eviction
off, enabled instances' reservations count even while they are stopped,
and the first Measure, with them enabled, gave sd-server 6.8 GiB and
failed in the VAE encode.

| Check | Result |
|---|---|
| Download | Only the two LoRAs (1.2 GB) were fetched; the experts, text model and VAE were already there from the 20-step set. Done in under a minute |
| Launch | `--lora-model-dir /data/models/gen-models/Kijai--WanVideo_comfy/LoRAs/Wan22-Lightning/old`, no flag per LoRA file. sd-server loaded both LoRAs on every request |
| Measure at Short | About 3 min 15 s (the 20-step set: about 28 min). Peak 28.6 GiB (whole GPU); reservation capped at 29.2 GiB, all the card could give. Whole card 31.2 GB (`rocm-smi`) |
| Start after Measure | Refused: "needs 29.2 GiB, but only 29.0 GiB is left after other programs 0.4 GiB and the safety margin". Other programs had grown by 0.2 GiB since Measure. The reservation was set to 28.8 GiB by hand (finding 11) |
| 1. Image-to-video at the defaults, WebM | 200 s (sampling 82 + 89 s); 33 frames, 5.1 MB, VP8. The frames match the straight-to-sd-server 4 + 4 run; the fox looks up and smiles |
| 2. Start-and-end, WebM | 196 s; 33 frames, 4.6 MB; ends on the end picture |
| 3. `extra: {"lora": []}` | 400 `bad_value`: "extra may not set "lora"; Wan2.2 I2V A14B Lightning (Q4, 4 steps) sends its own LoRAs" |
| Measure at Medium (49 frames at 1280×720) | 1,475 s (about 25 min). Peak 30.5 GiB, more than the 29.0 GiB the card could give, so Measure refused the size and the instance kept Short, as Phase 05 set out. Whole card 33.9 GB |
| Measure at Full | Stopped after it started: Full has more video than Medium, which already did not fit |

### compute2 (RTX A4000 16 GB, CUDA)

| Check | Result |
|---|---|
| Download | Only the two LoRAs, as on asmodean |
| Measure at Short, first try | Sampling worked (71 + 102 s), then the tiled decode crashed: `ggml_cuda_compute_forward: CONCAT failed`, `CUDA error: unspecified launch failure` (finding 1) |
| Measure at Short, second try | Crashed the same way, this time encoding the start picture, before sampling |

As for the 20-step set, the Lightning set is not usable on a 16 GB card
until sd.cpp's CUDA crash is fixed. The test instance was deleted.

### Fixed in this phase

- **A failed video job's reason was lost.** sd-server master-1015 reports
  a failed or cancelled job's `error` as `{"code", "message"}`, not a
  string, so toolchest could not read the job status and reported
  "json: cannot unmarshal object into Go struct field vidJobStatus.error
  of type string". It now reads both forms and shows the message. This
  affected every failed video job on master-1015, not only this set.

### Findings

11. **A reservation capped at the room cannot always be started.** When a
    measured peak fills the card, Measure caps the reservation at exactly
    the room left. A few minutes later other programs had grown by
    0.2 GiB, and the start check refused the instance. Fixed after the
    check (below).
12. **Measure blames memory for any crash.** On compute2 it said "probably
    ran out of GPU memory at this size … Try a smaller size" for the CUDA
    `CONCAT failed` crash, where a smaller size would not help. Fixed
    after the check (below).
13. **lightx2v's own Lightning files crash sd-server** (master-1015) as
    they load; the I64 `alpha` values are the likely cause. Not yet
    reported upstream.

### Fixed after the check (decided with the user)

- **Slack under a capped reservation** (finding 11): a capped reservation
  is the room less 0.5 GiB, but never below the measured peak, nor above
  the room. Checked on asmodean on 2026-10-27: the Lightning set at Short
  (peak 28.6 GiB) got 28.7 GiB in a 29.2 GiB room, and the instance then
  started with no hand-set reservation. TI2V Short on the A4000 (peak
  15.3 GiB over a 14.7 GiB room) still gets the room, as before.
- **Crash or memory** (finding 12): when the backend stops during
  Measure, Measure reads the end of its log. An out-of-memory message
  keeps the memory wording; a crash (a CUDA or HIP error, an abort, a
  segfault, `GGML_ASSERT`) is quoted with "a smaller size may not help".
  Checked with the compute2 and asmodean log lines in
  `TestBackendStoppedError`; not reproduced on hardware. On asmodean,
  squeezed to about 2.5 GiB, sd-server did not stop: the job failed with
  "generate_video returned no results", which toolchest now reads (the
  object-error fix above).

## Phase 07 — LTX-2.3 distilled 1.1, with sound

2026-10-27, with this branch's toolchest build and the same
`sd-master-1015-e41fbcf` builds. Every video is 8 steps at CFG 1, seed 42.
On asmodean the check again used the disabled `qwen-turbo-local` instance,
pointed at the LTX set with ti2v disabled, and put both back afterwards.
compute2 used a new instance, `ltx`, on GPU 1.

### The decode: spatial tiles hang, time tiles work

The entry first had `--vae-tiling`, the recipe's default.

| Run | Result |
|---|---|
| asmodean, Measure at Short, `--vae-tiling` | Hung in the video decode until Measure's 2-hour limit: "the test generation failed: context deadline exceeded". The R9700 sat at 3% busy with one CPU core at 100%. A job through the API hung the same way after sampling (22 s) and the sound decode (1.5 s); gdb showed it waiting in `hipGraphLaunch` under `LTXVideoVAE::_compute` ← `compute_with_temporal_tiling` ← `process_tiles_2d` (finding 14) |
| compute2, Measure at Short, `--vae-tiling` | Crashed in the tiled decode at tile 7 of 15: `ggml_cuda_compute_forward: CONCAT failed` (finding 1) |
| asmodean, straight to `sd-cli`, no tiling, Short | 33 s in all: text model 5 s, sampling 22 s, sound 1.4 s, video decode 4 s. Whole card 27.6 GiB |
| The same at Full (121 frames at 1280×720), no memory limit | 241 s: sampling 204 s, decode 28 s. Whole card 31.8 GiB, the whole card |
| The same at Full with `--temporal-tiling --max-vram 28.4` | 250 s: decode 38 s. Peak 29.6 GiB above the card's starting use. 121 frames, no seams: the frame-to-frame difference has no jump anywhere but the last frame, and the untiled video has the same jump there |

The entry now has `--temporal-tiling` in place of `--vae-tiling`, with a
comment saying why. Between the two builds, Measure without any tiling
granted Short (peak 27.6 GiB, reservation 28.4 GiB) and refused Medium and
Full (peaks 30.9 and 30.8 GiB, over the 29.0–29.2 GiB room), so tiling in
time is what lets Full fit.

### asmodean (R9700 32 GB, ROCm)

| Check | Result |
|---|---|
| Download | 25.6 GB, no token needed |
| Launch | `--steps 8 --cfg-scale 1 --sampling-method euler --diffusion-fa --temporal-tiling`; all 25.2 GB of weights on the card |
| Measure at Short (33 frames, 768×512) | About 70 s. Peak 27.7 GiB (whole GPU); reservation capped at 28.1 GiB, leaving 0.5 GiB of slack |
| 1. Text-to-video at the defaults, WebM | 36 s; 33 frames, 768×512, 5.2 MB. VP8 video and a PCM sound track (mean −37 dB: birdsong and wind) |
| 2. Image-to-video, WebM | 34 s; 4.8 MB; starts from the fox picture; has sound |
| 3. Start-and-end, WebM | 38 s; 4.8 MB; the fox picture to the end picture; has sound |
| 4. Text-to-video, animated WebP | 26 s; 33 frames, 4.8 MB, `image/webp`; no sound track |
| Measure at Full (121 frames, 1280×720) | About 28 minutes. Peak 29.7 GiB (whole GPU); reservation capped at 28.8 GiB, all the card can give, 0.9 GiB under the peak and inside the 1 GiB allowance. Granted: the instance's limit became Full |
| 5. Full, text-to-video, WebM | 260 s; 121 frames at 1280×736 (sd-server rounds 720 up to a multiple of 32), 5.0 s, 39.8 MB, with sound. The whole card went from 0.6 to 29.6 GiB: 29.0 GiB for the video against the 28.8 GiB reservation (finding 16) |
| Full, start-and-end, WebM | 675 s. The two pictures make a video about 2.5 times slower, which is part of why Measure at Full is slow; it does not explain all of the 28 minutes, and nothing else was using the card |
| Players | Job 3's WebM played in Chromium (the desktop app's browser) with its sound decoded (`webkitAudioDecodedByteCount` 526,080). Not opened in Firefox |

Jobs 1–4 ran on the build without tiling; Measure at Short and job 5 ran on
the build with `--temporal-tiling`.

### compute2 (RTX A4000 16 GB, CUDA)

| Check | Result |
|---|---|
| Download | 25.6 GB |
| Measure at Short, `--vae-tiling` | Crashed (above) |
| Measure at Short, no tiling | About 3 minutes. Peak 15.0 GiB; reservation 14.3 GiB, all the card could give |
| 1. Text-to-video, no tiling | 140 s; 33 frames, 5.1 MB, with sound; the same fox in the grass as on asmodean |
| Measure at Short, `--temporal-tiling` | Peak 15.0 GiB; reservation 14.4 GiB |
| 1. Text-to-video, `--temporal-tiling` | Crashed while sampling, at step 2 of 8: `CUDA error: unspecified launch failure` in `ggml_backend_cuda_synchronize`, before the decode, so not caused by the tiling |

So on a 16 GB card LTX-2.3 makes Short videos only some of the time: the
same CUDA crashes as the Wan sets (finding 1), this time also in sampling.

### Fixed in this phase

- **The LTX entry decodes in time tiles**, not spatial tiles (above).
- **Measure's crash message had progress-bar text in it.** An error that
  sd-server prints while its progress bar shows lands on the bar's line,
  so the message quoted the whole bar. Measure now quotes only what comes
  after the bar's last redraw. Found with the compute2 crash above.
- **A cap below the peak was described as leaving slack.** On the A4000
  the reservation is all the room, below the peak, but the note said
  "leaving up to 0.5 GiB free". That case now says the reservation is all
  the card can give and how far the peak is above it, within the 1 GiB
  allowance.
- **Editing an instance kept the last Measure note or error.** Changing an
  instance's set, build, GPU or arguments drops its measurement, but the
  last note or error still showed, so `qwen-turbo-local`, put back to its
  image set, showed an LTX video-size error. It is now dropped too
  (checked on asmodean).

### Findings

14. **The spatially tiled LTX video decode hangs on ROCm** (master-1015),
    as the Wan2.1 VAE's tiled encode does (finding 7). `--temporal-tiling`
    works. Not yet reported upstream.
15. **A job sent while its instance is starting is refused.** With
    eviction off, jobs 1–4, sent right after Start, got 503
    `backend_unavailable` "backend qwen-turbo-local is not running". Image
    jobs behave the same. Not changed.
16. **A video job reports the requested size, not the size made.**
    sd-server rounds LTX sizes up to multiples of 32 and does not say so in
    its result, so job 5 reported 1280×720 for a 1280×736 file. Not
    changed.
17. **Measure at Full is slow for LTX-2.3**: 28 minutes, against 4.5
    minutes for a Full video from a prompt. Measure always sends a start
    and an end picture when the set takes them, and those make the video
    about 2.5 times slower. Not changed.

### Not run

- Opening the videos in Firefox.
- Video on a Vulkan build. No video set was tried on a Vulkan build in
  this project.
- Medium on asmodean with `--temporal-tiling`: Full is granted, and Medium
  has less video.

## Success criteria

From `overview.md`, checked at the end of Phase 07.

| Criterion | Result |
|---|---|
| All three catalog sets download from the Models page with no Hugging Face token, and their licences show in the licence column | Met. Phases 04, 06 and 07; the Models page on asmodean shows Apache-2.0 for the Wan sets and both licences for LTX-2.3 |
| On the R9700, each set is measured with Measure and gets a reservation | Met. TI2V 5B 26.0 GiB (Phase 04), I2V A14B 29.0 GiB (Phase 06), LTX-2.3 28.1 GiB (Phase 07) |
| …makes a text-to-video (TI2V, LTX), an image-to-video (all three) and a start-and-end video (A14B, LTX) through the job API as WebM, and at least one as animated WebP | Partly met. TI2V and LTX-2.3: all of them, WebP included (Phases 04, 07). A14B: image-to-video and start-and-end through the job API with the Lightning set (Phase 08), which runs the same two experts; the 20-step set's videos were made through Measure and straight to sd-server (Phase 06). No A14B video was made as WebP |
| …has its time per video and peak memory in the verification notes and the catalog notes | Met |
| An LTX-2.3 WebM plays with sound in a browser | Met in Chromium (Phase 07). Firefox not run |
| On a 16 GB A4000, TI2V 5B is measured and makes a video through the job API; A14B and LTX-2.3 are tried and written down | Met. Phases 05, 06, 08 and 07 |
| A request over the model's limit or the measured size, or for a kind of video the set cannot make, is refused with a message, and nothing is queued | Met. Phases 04, 05 and 06 |
| After measuring at the largest preset, a video of that size stays within the reservation on the R9700 | Met for LTX-2.3, within the 1 GiB allowance: 29.0 GiB for a 28.8 GiB reservation (Phase 07). For the Wan sets Measure refuses the largest preset on the R9700, so no reservation promises it |
| On a 16 GB A4000, Measure gives a reservation the card can hold, with no value set by hand | Met. Phase 05 (TI2V), Phase 06 (A14B) and Phase 07 (LTX-2.3) |
| The waiting route returns the same finished job as polling does | Met. Phase 04 |
| With eviction on, a video job for a stopped instance starts it, making room as image jobs do, and locked instances are left alone | Met. Phase 04 |
| Image and music jobs, Haruspex's routes and `meta.api` values behave as before | Met: the existing tests pass |
| The help page documents the video API, the three sets and their measured reservations | Met, with the Lightning set as a fourth |

## Release notes

Video generation. llama-toolchest can now run stable-diffusion.cpp's
sd-server for video, with four catalog sets: Wan2.2 TI2V 5B (text-to-video
and image-to-video), Wan2.2 I2V A14B and its 4-step Lightning version
(image-to-video and start-and-end videos), and LTX-2.3 distilled 1.1
(text-to-video, image-to-video and start-and-end videos, with sound).
Programs make videos with the job API (`POST /v1/videos/jobs`, then poll,
or use the waiting route), and download them as WebM (VP8; LTX-2.3 adds a
sound track) or silent animated WebP; videos are kept for 24 hours. Measure
tests a video instance at a size you pick and refuses sizes the card cannot
hold; requests over the measured size are refused. The Wan sets are under
Apache-2.0. LTX-2.3 is under the LTX-2 Community Licence (free under US$10
million yearly revenue) and its Gemma 3 text model under Google's Gemma
Terms of Use. On a 16 GB card stable-diffusion.cpp's CUDA build crashes
often in video; a 32 GB card is the tested size.
