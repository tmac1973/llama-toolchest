# Video Generation — Hardware Verification

Results of the hardware checks: Phase 04 (Wan2.2 TI2V 5B), Phase 05 (measured
video size), and Phases 06 and 07 later. Each entry gives the machine, the date and the
build refs.

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
- Measure for the A14B pair and LTX-2.3 (Phases 06 and 07).
