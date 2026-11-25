# Video upscaling — field test (Phase 08)

2026-11-24. Test build 2.66.0-vupscale-test2 (all phases plus the two fixes below), swapped into the container on both machines; the 2.66.0 release is kept in each container as `/usr/bin/llama-toolchest.2.66.0`.

| Machine | GPU | stable-diffusion.cpp |
| --- | --- | --- |
| compute2 | RTX A4000 16 GiB (GPU 2), CUDA | `sd-master-1116-bf31b9c-cuda` (the minimum build) |
| asmodean | Radeon AI PRO R9700 32 GiB, ROCm 10 | `sd-master-1122-3b12781-rocm-d17328` |

## Clips

- **A:** a Wan2.2 TI2V 5B video made by asmodean's `ti2v` instance: 832×480, 33 frames, 24 fps, VP8 WebM as sd.cpp writes it.
- **B:** A looped to 81 frames, VP8 with every frame a keyframe (ffmpeg on the dev machine). `ti2v` is measured only for 33 frames, so it could not make 81.
- **C:** A scaled to 512×288, the same encoding. (`ti2v` ignored a 512x288 size and made 832x480 again; that is outside this feature.)
- **D:** C as MJPEG AVI with a 440 Hz tone as PCM16 audio.
- An MP4 of A for the refusal test.

`TestProbeField` read all four with the right size, frame count, rate and audio. A is a real sd.cpp output.

## Findings

1. **`--max-vram` breaks DGL upscaling.** Inside a 10.5 GiB `--max-vram` (the reservation from a 9.3 GiB measured peak), RelayVSR ×2 at 832×480 failed every time with `diffusion output shape [] does not match input [104, 60, 2, 48, 1]`. sd-server's log: `execution engine cannot make enough memory available on CUDA0: need 2747 MB device / 6434 MB budget` and `RelayVSR Wan segment 31/31 failed during weight preparation`. A hand-started sd-server on the same GPU with no `--max-vram` ran the job and peaked at 9.3 GiB (nvidia-smi). Measure had worked because it runs inside the whole room. **Fixed:** video upscaler instances run with no `--max-vram`; the reservation is the measured peak plus 10%, for planning, as for the ESRGAN upscaler (`fix(generation): run video upscalers without --max-vram`).
2. **Out of memory is reported as an empty result.** When the card itself is too small, the job's error is again `diffusion output shape [] does not match input` (RelayVSR ×4 at 832×480 on the A4000, which needs about 27 GiB) or `VAE encode failed` (SeedVR2 window 5 at 832×480), while sd-server logs the real reason. SeedVR2 window 5 at 512×288 instead crashed sd-server with `CUDA error: out of memory`, which Measure already reported as "probably ran out of GPU memory". **Fixed:** these texts map to "sd.cpp ran out of GPU memory for this clip size or window", and Measure says to try a smaller clip or window (`fix(api): name sd.cpp's DGL memory failures as out of GPU memory`). Worth an upstream issue: a step that fails weight preparation should fail the job with that reason.
3. **The frame count matters.** RelayVSR ×2 at 832×480 on the A4000: 6.8 GiB for 33 frames, 8.6 GiB for 81. The frame check in the measured ceiling stays.

## Results

Measured at 832×480, 81 frames (window 16 for RelayVSR, 1 for SeedVR2) unless noted. Times are wall time for clip B (81 frames at 832×480), as the client sees it.

| Set | Machine | Peak | Reservation | B to webm | B to avi | D to avi (sound) | D to webm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RelayVSR ×2 | compute2 | 8.6–9.3 GiB | 9.5–10.5 GiB | 35 s, 1664×960, 81 f | 25 s | 5 s, 1024×576, sound kept | 5 s, `audio_dropped` |
| RelayVSR ×2 | asmodean | 9.3–10.6 GiB | 10.5–12.0 GiB | 81 s | 66 s | 10 s, sound kept | 10 s, `audio_dropped` |
| RelayVSR ×4 | compute2 | does not fit; 10.8 GiB at 512×288 | 12.0 GiB (512×288) | – (over the measured size) | – | 15 s, 2048×1152, sound kept | – |
| RelayVSR ×4 | asmodean | 27.2 GiB | 27.3 GiB (capped to the card) | 317 s, 3328×1920, 81 f | 282 s | 41 s, sound kept | 40 s, `audio_dropped` |
| SeedVR2 3B | compute2 | 9.6 GiB | 11.0 GiB | 317 s, 1664×960, 81 f | 307 s | 50 s, sound kept | 51 s, `audio_dropped` |
| SeedVR2 3B | asmodean | 10.0 GiB | 11.0 GiB | 251 s | 241 s | 35 s, sound kept | 35 s, `audio_dropped` |
| SeedVR2 3B, window 5 | compute2 | does not fit (also at 512×288) | – | – | – | – | – |
| SeedVR2 3B, window 5 | asmodean | 24.1 GiB | 26.5 GiB | 723 s | – | – | – |

Every output had the input's frame count, its size times the scale, and its frame rate. AVI output carries the clip's sound (as 32-bit float PCM); webm output has none, and the job says so.

**Quality** (frame 16 of B, and frame-to-frame change against the input): both models are much sharper than a bicubic upscale, SeedVR2 most of all, with no colour shift and no tile seams. RelayVSR's window joins (frames 15, 30, 45, 60, 75) change no more than the input does there (ratio 0.3–1.1). SeedVR2 at window 1 changes 1.5 times as much as the input from frame to frame: more shimmer, as expected when each frame is treated on its own.

## Refusals and lifecycle

| Check | Result |
| --- | --- |
| MP4 | 400 `unsupported_video`: "MP4 is not supported: send AVI (MJPEG or RGB), WebM (VP8) or animated WebP" |
| Clip over the measured frames (RelayVSR ×2 measured at 33, B has 81) | 400 `over_measured_size`, with both sizes |
| Window 4 on SeedVR2 | 400 `invalid_window_size` |
| Instance pinned to `sd-master-1015` (asmodean) | Start refused with the master-1116 message; the Server page shows the build note; a job gets 503 with the same text |
| Cancel a running SeedVR2 job | 204; the job shows `cancelled`; the instance stopped and GPU 2 went back to 4 MiB; the next job ran normally (56 s) |
| Restart the container during a job | The job shows failed, "the server restarted before the job finished"; the instance's `input/` and `output/` and the incoming folder are empty |
| Old-build instance and the work folders after every job | No files left behind |

## Custom set

SeedVR2 7B Sharp on compute2: the set was built in the custom set form (the DiT as a Hugging Face link, which the form downloaded; the VAE reused from the 3B set). Measured at 832×480, 33 frames, window 1: peak 14.0 GiB, so the reservation is capped at 14.0 GiB, all the 16 GiB card can give the instance. Clip A (33 frames) went to 1664×960 in 155 s. That is 4.7 s per frame against the 3B set's 3.9 s. 81 frames was not tried: with no room left at 33 frames, it would most likely not fit this card.

## State left behind

- compute2: test build 2.66.0-vupscale-test2; instances `relayvsr-x2`, `relayvsr-x4`, `seedvr2-3b` and `seedvr2-7b-sharp` on GPU 2, stopped. The 2.66.0 release is `/usr/bin/llama-toolchest.2.66.0`.
- asmodean: test build 2.66.0-vupscale-test2; instances `relayvsr-x2`, `relayvsr-x4`, `seedvr2-3b` on GPU 0, stopped; all 16 generation ports are now in use. The 2.66.0 release is `/usr/bin/llama-toolchest.2.66.0`.

## Upstream fixes: out-of-memory code and DGL cancel (PR #715)

2026-11-25, compute2 (RTX A4000, GPU 2). Test build 2.67.0-dgl715-test; the 2.67.0 release is kept as `/usr/bin/llama-toolchest.2.67.0`. stable-diffusion.cpp `sd-master-1132-5fa203c-cuda`, which has leejet/stable-diffusion.cpp#2419 (`out_of_memory` error code) and #2420 (DGL cancel route). The clip was the 512×288, 33-frame WebM (clip C), so outputs are 1024×576.

| Check | Build | Result |
| --- | --- | --- |
| Cancel a running SeedVR2 3B job, 20 s in | 1116 | 204; sd-server answered 404 to the cancel route; toolchest logged "before master-1132" and stopped the instance; GPU 2 back to 4 MiB |
| Measure RelayVSR ×4 at 832×480, 33 frames (does not fit 16 GiB) | 1132 | Measure error: "the test upscale ran out of GPU memory at this size; try a smaller clip or window (sd.cpp: … RelayVSR Wan segment 26/31 (relayvsr.blocks.25) failed during weight preparation: execution engine cannot make enough memory available on CUDA0: need 1016.03 MB device / 13636.72 MB budget, available 993.56 MB device / unlimited budget)" |
| Measure SeedVR2 3B, 832×480, 33 frames, window 1 | 1132 | peak 9,813 MiB, reservation 11,264 MiB (same as on 1116) |
| Measure RelayVSR ×2, 832×480, 33 frames, window 16 | 1132 | peak 9,489 MiB, reservation 10,752 MiB (about the same as on 1116) |
| Cancel a running SeedVR2 3B job, 20 s and 15 s in (twice) | 1132 | 204; the job showed `cancelled` 2 s later; the instance kept running with its weights loaded (4.5 GiB on GPU 2) |
| A job right after each cancel | 1132 | succeeded in 45–46 s, 1024×576, 33 frames |

The out-of-memory message now says which step ran short and by how much. The old text markers would have matched this message too ("cannot make enough memory"), so this run does not show which check caught it; the unit test covers the `out_of_memory` code.

One submit right after the first 1132 cancel returned no job ID; the test script dropped the reply, so its text is lost. The same sequence, repeated, worked; it could not be reproduced.

Not tested: `--max-vram` for upscaler instances (still off), and RelayVSR cancel (a job on this clip takes about 5 s).

### `--max-vram` on master-1132

The same day, by hand: sd-server `sd-master-1132-5fa203c-cuda` started in the container on GPU 2 with the instance's `--model-dir` folders and `GGML_CUDA_REUSE_BUFFERS=0`, one DGL job posted to `/sdcpp/v1/dgl/jobs` with toolchest's own rendered script (default window, seed 1) on the Measure clip (832×480, 33 frames). Peak is nvidia-smi's whole-card figure, sampled every 0.5 s, so it includes the CUDA context.

| Set | `--max-vram` | Result | Time | Peak |
| --- | --- | --- | --- | --- |
| RelayVSR ×2 | none | completed | 14 s | 10,753 MiB |
| RelayVSR ×2 | 10.5 | completed | 14 s | 10,863 MiB |
| RelayVSR ×2 | 9 | completed | 14 s | 9,355 MiB |
| RelayVSR ×2 | 7 | completed | 17 s | 7,801 MiB |
| RelayVSR ×2 | 5 | completed | 18 s | 5,681 MiB |
| SeedVR2 3B | none | completed | 125 s | 9,817 MiB |
| SeedVR2 3B | 10.5 | completed | 126 s | 9,817 MiB |
| SeedVR2 3B | 8 | completed | 144 s | 9,471 MiB |
| SeedVR2 3B | 6 | completed | 136 s | 6,655 MiB |
| RelayVSR ×4 | 14 | `out_of_memory` (segment 26/31) | 7 s | 14,275 MiB |
| RelayVSR ×4 | 10 | `out_of_memory` (segment 14/31) | 6 s | 10,701 MiB |

Finding 1 above is fixed upstream: under a limit, master-1132 keeps part of the weights in RAM and the job completes, a little slower. The peak can still go past the limit (SeedVR2 at 8 GiB peaked 1.3 GiB over it), so a limit needs a measured peak, as for other kinds. A limit does not help when the runtime memory alone is too large: RelayVSR ×4 at 832×480 held about 13 GB of runtime memory and failed under both limits. sd-server's own job error was `{"code": "out_of_memory", ...}`, which is what PR #715 reads.

Toolchest still starts upscaler instances without `--max-vram`. Turning it back on for builds from master-1129 is a separate change.
