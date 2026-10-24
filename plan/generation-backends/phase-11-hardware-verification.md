# Phase 11 — Hardware Verification and Release Notes

**Depends on:** Phases 01–10 · **Enables:** release

## Goal

Check every success criterion in `overview.md` on real hardware, in both
install modes, with ROCm, CUDA and Vulkan. Record the measured numbers in the
Help page and release notes, so users know what to expect.

No new features are added in this phase. Bugs found here are fixed in this
phase, each with its own `fix(...)` commit.

## Files touched

- `plan/generation-backends/verification.md` (new): the results table for each
  machine and mode, with dates and build refs.
- `web/templates/help.html`: replace the reference numbers in the Generation
  section with the measured values.
- Any source file that needs a fix found during verification. Each fix comes
  with a regression test.

## Steps

1. **asmodean (R9700, 32 GB, ROCm 7.1.1, Fedora 44), host mode (`.rpm`),
   on a clean data directory.** Move the development data from Phases 01–10
   aside first, so set and instance IDs such as `qwen-turbo-local` are free.
   1. Feature off: run `go test ./...` and compare the UI, `/v1/models` and
      the GPU map against a build from before the feature.
   2. Build llama.cpp, sd.cpp and acestep.cpp with the `rocm` profile, then
      with the `vulkan` profile.
   3. **Ming-Image on its own.**
      1. With no LLM loaded, download the `ming-image` catalog set, then
         create and measure its instance.
      2. Record the peak and the reservation in MiB. Expect a peak above
         16,500 MiB, because the weights alone are 17.3 GB.
      3. Confirm `/v1/images/generations` with `model: ming-image` returns an
         RGBA PNG.
      4. **Find out what fits beside Ming-Image.**
         1. Try to enable the ACE-Step instance from step 1.4 next to it, once
            that instance exists. Re-run this sub-step after step 1.4.4.
         2. Then try loading Qwen3-4B Q8_0 next to Ming-Image.
         3. Record which of the two the fit check accepts.
         4. Put the measured result in `verification.md` and the Help text
            (Phase 09).
      5. Disable the instance.
   4. **Custom set, three at once.**
      1. Create a custom set named "qwen-turbo-local" from the Qwen-Image 2.1
         Turbo files in `~/models/gen-test`.
         - Recipe: `qwen-image-21-turbo`.
         - Extra launch arguments: `--sigmas qwen-image-2.1-turbo --cfg-scale 1
           --sampling-method euler --diffusion-fa --vae-tiling`. These are the
           settings measured on 2026-10-24.
      2. Create its instance and measure it with nothing else running. Accept
         the reservation. Expect a peak near 10,283 MiB and a reservation of
         11,776 MiB.
      3. Stop the Qwen instance, because measuring needs the GPU to itself.
      4. Download `ace-step-xl-sft`, then create and measure its instance.
         Expect a peak near 7,239 MiB and a reservation of 8,192 MiB.
      5. Start both instances.
      6. Confirm that Gemma 4 12B Q6 (about 12,150 MiB) is refused. Then load
         Qwen3-4B Q8_0 (`unsloth/Qwen3-4B-GGUF`), which fits in the 8,746 MiB
         left.
   5. Generate through `/v1/images/generations` with
      `model: "qwen-turbo-local"`.
   6. Confirm `/v1/music/jobs` returns a 60 s MP3, and that `/release_task` →
      `/query_result` → `/v1/audio` works with no `model` field.
   7. Send two image requests and one music request at once. Check that:
      - The started and finished timestamps in
        `GET /api/gen/queue/history?gpu=0` (Phase 07) never overlap.
      - While they run, the dashboard shows the jobs running and waiting on
        GPU 0, and the slowed-chat banner.
      - The banner clears when the queue is empty.
   8. Confirm that **enabling** the disabled Ming-Image instance now is
      refused by the fit check (Phase 06 step 3), with a message that names
      the other allocations.
   9. Confirm that loading Qwen3.6 27B Q4 is refused with the 507 VRAM message.
   10. Confirm that no server logs an out-of-memory error.
   11. Repeat the Ming-Image and ACE-Step runs with the Vulkan builds, and note
       the "estimated (whole-GPU)" reservation label.
2. **asmodean, container mode (ROCm image):** repeat steps 1.3–1.7. For the
   container install, answer yes to the generation web UI ports prompt
   (Phase 10). Then open
   both web UIs through the dashboard links on the published ports, in plain
   mode and in `--secure` mode. Then install once more with the host port
   range start set to 9081, and confirm the dashboard links point at 9081 and
   9082.
3. **compute2 (3× A4000, 16 GB each, CUDA), container mode:**
   1. Build sd.cpp and acestep.cpp with the `cuda` profile.
   2. Put an LLM on GPU 0, FLUX.2 klein on GPU 1 and ACE-Step turbo on GPU 2.
   3. Confirm that image and music jobs run at their stand-alone speed while
      chat is active, because they are on separate GPUs. Record the stand-alone
      and concurrent times.
   4. Try Ming-Image on one 16 GB card.
      - If it does not fit, edit the instance and add `--offload-to-cpu` to
        its extra launch arguments (`Instance.ExtraArgs`, Phase 05).
      - Record the result and timing in `verification.md`.
      - Record it in the catalog entry's `Notes` too, with a `fix(generation)`
        commit that updates `catalog.go`.
4. **asmodean with an RTX 3090 added next to the R9700 (one server, two GPU
   makers), host mode.**
   - This needs ROCm and CUDA toolkits both installed on the host. The
     container images are built for one vendor each.
   - Run this step once the 3090 is installed. Until then, `verification.md`
     marks it "not run (hardware not installed)", and the release notes say
     that mixed-vendor servers have not been verified.

   Steps:
   1. Build sd.cpp with the `rocm` profile and acestep.cpp with the `cuda`
      profile.
   2. Create a Qwen-Image instance on the R9700 and an ACE-Step instance on
      the 3090.
   3. Confirm the instance create form refuses a ROCm build on the 3090 and a
      CUDA build on the R9700 (Phase 05 step 1).
   4. Confirm each process sees only its own GPU. Its log names the R9700 or
      the 3090 respectively, through `HIP_VISIBLE_DEVICES` or
      `CUDA_VISIBLE_DEVICES`.
   5. Confirm `ProcessVRAM` reads the sd-server PID from `rocm-smi` and the
      ace-server PID from `nvidia-smi`.
   6. Confirm the GPU map shows each reservation on the right card.
   7. Confirm the queue history shows the image and music jobs running at the
      same time, because they are on different GPUs.
   8. Load an LLM on one card and confirm the slowed-chat banner appears only
      for jobs on that card.
5. **Haruspex:** point Haruspex's image provider at
   `http://asmodean:3000/v1`, and confirm Ming-Image generation and alpha
   output work end to end.
6. **Release notes:** write the release-notes paragraph for release-please,
   based on the `feat` commits from Phases 01–10. It covers the feature switch,
   supported backends and GPU profiles, the Vulkan host-only rule, and the
   web-UI-not-queued note.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make installer-test`

## Test plan

This phase is the test plan. `verification.md` must show a pass for every
success criterion in `overview.md`, each with the machine, mode, date and build
refs.

## Commit

`docs(generation): record hardware verification results`

Any fixes found during verification each get their own `fix(...)` commit
before it.

## Rollback

`verification.md` and the Help text changes can be reverted on their own. Each
fix commit made in this phase can be reverted on its own, together with its
regression test.
