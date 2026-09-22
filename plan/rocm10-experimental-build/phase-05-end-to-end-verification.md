# Phase 05 — End-to-end verification

**Depends on:** Phases 02, 03, 04 · **Enables:** Phase 06 (the documentation
states verified facts, including any failure)

## Goal

Find out whether the experimental path actually works, on real hardware, through
the route a user takes. The overview sets this as the done bar: the image builds,
the app starts, the GPU is visible inside the container, a `llama.cpp` build with
`GGML_HIP=ON` completes, and a model loads and generates tokens. This phase
writes no product code; it produces a findings document, and if something fails
the failure is recorded and carried into the documentation rather than dropped.

The machine: Radeon RX 9070 XT (gfx1201, RDNA 4) plus a gfx1036 integrated GPU,
kernel `7.2.3-1-cachyos`, ROCm 7.2.4 host userspace, podman 6.1.1. Two 4B/9B
Qwen3.5 GGUF models are already installed, which is enough to test a load.

## Files touched

- `plan/rocm10-experimental-build/findings-verification.md` — new; the result of
  each check below, pass or fail, with the output that shows it.

## Steps

1. **Record the starting state** so it can be restored: the current
   `ROCM_VARIANT` in `.env` (absent means stable), the active build
   (`b10453-rocm-optimized` at the time of writing), and
   `podman images localhost/llama-toolchest`. Note the ref that build was made
   from — every llama.cpp build in this phase uses that same ref, so the
   throughput comparison in Step 8 measures ROCm versions rather than two
   different llama.cpp revisions.
2. **Make a stamped baseline build on the stable image.** Still on ROCm 7.2.4 —
   and on a container built with `--from-source`, or nothing will be stamped at
   all (see Step 3) — build llama.cpp once from the Builds page at the ref from
   Step 1. This is
   necessary, not optional: Phase 04 stamps only new builds, and the existing
   builds carry no stamp, so without this step there is nothing that can ever be
   flagged and Steps 7 and 9 could not be verified at all. Confirm the new build
   shows `rocm 7.2.4` in the Built against column and is not flagged. Record its
   build ID, and its generation and prompt throughput on
   `Qwen3.5-4B-UD-Q4_K_XL` — that is the baseline Step 8 compares against.
3. **Install through the real route.** Run
   `./setup.sh install --container --from-source --rocm-image 10.0.0-full`.

   `--from-source` is not optional here, and the reason is easy to miss: the
   Dockerfiles install the *released* package from GitHub, so a plain install
   produces a container running the last release — which has neither the build
   stamp from Phase 04 nor the variant work from Phase 03. Steps 2, 7 and 9
   would all be unverifiable against it. `--from-source` builds this tree and
   installs it over the packaged binary, keeping the package's dependencies.

   Record, in order: that the
   tag validation passed, what the kernel pre-flight said (this kernel is well
   above the 6.12 floor RDNA 4 needs, so expect the "new enough" line rather
   than a warning), and that the switch warning
   appeared because the previous variant was stable.
4. **Confirm the image and the app.** The container starts, the UI answers on the
   management port, and the Builds page loads. Record the built image size.
   The Builds page should now name the toolchain at the top —
   "Running rocm 10.0.0 · built on docker.io/rocm/dev-ubuntu-24.04:10.0.0-full"
   — which also confirms the container is running this tree rather than the
   release, since the release has no such line.
5. **Confirm the GPU inside the container.**
   `podman exec llama-toolchest rocminfo | grep -E 'gfx1201|Marketing Name'`
   finds the RX 9070 XT. Also record `podman exec llama-toolchest cat
   /opt/rocm/core/.info/version` — it must read `10.0.0`, and it is the path the
   build stamp reads in this image (Phase 01 found `/opt/rocm/.info/version`
   absent here).
6. **Build llama.cpp.** From the Builds page, build the ROCm profile with
   `GGML_HIP=ON` at the same ref used in Steps 1 and 2 — not merely "a recent
   ref", so the comparison holds. Do not enable `GGML_HIP_ROCWMMA_FATTN`: the overview
   makes it a non-goal, upstream removed the path at b10332, and whether the base
   image ships rocWMMA headers is irrelevant to this result. Record the ref, the
   build duration, and the outcome.
   Two defined follow-ups, both of which are work belonging to this phase rather
   than notes for later:
   - If Phase 02 test 8 recorded the HIP cmake config somewhere other than
     `/opt/rocm/lib/cmake/hip/hip-config.cmake`, apply the cache-variable
     workaround now — point cmake at the ROCm root the way `Dockerfile.rocm`'s
     comments describe for Fedora — and record the change.
   - If the build fails for any other reason, capture the cmake and compiler
     output. The HIP cmake-config path is the most likely cause; fix it here, in
     `Dockerfile.rocm-next`, and note the fix in the findings so Phase 06 can
     mention it if it affects anyone else.
7. **Confirm the stamp and the flag.** Three things on the Builds page:
   - the new build shows `rocm 10.0.0` and is not flagged;
   - the **Step 2 baseline build** shows `rocm 7.2.4` and *is* flagged, with a
     tooltip naming both versions. This is the first time Phase 04's flag is
     exercised against two genuinely different ROCm lines, and it works only
     because Step 2 created a stamped 7.2.4 build;
   - the builds that predate Phase 04, such as `b10453-rocm-optimized`, still
     show an em-dash and the not-recorded tooltip, and are **not** flagged. Phase
     04 never guesses at an unstamped build, so this is correct behaviour, not a
     gap.
8. **Load a model and generate.** Make the new ROCm 10 build active, load
   `Qwen3.5-4B-UD-Q4_K_XL`, and send a short chat completion. Record generation
   and prompt throughput and compare against the Step 2 baseline figures — same
   model, same llama.cpp ref, different ROCm. Record the comparison whether or
   not it is favourable; Phase 06 documents this number, and a large regression
   is worth knowing before recommending the path to anyone.
9. **Confirm a flagged build still activates.** Select the Step 2 baseline build
   (now flagged) as active and confirm the app allows the selection — the flag
   informs, it does not block. It will then fail to load, which is the expected
   behaviour and exactly what the tooltip warned about. Record that the failure
   matches the prediction.
10. **Test the second ROCm line.** Re-run
    `./setup.sh install --rocm-image 7.14.1-full`, confirm the switch warning
    fires again, the image rebuilds on the 7.14 base, and
    `/opt/rocm/.info/version` reports a 7.14 version. This proves the version pin
    is not hard-coded to 10 and gives the docs a second confirmed tag.
11. **Return to stable and confirm nothing was lost.** Two parts, in this order,
    because the first changes what the second would observe:
    - *First*, while the stored variant is still `next` from Step 10, start
      `./setup.sh install` with no flags and confirm the menu marks option 2
      `(current)` with `2` as its default — the experimental variant is not
      silently dropped on a re-run. Answer `1`, confirm the summary now shows
      `Dockerfile.rocm`, then **decline at the confirmation prompt**. Declining
      matters: completing this install would store `stable` in `.env`, and the
      switch warning in the next part could then no longer fire, because the
      stored and selected variants would already agree.
    - *Then*, with the stored variant still `next`, run
      `ROCM_VARIANT=stable ./setup.sh install` and let it complete. An explicit
      selection rather than "accept the default", because the default follows
      what is installed (Phase 03 Step 4) and is therefore not a fixed value.
    Confirm: the switch warning fires, the stable
    Fedora image is rebuilt, the Step 2 baseline build is no longer flagged and
    loads a model again, and the original `b10453-rocm-optimized` build is still
    present and still loads. This is the property the "delete nothing" decision
    exists for, and it is the most important single check in this phase.
12. **Write `findings-verification.md`** with a row per step: what was run, what
    happened, pass or fail. End with a **Verdict** section stating whether the
    experimental path works on this hardware, any caveat a user should know, and
    the measured throughput comparison. If any step failed, state plainly what
    failed and whether the option should still ship — the overview's answer is
    that it ships as experimental with the failure documented.

## Build gate

```
go build ./... && go vet ./...
go test ./...        # the two internal/builder tag.gpgsign failures are expected here
bash -n setup.sh
```

Everything from Phases 02–04 must still pass after any fix made during this
phase. The two `internal/builder` tests that fail on this machine because of
`tag.gpgsign` are a known local-environment issue and are not a regression. No new gate of its own: this phase's gate is that every step above has a
recorded pass-or-fail.

## Test plan

This phase *is* the test plan for the whole project. The scenarios are the
numbered steps; each records an observed result rather than an expectation. The
checks that most determine whether the work shipped correctly:

- Step 6 — whether a `GGML_HIP` build succeeds against ROCm 10 at all.
- Step 8 — whether a model loads and generates, and at what speed.
- Step 11 — whether returning to stable restores the previous builds untouched.

A failure in step 6 or 8 is a documented caveat, not a blocker on shipping the
option. A failure in step 11 is a blocker: it would mean switching destroys
working state, which the design promises it does not.

## Commit

```
docs(plan): record the ROCm 10 end-to-end verification results
```

## Rollback

The only file added is in the plan folder. The machine is returned to its
starting state by step 11, which is part of the phase rather than a cleanup
afterwards. If the phase is abandoned midway, run
`ROCM_VARIANT=stable ./setup.sh install` to get back to the stable image — an
explicit selection, not "accept the default", since the default follows whatever
variant is currently installed. Then rebuild `llama.cpp` from the Builds page if
a ROCm 10 build was made active.
