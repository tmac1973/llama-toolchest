# Phase 06 — VRAM Reservations and Fit Planning

**Depends on:** Phase 05 (running instances and their PIDs) · **Enables:**
Phase 07 and Phase 08 (APIs only route to instances that have a reservation),
Phase 09 (dashboard shows reservations)

## Goal

Every instance gets a fixed VRAM reservation on its GPU.

- **Setting it:** a **measured test run** loads the set, runs one generation
  and records that process's peak VRAM. It suggests the peak plus a margin as
  the reservation, and the user can accept or edit it.
- **Fit planning:** the LLM fit planner subtracts reservations from each card's
  budget. The VRAM those processes actually hold is no longer counted as
  "other programs", so nothing is counted twice.
- **Start check:** an instance only starts when its reservation fits next to
  the LLM models and the other reservations.
- **sd-server enforcement:** sd-server is also given its reservation as
  `--max-vram`.

The asmodean test showed why this is needed. sd-server planned its memory from
the VRAM free when it started and did not know about ace-server.

## Files touched

- `internal/monitor/procvram.go` (new): `ProcessVRAM(pid) (mib int, gpu int,
  ok bool)`.
  - NVIDIA: `nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id
    --format=csv,noheader,nounits`, mapped to the monitor index by bus ID.
  - ROCm: `rocm-smi --showpids` with the per-PID VRAM column. If rocm-smi is
    missing, it falls back to reading `/sys/class/kfd/kfd/proc/<pid>/vram_*`.
  - Vulkan: no per-process data. It returns `ok=false`.
  - **A PID that does not appear in the tool's output** also returns
    `ok=false`, never a zero reading. This happens inside a container PID
    namespace, where the tools list host PIDs.
  - All VRAM figures in this plan and in the code are MiB (2^20 bytes), the
    unit the monitor already uses.
- `internal/monitor/procvram_test.go` (new): parser tests that use captured
  command output.
- `internal/generation/measure.go` (new): `Measure(ctx, inst)`, which runs the
  measured test run.
- `internal/generation/manager.go`: add a `measuring` state per instance,
  plus `IsMeasuring(id)`.
- `internal/models/fit.go`:
  - `GPUSpec` gains `ReservedMiB int`.
  - `cardBudgetGiB` subtracts it as well as `OtherUsedMiB`.
  - `FitBudgets` and `PlanCards` show the reservation separately.
- `internal/api/autoconfig.go`: `Server.hardware()` (L66) fills `ReservedMiB`
  per GPU from the enabled instances, and only while `GenerationEnabled` is
  on. With the switch off, every VRAM figure is the same as today.
- `internal/generation/host.go` (new): the `Host` interface that
  `internal/api` passes to `generation.Manager` and `Measure` at construction.
  `generation` never imports `api`. The interface has three methods:

  | Method | Implemented with |
  |---|---|
  | `GPUBusy(gpu int) (reason string, busy bool)` | `gpuBusyReason` |
  | `GPUUsedMiB(gpu int) int` | `monitor.Current()` |
  | `FitCheck(inst Instance) error` | step 3 |

  `ProcessVRAM` is called directly, because `generation` may import
  `monitor`.
- `internal/api/other_vram.go`: `attributeVRAM` subtracts the measured VRAM of
  running instance PIDs before computing "other". When a backend reports
  `ok=false`, it subtracts that instance's reservation instead.
- `internal/api/gen_instances.go`:
  - `POST /api/gen/instances/{id}/measure`, with progress as SSE.
  - `PUT /api/gen/instances/{id}/reservation`.
  - A start check before `Start`.
- `internal/api/gpu_map.go`, `internal/api/gpu_view.go`: show each
  reservation as its own coloured segment on the GPU map, labelled with the
  instance name.
- `web/templates/partials/gen_instances.html`: a "Measure" button, the measured
  peak, a reservation field and a fit warning. Each has a tooltip that explains
  how to read it.

## Steps

1. **Run the measured test run, `Measure(inst)`:**
   1. Refuse to run while a benchmark job is busy on the same GPU
      (`gpuBusyReason`, `autoconfig.go` L689), or while another generation
      instance on that GPU is running. The message says "Stop <ids> first".
      Measuring needs the GPU to itself, so the peak belongs to this instance
      alone.
      - Phase 07 extends this to the generation job queue.
   2. Mark the instance as `measuring` in `generation.Manager`. Phase 07 will
      not route API requests to an instance in this state.
   3. If the instance is running, stop it. Wait 2 s, then record the GPU-wide
      used VRAM as the baseline.
   4. Start it with `LaunchOptions{OmitMaxVRAM: true}`.
   5. Run one fixed generation directly against the backend's port:
      - **sd:** `POST /v1/images/generations` at the set's
        `DefaultWidth` × `DefaultHeight` (Phase 02).
      - **ace:** `POST /lm`, then `POST /synth`, polling `/job`. The request is
        fixed in `measure.go` as `measureSongRequest`, the same request used
        on asmodean on 2026-10-24:
        - caption: "Upbeat indie folk song with acoustic guitar, warm male
          vocals, hand claps and a catchy chorus"
        - lyrics: two verses and two choruses, copied from
          `~/scratch/acestep-test/req.json` on asmodean into the Go source
        - `duration: 60`, `vocal_language: "en"`, `use_cot_caption: false`,
          `seed: 42`
   6. Sample `ProcessVRAM(pid)` every 300 ms during the run and keep the
      maximum.
      - If `ProcessVRAM` returns `ok=false` (Vulkan), use the peak GPU-wide
        used VRAM during the run minus the baseline.
      - In that case set `MeasureEstimated=true`; the UI shows the label
        "estimated (whole-GPU)".
   7. Stop the instance and clear the `measuring` state.
      - **If it was running before `Measure` began,** start it again normally,
        so `--max-vram` is applied and the start check runs.
      - **If it was stopped before,** leave it stopped.
2. **Suggest a reservation:** the measured peak × 1.10, rounded up to the next
   512 MiB. Save `MeasuredPeakMiB` and `MeasuredAt`. Set `ReservedMiB` to the
   suggestion only when it is 0. Otherwise show the suggestion next to the
   current value.
3. **Check whenever a reservation would start to count, and before starting an
   instance.** `Host.FitCheck` runs when:
   - an instance is **enabled** or **created** (which enables it)
   - its reservation is changed through `PUT /reservation`
   - a new measurement would set the reservation
   - the instance is **started**, inside `generation.Manager.Start`. This
     covers the UI button, auto-start (Phase 05 step 8) and the restart at the
     end of `Measure`.

   Together these keep the Phase 02 promise: an enabled instance can always
   start.
   - If enabling or a reservation change fails the check, it is refused, and
     the message names the allocations on that card.
   - If a new measurement does not fit, the measured peak is saved, the
     reservation is left unchanged, and the instance is disabled. The UI shows
     the fit message.
   - `Measure`'s own first start (step 1.4) uses the instance's current
     `ReservedMiB`, which is 0 for a new instance. It needs at least 1 GiB of
     the card budget free. If the final restart fails the check, the instance
     stays stopped and the UI shows the fit message next to the new
     measurement.
   
   Build `Hardware` with `ReservedMiB` for the other **enabled** instances on
   that GPU, whether they are running or stopped (see the meaning of
   `Enabled` in Phase 02). Then check:

   (LLM models **currently loaded** by the router on that GPU) +
   (other reservations) + (this reservation) ≤ card budget

   - "Currently loaded" means the router's loaded-model list
     (`ListModels`), passed through `ComputeAllocations`.
   - Configured presets that are not loaded do not count.

   - If the check fails, refuse to start and show which allocations use the
     card.
   - An instance with `ReservedMiB == 0` can start only through `Measure`.
4. **Check before loading an LLM.** The fit planner already sees the
   reservations through `hardware()`, so autoconfig, the GPU map and router
   loads respect them.
5. **Handle the case where the router's actual use exceeds its plan.** The
   router loads LLM models on demand (LRU through `--models-max`). A request
   could therefore load a model that does not fit next to the reservations.
   - In `ensureModelLoadedForRequest` (`proxy.go` L126), when
     `GenerationEnabled` is on and any **enabled** instance (running or
     stopped, as in step 3) reserves VRAM on a GPU the model uses, run this
     check on that GPU:

     (planned VRAM of the requested model) +
     (planned VRAM of the LLMs already loaded) +
     (all reservations) ≤ card budget

     - When `ModelsMax` is 1, the loaded model is about to be replaced, so it
       is left out of the sum.
     - When `ModelsMax` is above 1, every loaded model counts. If the sum does
       not fit, the user unloads a model first; toolchest does not guess which
       model the router would evict.
   - If it does not fit, refuse with HTTP 507 and the message "Not enough VRAM:
     <n> GiB reserved for generation backends on GPU <i>".
6. **Enforce the reservation for sd-server.** `LaunchSpec` already passes
   `--max-vram` from `ReservedMiB` (Phase 05). That makes sd.cpp's own memory
   plan stay inside the reservation. ace-server has no such option. Its
   reservation is a planning figure based on the measured peak, and the UI
   says so in the reservation tooltip.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/monitor/... ./internal/models/... ./internal/generation/... ./internal/api/...`

## Test plan

- **`procvram_test.go`:** parse captured `nvidia-smi` and `rocm-smi
  --showpids` output, and map the bus ID to the monitor index.
- **`fit_test.go`:** add a case for a 32,624 MiB card with these allocations:

  | Allocation | Size |
  |---|---|
  | Other programs | 1,300 MiB |
  | Qwen-Image reservation | 11,776 MiB (measured peak 10,283 MiB × 1.10, rounded up to 512 MiB) |
  | ACE-Step reservation | 8,192 MiB (measured peak 7,239 MiB × 1.10, rounded up) |

  - The case checks the budget left after the max(1 GiB, 8% of the whole
    card) margin:
    32,624 − 1,300 − 2,610 − 11,776 − 8,192 = 8,746 MiB.
  - A 5,800 MiB LLM must fit. This is Qwen3-4B Q8_0 with 8K context.
  - A 12,150 MiB LLM must not fit. This is Gemma 4 12B Q6 with 8K context,
    measured on 2026-10-24.
- **507 test:** with `ModelsMax=2`, one loaded model plus a requested model
  that each fit alone but not together is refused.
- **`procvram` test:** a PID missing from the tool output returns `ok=false`.
- **`other_vram` test:** a running instance's PID usage is not counted as
  "other".
- **Start check test:** two instances whose reservations together exceed the
  card are refused, with a message that names both.
- **Manual on asmodean (R9700, 32 GB):**
  - Measure ACE-Step XL SFT with nothing else on the GPU. Expect a peak near
    7,239 MiB and a reservation of 8,192 MiB.
  - Stop and disable it, then measure Ming-Image the same way. Expect a peak
    above 16,500 MiB: the weights alone are 17.3 GB. Record the figure.
  - Disable Ming-Image. Create and measure an instance of the `qwen-turbo-dev`
    custom set from Phase 04.
    Expect a peak near 10,283 MiB and a reservation of 11,776 MiB. Then
    enable ACE-Step again.
  - Confirm that loading Gemma 4 12B Q6 (about 12,150 MiB) is refused.
    - On 2026-10-24 this combination ran with only 1.8 GB of the card spare.
    - The 10% headroom on each measured peak refuses it on its own. Even with
      no safety margin, only 11,356 MiB would be left.
    - The planner's margin, max(1 GiB, 8% of the whole card's VRAM), then
      takes a further 2,610 MiB.
  - Confirm that Qwen3-4B Q8_0 (`unsloth/Qwen3-4B-GGUF`, 4.28 GB file) loads.
  - Confirm that a larger LLM (Qwen3.6 27B Q4, 17.6 GB) is refused with the
    507 message.
  - Run the three-server load from `~/scratch/coexist.py` with its URLs
    changed to the instances' own ports (the image and music routes on port
    3000 arrive in Phases 07 and 08). Confirm there are no out-of-memory
    errors.
- **Manual on compute2 (CUDA):** measure a FLUX.2 klein instance on GPU 1
  while an LLM runs on GPU 0, and confirm the GPU map shows the reservation
  only on GPU 1.

## Commit

`feat(generation): reserve VRAM per backend instance and plan LLM fit around it`

## Rollback

Revert the commit. The `ReservedMiB` and measured fields stay in
`generation.json`. The Phase 05 binary ignores them for fit purposes but still
passes `--max-vram`, which is safe. The fit planner goes back to counting
backend VRAM as "other programs", which is conservative.
