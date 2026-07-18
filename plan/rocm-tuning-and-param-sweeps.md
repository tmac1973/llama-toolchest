# ROCm Tuning and Parameter Sweeps

Make llama-toolchest able to find and apply optimal settings for AMD
GPUs under ROCm. Three strands, in dependency order:

1. **Fix benchmark config overrides**, which are currently inert and
   record fabricated config in results.
2. **Generalize benchmark jobs into an N-axis parameter sweep**, so a
   `-ub` ladder is a first-class experiment rather than seven hand-made
   jobs.
3. **Surface batch/ubatch in model config**, and **make ROCm build
   flags and runtime env vars architecture-aware**, prefilled from the
   detected gfx target.

The driving case: prompt processing on gfx1201 (Radeon AI PRO R9700)
sitting at ~799 t/s avg PP on Qwen3.6-27B under ROCm 7.2.4, with no way
to sweep `-ub` and no `-ub` setting to sweep.

---

## Goals

1. Config overrides declared on a benchmark job actually reach
   llama-server, and recorded config snapshots are true.
2. Any override axis can take a **list** of values; the job matrix
   expands over them. `{models} × {builds} × {presets} × {axes…}`.
3. `--batch-size` / `--ubatch-size` are first-class model config fields,
   editable in the UI and emitted to the preset INI.
4. ROCm build options carry machine-readable architecture metadata
   instead of English prose, so the UI can prefill and, more
   importantly, **warn when a flag is contraindicated for the detected
   hardware**. Two currently-wrong toggles get corrected.
5. A curated set of ROCm runtime env vars is settable and applied to the
   router process.
6. Existing benchmark data whose config snapshot is untrustworthy is
   flagged rather than silently presented.

---

## Non-goals

- Per-model env vars. The router hosts all models in one process
  (`internal/process/manager.go:103-125`), so env is process-global by
  construction. Env toggles are app settings, not model settings.
- Free-form env var maps. Curated named toggles only.
- Auto-tuning, i.e. running a sweep and applying the winner without
  asking. Sweeps report; the user decides.
- Concurrent cells or parallel jobs. Still strictly serial.
- Vulkan/CUDA arch-aware defaults. ROCm only this pass; the mechanism
  should generalize but we won't populate other backends' matrices.

---

## Phase 1 — Make config overrides real

### The bug

`applyOverrides` (`internal/benchmark/job_runner.go:310`) merges
`ConfigOverrides` onto the model's saved `ConfigSnapshot` and returns
`cfg` at `job_runner.go:260`. That value's **only** consumer is
`run.Config` (`job_runner.go:271`) — the snapshot persisted for display.

Nothing applies it:

- `RunConfig` (`job_runner.go:284-292`) carries no config.
- `JobEnv` (`job_runner.go:19-57`) exposes no config-writing method.
- `jobs_env.go` uses the registry read-only (`Get`, `GetConfig`,
  `RouterName` at `:124-137`).

So a job with `ngl=50` benchmarks the model's saved config and then
records `gpu_layers: 50`. The compare view shows configs differing
across runs that were in fact identical. Every override field is
affected, not just the six with no `ConfigSnapshot` home.

Secondary, and subsumed by the fix: `ConfigOverrides` declares 15 fields
(`job.go:68-84`); `ConfigSnapshot` has 9 (`benchmark.go:77-87`).
`DraftModelPath`, `Temperature`, `TopP`, `TopK`, `MinP`,
`RepeatPenalty` round-trip through JSON and vanish.

### The fix — ephemeral preset layer

Never mutate the user's saved config. `Manager.Start()` already accepts
`RouterConfig.PresetPath` (`manager.go:112-114`), so:

1. Add `GeneratePresetINIFor(configs map[string]ModelConfig)` to
   `internal/models/preset.go`, factored out of the existing
   `GeneratePresetINI()` so both share one emitter.
2. New `JobEnv` method: `ApplyEphemeralConfig(ctx, modelID, cfg) error`.
   Writes `<dataDir>/config/bench-preset.ini` and restarts the router
   pointed at it.
3. New `JobEnv` method: `ClearEphemeralConfig(ctx) error`. Restarts
   against the real `preset.ini`. Called via `defer` on every job exit
   path, and on queue startup to recover from a crash mid-job.
4. `runCell` calls `ApplyEphemeralConfig` with the merged `cfg` before
   `q.runner.Run`.

Because `config/models.json` and `config/preset.ini` are untouched, a
hard crash costs at most a stale `bench-preset.ini` that nothing reads.

### Reconciling the two structs

Widen `ConfigSnapshot` to cover every field `ConfigOverrides` can set,
plus the new batch fields from Phase 3. `applyOverrides` then handles
all of them. Add a test asserting the two structs stay in sync by
reflection over field names — the drift is what caused this.

### Flagging bad history

Schema v3 migration in `internal/benchmark/benchmark.go` (bump
`schemaVersion` at `:300`, extend `Store.load()` at `:648`): for every
run whose job has non-nil `Overrides`, set
`ConfigUnverified bool` on the run. Compare and detail views render a
badge — "config not applied; recorded values may not reflect this run".
Keeps the throughput numbers, which are real, while killing the
misleading config columns.

---

## Phase 2 — N-axis parameter sweeps

### Data model

Replace the scalar pointers in `ConfigOverrides` with value lists.
Sketch:

```go
type SweepAxis struct {
    Field  string   `json:"field"`   // "ubatch_size", "gpu_layers", …
    Values []string `json:"values"`  // parsed per field type
}

type JobDefinition struct {
    // … existing
    Overrides *ConfigOverrides `json:"overrides,omitempty"` // fixed, all cells
    Sweeps    []SweepAxis      `json:"sweeps,omitempty"`    // expanded
}
```

Keeping `Overrides` for fixed values and adding `Sweeps` for expanded
ones avoids forcing every existing job through a migration and keeps
"set ngl=99 for the whole job" ergonomic.

### Expansion

`ExpandCells` (`job_runner.go:348`) grows from three fixed axes to
three plus N. `JobCell` gains `SweepValues map[string]string`.

**Ordering matters for cost.** Builds stay outermost so
`EnsureBuildActive` fires once per build (existing rationale at
`job_runner.go:346`). Config sweeps must be *inside* builds but should
group by value, since each distinct config costs a router restart. Cell
count and estimated restart count both belong in the UI matrix preview
(`benchmarks.html:322`).

### Registry of sweepable fields

One table mapping field name → type, parser, validator, label, and
which `ConfigSnapshot` field it sets. Drives the form, the parser, and
`applyOverrides` from a single source, so the next field added doesn't
repeat this phase's drift bug.

### UI

Extend the overrides `<details>` in `job_form.html:42-108`. Each field
gets a mode: fixed value or sweep list (comma-separated). Live matrix
preview updates cell + restart estimate. `readOverrides`
(`benchmarks.html:338-353`) and the edit-mode repopulate at `:300-308`
follow.

Add a **"Find best ubatch"** preset button that fills a `-ub` sweep of
`32,64,128,256,512,1024,2048` against the selected model, since that is
the concrete motivating workflow.

---

## Phase 3 — batch / ubatch in model config

`-b` / `--batch-size` and `-ub` / `--ubatch-size` appear nowhere in the
codebase today; the only way to set them is typing into the free-text
`ExtraFlags` box.

Touch points:

1. `internal/models/registry.go:97-148` — add `BatchSize`,
   `UBatchSize int`. Defaults at `:277-288` — leave both 0, meaning
   "don't emit, let llama.cpp default" (2048/512), so existing models
   are unchanged.
2. `internal/models/preset.go:100` — emit `batch-size` / `ubatch-size`
   INI keys when non-zero.
3. `internal/models/registry.go:179` — mirror in `EffectiveFlagsFor`.
   **This duplication is the real hazard**: `writeConfigParams` and
   `EffectiveFlagsFor` are two hand-maintained lists of the same flags.
   Factor them onto one flag-descriptor table as part of this phase.
4. `internal/api/service.go:695+` — parse the form fields.
5. `web/templates/partials/model_config.html` — inputs near `parallel`
   (`:136`). Help text: `-ub` ≤ `-b`; larger `-ub` generally improves
   prefill at the cost of compute-buffer memory; the optimum is
   machine-specific, so measure with a sweep.

   **Do not** claim small `-ub` helps hybrid/recurrent models. That
   claim traces to a single unversioned Windows anecdote (Qwen3.5-27B on
   an RX 9070 XT: pp512 of 59.5 t/s at ub4, 582.4 at ub64, 14.7 at
   ub128). A 40x collapse between ub64 and ub128 with flat TG is the
   signature of a **VRAM allocation cliff on a 16GB card**, not a
   property of DeltaNet layers. No llama.cpp issue or PR establishes the
   architectural claim; the genuine hybrid-model issues (#22384, #25004,
   #22746) are cache/checkpoint **correctness** bugs causing full prompt
   reprocessing — which produces the repeated-slow-prefill symptom people
   attribute to ubatch, and is the likely origin of the folklore.
   Hardcoding a small `-ub` for hybrid architectures would badly hurt
   prefill on a machine with adequate VRAM. This is precisely why the
   deliverable is a sweep, not a heuristic.
6. `ConfigSnapshot` + sweep field registry — so they're sweepable.

---

## Phase 4 — Architecture-aware ROCm defaults

### What exists

- `internal/builder/detect.go:44-73` parses `rocminfo` into
  `Backend.GPUs []string` of gfx IDs.
- `profiles.go:135-141` already feeds those into `AMDGPU_TARGETS`,
  falling back to a hardcoded `gfx1100`.
- `setup.sh:112-160` has generation buckets (RDNA4 `gfx1200|1201`,
  RDNA3 `gfx110x`, RDNA2 `gfx103x`, …) for `HSA_OVERRIDE_GFX_VERSION`.

### What's missing

`ProfileOptions(profile)` (`profiles.go:29`) switches only on backend
name and takes no GPU argument, so every ROCm option defaults `false`
regardless of hardware. The architecture knowledge exists **only as
English prose inside `Description` strings** (`profiles.go:90,103`) —
unreachable by code and unverifiable.

### Change

1. Add an arch-family classifier in `internal/builder`, sourced from the
   `setup.sh` buckets so the two don't drift: gfx ID → `rdna2` /
   `rdna3` / `rdna4` / `cdna3` / …
2. `BuildOption` gains:
   ```go
   RecommendedFor []string // arch families
   Rationale      string   // why, shown in the UI
   ```
3. `ProfileOptions(profile string, gpus []string)` computes each
   option's effective default as
   `Default || archMatches(RecommendedFor, gpus)`. Two call sites:
   `builder.go:243`, `api/build.go:25,191`.
4. `api/build.go:63-66` renders recommended toggles pre-checked with a
   visible "recommended for RDNA4 (gfx1201)" badge and the rationale.
   Per the design decision: **prefilled and visible, never silently
   applied.** The effective-flags preview at `build.go:90-92` already
   shows the true command line and must keep agreeing.
5. Make `AMDGPU_TARGETS` editable in the Builds UI — `help.html:149`
   already claims it's overridable and it isn't.

### Runtime env vars

Curated named toggles, app-global (see Non-goals). Add to
`internal/config/config.go:13-25` and the Settings UI, applied in
`internal/process/manager.go:125` alongside the existing
`pinCUDADeviceOrder` / `appendLibraryPath` composition — same pattern,
"don't clobber a value the user already exported in the environment."

### Flag matrix

Verified against `ggml-org/llama.cpp` HEAD `571d0d5` (2026-07-18) by
reading source, not docs. **The research inverted this phase's premise:
upstream has absorbed the per-architecture tuning into runtime
heuristics, so the correct default for nearly every flag is "leave it
alone." The valuable work here is removing two wrong toggles, not adding
recommended-on ones.**

#### Finding 1 — `GGML_HIP_ROCWMMA_FATTN` — MEASURED, HYPOTHESIS REJECTED

> **This section's original conclusion was wrong and is retained below
> only to record the reasoning error.** Measured on 4× Radeon AI PRO
> R9700 (gfx1201), ROCm 7.2.4, llama.cpp `571d0d5`, Qwen3.6-27B-MTP
> UD_Q8_K_XL, preset `internal-long-ctx` (25125-token prompt, 512 gen),
> spec decoding `draft-mtp`, ctx 262144:
>
> | Build | PP t/s | TG t/s | TTFT ms |
> |---|---|---|---|
> | `GGML_HIP_ROCWMMA_FATTN=ON` | **2143.5** | **33.96** | 11722 |
> | flag absent | 1726.4 | 31.49 | 14554 |
>
> **The flag is +24.2% PP and +7.8% TG. Keep it ON for this hardware and
> model.** Job `job-1784394108240-1c5493a0`.
>
> **Why the prediction failed.** The two branches are not
> interchangeable — their eligibility differs. rocWMMA (`fattn.cu:504`)
> needs only `K->ne[1] % FATTN_KQ_STRIDE == 0` and head size ∉ {40, 72,
> 192, 512, 576}. Native AMD WMMA (`:525`) *additionally* needs
> `gqa_opt_applies` (`:380` — `gqa_ratio >= 2 && mask && max_bias == 0`)
> and `Q->ne[0] <= 128`. The MFMA branch (`:512`) is CDNA-only and never
> fires on RDNA4. So when the native path declines, turning the flag off
> drops to the **generic tile kernel**, not to native MMA. The original
> analysis established dispatch *order* and then asserted a conclusion
> that only holds when both branches are eligible.
>
> The upstream PR benchmarks (#18481, #22880) used llama-8B Q4_0 — a
> different attention shape. They are not evidence about a 27B hybrid
> with MTP.
>
> **Consequence for Phase 4:** do not encode a `RecommendedFor` or a
> contraindication for this flag. Eligibility depends on head size, GQA
> ratio, and masking — i.e. on the *model*, not just the architecture.
> A per-arch table is the wrong shape for it. This is a per-model,
> measured question, which is an argument for the sweep tooling rather
> than for baked-in defaults.
>
> Caveats on the measurement: n=1 per build, single prompt length,
> single preset, and spec decoding active. Repeat with repetitions and a
> couple of prompt lengths before treating 24% as the exact number. The
> direction is clear; the magnitude is not yet.

<details>
<summary>Original (rejected) reasoning, kept for the record</summary>

`ggml_cuda_should_use_wmma_fattn()` returns true for RDNA3
unconditionally when the flag is defined, and for RDNA4 when rocWMMA ≥
2.0 (`fattn-wmma-f16.cuh:26-45`). The WMMA branch is tested at
`fattn.cu:504`, **before** the native AMD MFMA branch (`:512`) and the
native AMD WMMA branch (`:525`). Compiling the flag ON therefore
*shadows* the newer native MMA kernels for head sizes 64/128.

Those kernels exist specifically to beat rocWMMA — PR #18481 (RDNA4,
merged 2026-01-13) measured 2.13x at batch 4 / 1.32x at 512; PR #22880
(RDNA3, merged 2026-05-14) measured up to 2.23x on gfx1151 and describes
the rocWMMA kernel as retired for RDNA3/4.

**Action:** invert the option. Default OFF for every architecture,
rewrite `profiles.go:88-92`'s description (which currently recommends it
for exactly the architectures it hurts), and surface a warning when
enabled on RDNA3/RDNA4.

**This directly implicates the current gfx1201 server**, which builds
with `-DGGML_HIP_ROCWMMA_FATTN=ON` under ROCm 7.2.4 (rocWMMA ≥ 2). It is
the first thing to A/B against the ~799 t/s PP baseline.

</details>

#### Finding 2 — `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F` no longer exists

Zero occurrences at HEAD. It was replaced by a **runtime** env var,
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` (`ggml-cuda.cu:1630`), accepting
`auto|f16|fp16|bf16|f32|fp32`.

So the toggle at `profiles.go:101-105` (ROCm) and `:66-71` (CUDA) is
passing a `-D` that current llama.cpp ignores — a no-op that the
effective-flags preview presents as active. **Action:** drop it from
`ProfileOptions` for both backends, and re-expose the capability as a
runtime env toggle (below). Keep in mind builds pinned to older refs may
still honour it; if that matters, gate on ref rather than deleting.

#### Finding 3 — never force MMQ or cuBLAS on AMD

`ggml_cuda_should_use_mmq()` (`mmq.cu`) already carries hand-tuned
per-architecture heuristics: RDNA4 returns true unconditionally ("MMQ is
consistently faster than dequantization + hipBLAS", PR #18537); RDNA3
branches on expert count and quant type; CDNA3 returns true with a
comment that rocBLAS/Tensile "performs very poorly" as of ROCm 7.0.
`GGML_CUDA_FORCE_MMQ` / `FORCE_CUBLAS` are compile-time `#define`s that
short-circuit this logic entirely. Keep them exposed as opt-in (FORCE_MMQ
has a legitimate VRAM-reduction use), never defaulted.

#### Finding 4 — `GGML_HIP_FORCE_ROCWMMA_FATTN_GFX12` is not real

No CMake option, no source reference, nothing in `git log -S`. Folklore.
Do not implement.

#### Current upstream defaults — already correct, don't touch

`GGML_HIP_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`, `GGML_HIP_MMQ_MFMA=ON`. The
HIP CMakeLists also already applies `-funsafe-math-optimizations`
(alongside `-ffast-math -fno-finite-math-only`, deliberately — plain
`-ffast-math` breaks ggml's INFINITY masking and yields NaNs). That makes
the "HIP Fast Math" toggle at `profiles.go:94-99` redundant on current
refs, as its own description already half-admits.

#### Runtime env toggles to surface

| Var | Default | Rationale |
|---|---|---|
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | unset (`auto`) | The real successor to COMPUTE_16F. The FP16-accumulator knob. |
| `ROCBLAS_USE_HIPBLASLT` | unset | **Not defaulted anywhere.** No-op on gfx12 (hipBLASLt is already rocBLAS's default backend there), and llama.cpp's own source notes it crashes on gfx942. Worth an A/B on gfx1151/gfx1100 only. |
| `GGML_CUDA_DISABLE_GRAPHS` | unset | Debug/compat lever. Expect a small TG *loss*, not a gain. |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | unset | Survivability, not speed. |

**Explicitly rejected:** `HSA_NO_SCRATCH_RECLAIM` (no llama.cpp or ROCm
evidence; PyTorch-thread folklore). `HSA_ENABLE_SDMA=0` (stability
workaround, not perf). `HSA_OVERRIDE_GFX_VERSION` on **gfx1201** — it is
officially supported in current ROCm, so overriding is actively harmful.
`setup.sh:140` currently maps `gfx1200|gfx1201` to an override; that
mapping should be re-checked against the target ROCm version.

#### Per-architecture recommendation table

| Arch | Deviations from upstream default |
|---|---|
| RDNA2 gfx103x | none |
| RDNA3 gfx110x | none — native MMA FA since #22880 |
| RDNA3.5 gfx1151 | none at build time; `ROCBLAS_USE_HIPBLASLT=1` worth measuring |
| RDNA4 gfx120x/1201 | `GGML_HIP_ROCWMMA_FATTN=ON` **measured +24% PP / +8% TG** on Qwen3.6-27B-MTP — but see Finding 1: this is model-dependent, not arch-dependent |
| CDNA gfx90a | none |
| CDNA3 gfx942 | none; **avoid** `ROCBLAS_USE_HIPBLASLT` |

**Phase 4 is now the lowest-value phase and should be scoped down.** The
one flag we measured turned out to be model-dependent rather than
architecture-dependent, which is evidence that the whole
`RecommendedFor []string` premise is the wrong abstraction — arch is not
the variable that determines the answer. What survives:

- Correct the two demonstrably wrong things (Finding 2's dead
  `COMPUTE_16F` toggle; the `setup.sh:140` gfx1201 `HSA_OVERRIDE`
  mapping).
- Fix `GGML_HIP_ROCWMMA_FATTN`'s description to describe *eligibility*
  (head size ≤ 128, GQA ≥ 2, masked, no ALiBi → native path available;
  otherwise rocWMMA covers cases the native kernel declines) instead of
  naming architectures.
- Drop the arch-recommendation engine until there's evidence a per-arch
  answer exists for any flag. There currently isn't one.

The effort saved here belongs in Phases 1–2, which is what actually
produced a real answer.

#### Version sensitivity

ROCm version outweighs every flag here. ROCm issue #2865 (opened
2026-01-26, still in triage) reports gfx1151 prompt processing at 545 t/s
on ROCm 7.2 vs 1648 t/s on 6.4.4 — a 3x regression with TG unaffected.
Consider a warning for known-bad ROCm/rocWMMA version combinations; the
build already records enough context to do it.

---

## Risks

- **Router restarts dominate sweep wall-clock.** A 7-value `-ub` ladder
  on a 27B model is 7 full model loads. Show an estimate before the user
  commits, and make cancellation land promptly between cells.
- **Changing build option defaults changes what users get.** Someone who
  rebuilds the same git ref after this ships may get different flags,
  hence a different flag-hash build ID (`builder.go:297-298`). Inverting
  `GGML_HIP_ROCWMMA_FATTN` in particular will silently change the
  attention kernel for anyone who had it on. Call it out in release
  notes, and keep prior builds intact so an A/B is possible.
- **The research is source-verified but not end-to-end benchmarked.**
  The rocWMMA shadowing is certain from dispatch order; the *magnitude*
  of the win from turning it off rests on maintainers' PR benchmarks,
  not an independent A/B on gfx1201. Measure before believing — which
  the Phase 2 sweep makes cheap.
- **`ConfigSnapshot` widening touches persisted data.** New fields must
  be additive with zero values meaning "unset"; the v3 migration should
  not attempt to backfill config it cannot know.
- **The two flag emitters.** If Phase 3 doesn't unify `preset.go` and
  `EffectiveFlagsFor`, the UI will eventually lie about the command
  line the way the benchmark config snapshot does now.

---

## Sequencing

Phase 1 is a prerequisite for Phase 2 (sweeps need working config
application) and is independently a correctness fix worth shipping
alone. Phase 3 is independent and small — it can land first if you want
`-ub` in the UI immediately. Phase 2 depends on 1; sweeping `-ub`
specifically depends on 3. Phase 4 is fully independent of 1–3.

Suggested order: **3 → 1 → 2 → 4**, which puts a manually-settable `-ub`
in your hands on day one, then makes sweeping it trustworthy.

---

## Phase 0 — CSV exports must carry CMake flags

`writeCSVExport` emits `build_id`, `build_profile`, `git_ref`
(`bench_export.go:80,159`) but not `cmake_flags`. JSON exports already
carry the full `BuildSnapshot` including flags, so this gap is CSV-only.

It bites exactly the flag-comparison workflow this document exists for:
two builds off one git ref differing only in flags appear in CSV as two
rows with the same `git_ref` and profile, distinguishable only by an
opaque ID. Add a `cmake_flags` column (sorted `K=V` joined, matching
`effectiveCMakeFlags` at `api/build.go:102-115`) to both the cells and
summary schemas.

Small, independent, and unblocks trustworthy comparison spreadsheets.

---

## ~~Before any of this~~ — DONE: the gfx1201 experiment

The research produced one actionable result that needs no code. Current
builds use `-DGGML_HIP_ROCWMMA_FATTN=ON`, which on RDNA4 with rocWMMA ≥
2.0 routes flash attention through the retired rocWMMA kernel instead of
the native MMA kernel added in PR #18481.

**Run 2026-07-18. Result: keep the flag ON (+24% PP, +8% TG).** See
Finding 1 for numbers and for why the prediction was wrong.

The methodological point is worth keeping: the A/B cost two builds and
95 seconds of benchmarking, and it overturned a conclusion drawn from
reading upstream source and merged PR benchmarks. Source analysis
established a real mechanism but could not tell us which branch a given
*model* would take. Every remaining recommendation in this document is
of the same epistemic type as the one that just failed — measure before
baking anything into defaults.
