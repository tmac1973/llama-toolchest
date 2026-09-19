# Phase 06 — Hardware fit planner

**Depends on:** 02 (`ProfileNote`), 05 · **Enables:** 09 (autoconfigure's code-decided settings),
11 (autotune knows how many GPUs are assigned and whether layers are on the
CPU)

## Goal
A pure function that, given a model, the hardware, and a context size class,
returns the settings that make the model fit, with a plain-language reason for
each one. It builds on the existing VRAM estimator. It touches no HTTP or LLM
code, so it can be tested thoroughly with table-driven tests.

## Files touched
- `internal/models/fit.go` (new): `PlanFit`, the types, and the size classes.
- `internal/models/fit_test.go` (new).
- `internal/models/registry.go`: change the default `Threads: 8` for a new
  model (line 447) to `ThreadsFor(runtime.NumCPU())`, where `ThreadsFor` is
  defined in `fit.go`.

## Steps
1. **Types.**
   ```go
   type ContextClass string // "short" | "medium" | "long" | "max"
   var ContextClassTokens = map[ContextClass]int{"short": 8192, "medium": 32768, "long": 131072} // "max" = model.ContextLength
   type Hardware struct {
       GPUs []GPUSpec // Index, Name, VRAMTotalMiB, IsIGPU
       LogicalCores int
       RAMTotalMiB  int // from monitor.Metrics.Memory.TotalMB
   }
   type FitResult struct {
       Config ModelConfig      // only the fields below are set; the caller merges them
       Notes  []ProfileNote    // Origin "hardware fit"
       EstimateGiB, BudgetGiB float64
       Fits bool               // false only when nothing below makes it fit
   }
   func PlanFit(m *Model, base ModelConfig, hw Hardware, class ContextClass) FitResult
   ```
2. **Budget.**
   - Use the dedicated (non-iGPU) GPUs, or the iGPU when it is the only GPU.
   - The budget is the sum of `VRAMTotalMiB` minus a safety margin of
     max(1 GiB, 8%) per card.
   - With more than one card, set `GPUAssign` to `"all"` and let the existing
     `ResolveGPUAssign` produce the split.
3. **Context.** Target = `ContextClassTokens[class]`, capped at
   `m.ContextLength`. "max" uses `m.ContextLength`.
4. **Fit loop.** Evaluate `VRAMEstimateForConfigOn` at each step and stop at
   the first step that fits:
   1. Full GPU layers (999), f16 KV cache, the target context.
   2. KV cache `q8_0`. Note: "Stores the conversation memory at 8-bit to fit
      the requested context. The effect on quality is very small."
   3. MoE models only (`ExpertCount > 0`): raise `CPUMoE` from 1 to
      `NLayers` and take the smallest value that fits. Note: "N layers of
      expert weights kept in system memory so the model fits."
   4. Dense models: lower `GPULayers` from `NLayers` to 0 and take the
      largest value that fits. Note: "Only N of M layers fit on the GPU; the
      rest run on the CPU, which is slower."
   5. Halve the context (not below 4096) and repeat from step 2. Note: "The
      requested context did not fit; reduced to N tokens."

   **RAM check:** at every step, the bytes placed on the CPU (the
   `CPURAMGiB` from Phase 05 plus the weights of layers not on the GPU) must
   be at most `RAMTotalMiB` minus max(4 GiB, 10%). A step that exceeds this
   does not fit.

   If nothing fits, return `Fits=false` with the smallest estimate and the
   note "This model is too large for this machine even with offloading.
   Choose a smaller quantization."
5. **Batch sizes.** Leave `BatchSize`/`UBatchSize` at 0 (llama.cpp defaults);
   autotune tunes them. Note: "Left at llama.cpp's defaults; Autotune can
   measure better values."
6. **Flash attention and parallel slots.**
   - `FlashAttention = true` unless `ValidateFlashAttention` rejects it for
     this model.
   - `Parallel = 1`. Note: "One conversation at a time uses all of the
     context for that conversation."
7. **Threads.**
   - `ThreadsFor(cores)` returns `max(1, cores/2)`, which is physical
     cores on SMT machines.
   - Set it only when layers or experts are on the CPU. Otherwise leave the
     base value.
8. **Fit label.** Record `EstimateGiB` and `BudgetGiB` so the review screen
   can show "Estimated 14.2 GiB of 15.0 GiB available".
9. **Where the hardware comes from.** `Hardware` is built by the caller from
   `s.monitor.Current()` (`internal/monitor/monitor.go:95`). `fit.go` imports
   nothing from `monitor`.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
Table tests with synthetic `Model` values:
- An 8B dense model on a 24 GiB card with the medium class: full offload,
  f16, 32K.
- The same model with the max class on a 12 GiB card: the context is reduced
  and q8_0 is applied, with notes.
- A 30B-A3B MoE model on a 16 GiB card: `CPUMoE > 0`, all layers on the GPU,
  and threads set.
- A 70B dense model on 8 GiB: partial `GPULayers`.
- A model larger than RAM plus VRAM (with the margins): `Fits=false`.
- An MoE model that fits the GPU only if more experts move than RAM can
  hold: `Fits=false`.
- Two cards: the budget is summed and `GPUAssign="all"`.
- An iGPU with a dedicated card: the iGPU is ignored.

Each case also checks that every changed field has exactly one note.

## Commit
`feat(models): plan settings that fit a model to the machine`

## Rollback
Revert the commit. Nothing calls `PlanFit` until Phase 09. The default
threads change affects only models added afterwards.
