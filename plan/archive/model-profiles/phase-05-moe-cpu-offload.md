# Phase 05 — MoE expert offload to the CPU (`cpu_moe`)

**Depends on:** 02 (a profile captures the new field through
`profileFields`), 04 (the snapshot builder this phase extends) · **Enables:** 06 (the fit planner uses `cpu_moe` for MoE
models that do not fit), 10/11 (autotune tunes threads when layers are on the
CPU)

## Goal
Add llama-server's `--n-cpu-moe N` as a normal model setting. It keeps the
expert weights of the first N layers in system RAM, and it is the fastest way
to run an MoE model larger than VRAM. This phase also reads the expert counts
from the GGUF so the app knows which models are MoE, and includes the setting
in the VRAM estimate.

## Files touched
- `internal/models/gguf.go`:
  - `GGUFMeta` gains `ExpertCount` and `ExpertUsedCount`, read from
    `{arch}.expert_count` and `{arch}.expert_used_count`;
  - `ApplyTo` copies them to `Model`.
- `internal/models/registry.go`:
  - `Model` gains `ExpertCount`, `ExpertUsedCount` and `ExpertBytes int64`
    (the total size of the expert tensors, which match `*_exps*`), and
    `NextNLayers int` (`json:"nextn_layers,omitempty"`), copied from
    `GGUFMeta.NextNPredictLayers` for files that are not MTP heads. Non-zero
    means the model has built-in MTP draft layers; Phases 08, 09 and 11 read
    it;
  - `ModelConfig` gains `CPUMoE int` (`json:"cpu_moe,omitempty"`);
  - bump the GGUF backfill version (the numbered list near line 661) so
    `BackfillGGUFMeta` fills in existing models.
- `internal/models/gguf.go`: while reading tensor info, sum the byte sizes of
  tensors whose names contain `_exps` into `ExpertBytes`.
- `internal/models/preset.go` (`writeConfigParams`): emit `n-cpu-moe = N`
  when `CPUMoE > 0`. `EffectiveFlagsFor` shows `--n-cpu-moe N`.
- `internal/models/vram.go` (`VRAMBreakdownForConfigOn`): subtract
  `ExpertBytes * CPUMoE / NLayers` from the weights when `CPUMoE > 0`, and
  add a `CPURAMGiB` field to `VRAMBreakdown` that reports what moved.
- `internal/api/service.go` (`handleUpdateModelConfig`) and
  `web/templates/partials/model_config.html`:
  - a "CPU expert layers" number input, shown only when
    `Model.ExpertCount > 0`;
  - validation: 0 ≤ N ≤ NLayers.
- `internal/benchmark/benchmark.go` (`ConfigSnapshot`) and
  `internal/benchmark/snapshot.go` (`SnapshotFromConfig`, Phase 04): add
  `CPUMoE`.
- `internal/api/jobs_env.go` (`applySnapshotToConfig`, line 657): copy
  `CPUMoE` from the snapshot, so a swept or overridden value is actually
  launched.
- `internal/benchmark/sweep.go`: add a `cpu_moe` sweep field (integer, with
  restart), so a user can also sweep it by hand.
- Tests next to each file.

## Steps
1. **GGUF parsing.** Parse the two metadata keys, and copy
   `NextNPredictLayers` to `Model.NextNLayers` in `ApplyTo` when
   `!meta.IsMTPHead()`. The tensor-info loop
   already walks tensor names (used for `HasBlockTensors`); accumulate the
   `_exps` bytes there using the tensor type size that `vram.go` already uses.
2. **Backfill.** Add the fields to `Model`, and bump the metadata version so
   that `BackfillGGUFMeta` re-reads models registered earlier.
3. **Config field.** Add the field, the INI line and the effective flag.
4. **VRAM.**
   - Split the weights term, with `movedGiB = ExpertBytes/NLayers * min(CPUMoE, NLayers)`.
   - The GPU estimate drops by `movedGiB`, and `CPURAMGiB` shows it.
   - The panel's VRAM banner shows "N GiB of expert weights in system RAM"
     when it is non-zero.
5. **Form field.**
   - Tooltip: "Mixture-of-experts models only. Keeps the expert weights of
     this many layers in system memory instead of GPU memory. Use it when the
     model does not fit on the GPU: it is much faster than lowering GPU
     layers. 0 keeps everything on the GPU."
   - An always-visible "Effective: `--n-cpu-moe 12`" line under it, following
     the existing effective-flag pattern.
6. **Snapshot and sweep.** Add the snapshot field and the sweep field; the
   sweep field uses the existing integer pattern, e.g. `ubatch_size`.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- `ApplyTo` sets `NextNLayers` for a main model with NextN layers and
  leaves it 0 for an MTP head.
- GGUF test: a synthetic GGUF header with `qwen3moe.expert_count = 128` and
  two `_exps` tensors gives the right counts and bytes (use the existing
  synthetic-GGUF test helpers in `gguf_test.go`).
- The preset INI contains `n-cpu-moe = 10` for `CPUMoE: 10` and nothing for 0.
- VRAM: setting `CPUMoE = NLayers` lowers the estimate by `ExpertBytes`.
- Render: the field is hidden for a dense model and shown for an MoE model.
- Job runner test: a `cpu_moe` sweep cell writes `n-cpu-moe` into the
  benchmark preset INI.
- Manual: on an MoE GGUF, set 10, restart, and check that `ps` shows
  `--n-cpu-moe 10` in the router's child process arguments and that the GPU
  memory drops.

## Commit
`feat(models): keep MoE expert layers in system memory with cpu_moe`

(The same commit records `NextNLayers`; mention it in the commit body.)

## Rollback
Revert the commit. `cpu_moe` values saved in configs or profiles are ignored
by older code, because unknown JSON fields are dropped on the next save. That
removes the value but has no other effect.

## As implemented

- **Expert layer span.** Besides the expert byte total, the parser records
  which layers carry experts (`ExpertLayerFirst`, `ExpertLayers`).
  `--n-cpu-moe N` counts layers from 0, dense leading layers included, so
  the VRAM estimate moves only the expert bytes of layers that have them.
- **Split models.** Expert tensors are summed over every shard. The shard
  scan now always runs for a split file, not only when the embedding tables
  were missing.
- **Partial GPU offload is now estimated.** `CPUWeightBytes` also accounts
  for `gpu_layers` below the layer count, which the estimator ignored
  before; Phase 06 needs this for dense models that do not fit. A
  `gpu_layers` of 0 is still read as "not set", as before, because many
  callers build configs with only the fields they care about. The fit
  planner therefore never goes below 1 GPU layer.
- **`VRAMBreakdown.CPURAM`** reports what stays in system memory. The panel
  shows it as "About N GiB of the model's weights stay in system memory with
  these settings", with a tooltip.
- **Evaluations.** `cpu_moe` is passed to capability evaluations
  (`--n-cpu-moe` in `evaluate.MapConfigFlags`) and is classified
  `AffectsEval`. It decides where weights live, like `gpu_layers`, and a
  large MoE model that only fits with it would otherwise fail to load for an
  evaluation.
- **Save check.** The config save refuses a negative value or one above the
  model's layer count.
