# Phase 02 — Downstream Go call sites

**Depends on:** phase 01 · **Enables:** phases 03 and 04. Everything that reads
`SpecType` as a single value learns about the second slot, and
`ConfigOverrides` grows the fields the sweep will set.

## Goal

Phase 01 changed what `SpecType` means; this phase updates every consumer that
compared it to a literal or assumed one mode. Three of those comparisons are
now wrong rather than merely narrow: VRAM accounting only counts a draft head
for `spec_type == "draft"`, so an EAGLE-3 head would be invisible to the
placement estimate; the measured-memory load signature would treat two
different assist settings as the same load; and the run comparison label would
show only half of a combined configuration. This phase also relaxes
`FindDraftCandidates` for the three head-based methods and adds the assist
fields to `ConfigOverrides`, so phase 04 is purely encoding and UI.

## Files touched

- `internal/models/vram.go` — `AuxFilesVRAMGB` counts the draft head for all
  five draft methods.
- `internal/models/registry.go` — `FindDraftCandidates` takes the mode and
  relaxes its filter for head-based methods.
- `internal/api/vram_corpus.go` — the fixture emitter records the mode-agnostic
  draft path.
- `internal/api/memory_measured.go` — the load signature includes the assist
  slot.
- `internal/api/service.go` — pass the selected mode to `FindDraftCandidates`.
- `internal/api/jobs_env.go` — snapshot, restore and diff the assist fields.
- `internal/benchmark/compare_labels.go` — the speculative label shows both
  slots.

`ConfigOverrides`, `ConfigSnapshot` and the override merge already gained their
assist fields in phase 01, which could not split the shared parameter table
without them.
- `internal/evaluate/evaluate.go` — update the field list in the comment at
  line 176.
- `web/templates/partials/job_detail.html` — show the assist alongside the mode.
- `internal/models/vram_test.go` — draft-head accounting for the new methods.
- `internal/models/registry_test.go` — the relaxed `FindDraftCandidates` filter.
- `internal/benchmark/job_runner_test.go` — the override round trip for a stored
  job carrying the legacy shape.

## Steps

1. **`AuxFilesVRAMGB` (`internal/models/vram.go:253`).** Replace
   `if cfg.SpecType == "draft" && cfg.DraftModelPath != ""` with
   `if IsDraftMode(cfg.SpecType) && cfg.SpecType != "draft-mtp" && cfg.DraftModelPath != ""`
   — unqualified, since `vram.go` and `specmodes.go` are both package `models`.
   `draft-mtp` is excluded because its head is already counted from `MtpPath`
   two lines above — so every draft method that loads a head still contributes
   exactly once. The other four all load `DraftModelPath` through
   `--model-draft`. The n-gram assist loads no file and adds nothing here —
   state that in a comment so the omission reads as deliberate.

2. **`FindDraftCandidates` (`internal/models/registry.go:1265`).** Change the
   signature to `FindDraftCandidates(id, mode string)`. When
   `IsHeadBasedDraftMode(mode)` is true, skip the architecture-match and the
   40%-size checks — a converted EAGLE-3 / DFlash / DSpark head passes neither
   — and keep only the embedding-model and MTP-head exclusions. For every other
   mode the behaviour is exactly as today. Document why: the head is not a
   smaller model of the same family, it is a trained extra layer, so the filter
   that makes `draft` safe makes these three empty.

3. **`internal/api/service.go:724`.** Pass `cfg.SpecType` to the new signature.

4. **`internal/api/vram_corpus.go:184`.** The emitted fixture line currently
   tests `SpecType == "draft"`. Use `models.IsDraftMode(cfg.SpecType)` so a
   corpus entry captured under any draft method records its path. Regenerate no
   fixtures — existing corpus files stay valid, since the condition only widens.

5. **`internal/api/memory_measured.go:203`.** `memoryFingerprint` formats the
   memory-relevant config fields into a readable string; extend its format with
   `assist=%s/%d/%d/%d/%d/%d/%d` carrying `cfg.SpecAssist`, `cfg.AssistNMax`,
   `cfg.AssistNMin`, `cfg.AssistNMatch`, `cfg.AssistSizeN`, `cfg.AssistSizeM`,
   `cfg.AssistMinHits`, placed after the existing `spec=`/`draft=`/`mtp=`
   entries. Without it, a measurement taken with the assist on is reported for a
   launch with it off.

   No version or salt is needed: the fingerprint is computed at load time and
   compared against the current config in the same process, never persisted, so
   there are no stale stored values to invalidate. The comment above the
   function says so — leave it accurate.

6. **`internal/benchmark/compare_labels.go:100`.** Replace
   `func(r BenchmarkRun) string { return r.Config.SpecType }` with a helper that
   joins the two slots the way the flag does:

   ```go
   func specLabel(c RunConfig) string {
       switch {
       case c.SpecType != "" && c.SpecAssist != "":
           return c.SpecType + " + " + c.SpecAssist
       case c.SpecType != "":
           return c.SpecType
       case c.SpecAssist != "":
           return c.SpecAssist
       }
       return ""
   }
   ```

   The `" + "` separator (not `","`) keeps the comparison table readable and
   matches the `+` the sweep encoding uses in phase 04. Runs recorded before
   this change have `SpecAssist` empty and a draftless name in `SpecType`, so
   they keep rendering exactly as they did.

7. **`web/templates/partials/job_detail.html:31`.** Extend the override summary
    line: `{{if .SpecAssist}} · spec_assist={{deref .SpecAssist}}{{end}}` after
    the existing `spec_type` clause.

8. **`internal/api/jobs_env.go`.** Add `SpecAssist` and the six assist values to
    the snapshot struct (near line 398), to the restore path (near line 669),
    and to the flag diff (near line 805) as
    `add("spec-assist", base.SpecAssist, merged.SpecAssist)` plus
    `add("ngram-mod-n-max", …)`, `add("ngram-mod-n-min", …)`,
    `add("ngram-mod-n-match", …)`, `add("size-n", …)`, `add("size-m", …)`,
    `add("min-hits", …)`.

9. **`internal/evaluate/evaluate.go:176`.** The comment lists the config fields
    the evaluation path deliberately ignores. Add `SpecAssist` and the assist
    parameters to that list — speculative decoding is excluded from capability
    evaluations, and the comment is the only record of why.

## Build gate

```
make build
go test ./...
go vet ./...
```

## Test plan

- `internal/models/vram_test.go`: a config with `SpecType: "draft-eagle3"` and
  a `DraftModelPath` pointing at a temp file of known size reports that size in
  `AuxFilesVRAMGB`; the same config with `SpecAssist: "ngram-mod"` and no draft
  method reports zero.
- A `FindDraftCandidates` test: with a registered model whose architecture
  differs from the target and whose size is 90% of it, `mode: "draft"` returns
  nothing and `mode: "draft-eagle3"` returns it.
- A `job_runner` round-trip test: `ConfigOverrides{SpecType: ptr("ngram-mod"),
  DraftMax: ptr(64)}` — the shape a stored job holds — merges to a snapshot that
  launches as `--spec-type ngram-mod --spec-ngram-mod-n-max 64`, unchanged from
  before. The snapshot itself keeps the legacy shape: it records what was
  requested, and the launch path normalises.
- Manually: open a saved benchmark job that used `ngram-mod` and confirm the
  job detail page still renders its override summary; re-run one cell and
  confirm the launched command is unchanged.

## Commit

```
refactor(spec): teach every SpecType consumer about the n-gram assist slot

VRAM accounting counted a draft head only for spec_type=draft, so an EAGLE-3,
DFlash or DSpark head was invisible to the placement estimate. The measured-
memory load signature ignored the assist entirely, so a measurement taken with
it on would be reused for a launch with it off. The comparison label showed
half of a combined run.

Relaxes FindDraftCandidates for the three head-based methods, whose heads
match neither the architecture nor the size filter that makes the draft-model
picker safe.
```

## Rollback

Revert the commit. Safe to leave partially applied only if the
`ConfigOverrides` field additions land with `job_runner`'s use of them —
otherwise a job saved with assist overrides silently drops them at run time,
which produces a mislabeled result rather than a visible failure. The
memory-signature change invalidates previously measured entries by design; on
revert they are simply measured again.
