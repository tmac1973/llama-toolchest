# Phase 01 — Data model, flag emission, and the three new draft methods

**Depends on:** nothing · **Enables:** phases 02, 03, 04 and 06 (phase 05 is
independent of it). This is the foundation: `SpecType` becomes strictly the draft method, `SpecAssist` strictly
the draftless one, and `specDecodingParams` emits one comma-joined `spec-type`
value plus each slot's own flags.

## Goal

Split speculative decoding into two independent slots in `ModelConfig` and make
the launch path emit both. After this phase a config with
`spec_type: draft-mtp`, `draft_max: 3`, `spec_assist: ngram-mod`,
`assist_n_max: 64`, `assist_n_min: 48`, `assist_n_match: 24` produces
`--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3
--spec-ngram-mod-n-max 64 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24`
on the command line and a single `spec-type = draft-mtp,ngram-mod` line in the
preset INI. The three merged-but-unreachable draft methods
(`draft-eagle3`, `draft-dflash`, `draft-dspark`) are added here, because under
the new model they are just three more rows in the draft table. The dead
`--spec-ngram-mod-n-match` and the never-offered `--spec-<mode>-min-hits` are
fixed as part of writing the assist emitter, and the m-gram parameter is dropped
from `ngram-mod`'s row in the parameter table, so nothing downstream offers it
for that mode. The form still renders its own hard-coded m-gram input until
phase 03 rewrites the section; until then the posted value is migrated by
`NormalizeSpec` into `AssistSizeM` and — for `ngram-mod` — discarded, which is
exactly what it was worth before.

No new UI in this phase. The form keeps its single Mode picker and still posts
`spec_type` with a draftless name for a draftless-only model; the tolerant
reader added here turns that into the new shape as the value is parsed, so
nothing breaks between phases.

## Files touched

- `internal/models/specmodes.go` — **new**. Mode classification and the two
  ordered name lists both forms and the sweep will render from.
- `internal/models/registry.go` — add `SpecAssist` and the six `Assist*`
  fields to `ModelConfig`; add `BackfillSpecAssist`; update the `SpecType`
  field comment.
- `internal/models/specparams.go` — replace `SpecModeParams` with
  `SpecDraftParams` and `SpecAssistParams`.
- `internal/models/specflags.go` — rewrite `specDecodingParams` as a draft
  block plus an assist block feeding one joined `spec-type` param.
- `internal/api/server.go` — call `BackfillSpecAssist` alongside
  `BackfillGGUFMeta` at startup.
- `internal/models/spec_smoke_test.go` — extend with combination, new-mode and
  migration cases.
- `internal/models/preset_test.go` — pin the single `spec-type` INI line.
- `internal/api/service.go` — `applySpecDefaults` splits into
  `applyDraftDefaults` and `applyAssistDefaults`; the interim single-picker form
  parser routes a draftless `spec_type` through `NormalizeSpec`. The two-picker
  rework is phase 03.

## Steps

1. **Create `internal/models/specmodes.go`.**

   ```go
   // DraftModes are the speculative modes that load a drafter through
   // --model-draft or a head baked into the main GGUF. At most one may be
   // active: llama.cpp crashes at startup on two (upstream issue #27897).
   func DraftModes() []SpecMode   // draft, draft-mtp, draft-eagle3,
                                  // draft-dflash, draft-dspark
   // AssistModes are the draftless n-gram methods. llama.cpp sanctions
   // mixing exactly one of these with one draft mode.
   func AssistModes() []SpecMode  // ngram-mod, ngram-simple, ngram-cache,
                                  // ngram-map-k, ngram-map-k4v
   func IsDraftMode(name string) bool
   func IsAssistMode(name string) bool
   // HeadBasedDraftModes are the draft modes whose --model-draft target is a
   // converted head rather than a smaller model of the same architecture.
   func IsHeadBasedDraftMode(name string) bool // eagle3, dflash, dspark
   ```

   `SpecMode` carries `Name` and `Label` (`"draft"` → `"Draft Model"`,
   `"draft-mtp"` → `"MTP (self-speculation)"`, `"draft-eagle3"` → `"EAGLE-3"`,
   `"draft-dflash"` → `"DFlash"`, `"draft-dspark"` → `"DSpark"`,
   `"ngram-mod"` → `"N-gram Mod"`, `"ngram-simple"` → `"N-gram Simple"`,
   `"ngram-cache"` → `"N-gram Cache"`, `"ngram-map-k"` → `"N-gram Map-K"`,
   `"ngram-map-k4v"` → `"N-gram Map-K4V"`). These labels are the single source
   for every mode *name* the two forms and the sweep render, so those cannot
   drift. The empty slot deliberately has no `SpecMode` entry, because it reads
   differently in each place: "Disabled" in the draft picker, "None" in the
   assist picker, "Off (no speculative decoding)" in the sweep choices list.
   Each surface writes its own.

   Add `NormalizeSpec(c *ModelConfig)`: if `c.SpecType` holds an assist mode
   name, move it to `c.SpecAssist`, move `DraftMax`→`AssistNMax`,
   `DraftMin`→`AssistNMin`, `NgramSizeN`→`AssistNMatch` for `ngram-mod` or
   `AssistSizeN` for the other four, `NgramSizeM`→`AssistSizeM` (dropped for
   `ngram-mod`, which has no m-gram size), clear the four legacy fields and
   clear `SpecType`. It is idempotent and does nothing when `SpecType` is
   already a draft mode or empty. This is the tolerant reader; it runs on
   configs reconstructed from stored benchmark history as well as on live ones.

2. **Extend `ModelConfig`** in `internal/models/registry.go`, immediately after
   the existing speculative block:

   ```go
   // Draftless n-gram assist. Runs alongside the draft method above:
   // llama.cpp accepts a comma-separated --spec-type list mixing one draft
   // method with one draftless one, and each has its own draft-length flags.
   SpecAssist     string `json:"spec_assist,omitempty"`      // "", "ngram-mod", "ngram-simple", …
   AssistNMax     int    `json:"assist_n_max,omitempty"`     // --spec-ngram-mod-n-max
   AssistNMin     int    `json:"assist_n_min,omitempty"`     // --spec-ngram-mod-n-min
   AssistNMatch   int    `json:"assist_n_match,omitempty"`   // --spec-ngram-mod-n-match
   AssistSizeN    int    `json:"assist_size_n,omitempty"`    // --spec-<mode>-size-n
   AssistSizeM    int    `json:"assist_size_m,omitempty"`    // --spec-<mode>-size-m
   AssistMinHits  int    `json:"assist_min_hits,omitempty"`  // --spec-<mode>-min-hits
   ```

   Leave `DraftMax`, `DraftMin`, `DraftPMin` where they are — they now belong
   unambiguously to the draft method. Leave `NgramSizeN` and `NgramSizeM` in
   the struct as legacy fields, marked
   `// legacy; the form posts these until phase 03, and NormalizeSpec migrates them into Assist*`.
   They must keep being written by the interim form parser (step 6) — that is
   how a draftless save made between this phase and phase 03 reaches
   `NormalizeSpec` at all. Nothing else reads them, and phase 03 removes the
   inputs that post them. Update the `SpecType` comment to
   `// "", "draft", "draft-mtp", "draft-eagle3", "draft-dflash", "draft-dspark" — draft methods only; the draftless mode lives in SpecAssist`.

3. **Rewrite `internal/models/specparams.go`** as two functions with disjoint
   key namespaces:

   ```go
   func SpecDraftParams(mode string) []SpecModeParam
     "draft"         -> draft_max=16, draft_min=0, draft_p_min=0.75
     "draft-mtp"     -> draft_max=6,  draft_min=0, draft_p_min=""
     "draft-eagle3"  -> draft_max="", draft_min="", draft_p_min=""
     "draft-dflash"  -> draft_max="", draft_min="", draft_p_min=""
     "draft-dspark"  -> draft_max="", draft_min="", draft_p_min=""

   func SpecAssistParams(mode string) []SpecModeParam
     "ngram-mod"     -> assist_n_max=64, assist_n_min=48, assist_n_match=24
     "ngram-simple"  -> assist_size_n=12, assist_size_m=48, assist_min_hits=1
     "ngram-map-k"   -> assist_size_n=12, assist_size_m=48, assist_min_hits=1
     "ngram-map-k4v" -> assist_size_n=12, assist_size_m=48, assist_min_hits=1
     "ngram-cache"   -> (none; the mode takes no tunables)
   ```

   The three new draft modes deliberately carry empty defaults: llama.cpp's own
   `--spec-draft-n-max 3` / `-n-min 0` apply, and there is no verified guidance
   to present as a recommendation. The labels are "Draft tokens max",
   "Draft tokens min", "Draft probability min" for the draft block and
   "Draft tokens max", "Draft tokens min", "Match length", "Lookup size",
   "Draft size", "Minimum hits" for the assist block.

4. **Rewrite `specDecodingParams`** in `internal/models/specflags.go`:

   ```go
   func specDecodingParams(c *ModelConfig) []specParam {
       cfg := *c
       NormalizeSpec(&cfg)          // tolerate a legacy draftless SpecType
       var modes []string
       var params []specParam
       if IsDraftMode(cfg.SpecType) {
           modes = append(modes, draftTypeName(cfg.SpecType))
           params = appendDraftParams(params, &cfg)
       }
       if IsAssistMode(cfg.SpecAssist) {
           modes = append(modes, cfg.SpecAssist)
           params = appendAssistParams(params, &cfg)
       }
       if len(modes) == 0 {
           return nil          // never emit "none" — an empty slot contributes nothing
       }
       return append([]specParam{{"spec-type", strings.Join(modes, ",")}}, params...)
   }
   ```

   The draft method comes first in the joined value. Order is behaviourally
   irrelevant (`common_speculative_init` builds a bitmask and walks a fixed
   priority order) but fixing it keeps INI diffs stable.

   `draftTypeName` maps the internal `"draft"` to llama.cpp's `"draft-simple"`
   and passes the other four through — the existing legacy-compat rename, moved
   out of the switch.

   `appendDraftParams` is today's `draft` / `draft-mtp` bodies merged: emit
   `model-draft` from `MtpPath` when the mode is `draft-mtp` and `MtpPath` is
   set and not disabled, from `DraftModelPath` for the other four modes when
   set; then `appendDraftResourceParams` whenever a `model-draft` was emitted;
   then `appendDraftSamplingParams` always. Self-speculation MTP (empty
   `MtpPath`) keeps emitting sampling flags only, exactly as today.

   `appendAssistParams` is new:

   ```go
   case "ngram-mod":
       assist_n_max   -> spec-ngram-mod-n-max
       assist_n_min   -> spec-ngram-mod-n-min
       assist_n_match -> spec-ngram-mod-n-match   // fixes the dead field
   case "ngram-simple", "ngram-map-k", "ngram-map-k4v":
       prefix := "spec-" + mode
       assist_size_n   -> prefix + "-size-n"
       assist_size_m   -> prefix + "-size-m"
       assist_min_hits -> prefix + "-min-hits"    // fixes the second dead field
   case "ngram-cache":
       // no tunables
   ```

5. **Add `BackfillSpecAssist`** to `internal/models/registry.go`, modelled on
   `BackfillGGUFMeta`: take the write lock, run `NormalizeSpec` over every
   entry in `r.data.Configs`, count the ones it changed, and `r.save()` if the
   count is non-zero. Return the count so the caller can log it. Call it from
   `internal/api/server.go` next to the existing `BackfillGGUFMeta()` call at
   line 217, logging `"migrated draftless speculative configs", "configs", n`
   when non-zero.

6. **Update `internal/api/service.go` for the interim single-picker form.**
   Replace `applySpecDefaults` with the two functions phase 03 will wire to the
   two pickers, since they are needed here anyway:

   - `applyDraftDefaults(cfg)` zeroes `DraftMax`, `DraftMin`, `DraftPMin` and
     refills them from `SpecDraftParams(cfg.SpecType)`.
   - `applyAssistDefaults(cfg)` zeroes the six assist fields and refills them
     from `SpecAssistParams(cfg.SpecAssist)`.

   In the form parser, keep parsing `draft_max`, `draft_min`, `draft_p_min`,
   `ngram_size_n` and `ngram_size_m` from the request exactly as today — the
   form still posts those names. Then, in this order:

   ```go
   prevSpecType, prevSpecAssist := cfg.SpecType, cfg.SpecAssist
   cfg.SpecType = r.FormValue("spec_type")
   // … parse the numeric fields as today, into the legacy field names …
   models.NormalizeSpec(cfg)   // routes a draftless choice into SpecAssist
                               // and its values into the Assist* fields
   if cfg.SpecType != prevSpecType     { applyDraftDefaults(cfg) }
   if cfg.SpecAssist != prevSpecAssist { applyAssistDefaults(cfg) }
   ```

   `NormalizeSpec` runs *before* the two comparisons, so both see the normalised
   slots and a draftless mode change applies assist defaults rather than none —
   which is what keeps the app behaving as it does today at the end of this
   phase. Splitting the defaults in two is what stops a change in one slot
   wiping the other's tuned values, the bug a single `applySpecDefaults` would
   cause the moment a second slot exists. Phase 03 replaces the parsing half of
   this with the two pickers' own field names; the two default functions and the
   ordering survive unchanged.

7. **Tests.** In `internal/models/spec_smoke_test.go`:
   - `TestSpecCombinedFlags` — `draft-mtp` + `ngram-mod` produces exactly one
     `--spec-type draft-mtp,ngram-mod`, plus `--spec-draft-n-max 3`,
     `--spec-ngram-mod-n-max 64`, `--spec-ngram-mod-n-min 48`,
     `--spec-ngram-mod-n-match 24`, and contains no `none`.
   - `TestSpecNgramModMatchEmitted` — `assist_n_match` reaches the command line
     (the bug this fixes).
   - `TestSpecMinHitsEmitted` — `ngram-simple` emits
     `--spec-ngram-simple-min-hits`.
   - `TestSpecNewDraftModes` — each of `draft-eagle3`, `draft-dflash`,
     `draft-dspark` with a `DraftModelPath` emits its own `--spec-type` name
     and `--model-draft`.
   - `TestSpecLegacyDraftlessConfig` — a config built with
     `SpecType: "ngram-mod"`, `DraftMax: 64`, `DraftMin: 48`, `NgramSizeN: 24`
     emits byte-identical flags before and after normalisation, except that
     `--spec-ngram-mod-n-match 24` is now present.
   - `TestSpecUnchangedForExistingModes` — table over the existing smoke-test
     fixtures asserting each emits the same flag set it did before, with one
     allowed addition: an `ngram-mod` fixture also emits
     `--spec-ngram-mod-n-match`, the bug fix. No fixture gains a `min-hits`
     flag: nothing migrates into `AssistMinHits`, because no legacy field feeds
     it. The flag reaches the command line only once a config is re-saved
     through the form, at which point `applyAssistDefaults` (step 6) fills in
     the 1 — which is llama.cpp's own default, so the launch does not change
     either way.
   In `internal/models/preset_test.go`, `TestPresetSingleSpecTypeLine` — the
   generated INI for a combined config contains exactly one line starting
   `spec-type` and its value is `draft-mtp,ngram-mod`.

## Build gate

```
make build
go test ./internal/models/... ./internal/api/...
go vet ./...
```

There is no `make test` target in this project: `go test ./...` is the suite,
and `make js-test` runs the page JavaScript.

## Test plan

- The unit tests above.
- Manually: load an existing instance whose registry has a
  `spec_type: ngram-mod` model, start the server, confirm the startup log
  reports the migration count once and not on the second start, and confirm
  `~/.config/llama-toolchest/` (or the configured registry path) now holds
  `spec_assist: ngram-mod` with `assist_n_max: 64`, `assist_n_min: 48`,
  `assist_n_match: 24` and no `ngram_size_m`.
- Diff the generated preset INI before and after for a model using each
  existing mode; only the `ngram-mod` model should change, gaining one
  `n-match` flag.

## Commit

```
feat(spec): split speculative decoding into a draft method and an n-gram assist

llama.cpp accepts a comma-separated --spec-type list mixing one draft method
with one draftless method, and each has its own draft-length flags. SpecType
now holds only the draft method; the new SpecAssist holds the draftless one
with its own parameters, so MTP at 3 tokens alongside ngram-mod at 64 is
expressible for the first time.

Adds the three merged draft methods that were unreachable (draft-eagle3,
draft-dflash, draft-dspark), emits --spec-ngram-mod-n-match and
--spec-<mode>-min-hits, which were offered or defaulted but never passed to
llama-server, and drops the m-gram field from ngram-mod, which has no m-gram
size.

A one-shot backfill migrates saved draftless configs; NormalizeSpec keeps
reading the old shape wherever it survives.
```

## Rollback

Revert the commit. The `Assist*` JSON fields are `omitempty`, so a registry
already migrated by `BackfillSpecAssist` loses its n-gram parameters on the
reverted build — `spec_assist` and `assist_*` become unknown keys and the mode
reads as disabled. Back the registry file up before first running the migrated
build if that matters. Nothing else in this phase is stateful; leaving it
partially applied is safe as long as `specmodes.go` and the struct fields land
together.
