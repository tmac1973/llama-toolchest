# Phase 03 — Model config form: two pickers

**Depends on:** phases 01, 02 · **Enables:** users can actually build a combined
configuration. Independent of phase 04, which does the same job for the
benchmark surface.

## Goal

Replace the single Mode dropdown with two always-visible pickers — "Draft
method" and "N-gram assist" — each with its own parameters underneath, an
always-visible effective `--spec-type` line, and help copy that explains the one
rule a user must understand to read their own results: when both are active the
draftless method takes precedence for a step. Because the five draft methods
live in one dropdown and the five n-gram methods in the other, selecting two
draft methods is impossible, which is what keeps the open upstream startup crash
([#27897](https://github.com/ggml-org/llama.cpp/issues/27897)) out of reach.

## Files touched

- `web/templates/partials/model_config.html` — the speculative decoding section
  (lines 275–400) is rewritten; the help dialog above it gains a paragraph.
- `internal/api/service.go` — form parsing for the two slots, and the template
  data gains the mode lists and the effective `--spec-type` string.
  `applyDraftDefaults` / `applyAssistDefaults` already exist from phase 01.
- `internal/models/registry.go` — `ValidateSpec` joins `ValidateBatchSizes`
  (line 247) and `ValidateFlashAttention` (line 270) as a `*ModelConfig` method.
- `internal/api/service_test.go` — the three handler tests below.
- `internal/models/registry_test.go` — `ValidateSpec` accepts a draft name in
  `SpecType` and an assist name in `SpecAssist`, and rejects each in the other
  slot.

## Steps

1. **Template data.** In the handler that renders the config form
   (`internal/api/service.go` around line 788), add to the struct:
   `DraftModes []models.SpecMode`, `AssistModes []models.SpecMode`,
   `DraftParams []models.SpecModeParam`, `AssistParams []models.SpecModeParam`,
   and `EffectiveSpecType string`. Fill them from `models.DraftModes()`,
   `models.AssistModes()`, `models.SpecDraftParams(cfg.SpecType)`,
   `models.SpecAssistParams(cfg.SpecAssist)`, and the joined mode names
   (`""` when neither slot is set). `DraftCandidates` is already there; it now
   comes from `FindDraftCandidates(id, cfg.SpecType)`.

2. **The two pickers.** Replace the single `<select name="spec_type">` with:

   ```
   Draft method   [ Disabled | Draft Model | MTP (self-speculation)
                    | EAGLE-3 | DFlash | DSpark ]
   N-gram assist  [ None | N-gram Mod | N-gram Simple | N-gram Cache
                    | N-gram Map-K | N-gram Map-K4V ]
   ```

   Both `<select>`s are rendered by ranging over `.DraftModes` / `.AssistModes`
   so the labels come from `specmodes.go`. Keep the existing
   `{{if not .DraftCandidates}} (no compatible models){{end}}` suffix on the
   Draft Model option. Both are always visible — a user cannot discover the
   combination if the second picker only appears after choosing the first.

   Per the project's UI conventions, the explanation lives in each picker's
   `title` tooltip rather than as static prose:
   - Draft method: "A model or trained head proposes tokens that the main model
     verifies in parallel. Only one may run at a time."
   - N-gram assist: "Matches repeated text in the prompt and the generation so
     far, and proposes the continuation. Costs no extra memory and runs
     alongside the draft method. When both propose tokens for the same step,
     the n-gram proposal is used."

3. **Draft slot body.** Shown when the draft picker is not Disabled:
   - The draft model picker (`draft_model_path`) for `draft` and for the three
     head-based methods. Its tooltip differs for the head-based ones: "The
     converted EAGLE-3 / DFlash / DSpark head for this model. The list is not
     filtered by architecture or size, because a head is a trained extra layer
     rather than a smaller model of the same family."
   - The MTP drafter head path and the Load MTP Head switch, unchanged, for
     `draft-mtp` with `.HasMTP`.
   - The parameters, rendered by ranging over `.DraftParams` so the fields and
     their placeholders come from `SpecDraftParams`. For the three new methods
     every default is empty, so each field shows llama.cpp's own default as its
     placeholder ("3", "0", and blank).
   - The draft-model resource block (ctx size, GPU layers, device, CPU MoE, KV
     cache quant), shown whenever a `--model-draft` will actually be emitted:
     `draft`, any head-based method with a path set, or `draft-mtp` with
     `.HasMTP`.
   - The existing MTP + parallel warning and the gemma-4 build note, unchanged,
     under `draft-mtp`.

4. **Assist slot body.** Shown when the assist picker is not None: range over
   `.AssistParams` (the selected assist mode's parameters — a flat
   `[]models.SpecModeParam`; phase 04's `SweepChoice.AssistParamsByMode` is a
   different, per-mode-keyed thing with a deliberately different name). For `ngram-mod` that is Draft tokens max / Draft tokens min
   / Match length; for `ngram-simple`, `ngram-map-k` and `ngram-map-k4v` it is
   Lookup size / Draft size / Minimum hits; `ngram-cache` renders nothing and
   shows the line "This mode takes no settings." The m-gram field no longer
   appears under `ngram-mod` at all — it was meaningless there.

   Field names are `assist_n_max`, `assist_n_min`, `assist_n_match`,
   `assist_size_n`, `assist_size_m`, `assist_min_hits`.

5. **Effective flags line.** Below both slots, always visible (not a tooltip, not
   revealed on hover):

   ```
   Effective --spec-type: draft-mtp,ngram-mod
   ```

   When both slots are empty the line reads "Speculative decoding is off — no
   `--spec-type` flag is passed." It must not print `none`: that is a real
   llama.cpp value which discards every other mode in a list, the launch never
   emits it, and showing it here would teach the wrong thing. This line is the
   one place a user can see the comma-joined list they built, and the thing to
   quote when a launch fails.

6. **Re-rendering on change needs no new markup.** The config form already
   carries `hx-trigger="submit, change delay:500ms"`
   (`web/templates/partials/model_config.html:11`), so a change to either
   `<select>` re-posts the form and the server re-renders the partial with the
   right parameter fields. Add no `hx-post` or `hx-trigger` attributes to the
   selects, and leave the 500 ms delay alone: it debounces, so changing both
   pickers inside the window posts once carrying both values, which is the
   behaviour we want rather than a problem to work around.

7. **Form parsing (`internal/api/service.go`, around line 922).** Track the
   previous value of each slot separately:

   ```go
   prevSpecType, prevSpecAssist := cfg.SpecType, cfg.SpecAssist
   cfg.SpecType   = r.FormValue("spec_type")
   cfg.SpecAssist = r.FormValue("spec_assist")
   ```

   Parse the three draft parameters as today, and the six assist parameters the
   same way (`Atoi`, `> 0`, else zero). Then:

   ```go
   if cfg.SpecType != prevSpecType     { applyDraftDefaults(cfg) }
   if cfg.SpecAssist != prevSpecAssist { applyAssistDefaults(cfg) }
   ```

   Both functions already exist from phase 01 step 6 and are unchanged here.
   Remove the `models.NormalizeSpec(cfg)` call phase 01 put in this parser, and
   remove the `ngram_size_n` / `ngram_size_m` parsing — the form now posts the
   two slots and the six `assist_*` names directly, and can no longer put a
   draftless name in `spec_type` or a value in a legacy field. `NormalizeSpec`
   stays in `specDecodingParams` and in the benchmark merge path, where stored
   history still arrives in the old shape.

8. **Validation.** Add `func (c *ModelConfig) ValidateSpec() error` to
   `internal/models/registry.go`, beside `ValidateBatchSizes` (line 247) and
   `ValidateFlashAttention` (line 270), and call `cfg.ValidateSpec()` alongside the existing
   `ValidateBatchSizes` / `ValidateFlashAttention` calls: reject a `spec_type`
   that is not empty and not a draft mode, and a `spec_assist` that is not empty
   and not an assist mode. The UI cannot produce either, but a hand-edited
   registry or a crafted POST can, and an unknown name is a hard llama-server
   startup failure with a message the user will not connect to this form.

9. **Help dialog.** Add a section to the existing dialog above the section:

   > **Running both at once.** A draft method and an n-gram assist can run
   > together. They cost each other nothing on new text, and on text the model
   > is repeating — a file it is rewriting, a structured response, a tool-call
   > loop — the n-gram assist is much faster than a draft model alone.
   >
   > When both propose tokens for the same step, the n-gram proposal is the one
   > that gets used. That is worth knowing when you read a benchmark result: if
   > a combined run is slower than the draft method alone on new text, it is
   > because low-quality n-gram matches are displacing good draft proposals. The
   > benchmark sweep is how you find out.

   Plain language, no slang, and it explains how to read the number rather than
   only what the setting does.

## Build gate

```
make build
go test ./internal/models/... ./internal/api/...
go vet ./...
```

## Test plan

- A handler test posting `spec_type=draft-mtp`, `draft_max=3`,
  `spec_assist=ngram-mod`, `assist_n_max=64`, `assist_n_min=48`,
  `assist_n_match=24` saves a config whose `EffectiveFlags()` contains
  `--spec-type draft-mtp,ngram-mod`.
- A handler test that changing only `spec_assist` on an existing config leaves
  `DraftMax` untouched — the regression `applySpecDefaults` would have caused.
- A handler test posting `spec_type=ngram-mod` returns 400 from
  `ValidateSpec`.
- Manually, in the browser: select MTP, confirm the three draft fields appear
  with 6/0/blank; select EAGLE-3, confirm they appear empty with placeholders
  3/0 and the draft model list is no longer filtered; select N-gram Mod as the
  assist and confirm Match length appears and the m-gram field does not;
  confirm the effective line reads `draft-mtp,ngram-mod`; save, and check the
  generated preset INI has one `spec-type` line.
- Confirm a model migrated in phase 01 opens with Draft method on Disabled and
  N-gram assist on N-gram Mod, its parameters intact.

## Commit

```
feat(ui): pick a draft method and an n-gram assist independently

The speculative decoding section now has two pickers instead of one. The five
draft methods live in the first and the five n-gram methods in the second, so
two draft methods cannot both be selected -- the combination that crashes
llama-server at startup upstream.

Changing one slot no longer resets the other's tuned values, and the effective
--spec-type value is shown so a failing launch can be read off the form. Help
copy explains that the n-gram proposal wins a contested step, which is what a
combined benchmark result has to be read against.
```

## Rollback

Revert the commit. The template and handler change together; reverting either
alone leaves the form posting fields the handler ignores (values silently lost
on save) or reading fields the form never posts (values silently zeroed). Config
data written by this phase is the same shape phase 01 established, so a revert
to phase 02 keeps every saved config launchable — only the form loses the second
picker.
