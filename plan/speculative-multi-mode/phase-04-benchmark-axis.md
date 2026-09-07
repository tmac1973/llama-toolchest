# Phase 04 — Benchmark sweep axis and job form

**Depends on:** phases 01, 02 · **Enables:** phase 06. This is what makes the
combination measurable rather than argued about, and it is the fiddliest part of
the project.

## Goal

Extend the `spec_type` sweep axis so a cell can name a draft method, an n-gram
assist, or both, with each slot's parameters. The encoded form joins the two
mode names with `+` and keeps `,` between parameters, because the field's
`Separator` is already `","` and the sweep splits value lists on it. In the job
form, each draft-method choice's expandable section gains an "N-gram assist"
dropdown and the assist's parameter fields. Eleven choices (five draft methods, five n-gram
methods, and off) then reach thirty-six values: off, each mode alone, and all
twenty-five draft-plus-assist pairs. Keeping the list short matters because
`spec_type` is registered without a custom-entry box, so the list is the only
way in.

## Files touched

- `internal/benchmark/sweep.go` — `specValue`, `parseSpecValue`,
  `applySpecValue`, `encodeSpecValue`, `canonicalSpecValue`, and the
  `spec_type` choices table.
- `internal/benchmark/sweep_test.go` — round-trip and rejection cases.
- `web/templates/partials/job_form.html` — the assist sub-picker inside a
  choice's settings block.
- `web/templates/benchmarks.html` — `updateSpecValue` and `addParamValue`
  handle two slots; the live cell-count estimate keeps splitting on `,`.
- `web/jstest/dom.js`, `web/jstest/params_test.js` — fixtures and assertions
  for the new value shape.

## Steps

1. **Encoding.** The accepted shapes become:

   ```
   none                                          speculative decoding off
   draft-mtp                                     draft method, inherit params
   draft-mtp:draft_max=3                         draft method with params
   ngram-mod:assist_n_max=64                     assist alone
   draft-mtp+ngram-mod                           both, inherit params
   draft-mtp+ngram-mod:draft_max=3,assist_n_max=64,assist_n_match=24
   ```

   Parameter keys are globally unique across the two slots
   (`draft_*` vs `assist_*`), so one flat `key=value` list after the single `:`
   still works and no per-mode grouping is needed. The mode part is split on
   `+`; at most two segments, at most one from `DraftModes()` and at most one
   from `AssistModes()`.

2. **`specValue`** grows a second mode field:

   ```go
   type specValue struct {
       mode   string            // draft method, "" if none
       assist string            // n-gram assist, "" if none
       params map[string]string // draft_* and assist_* keys
   }
   ```

   The raw value `"none"` parses to both fields empty and an empty map — the
   same representation the current parser produces for it, and the reason
   `applySpecValue` can write two empty slot pointers without a special case.

3. **`parseSpecValue`.** Keep the existing `if v == "none"` early return at the
   top, before any splitting: `none` is the choices table's off entry and is
   never a mode name, so the mode-name validation below must not see it. Then,
   after `strings.Cut(v, ":")`, split the mode part on `"+"`. Reject more than two segments; reject two draft modes with
   `"%q and %q are both draft methods; only one can run at a time"`; reject two
   assist modes with the matching message; reject a segment that is neither with
   the existing `"%q is not a speculative decoding mode"`. Build the allowed-key
   set from `models.SpecDraftParams(mode)` plus
   `models.SpecAssistParams(assist)` so a key belonging to a slot the value does
   not use is rejected — `draft-mtp:assist_n_max=64` is a mistake worth catching
   at parse time rather than silently dropping. Keep the existing per-key type
   check: `draft_p_min` parses as a float, everything else as an integer.

4. **`applySpecValue`.** Write `o.SpecType = &sv.mode` and
   `o.SpecAssist = &sv.assist` — both always, including when empty, since an
   explicitly empty slot in a swept value means "off for this cell", not
   "inherit". Extend the key switch with the six assist keys onto the pointers
   phase 02 added.

5. **`encodeSpecValue`.** Change the signature to
   `encodeSpecValue(mode, assist string, params []models.SpecModeParam) string`.
   Join the non-empty mode names with `+`, then append `":"` and the non-empty
   parameter defaults joined with `","`. A choice with no parameters at all
   encodes to the bare mode string, as today.

6. **`canonicalSpecValue`.** Render `mode`, then `"+"+assist` if set, then the
   sorted `key=value` list — so `draft-mtp+ngram-mod:assist_n_max=64,draft_max=3`
   and the same value with the pairs reversed dedup to one cell. An unparseable
   value still returns the trimmed raw string for the parser to reject with a
   real message later.

7. **Choices table** (`sweep.go:655`). Rebuild it from `specmodes.go` so the
   labels cannot drift from the config form:

   ```
   none            Off (no speculative decoding)
   draft           Draft Model
   draft-mtp       MTP (self-speculation)
   draft-eagle3    EAGLE-3
   draft-dflash    DFlash
   draft-dspark    DSpark
   ngram-mod       N-gram Mod
   ngram-simple    N-gram Simple
   ngram-cache     N-gram Cache
   ngram-map-k     N-gram Map-K
   ngram-map-k4v   N-gram Map-K4V
   ```

   Eleven entries: five draft, five assist, and off. The post-registration loop
   that attaches `Params` and `Mode` to each choice now attaches
   `SpecDraftParams` to a draft choice and `SpecAssistParams` to an assist
   choice, and sets a new `SweepChoice.Slot` field (`"draft"` or `"assist"`) so
   the template knows whether to render the assist sub-picker.

8. **`SweepChoice`** gains `Slot string` and `AssistModes []models.SpecMode`
   (populated only on draft choices, so the template can render the dropdown
   without a template-side call), plus
   `AssistParamsByMode map[string][]models.SpecModeParam`
   keyed by assist mode name so every assist mode's fields can be rendered
   hidden and revealed by the dropdown — the choices are rendered once, server
   side, and the browser only re-encodes.

9. **Job form template** (`web/templates/partials/job_form.html:150`). Inside
   `<details class="spec-params">`, after the existing parameter inputs, and
   only when `.Slot` is `"draft"`:

   ```html
   <label class="spec-param" title="Runs alongside the draft method. Costs no
          extra memory. When both propose tokens for the same step, the n-gram
          proposal is used.">
     <small>N-gram assist</small>
     <select class="spec-assist-select" onchange="updateSpecValue(this)">
       <option value="">None</option>
       {{range .AssistModes}}<option value="{{.Name}}">{{.Label}}</option>{{end}}
     </select>
   </label>
   {{range $mode, $params := .AssistParamsByMode}}
   <div class="spec-assist-params" data-assist="{{$mode}}" hidden>
     {{range $params}}
     <label class="spec-param" title="Clear the field to use the model's saved value.">
       <small>{{.Label}}</small>
       <input type="text" inputmode="decimal" class="spec-param-input"
              data-key="{{.Key}}" value="{{.Default}}"
              placeholder="model's saved value" oninput="updateSpecValue(this)">
     </label>
     {{end}}
   </div>
   {{end}}
   ```

   Toggle the blocks with `el.hidden`, and have `updateSpecValue` skip inputs
   inside a hidden block so a non-selected assist's defaults never reach the
   encoded value.

10. **`updateSpecValue`** (`benchmarks.html:430`). Read the box's `data-mode` as
    today, then read `.spec-assist-select` if present. Show the matching
    `.spec-assist-params` block and hide the rest. Collect pairs from every
    `.spec-param-input` that is not inside a hidden block. Build
    `mode + (assist ? "+" + assist : "") + (pairs.length ? ":" + pairs.join(",") : "")`.
    The rest of the function — find the checkbox by `data-mode`, set its value,
    check it, `syncParamRow`, `updateMatrixCount` — is unchanged.

11. **`addParamValue`** (`benchmarks.html:488`). This is the restore path when a
    saved job is edited, and today it takes `raw.split(':')[0]` as the mode.
    Split that on `"+"`: the segment matching a checkbox's `data-mode` selects
    the checkbox; the other segment, if any, is set on that choice's
    `.spec-assist-select` and its params block revealed before the `key=value`
    pairs are distributed. Distribute each pair to whichever input carries that
    `data-key`, in the draft block or the revealed assist block. A saved job
    holding a bare `ngram-mod` value still matches the `ngram-mod` choice's own
    checkbox, so old jobs restore unchanged.

12. **Cell-count estimate.** No change needed: the field's `Separator` is still
    `","`, and `+` appears only inside a single value. Add a case to the JS
    tests that pins this, because it is exactly the kind of thing a later
    separator change would break silently.

13. **JS tests** (`web/jstest/`). In `dom.js`, extend the `spec_type` row
    fixture with an assist select and an assist params block on the `draft-mtp`
    choice. In `params_test.js`:
    - `spec_type` still has no custom box (the existing assertion, kept).
    - Editing an assist parameter produces
      `draft-mtp+ngram-mod:draft_max=3,assist_n_max=64`.
    - Setting the assist select back to None drops the `+ngram-mod` segment and
      the `assist_*` pairs.
    - A saved value `draft-mtp+ngram-mod:draft_max=3,assist_n_max=64` restores
      onto the checkbox, the assist select and both inputs.
    - A saved bare `ngram-cache` value still survives a round trip (the existing
      assertion, kept).

## Build gate

```
make build
make js-test
go test ./internal/benchmark/...
go vet ./...
```

## Test plan

- Go round-trip tests in `sweep_test.go`: every shape in step 1 parses, encodes
  back to itself under `canonicalSpecValue`, and applies to the expected
  `ConfigOverrides` pointers.
- Rejection tests: `draft+draft-mtp`, `ngram-mod+ngram-simple`,
  `draft-mtp+ngram-mod+ngram-simple`, `draft-mtp:assist_n_max=64`,
  `ngram-mod:draft_p_min=0.5`, and `draft-mtp:draft_max=abc` each return an
  error naming the problem.
- `none` never appears in an emitted value, and a value with an empty draft slot
  encodes as `ngram-mod…` rather than `none+ngram-mod` — the poison-value rule
  from `common_speculative_types_from_names`.
- Manually: build a job with MTP checked and its assist set to N-gram Mod, plus
  MTP checked with assist None — confirm the cell count reads 2, submit, and
  confirm the two cells launch with and without `,ngram-mod` in `--spec-type`.
  Then edit the saved job and confirm both values restore into the form
  unchanged.

## Commit

```
feat(benchmarks): sweep a draft method and an n-gram assist together

spec_type sweep values now name both slots, joined with "+" so the field's
existing "," separator keeps splitting value lists:
draft-mtp+ngram-mod:draft_max=3,assist_n_max=64. Parameter keys are unique
across the two slots, so one flat key=value list still covers both.

Each draft-method choice in the job form gains an n-gram assist dropdown with
that mode's own settings. Eleven choices then reach thirty-six values -- off,
each mode alone, and all twenty-five draft-plus-assist pairs -- without growing
a choices list that has no custom-entry box to fall back on. The five draft
methods and five n-gram methods replace the eight choices offered before.
```

## Rollback

Revert the commit. A benchmark job saved with a combined value becomes
unparseable on the reverted build — `parseSpecValue` rejects
`draft-mtp+ngram-mod` with "is not a speculative decoding mode" — so such a job
fails to load rather than running the wrong thing, which is the right failure.
Delete or edit those jobs before reverting. Jobs saved before this phase are
unaffected. Safe to leave the Go half applied without the template half: the
parser accepts more shapes than the form can produce.
