# Multiple speculative decoding modes — Project Overview

Source report: [`report.md`](report.md)
(written 2026-09-07, verified against the managed llama.cpp clone at `b10453`).

## Problem

llama.cpp has accepted a comma-separated `--spec-type` list for some time. A
draft-model method and a draftless n-gram method can run together, and the
measured effect is large: on PR #27210's benchmark table — run on hardware close
to ours — MTP alone reaches 82 tokens/second on echo-heavy work, while
MTP + N-gram Mod reaches 324, with no measurable cost on novel generation
(51.65 vs 51.68 reasoning, 68.36 vs 69.28 code). Echo-heavy work is file
rewrites, structured output and tool-call loops, which is a large share of how
these servers are actually used.

llama-toolchest cannot express that. `ModelConfig.SpecType` is a single string,
and the four draft-length fields around it (`DraftMax`, `DraftMin`,
`NgramSizeN`, `NgramSizeM`) change meaning depending on which mode is selected —
`specflags.go` switches on `SpecType` to decide whether `DraftMax` becomes
`--spec-draft-n-max` or `--spec-ngram-mod-n-max`. A configuration with MTP
drafting 3 tokens and N-gram Mod drafting 64 is not merely unsupported, it is
inexpressible in the current struct.

Two smaller defects sit in the same code. `--spec-ngram-mod-n-match` is offered
in the config form and stored, but `specDecodingParams` never emits it, so
tuning it does nothing; and `ngram_size_m` is offered for `ngram-mod`, which has
no m-gram size at all. Separately, three merged and documented spec types —
`draft-eagle3`, `draft-dflash`, `draft-dspark` — are present in the build we
ship, have converted checkpoints on Hugging Face for models we run, and appear
in neither the mode picker nor the benchmark axis.

## Goals

- Let a model configuration run one draft-model method together with one
  draftless n-gram method, with independent parameters for each.
- Emit a single comma-joined `spec-type` line in the preset INI, since
  `common/preset.cpp` parses a section into a map and silently keeps only the
  last value of a repeated key.
- Make the combination sweepable on the benchmark axis so its effect on this
  hardware is measurable rather than argued about.
- Add `draft-eagle3`, `draft-dflash` and `draft-dspark` to the mode picker and
  the benchmark axis choices.
- Emit `--spec-ngram-mod-n-match` and `--spec-<mode>-min-hits`, both of which
  are offered or defaulted today but never passed to llama-server, and stop
  offering the meaningless m-gram field for `ngram-mod`.
- Add a recall-workload benchmark preset, so the combination's main benefit can
  be measured here at all — every existing internal preset asks the model to
  produce new text, which is exactly the workload where an n-gram method
  measures nothing.
- Make it structurally impossible for the UI to select two draft-model methods
  at once, which is the open upstream startup crash
  ([#27897](https://github.com/ggml-org/llama.cpp/issues/27897)).
- Keep every existing configuration working: a config naming one mode must emit
  exactly the flags it emits today, with the single deliberate exception of the
  two flags that were dropped by the bugs above and now appear.
- Measure the combination on this hardware and record the result.

## Non-goals

- **`--spec-chain` is not built.** The flag does not exist anywhere in
  llama.cpp; the underlying feature request
  ([#23184](https://github.com/ggml-org/llama.cpp/issues/23184)) is closed as not
  planned. There is nothing to wrap.
- **`draft-mtp-adaptive` is not added.** It is an open, merge-blocked PR
  ([#27210](https://github.com/ggml-org/llama.cpp/pull/27210)). An unknown mode
  name is a hard llama-server startup failure, and the toolchest has no
  build-aware capability gating. It is revisited when
  [`../builder-refs.md`](../builder-refs.md) lands, as the first customer for
  that gating. Until then a user can put `--spec-type draft-mtp-adaptive` in
  Extra Flags against a hand-built binary.
- **No general N-mode list.** llama.cpp's own rule is "a draft method may be
  mixed with a draftless method", not "any set of modes". The data model encodes
  exactly that rule; three simultaneous modes are not supported and nothing asks
  for them.
- **No user-controllable mode ordering.** `common_speculative_init` turns the
  list into a bitmask and walks a fixed priority order, so `draft-mtp,ngram-mod`
  and `ngram-mod,draft-mtp` are byte-identical. There is no ordering to expose.
- **Stored benchmark history is not rewritten.** Old jobs and results keep their
  `spec_type: ngram-mod` shape on disk and are normalised on read.
- **No build-capability detection.** The three new spec types are added
  unconditionally because they are in the build we ship.

## Users & primary flow

The user is the operator of this llama-toolchest instance, configuring and
benchmarking local llama-server deployments.

**Configuring a model.** In the model config form's Speculative Decoding
section, two pickers now sit side by side and are always visible:

1. **Draft method** — Disabled, Draft Model, MTP (self-speculation), EAGLE-3,
   DFlash, DSpark. Its parameters (draft tokens max/min, draft probability min)
   and its model/head path appear underneath when a method is chosen.
2. **N-gram assist** — None, N-gram Mod, N-gram Simple, N-gram Cache,
   N-gram Map-K, N-gram Map-K4V. Its own parameters appear underneath.

Because the draft methods live in one picker and the n-gram methods in the
other, two draft methods cannot be selected. Leaving Draft method on Disabled
and choosing an N-gram assist is the draftless-only configuration that
`spec_type: ngram-mod` expresses today. An effective-flags line under the
section shows the resulting `--spec-type` value so the user can see the
comma-joined list they have built.

**Benchmarking a combination.** On the benchmark job form, each draft-method
choice's expandable section gains an "N-gram assist" dropdown alongside its own
parameters, with the assist's parameters below it. Checking MTP with the assist
set to N-gram Mod produces the sweep value
`draft-mtp+ngram-mod:draft_max=6,assist_n_max=64,assist_n_min=48,assist_n_match=24`
— each slot's own recommended defaults, pre-encoded, exactly as the mode choices
already work today. Running that against MTP alone on a novel-generation prompt
set and an echo-heavy one is the C1-vs-C7 comparison from the source report, on
local hardware.

**Launching.** Nothing changes for the user. The launch path emits
`--spec-type draft-mtp,ngram-mod` plus each mode's own flags, and the preset INI
carries one `spec-type` line.

## Constraints

- **Go backend, `html/template` + htmx frontend.** No build step for the web
  assets; JS tests live in `web/jstest/` and run against a small DOM shim.
- **`none` must never appear in a list.** `common_speculative_types_from_names`
  returns `{NONE}` and discards everything else the moment it sees `none`. The
  emitter never writes `none` as a list member — an empty slot contributes
  nothing to the list.
- **An unknown mode name is fatal.** llama-server refuses to start with
  `unknown speculative type: <name>`. There is no graceful degradation, which is
  why only merged, shipped mode names are offered.
- **One `spec-type` line in the INI.** `common/preset.cpp` parses an INI section
  into a `std::map<common_arg, std::string>`; a repeated `spec-type =` line
  keeps only the last value.
- **The sweep encoding already uses commas.** `spec_type` sweep values are
  `mode:key=value,key=value` and the field's `Separator` is `","`, so the two
  mode names are joined with `+` instead.
- **`spec_type` has no custom-entry box** in the benchmark job form
  (`web/jstest/params_test.js:49` pins this), so every reachable value must come
  from the choices table and its expandable controls.
- **The three new draft methods load a head that fails the draft-candidate
  filter.** `FindDraftCandidates` requires the same architecture and under 40%
  of the target's size; a converted EAGLE-3 / DFlash / DSpark head passes
  neither.
- **17 non-test files reference `SpecType`**, plus 4 templates and 2 JS test
  fixtures. The work is wide rather than deep.
- **Plain language throughout.** No slang in labels, help copy or tooltips;
  written for a non-developer, ESL-inclusive reader.
- **UI conventions.** Tooltips carry the explanation rather than static prose;
  the effective `--spec-type` value is always visible, not revealed on hover.

## Success criteria

- A model configured with MTP at 3 draft tokens and N-gram Mod at 64/48/24
  launches llama-server with `--spec-type draft-mtp,ngram-mod`,
  `--spec-draft-n-max 3`, `--spec-ngram-mod-n-max 64`,
  `--spec-ngram-mod-n-min 48` and `--spec-ngram-mod-n-match 24`.
- The preset INI for that model contains exactly one `spec-type` line, reading
  `spec-type = draft-mtp,ngram-mod`. A test pins this.
- Every configuration that exists before the change emits an identical flag set
  after it — verified by a test over the existing spec smoke-test fixtures —
  except that an `ngram-mod` config now also emits `--spec-ngram-mod-n-match`,
  carrying the value the form has been storing and discarding. That one addition
  is the bug fix; every other flag is byte-identical. `--spec-<mode>-min-hits`
  becomes tunable and is emitted for configs saved after the change; a migrated
  config leaves it unset, because there is no legacy field to migrate from and
  llama.cpp's own default is the 1 the form now offers, so nothing changes for
  it either.
- A draftless-only config saved before the change (`spec_type: ngram-mod`) still
  launches with the same flags plus that one `n-match` addition, and its saved
  parameters land on the new fields after the one-shot backfill.
- Saved benchmark jobs and results that carry `spec_type: ngram-mod` still
  render their labels and still re-run correctly.
- The mode picker offers `draft-eagle3`, `draft-dflash` and `draft-dspark`, and
  each launches with a head selected from a picker that no longer filters by
  architecture and size.
- Selecting two draft-model methods is impossible from the UI.
- A benchmark job can sweep MTP alone against MTP + N-gram Mod in one run, and
  the compare table labels the two cells distinguishably.
- VRAM accounting counts a draft head for every draft method that loads one —
  `DraftModelPath` for `draft`, `draft-eagle3`, `draft-dflash` and
  `draft-dspark`, and `MtpPath` for `draft-mtp`, which is already counted —
  rather than only for `draft`; the n-gram assist loads no file and adds
  nothing.
- Tuning "match length" for N-gram Mod changes the launched command, as does
  "minimum hits" for N-gram Simple / Map-K / Map-K4V; the m-gram field is no
  longer offered for N-gram Mod.
- An `internal-echo` benchmark preset exists whose generation is recall of text
  already in the prompt, and `ngram-mod` alone measures at baseline under
  `internal-standard` while measuring well above it under `internal-echo`.
- The C1-vs-C7 sweep has been run on this hardware and its numbers are recorded
  in this plan folder.
- `make build`, `go test ./...` and `make js-test` pass. (There is no
  `make test` target; the JS suite runs inside `go test ./internal/api/` when
  node is present, and `make js-test` fails loudly when it is not.)

## Decisions

- **Scope** → All three buildable items: add the three merged-but-unreachable
  spec types, fix the N-gram Mod parameter bugs, and build the draft + assist
  combination. Skip `--spec-chain`; defer `draft-mtp-adaptive`.
- **Data model** → `SpecType` stays the draft method; a new `SpecAssist` holds
  the draftless one, with its own six fields: `AssistNMax`, `AssistNMin`,
  `AssistNMatch`, `AssistSizeN`, `AssistSizeM`, `AssistMinHits`. Not a general
  list of mode blocks.
- **Assist modes and visibility** → All five draftless modes are offered; the
  assist picker pairs with a draft method, so two draft methods cannot be
  selected. This is what makes the upstream crash structurally unreachable.
- **Sweep axis encoding** → One axis. `+` joins the two mode names, `,` keeps
  separating parameters: `draft-mtp+ngram-mod:draft_max=3,assist_n_max=64`.
  Parameter keys are unique across the draft and assist blocks, so one flat
  key=value list still works.
- **Head path for the new draft methods** → Reuse `DraftModelPath`, and relax
  `FindDraftCandidates` when the mode is `draft-eagle3`, `draft-dflash` or
  `draft-dspark`: drop the architecture-match and 40%-size checks, keep the
  embedding-model and MTP-head exclusions. No new path field, no auto-detection
  pass.
- **Draftless parameter fields** → One shared block. `SpecType` becomes strictly
  the draft method (or empty); `SpecAssist` strictly the draftless one. A
  draftless-only config becomes `spec_type: ""` + `spec_assist: ngram-mod`,
  which costs a one-time config migration and gives one place to fix the
  n-match bug.
- **Migration mechanics** → A one-shot backfill at registry load rewrites model
  configs and saves, following the `BackfillGGUFMeta` pattern. A separate
  tolerant `models.NormalizeSpec` helper on the read path maps a draftless name found
  in `SpecType` into `SpecAssist`, so stored benchmark jobs and results keep
  working without their history being rewritten.
- **Model config form layout** → Two always-visible pickers, "Draft method" and
  "N-gram assist", each with its parameters revealed underneath.
- **Benchmark job form** → Each draft-method choice's expandable section gains
  an "N-gram assist" dropdown plus the assist's parameter fields. The choices
  list is eleven entries — five draft methods, five n-gram methods and off, up
  from the eight offered today — and every combination remains reachable.
- **Naming** → "Draft method" and "N-gram assist".
- **Validation** → A final phase runs the C1-vs-C7 sweep on this hardware — MTP
  alone against MTP + N-gram Mod, across a novel-generation and an echo-heavy
  prompt set — and records the numbers in this plan folder.
- **Params table shape** → Split `SpecModeParams` into `SpecDraftParams(mode)`
  and `SpecAssistParams(mode)`, each with one key namespace, so a caller can
  never be handed a key it has no field for.
- **Defaults for the three new draft methods** → All fields blank, inheriting
  llama.cpp's own `--spec-draft-n-max 3` / `-n-min 0`. There is no verified
  guidance for EAGLE-3, DFlash or DSpark, and the benchmark axis is the right
  place to find good values rather than guessing them into the table.
- **`--spec-<mode>-min-hits`** → Fixed in the same work. It is the same class of
  gap as `n-match`, in the same function, and the assist parameter block is
  being written from scratch anyway.
- **Phase order** → Data model first, with the three new draft methods riding
  along in phase 01 rather than shipping ahead of it. The shared-block decision
  rewrites the mode picker and the choices table regardless, so this writes
  every surface once, in its final shape.
- **Echo workload** → Add an `internal-echo` preset (phase 05) before
  validating. The existing presets can only measure the half of the claim where
  the combination changes nothing.
