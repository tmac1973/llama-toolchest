# Report: multiple speculative modes, `--spec-chain`, and adaptive MTP

Written 2026-09-07. Verified against the managed llama.cpp clone at
`b10453` (`3cb7ffb1a`, 16 Aug 2026), upstream `master` docs, and the
GitHub issue/PR state on the day of writing.

## Bottom line

Three separate things were suggested. They are in three different states,
and only one of them is worth building now.

| Feature | Exists upstream? | Worth adding? |
|---|---|---|
| Several speculative modes at once (`--spec-type a,b`) | **Yes** — merged, documented, in the build we already ship | **Yes.** Real speed on echo-heavy work, and the toolchest cannot express it today. |
| `--spec-chain N` | **No.** The flag does not exist anywhere in llama.cpp. The feature request (#23184) is closed as not planned. | No. Nothing to wrap. |
| `draft-mtp-adaptive` | **Not merged.** Open PR [#27210](https://github.com/ggml-org/llama.cpp/pull/27210), 83 comments, merge blocked. | Not yet — but it is cheap to add later, and it is a good first customer for the "build from a PR" work. |

The claim that started this — that llama.cpp accepts a comma-separated
`--spec-type` list — is correct. The claims about `--spec-chain` and a
merged `draft-mtp-adaptive` are not.

---

## 1. What llama.cpp actually does today

### Multiple modes are a merged, supported feature

`common/arg.cpp` splits the `--spec-type` value on commas and *appends*
to a list, and repeating the flag is explicitly exempted from the
"specified multiple times" deprecation warning:

```cpp
{"--spec-type"}, common_speculative_all_types_str(),
"comma-separated list of types of speculative decoding to use ..."
const auto types_str = string_split<std::string>(value, ',');
params.speculative.types.insert(params.speculative.types.end(), ...);
```

`docs/speculative.md` states the rule plainly:

> An implementation with draft model can be mixed with an implementation
> without draft model.
> […]
> If a draft model is combined with a draftless decoding the draftless
> decoding has higher precedence.

Four details matter for how we would model this:

1. **It is a set, not a chain.** `common_speculative_init` turns the list
   into a bitmask and then walks a *fixed* priority order — all n-gram
   methods first, then all draft-model methods. `draft-mtp,ngram-mod` and
   `ngram-mod,draft-mtp` produce byte-identical behaviour. There is no
   user-controllable ordering to expose.
2. **At most one draft-model method.** The docs only sanction
   draft + draftless. Combining `draft-mtp` with another draft-model type
   while `--model-draft` is set currently crashes at startup — open issue
   [#27897](https://github.com/ggml-org/llama.cpp/issues/27897),
   "failed to create MTP context". This is a constraint the UI should
   enforce, not a warning to print.
3. **`none` is poison in a list.** `common_speculative_types_from_names`
   returns `{NONE}` and discards everything else the moment it sees
   `none`. We must never emit `none,` as a list member.
4. **An unknown name is fatal.** It throws
   `unknown speculative type: <name>` and llama-server refuses to start.
   There is no graceful degradation for a mode a given build does not know.

### Each mode has its own draft-length knobs

This is the part that breaks our current data model. The modes do *not*
share a draft length:

| Mode | Draft length flags | Defaults |
|---|---|---|
| `draft-simple`, `draft-mtp`, `draft-eagle3`, … | `--spec-draft-n-max` / `-n-min` | 3 / 0 |
| `ngram-mod` | `--spec-ngram-mod-n-max` / `-n-min` / `-n-match` | 64 / 48 / 24 |
| `ngram-simple`, `ngram-map-k`, `ngram-map-k4v` | `--spec-<mode>-size-n` / `-size-m` / `-min-hits` | 12 / 48 / 1 |

`common_speculative_n_max` takes the maximum across whatever is enabled,
so the two lengths coexist deliberately: MTP drafts 3 tokens, ngram-mod
drafts up to 64, both live.

### `--spec-chain` does not exist

Code search over `ggml-org/llama.cpp` returns zero hits for `spec-chain`
and zero for `spec_chain`. The underlying idea — feed the MTP draft into
ngram-mod so the n-gram *extends* it rather than replacing it — is
[issue #23184](https://github.com/ggml-org/llama.cpp/issues/23184), and
it is **closed as not planned**. The reporter's own estimate for the
pipelined version was 5–10% extra throughput.

Note also that the older PR "spec : allow for multiple spec types (chains
of speculators)" ([#22546](https://github.com/ggml-org/llama.cpp/pull/22546))
was closed unmerged; multi-type support arrived by another route. The word
"chain" in that title is probably where the `--spec-chain` idea came from.

### `draft-mtp-adaptive` is an open PR

[PR #27210](https://github.com/ggml-org/llama.cpp/pull/27210) adds
`--spec-type draft-mtp-adaptive` plus `--spec-draft-n-min-adaptive`. It
raises draft depth after consecutive full accepts and drops it under a
weighted miss pressure. Opened 17 Aug 2026, last touched today,
`mergeable_state: blocked`, +451/−26 across 10 files. It is not in any
released build, and the author's own guidance is hedged: worse on older
hardware, weaker on MoE, and pointless below `n-max 7`.

---

## 2. Value

The strongest evidence is the benchmark table in PR #27210 itself, which
happens to be run on hardware close to ours (2× Radeon AI PRO R9700,
ROCm, Qwen3.8-27B Q8_0, tokens/second, generation only):

| config | reasoning | prose | code | recall |
|---|---|---|---|---|
| C0 no speculation | 30.03 | 30.08 | 30.03 | 29.93 |
| C1 MTP fixed 3 | **51.68** | 55.89 | 69.28 | 81.94 |
| C5 ngram-mod alone | 30.03 | 30.10 | 29.98 | 292.88 |
| C7 **MTP fixed 3 + ngram-mod** | 51.65 | 55.90 | 68.36 | **324.10** |
| C3 adaptive 3..12 | 51.42 | **56.64** | **78.03** | 147.46 |
| C6 adaptive + ngram-mod | 51.48 | 56.61 | 73.06 | 321.25 |

Read the two rows we can build today, C1 and C7:

- **Combining costs nothing.** On reasoning, prose and code the combined
  row is within noise of MTP alone (51.65 vs 51.68, 55.90 vs 55.89,
  68.36 vs 69.28).
- **Combining is worth 4× on echo-heavy work.** 324 vs 82 tokens/second
  on the recall task — text the model has already seen and is repeating.
  That is the file-rewrite, structured-output, tool-call-loop case, which
  is a large share of how these servers actually get used.
- **Neither mode alone gets both.** ngram-mod alone is *zero* help on
  novel generation (30.0, exactly baseline). MTP alone gives up three
  quarters of the echo speed.

Set against that, the earlier report of `draft-mtp,ngram-mod` being ~4
tokens/second *worse* than MTP alone on novel text is plausible and
backend-dependent — the priority rule means a low-quality n-gram
suggestion displaces a high-acceptance MTP draft for that step. It is a
reason to make the combination measurable, not a reason to avoid it. We
already have the instrument for that: the benchmark sweep.

Adaptive MTP's own numbers are more interesting on paper (+13% on code
over fixed-3, and it removes the depth-tuning problem) but it is not
merged, and C6 vs C7 shows it adds nothing once ngram-mod is in the mix
for the echo case.

**Value verdict:** the combination is the one with a real, measured,
workload-shifting payoff, and it needs no new upstream code.

---

## 3. Viability in llama-toolchest

### What has to change

`SpecType` is a single string today, and every consumer treats it as one.
14 non-test Go files, 4 templates and the two JS test fixtures reference
it. The work is not deep, but it is wide.

**a. The data model is the real cost.** Today `DraftMax`, `DraftMin`,
`NgramSizeN`, `NgramSizeM` are *one* set of fields whose meaning depends
on the selected mode — `specflags.go` switches on `SpecType` to decide
whether `DraftMax` becomes `--spec-draft-n-max` or
`--spec-ngram-mod-n-max`. With two modes active, that same field would
have to be two different flags at once. Config C7 above — MTP at 3,
ngram-mod at 64 — is literally inexpressible in the current struct.

There are two ways out:

- *Per-mode parameter blocks.* `SpecModes []SpecModeConfig`, each with its
  own mode name and values. Correct, and it makes `specDecodingParams` a
  loop over blocks instead of a switch. Costs a config migration
  (`spec_type` + flat fields → a list) and a rewrite of the benchmark
  override plumbing, which passes each field as its own `*int`.
- *Keep the flat fields, add a second mode.* `SpecType` stays the draft
  method; a new `SpecAssist` holds the draftless one, with its own small
  set of fields. Much cheaper, no migration, and it directly encodes the
  "one draft method + one draftless method" rule that llama.cpp actually
  sanctions. It does not generalise to three modes — which nothing asks
  for.

I would take the second. The upstream rule is not "any set of modes"; it
is "a draft method may be mixed with a draftless method". Modelling
exactly that is smaller *and* more honest than a general list.

**b. The preset INI must emit one line.** `common/preset.cpp` parses an
INI section into a `std::map<common_arg, std::string>`, so a repeated
`spec-type =` line silently keeps only the last value. This is the router
bug that was reported. `specDecodingParams` currently returns one
`specParam` per flag and `preset.go` writes one line each, so emitting two
`spec-type` entries would produce exactly that bug. The fix is trivial —
join the mode names with a comma into a single entry — but it has to be
done deliberately, and a test should pin it.

**c. The benchmark axis has a comma collision.** `spec_type` sweep values
are encoded as `mode:key=value,key=value`, and the field's `Separator` is
`","` — commas already mean two different things in that string. A
combined mode cannot be written as `draft-mtp,ngram-mod` there without
choosing a new separator (`+` reads well: `draft-mtp+ngram-mod:...`) and
updating `parseSpecValue`, `encodeSpecValue`, the choices table, the live
cell-count estimate in `benchmarks.html`, and the JS param tests.

**d. Equality checks on `SpecType` become substring checks.** Three
places compare the mode by `==`: VRAM aux-file accounting
(`vram.go:253`, `SpecType == "draft"`), the measured-memory load
signature (`memory_measured.go:203`), and the run comparison label
(`compare_labels.go:100`). Each needs a small helper rather than a
literal comparison. The template conditionals in `model_config.html`
have the same shape and the same fix.

**e. The UI.** The mode picker becomes a draft-method picker plus a
draftless "assist" picker, with the assist's own parameters revealed
underneath. The help dialog needs a paragraph on what combining does and,
importantly, that the draftless method wins each step — that precedence
rule is the single thing a user needs to understand to read their own
benchmark results.

### Effort

| Piece | Size |
|---|---|
| Config field + migration-free default, INI/CLI emission, tests | Small |
| Model config form + help copy | Small |
| Benchmark axis encoding, choices, cell-count estimate, JS tests | Medium — the fiddliest part |
| VRAM / memory-signature / compare-label call sites | Small |

Call it a focused day's work, most of it in the benchmark surface rather
than in the launch path.

### Risk

Low, and mostly contained. The feature is additive: a config that names
one mode emits exactly the flags it emits today. The failure mode of the
combined path is a llama-server that will not start, which the job runner
already surfaces as a failed cell. The one thing to guard is (a) above —
never letting the UI offer two draft-model methods, since that is the
open upstream crash.

---

## 4. `draft-mtp-adaptive` specifically

Adding it is a two-line change *once it merges*: one entry in the mode
list, one entry in `SpecModeParams` (and `--spec-draft-n-min-adaptive`
as a fourth tunable). Adding it before it merges is a different question,
because an unknown mode name is a hard startup failure and the toolchest
has no build-capability gating — the gemma-4 note in the config form is a
static sentence of prose, not a check against the active build's commit.

The natural time to revisit is when
[`plan/builder-refs.md`](builder-refs.md) lands. Building
`refs/pull/27210/head` is exactly the case that plan exists for, and a
mode that only appears when the active build knows it would be the first
real use of build-aware capability gating. Until then, a user who wants
to try it can already put `--spec-type draft-mtp-adaptive` in Extra Flags
against a hand-built binary.

---

## 5. Two things found while checking this

Unrelated to the request, but in the same code:

1. **`--spec-ngram-mod-n-match` is never emitted.**
   `SpecModeParams` (`internal/models/specparams.go:33`) offers "N-gram
   size N" with a default of 24 for `ngram-mod`, and the config form
   renders the field (`web/templates/partials/model_config.html:380`),
   but the `ngram-mod` case in `specDecodingParams`
   (`internal/models/specflags.go:86-93`) only emits `n-max` and
   `n-min`. Whatever the user types in that
   box is stored and ignored. It happens to be harmless right now because
   llama.cpp's own `n_match` default is also 24 — so the field is dead
   rather than wrong — but a user who tunes it gets silence.
   `ngram_size_m` is worse: `ngram-mod` has no m-gram size at all, so
   that field is meaningless for the mode and should not be offered.
2. **Three merged spec types are unreachable.** `draft-eagle3`,
   `draft-dflash` and `draft-dspark` (`common/common.h:170-182` in the
   managed clone) are all in the build we ship, all
   documented, all with converted checkpoints on Hugging Face for models
   we run (Qwen3, gpt-oss, gemma-4). None appears in the mode picker
   (`web/templates/partials/model_config.html:277-287`) or in the
   benchmark axis choices (`internal/benchmark/sweep.go:654-661`).
   EAGLE-3 in particular claims higher acceptance than a same-size draft
   model, and it reuses the draft-model plumbing we already have. This
   may be a larger win than anything in this report and is cheaper than
   all of it.

---

## 6. Recommendation

1. **Build the combination** (`draft-mtp` or `draft` plus one draftless
   method), modelled as draft-method + assist rather than a general list.
   The upside is measured, large on the workloads these servers see most,
   and needs nothing from upstream.
2. **Fix the ngram-mod parameters** while in the file — emit `n-match`,
   drop the meaningless m-gram field.
3. **Add the three missing merged spec types** as a separate, smaller
   change. Probably do this first; it is the best value per hour here.
4. **Skip `--spec-chain`.** It does not exist and is not planned.
5. **Defer `draft-mtp-adaptive`** until it merges, and treat it as the
   first customer for build-aware capability gating once
   `plan/builder-refs.md` lands.
