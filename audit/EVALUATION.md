# Meta-Evaluation of Code-Duplication Audits — llama-toolchest

**Date:** 2026-06-24
**Evaluator:** Claude Opus 4.8 (separate session from the audit author)
**Subject:** Six independent code-duplication audits of the llama-toolchest Go
codebase (~16.5k LOC under `internal/` + `cmd/`), produced by five open-weight
models running locally plus one Claude Opus 4.8 run.

This report reads all six audits, cross-checks their concrete claims against the
actual source tree, and rates each for **correctness** (are the findings real and
accurately located?) and **coverage** (how much of the genuine duplication did it
find?).

> **Conflict-of-interest note.** One of the six audits under review was produced by
> Claude Opus 4.8, the same model family writing this evaluation. I verified its
> claims against source the same way as the others (line-by-line grep/read), and I
> call out its real coverage gaps below. But a reader should weight my praise of it
> accordingly.

---

## Verdict at a glance

| Audit | Findings | Verified correct | False/▽ fabricated | Unique real finds | Grade |
|---|---|---|---|---|---|
| **claude-opus-4.8** | 24 | all spot-checks passed | 0 | 16 (sole finder) | **A** |
| **qwen-3.6-27b** | 12 | all verified | 0 | 2 (shortSHA, GPU-alloc map) | **B+** |
| **qwen3-coder-next** | 6 | all verified | 0 | 0 | **B−** |
| **minimax-2.7** | 5 | 4 verified | **1 fabricated** | 3 (registry/monitor) | **C+** |
| **glm-4.7-flash** | 8 (themes) | ~2 themes real | metrics inverted; ▽ line nums | 0 | **D+** |
| **nemotron-3-super-120b** | 0 substantive | n/a | 0 | 0 | **F** |

"Grade" reflects correctness × coverage × usefulness of the writeup, not length.

---

## Methodology

For each audit I took its concrete, checkable claims — file, function, line range,
and the asserted pattern — and verified them against the tree with `grep`/`Read`.
Representative checks performed:

- Located every shard helper (`shardRe`, `findShards`, `isNonFirstShard`,
  `shardPattern`, `ExpandShards`) — they live in `internal/models/registry.go` and
  `internal/huggingface/client.go` **only**.
- Confirmed the capabilities-list triplication (`capabilities.go:17`,
  `models.go:175`, `v1models.go:14`) and the `IsEmbeddingModel(m.ModelID) ||
  IsEmbeddingModel(m.ID)` idiom at 6 sites.
- Confirmed spec-decoding flag emission in `preset.go` (`switch cfg.SpecType`,
  line 145) vs `registry.go` (`switch c.SpecType`, line 229).
- Confirmed router `/models/load` + `/models/unload` logic in both
  `process/manager.go` (254/278) and `benchmark/runner.go` (201–288).
- Confirmed `EstimateVRAM` (vram.go:18, `+ vramOverheadGB`) vs `estimateVRAM`
  (huggingface/client.go:185, no overhead).
- Confirmed `.so` copy loops (builder.go:511 & 525), logbus subscribe/broadcast
  (builder.go:148/185, manager.go:318/346), `shortModelName`/`shortenModelName`
  (gpu_assign.go:264, jobs_env.go:184), `cssID`/`domID` (server.go:129, hf.go:239),
  `safeShortSHA` (build.go:346), GPU-alloc map build (gpu_map.go:19, service.go:626),
  enabled-models filter (bench_jobs.go:270, bench.go:457, server.go:506).
- Verified minimax's registry/monitor finds: symlink-resolve+walk in
  `OrphanParts` (706) and `ScanModels` (797); `findMMProjInDir` (964) /
  `findMTPInDir` (1041); rocm `card[0-9]*` re-enumeration in `collectSysfs` (106)
  and `readVRAMSysfs` (155/157).
- Counted GLM's metric claims directly (`http.Error`, `respondJSON`,
  `json.NewEncoder`, `r.FormValue`).

Every line reference below was confirmed against the current tree.

---

## Per-audit assessment

### claude-opus-4.8 — Grade A

The most thorough by a wide margin: 24 findings spanning every package, ranked by
severity, each with two-or-more concrete call sites, a remedy, and a separate
**"Non-findings (checked, deliberately not flagged)"** section that demonstrates the
review distinguished real duplication from coincidental shape-sharing (e.g. it
correctly declines to flag the build-tagged `manager_unix/windows` terminate
variants and the already-correct `respond.go` abstraction).

**Correctness:** Every claim I spot-checked (spec-decoding, capabilities, router
HTTP, VRAM, `.so` loops, logbus, `shortModelName`, `cssID/domID`, content-type
decode, suffix-strip) was accurate, with line numbers on target. Zero false
positives found.

**Coverage gaps (real, not nitpicks):** It is the sole finder for 16 of its issues,
but three genuine duplications were missed that other models caught:
- symlink-resolve + `filepath.Walk` scaffold duplicated between `OrphanParts` and
  `ScanModels` (caught by minimax),
- `findMMProjInDir` / `findMTPInDir` structural twins (minimax),
- rocm sysfs GPU re-enumeration in `collectSysfs` vs `readVRAMSysfs` (minimax) —
  related to but distinct from Claude's #7/#14.
- `shortSHA`/`safeShortSHA` parameterizable pair and the GPU-allocation-map build
  duplication (both caught by qwen-3.6) are also absent.

So the **union of all six audits exceeds Claude's report** — useful to know if the
goal is exhaustive cleanup. Still, on its own it covers the most ground at the
highest precision.

### qwen-3.6-27b — Grade B+

The strongest open-weight audit. Twelve findings, all verified correct, all with
accurate `file:line` ranges and severity tags. It correctly identifies that
`isJSONContentType` **already exists** (`proxy.go:113`) and recommends reusing it
rather than re-deriving — a level of codebase awareness the other small models
lack. It contributes two real findings Claude missed (#4 shortSHA pair, #6
GPU-allocation-map build).

**Limitation:** Scope is almost entirely `internal/api`. It found nothing in
`models`, `builder`, `process`, `benchmark`, or `monitor`, so its coverage of the
non-API half of the codebase is near zero. Within its scope it is excellent and
essentially a high-precision subset of Claude's API findings plus two extras.

### qwen3-coder-next — Grade B−

Six findings, all correct, but shallow. Its two "high priority" picks (VRAM
estimation divergence, model-name trimming) are genuine and overlap Claude #18/#19.
The rest (registry iteration pattern, export-handler shape, itoa/ftoa,
suffix-trimming) are real but mostly low-value "missed-centralization" observations.
Line numbers are approximate but land in the right functions. A competent,
honest, unremarkable audit — no fabrications, no deep insight.

### minimax-2.7 — Grade C+

The most interesting failure-and-success mix. It is the **sole finder of three real
duplications** — the `OrphanParts`/`ScanModels` symlink-walk scaffold (~30 lines
each, verified), the `findMMProjInDir`/`findMTPInDir` twins (verified), and the
rocm sysfs card re-enumeration (verified). These required reading deep into
`registry.go`/`rocm.go` internals that even Claude skimmed. Genuine added value.

**But it contains a confident fabrication.** Its "Low" finding claims
`isNonFirstShard`/`findShards` are "copied verbatim" into
`internal/api/bench_export.go:210-237`, complete with invented line numbers and a
plausible rationale ("avoids an import cycle… `api` already imports `models`"). I
checked: **`bench_export.go` contains no shard helpers at all** — they exist only in
`registry.go` and `huggingface/client.go`. This is a hallucinated finding dressed in
specifics, which is more dangerous than a vague one because it reads as
authoritative. Its other claims (itoa/ftoa wrappers, the three structural finds)
are accurate. Net: high signal where it dug in, but cannot be trusted without
verification.

### glm-4.7-flash — Grade D+

Reframes the task from "code duplication" to "API inconsistency," which is partly
legitimate (form-parsing repetition and the content-type-decode pattern are real
and overlap qwen-3.6 #9). But it is undermined by **unreliable, partly inverted
metrics** and **placeholder line numbers**:

| GLM claim | Actual |
|---|---|
| `json.NewEncoder` used 66× | **4** |
| `respondJSON` used only 8× | **45** |
| `http.Error` 88× | 83 (close) |
| `r.FormValue` 60× | 59 (close) |

GLM's headline narrative — "respondJSON barely adopted, json.NewEncoder
everywhere" — is the **opposite of reality**: `respondJSON` is the dominant pattern
(45 uses) and raw `json.NewEncoder` is nearly extinct (4). It also cites references
like `hf.go:3`, `models.go:1`, `bench.go:8` that are clearly placeholders, not real
locations. The two genuinely useful themes are buried in fabricated quantification.

### nemotron-3-super-120b — Grade F

Effectively a non-audit. It examined a handful of functions (mostly the
`bench_export.go` `itoa`/`ftoa`/`metricMean` helpers and a few single-use
`gguf.go` functions) and concluded "no instances of harmful duplication." Its only
correct observations (itoa/ftoa are underused wrappers) are trivial and were also
caught by three other models. It missed every High and Medium finding in the
codebase. The shortest output (2.9 KB) for the least work.

---

## Consolidated finding map (who caught what)

Genuine duplications, with the audits that flagged each. "C"=claude,
"q6"=qwen-3.6, "qc"=qwen3-coder, "mm"=minimax, "g"=glm, "n"=nemotron.

| # | Finding | Sev | Caught by |
|---|---|---|---|
| 1 | Spec-decoding flag emission (preset.go ↔ registry.go) | High | C |
| 2 | Capabilities `[]string` built 3 ways | High | C, q6 |
| 3 | Router load/unload/list HTTP across packages | High | C |
| 4 | Router-state name-fallback lookup ×4 | High | C |
| 5 | Symlink-walk scaffold: OrphanParts ↔ ScanModels | High | **mm** |
| 6 | CSV export row-building dup | Med | C |
| 7 | Shard regex+expansion models ↔ huggingface | Med | C |
| 8 | nvidia/rocm SMI parse scaffold | Med | C |
| 9 | findMMProjInDir ↔ findMTPInDir twins | Med | **mm** |
| 10 | rocm sysfs GPU re-enumeration | Med | **mm** |
| 11 | logbus subscriber/broadcast ring buffer | Med | C |
| 12 | JSON-or-form decode boilerplate ×4 | Med | C, q6, g |
| 13 | `?ids=` parse + per-id Get ×3 | Med | C |
| 14 | Job submit-error (409/400) handling ×4 | Med | C |
| 15 | POST-JSON-body + status check | Med | C |
| 16 | rocm per-field strconv/TrimSpace idiom | Med | C |
| 17 | `.so` shared-lib copy loops ×2 | Med | C |
| 18 | Enabled-models filter loop ×3–4 | Med | C, q6, qc |
| 19 | GPU-allocation-map build ×2 | Med | **q6** |
| 20 | cssID ↔ domID sanitizer | Med | C, q6 |
| 21 | JSON-store load/persist boilerplate | Low | C |
| 22 | Find-by-ID linear scans (~24) | Low | C, qc |
| 23 | VRAM `*1.1` formula ×2 | Low | C, qc |
| 24 | `-GGUF`/`-gguf` suffix strip ×3 | Low | C, qc |
| 25 | shortModelName ↔ shortenModelName | Low | C, qc |
| 26 | URL-safe model-ID construction ×3 | Low | C |
| 27 | shortSHA ↔ safeShortSHA parameterizable | Low | **q6** |
| 28 | Builder single-build lookup ×3 (api side) | Low | q6 |
| 29 | FindOrphans/IncompleteRegistered sets ×2 | Low | C, q6 |
| 30 | DefaultDataDir/configSearchPaths per-OS | Low | C |
| 31 | Builder candidate-discovery + cmake guards | Low | C |
| 32 | env PATH/var scan-and-modify | Low | C |
| 33 | itoa/ftoa/metric* underused wrappers | Triv | C, q6, qc, mm, n, g |

**Totals:** Claude 24 · qwen-3.6 12 · qwen3-coder 6–7 · minimax 4 real · glm ~2 · nemotron ~1.
33 distinct genuine issues exist in the union; Claude covers ~24 of them, the
remaining ~9 come from qwen-3.6 (3 unique) and minimax (3 unique).

---

## Cross-cutting observations

1. **Bigger ≠ better, but precision tracks effort.** The two best audits (Claude,
   qwen-3.6) were also the longest and most specific. The worst (nemotron) was the
   shortest. Output length correlated with both coverage and willingness to read
   deep into files.

2. **Hallucinated specifics are the key risk.** Minimax's fabricated
   `bench_export.go` shard finding and GLM's inverted metrics are the two most
   harmful errors in the set — both are *confident and specific*, so a reader acting
   on them without verification would waste time or "fix" nonexistent code. Vague
   audits (nemotron) are useless but harmless; confidently-wrong audits are worse.

3. **Scope discipline varied.** GLM silently redefined "duplication" as "stylistic
   inconsistency." That's a defensible adjacent analysis, but presented as the
   assigned task it inflates the apparent finding count with non-duplication items.

4. **Complementarity is real.** No single audit was complete. Claude + qwen-3.6 +
   minimax together cover all 33 issues; any one alone misses 9–29 of them. For an
   exhaustive cleanup pass, the union (minus minimax's one fabrication) is the right
   work list.

5. **Verification is cheap and mandatory.** Every false claim here was caught in
   seconds with a single `grep`. None of the models verified their own line-number
   claims; the ones that happened to be accurate (Claude, qwen-3.6) were accurate by
   discipline, not by checking.

---

## Recommendation

If acting on these audits:

1. **Start from claude-opus-4.8's report** as the spine (highest coverage,
   zero false positives in spot-checks), prioritizing its High items #1–#4.
2. **Add qwen-3.6's #4 (shortSHA), #6 (GPU-alloc map),** and **minimax's three
   registry/monitor finds** (symlink walk, findMMProj/MTP, rocm sysfs) — the genuine
   gaps in Claude's coverage.
3. **Ignore minimax's shard-helper-in-bench_export finding** (fabricated) and
   **GLM's metrics** (inverted); salvage only GLM's form-parsing theme, already
   covered by qwen-3.6 #9.
4. **Skip nemotron** entirely.
5. Re-verify every line number before editing — they drift, and at least two models
   invented them.
