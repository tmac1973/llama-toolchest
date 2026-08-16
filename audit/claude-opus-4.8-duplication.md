# Code Duplication Audit — llama-toolchest

**Date:** 2026-06-24
**Scope:** Go source under `internal/`, `cmd/` (~18.5k LOC). Platform build-tagged
files (`manager_unix.go` / `manager_windows.go`) were checked but their per-OS
variants are legitimate and not counted as duplication. Test files and the large
`setup.sh` installer were out of scope for this pass.

This report records findings only — no code was modified. Line numbers are from the
state of the tree at audit time and may drift as the code changes.

## Summary

| # | Finding | Packages | Severity |
|---|---------|----------|----------|
| 1 | Spec-decoding flag emission duplicated 2–3× | models | **High** |
| 2 | Model capabilities `[]string` built 3 ways | api | **High** |
| 3 | Router load/unload/list HTTP logic duplicated across packages | benchmark, process | **High** |
| 4 | Router-state name-fallback lookup duplicated 4× | api | **High** |
| 5 | CSV export row-building duplicated | api | Medium |
| 6 | Shard regex + expansion duplicated across packages | models, huggingface | Medium |
| 7 | nvidia/rocm SMI parsing scaffold un-unified | monitor | Medium |
| 8 | Subscriber / log-broadcast ring-buffer machinery duplicated | builder, process | Medium |
| 9 | JSON-or-form request decoding boilerplate | api | Medium |
| 10 | `?ids=` parsing + per-id `Get` loop duplicated 3× | api | Medium |
| 11 | Job submit-error (409/400) handling duplicated 4× | api | Medium |
| 12 | Download-progress HTML rendering duplicated | api | Medium |
| 13 | Repeated POST-JSON-body + status-check block | process, benchmark | Medium |
| 14 | Per-field `strconv`+`TrimSpace` idiom in rocm.go | monitor | Medium |
| 15 | `.so` shared-library copy loops duplicated | builder | Medium |
| 16 | JSON-store load/persist boilerplate | benchmark, builder | Low |
| 17 | Linear "find by ID in slice" scans (~24×) | benchmark, builder | Low |
| 18 | VRAM `* 1.1` estimate formula duplicated | models, huggingface | Low |
| 19 | `-GGUF`/`-gguf` suffix-strip repeated 3× | models, api | Low |
| 20 | URL-safe model-ID construction duplicated 3× | huggingface, models, api | Low |
| 21 | Misc small api helpers (`cssID`/`domID`, SVG consts, SSE boilerplate, etc.) | api | Low |
| 22 | `DefaultDataDir` / `configSearchPaths` per-OS branch repeated | config | Low |
| 23 | Builder "find first existing binary" + cmake-flag guards | builder | Low |
| 24 | env PATH/var-prefix scan-and-modify | builder, process | Low |

---

## High severity

### 1. Speculative-decoding flag emission duplicated between INI writer and CLI-flag builder
- **Instance A:** `internal/models/preset.go:142-227` (`writeConfigParams`, the `switch cfg.SpecType` block plus KV-cache/mmproj/sampling emission)
- **Instance B:** `internal/models/registry.go:226-322` (`EffectiveFlagsFor`, the `switch c.SpecType` block)
- **Third partial copy:** `internal/models/registry.go:152-173` (`SamplingOverrides`) re-walks the same six sampling fields

Both functions walk identical `ModelConfig` fields in identical order and emit identical
llama.cpp flag semantics — `draft` → `draft-simple` with draft-resource overrides,
`draft-mtp` with the same nested `MtpPath`/draft-resource logic, the
`ngram-mod`/`ngram-simple`/`ngram-map-k*`/`ngram-cache` cases with their `spec-` prefix
construction, plus the KV-cache K/V quant pair, mmproj, and six sampling parameters. They
differ only in output syntax (`key = value` INI lines vs. `--key value` slice appends).
The comments are copy-pasted verbatim. This is the most fragile duplication in the
codebase: any llama.cpp flag rename must be edited in lockstep across 2–3 copies.

**Remedy:** Extract one canonical flag source — a function returning an ordered
`[]struct{key, value string}` (or a visitor taking an `emit(key, value)` callback)
consumed by both `writeConfigParams` (joins with ` = `) and `EffectiveFlagsFor`
(prepends `--`). The sampling block should draw from the same source as `SamplingOverrides`.

### 2. Model capabilities `[]string` built three different ways
- `internal/api/capabilities.go:39-61` (`buildCapabilities`, as booleans)
- `internal/api/models.go:174-185` (`handleModelInfo`)
- `internal/api/v1models.go:13-25` (`openAIModel`)

The same capability-derivation logic is copy-pasted: `IsEmbeddingModel(m.ModelID) ||
IsEmbeddingModel(m.ID)` → "embedding"/"chat", `m.SupportsTools` → "tools",
`m.HasBuiltinVision || (cfg != nil && cfg.MmprojPath != "")` → "vision". Verified
byte-for-byte identical between `models.go` and `v1models.go`; `buildCapabilities`
reimplements the same predicates. The bare embedding predicate alone recurs at
capabilities.go:37, models.go:17/39/175, server.go:502, service.go:589, v1models.go:14.

**Remedy:** Extract `modelCapabilities(m, cfg) []string` and an `isEmbeddingModel(m)`
wrapper; have all three sites and `buildCapabilities` reuse them.

### 3. Router load / unload / list HTTP logic duplicated across two packages
- `internal/benchmark/runner.go:198-295` — `ensureModelLoaded`, `unloadAllModels`, `listLoadedModels`, `unloadModel`
- `internal/process/manager.go:243-315` — `LoadModel`, `UnloadModel`, `ListModels`

Both packages independently implement the same router HTTP protocol: POST `/models/load`
and `/models/unload` with `{"model": name}` bodies, and GET `/models` decoding a status
struct. The "already loaded" tolerance, the `bytes.NewReader(body)` POST pattern, and the
`"loaded"`/`"loading"` status filtering are reimplemented. The two decode structs
(`runner.go:266-271` vs `manager.go:22-29`) describe the same endpoint with different
shapes. A router API change forces edits in two unrelated packages.

**Remedy:** Extract a shared router client (e.g. `internal/router`, or a method set on
`process.Manager`) exposing `LoadModel`/`UnloadModel`/`ListLoadedModels`, and have the
benchmark runner depend on it instead of re-rolling raw HTTP.

### 4. Router-state name-fallback lookup duplicated across four sites
- `internal/api/models.go:357-364` (`renderModelCard`): `routerName → m.ID → PublicName()`
- `internal/api/server.go:510-517` (`handleDashboard`): identical 3-step fallback
- `internal/api/proxy.go:178-188` (`lookupRouterState`): same names vs ListModels aliases
- `internal/api/models.go:332-347` (`routerKnownStates`) builds the map two of these re-implement inline

The "look up a model's router status by trying RouterName, then m.ID, then PublicName"
pattern is repeated verbatim. Closely related: three separate loops over
`s.process.ListModels()` populate a name→status map from `rm.ID`/`rm.Model`/`rm.Aliases`
(`models.go:332-347`, `server.go:484-497`, `proxy.go:178-187`).

**Remedy:** One shared map-builder (with an optional "only-resident" filter) plus a
`routerStateFor(known, m)` helper used by `renderModelCard` and `handleDashboard`.

---

## Medium severity

### 5. CSV export row-building largely duplicated between cells and summary
- `internal/api/bench_export.go:76-153` (`writeCSVCells`)
- `internal/api/bench_export.go:155-203` (`writeCSVSummary`)

Both repeat the identical `jobName` lookup (93-96 vs 169-172), the identical
`build := run.EffectiveBuild()` + leading-column block (JobID, jobName, ID, CreatedAt UTC,
ModelID, ModelName, Quant, build.ID/Profile/GitRef, Preset). The two header slices share
their first 11 columns verbatim.

**Remedy:** Extract `baseRunColumns(run, jobs) []string` and a shared leading-header constant.

### 6. Shard regex + shard-expansion logic duplicated across packages
- `internal/models/registry.go:1147-1182` — `shardRe` (`-(\d{5})-of-(\d{5})\.gguf$`), `isNonFirstShard`, `findShards`
- `internal/huggingface/client.go:189-261` — `shardPattern` (`^(.+)-(\d{5})-of-(\d{5})\.gguf$`), `groupShards`, `ExpandShards`

Two near-identical regexes parse the same `-NNNNN-of-NNNNN.gguf` naming, and both
`ExpandShards` and `findShards` reconstruct filenames with the same
`fmt.Sprintf("%s-%05d-of-%05d.gguf", ...)` loop. Two regex variants for one on-disk
convention risk drifting apart.

**Remedy:** Move the shard regex + a shared `ExpandShardNames`/`ParseShard` into one place
(the `models` package, already imported by `huggingface`) and have both callers use it.

### 7. nvidia.go and rocm.go share an un-unified GPU-collection scaffold
- `internal/monitor/nvidia.go:25-59` (`Collect`)
- `internal/monitor/rocm.go:35-100` (`collectROCmSMI`)

Both run a `*-smi` command, split output into trimmed rows, `strconv`/`ParseFloat` each
metric wrapped in `strings.TrimSpace`, and assemble a `GPUInfo`. The per-field parse idiom
repeats ~6× per function. Acquisition paths differ (fixed positional CSV vs header-indexed
CSV + sysfs), so full merge isn't warranted, but the row scaffold and parse helpers are shared.

**Remedy:** Add `atoiTrim(s) int` / `parseFloatTrim(s) float64` and a shared row-splitter;
both backends call them.

### 8. Subscriber / log-broadcast ring-buffer machinery duplicated
- `internal/builder/builder.go:148-200` (`SubscribeLogs`/`UnsubscribeLogs`/`broadcastLog` + history trim)
- `internal/process/manager.go:318-361` (`Subscribe`/`Unsubscribe`/`broadcast`)

Both implement the same pub/sub-with-replay: a `map[chan string]struct{}` of subscribers,
a bounded `[]string` history, non-blocking `select { case ch <- line: default: }` fan-out,
replay on subscribe. The non-blocking send idiom appears ≥5× total (builder.go:160-164,
194-198, 398-402, 702-705; manager.go:322-325, 356-359).

**Remedy:** Extract a reusable `logbus` type and embed it in both `Builder` and `Manager`;
optionally a `trySend(ch, line)` helper for the bare non-blocking send.

### 9. JSON-or-form request decoding boilerplate duplicated
- `internal/api/build.go:184-208` (`handleTriggerBuild`)
- `internal/api/hf.go:110-120` (`handleHFDownload`)
- `internal/api/settings.go:51-102` (`handleUpdateSettings`)
- `internal/api/service.go:702-825` (`handleUpdateModelConfig`)

All four branch on `Content-Type == "application/json"` →
`json.NewDecoder(r.Body).Decode(...)` (with near-identical 400-on-error) else
`r.ParseForm()` + `FormValue`.

**Remedy:** A `decodeJSONBody(w, r, &v) bool` helper for the JSON arm.

### 10. `?ids=` parameter parsing + per-id `bench.Get` loop duplicated
- `internal/api/bench.go:330-343` (`handleBatchDeleteBenchmarks`)
- `internal/api/bench.go:348-378` (`handleExportBenchmarks`)
- `internal/api/bench.go:401-415` (`handleCompareBenchmarks`)

Each repeats: read `ids` query, 400 if empty, split on `,`, trim, skip empties, then
`s.bench.Get(id)`.

**Remedy:** `parseIDsParam(r) ([]string, error)` + `runsForIDs(ids)`.

### 11. Job submit-error and not-found disambiguation duplicated
- 409/400 (`ErrJobAlreadyRunning`): `bench.go:293-300`, `bench_jobs.go:115-122`, `bench_jobs.go:164-171`, `bench_jobs.go:342-350`
- "synthetic" → 400-else-404: `bench_jobs.go:155-162`, `bench_jobs.go:319-327`
- Identical job create/update validation: `bench_jobs.go:87-99` vs `138-150`

The same `errors.Is(err, ErrJobAlreadyRunning)` 409/400 block appears 4×; the brittle
`strings.Contains(err.Error(), "synthetic")` 400/404 split appears 2×; the
decode + `name is required` + `model_ids/build_ids/presets required` validation is verbatim 2×.

**Remedy:** `writeJobSubmitError(w, err)` helper; a typed/sentinel error for "synthetic";
`decodeAndValidateJobRequest(w, r)`.

### 12. Download-progress HTML rendering duplicated
- `internal/api/hf.go:176-198` (`handleHFDownloadProgress`, SSE branch)
- `internal/api/hf.go:218-231` (`handleHFActiveDownloads`)

Both compute `pct` (guarding TotalBytes>0), `speedMB`, `downloadedGB`/`totalGB`, and emit a
`<progress>` + `%.1f / %.1f GB (%.1f MB/s) — %.0f%%` line.

**Remedy:** Extract `downloadProgressHTML(status)`.

### 13. Repeated HTTP POST-with-JSON-body + status-check block
- `internal/process/manager.go:253-264` (`LoadModel`), `:277-288` (`UnloadModel`)
- `internal/benchmark/runner.go:203-230` (`ensureModelLoaded`)

Identical sequence: marshal `{"model": name}`, POST, `defer resp.Body.Close()`, on non-200
read body and format `"... HTTP %d: %s"`. `LoadModel`/`UnloadModel` differ only in URL path
suffix and error verb. (Subset of finding #3 but worth its own helper.)

**Remedy:** `postModel(url, path, name string) error` parameterized on path + label.

### 14. Per-field `strconv`+`TrimSpace` lookup idiom repeated within rocm.go
- `internal/monitor/rocm.go:63-87` — five consecutive
  `if i, ok := colIdx[...]; ok && i < len(fields) { ...strconv...(TrimSpace(fields[i]))... }` blocks.

**Remedy:** A `getField(name) (string, bool)` closure over `colIdx`/`fields` plus the
parse helpers from finding #7 collapses each to one line.

### 15. Near-identical `.so` shared-library copy loops
- `internal/builder/builder.go:507-519` (scanning `buildDir/lib`)
- `internal/builder/builder.go:521-533` (scanning `bin/`)

Two back-to-back loops, byte-for-byte identical except source directory: same
`os.ReadDir`, same `HasSuffix(name,".so") || Contains(name,".so.")` filter, same
`copyFile` + `sendLog("    Installed lib: %s")`.

**Remedy:** `copySharedLibs(srcDir, outDir, sendLog)` called twice.

---

## Low severity

### 16. JSON-store load/persist boilerplate
- `internal/benchmark/benchmark.go:644-651, 762-775` (`benchmarkPath`/`load`/`persist`)
- `internal/builder/builder.go:722-754` (`buildsPath`/`loadBuilds`/`saveBuilds`)

`persist`/`saveBuilds` are structurally identical: `os.MkdirAll(dir, 0o755)`,
`json.MarshalIndent(x, "", "  ")`, `slog.Error` on marshal failure, `os.WriteFile(path, data, 0o644)`.

**Remedy:** A generic `saveJSON(path, v)` / `loadJSON(path, v)` helper.

### 17. Linear "find by ID in slice" scans (~24 sites)
- benchmark.go: `Get` (339), `Save` (360), `Delete` (375), `GetJob` (406), `SaveJob` (419), `DeleteJob` (441), `UpdateJobDefinition` (494), `hasJobLocked` (753)
- builder.go: `BuildStatus` (213), `Delete` (367), `finishBuild` (566), `HasBuild` (670), inline scans in `Build` (320-344)

The `for i := range slice { if slice[i].ID == id { ... } }` pattern repeats ~12× per store.
Individually trivial; collectively a smell.

**Remedy:** A generic `indexByID[T](items, id, key)` (Go 1.18+) collapses most.

### 18. VRAM `* 1.1` estimate formula duplicated
- `internal/models/vram.go:18-20` (`EstimateVRAM`)
- `internal/huggingface/client.go:185-187` (`estimateVRAM`)

Same `sizeBytes * 1.1 / (1024^3)` formula; the HF client re-implements a private copy
(differs only by the +0.2 `vramOverheadGB`) despite already importing `models`.

**Remedy:** Have `huggingface.estimateVRAM` delegate to `models.EstimateVRAM`.

### 19. `-GGUF` / `-gguf` suffix-strip repeated
- `internal/models/registry.go:25-26` (`OrgAndBase`), `internal/models/gpu_assign.go:270-271` (`shortModelName`), `internal/api/jobs_env.go:189-190`

Verbatim `TrimSuffix(x,"-GGUF"); TrimSuffix(x,"-gguf")` in three places.

**Remedy:** One exported `models.StripGGUFSuffix(name)`.

### 20. URL-safe model-ID construction duplicated
- `internal/huggingface/downloader.go:105-107` (`Start`)
- `internal/models/registry.go:865-867` (`ScanModels`)
- `internal/api/hf.go:369`

Same `ReplaceAll(modelID,"/","--")` + `TrimSuffix(filename,".gguf")` scheme reconstructed
independently; divergence would split downloader vs scanner IDs.

**Remedy:** A shared `DeriveModelID(modelID, filename)`.

### 21. Miscellaneous small api helpers
- `cssID` (`server.go:129-138`) and `domID` (`hf.go:239-248`): byte-identical `strings.Map` sanitizers.
- Enabled-models filter loop duplicated: `bench.go:454-460` vs `bench_jobs.go:268-273` → `s.enabledModels()`.
- Inline SVG icon literals declared twice in one function: `server.go:527-529` and `:556-557` → package-level consts.
- Orphan/incomplete membership scan: `service.go:493-506` re-scans `FindOrphans()`/`IncompleteRegistered()` for one id where `models.go:433-441` already builds the sets.
- `NewSSEWriter(w)` + 500-on-error boilerplate: `bench_jobs.go:368-372`, `hf.go:165-169`, `monitor.go:12-16` (already encapsulated by `sse.go` `StreamLines` for log streaming) → `mustSSE(w)`.

### 22. `DefaultDataDir` and `configSearchPaths` repeat per-OS branch logic
- `internal/config/paths.go:13-32` (`DefaultDataDir`)
- `internal/config/paths.go:64-89` (`configSearchPaths`)

Both `switch runtime.GOOS` over darwin/windows/default with the same base-dir rules;
`configSearchPaths` is essentially `DefaultDataDir`'s dirs with a filename appended
(plus an `/etc` system path).

**Remedy:** Factor `userBaseDirs() []string` (judgment call — the two have slightly
different fallbacks).

### 23. Builder candidate-discovery and cmake-flag guards
- "find first existing binary among candidates": `builder.go:482-491`, `:536-548`, `:823-834` (`findCUDAHostCompiler`), `:844-848` (`findNVCC`) → `firstExisting(candidates) string`.
- "already-set cmake flag" guard: `builder.go:435-447` (CMAKE_CUDA_COMPILER) and `:448-461` (CMAKE_CUDA_HOST_COMPILER) repeat probe → scan `cmakeArgs` for `-DFLAG=` → conditional append + log → `appendCMakeFlagIfUnset(...)`.

### 24. env PATH/var-prefix scan-and-modify
- `internal/builder/builder.go:854-874` (`prependPath`)
- `internal/process/manager.go:470-485` (`appendLibraryPath`, Windows branch), `:461-468` (`pinCUDADeviceOrder`)

Three functions across two packages walk `env []string`, test `HasPrefix(kv,"PREFIX=")`,
and rewrite or append. `prependPath` and the Windows `appendLibraryPath` branch are the
same PATH-prepend operation.

**Remedy:** A shared `internal/envutil` with `prependToPathVar` / `ensureEnvDefault`.

---

## Non-findings (checked, deliberately not flagged)

- `manager_unix.go` / `manager_windows.go`: legitimate build-tagged `terminate()` variants — no shared logic copied.
- `respond.go` (`respondJSON`/`respondHTML`/`isHTMX`) is already the correct shared abstraction — no duplication.
- `proxy.go` / `middleware.go` JSON error envelopes share a shape but differ enough in `type`/`code`/timing that consolidation is marginal.
- `detect.go` per-backend `detectROCm`/`detectCUDA`/`detectVulkan` share only a success/failure tail (`finalizeBackend`-worthy, Low); parsing differs per tool.
- TPS summaries `stats.go:6-35` (`ComputeSummary`) vs `benchy.go:145-179` (`summarizeBenchy`) operate on different input shapes — not true duplication.
- `head_dim = nEmbd/nHead` recurs (gguf.go:57-62, vram.go:86-89, gguf.go:248-251) but each guards/uses it differently — borderline.
- `resolveModelGPUs` vs `ResolveGPUAssign` (gpu_assign.go) both parse `tensor-N` but outputs/edge-cases differ.
- KV-cache and GGUF type-size lookup tables each appear exactly once — no duplicate tables.

## Recommended priority

The highest-leverage fixes, in order: **#1** (spec-decoding flags — most fragile,
silent-breakage risk), **#3** (router HTTP client — cross-package protocol drift),
**#2 / #4** (api capabilities and router-state lookups — many call sites), then the
Medium api boilerplate cluster (#9–#12) which is mechanical and low-risk to extract.
