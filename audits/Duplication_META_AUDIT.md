# Audit report: Code Duplication Audit (cleaned)

Line numbers below were re-validated against the current source and corrected. Overlapping
findings from the original 10-run sample were merged (original #7→#1, #21→#12, #22→#4, #25→#24,
#26/#27→#13), and refuted sub-claims were struck out. **23 distinct verified findings.**

> Every citation was re-checked against HEAD. Where the original audit's line numbers were stale
> they have been replaced with the current ranges. Sub-claims that did not survive re-verification
> are marked ~~struck through~~ with a correction.

## Remediation (2026-07-03, branch `refactor/dedup-audit-fixes`)

All findings below were fixed except #21 (see note). Note: the cited line numbers in the
findings describe the code **before** remediation.

| # | Fix |
|---|-----|
| 1, 2 | `specDecodingParams()` in new `internal/models/specflags.go` returns structured name/value pairs; `writeConfigParams` formats them as INI, `EffectiveFlagsFor` as `--flags`. Draft-resource and draft-sampling blocks are shared sub-helpers. |
| 3 | `handleDashboard` now calls `routerKnownStates()` + new `routerStateFor()` lookup helper instead of its inline block. |
| 4, 12, 14, 15 | New `(*Model).IsEmbedding()`, `(*Model).HasVision(cfg)`, `(*Model).Capabilities(cfg)` in the models package; all nine double-call sites and both capability-list builders now delegate. `filterModels()` unifies the two list-filter loops. |
| 5 | New generic `internal/broadcast.Broadcaster[T]` (ring history, non-blocking fan-out, race-tested); adopted by `builder.Builder` (per-build log streams), `process.Manager` (router logs), and `huggingface.download` (status, history size 1 replaces `lastStatus`). |
| 6 | `builder.ApplyOptionOverrides()` shared by `Builder.Build` and api `effectiveCMakeFlags`. Intentional behavior fix: a partial non-nil overrides map now applies option defaults for unspecified options, so the built flags always match the UI preview. |
| 7 | `downloadProgressHTML()` shared by the SSE progress stream and active-downloads panel. |
| 8 | `parseEnumParam()`; both normalizers are now one-liners. |
| 9 | `(*Server).resolveActiveBuild()` shared by `startRouter` and `handleStartBenchmark`; the benchmark path now also validates `Status == BuildStatusSuccess` (closing the failed-build hole the audit flagged). |
| 10 | `(*Builder).Find(id)`; all five linear-search sites delegate. |
| 11 | `huggingface.SafeModelID()` / `SafeFileID()` in new `names.go`; all six sites delegate. |
| 13 | `cssID` template func now delegates to `domID`. |
| 16, 17 | `(*Server).resolveRouterModel(rm)` shared by `handlePS` and `renderLoadedModelsJSON` (handlePS now matches by public name/router name too, not just registry ID). `renderLoadedModelsHTML` and `lookupRouterState` left as-is: different direction/output, no shared body to extract. |
| 18 | New `respondJSONStatus()`; `writeProxyError` (now code-parameterized) and the v1models 404 path delegate. `writeJSONExport` left as-is (needs `SetIndent` + `Content-Disposition`). |
| 19 | `parseTensorAssign()` shared by `ResolveGPUAssign` and `resolveModelGPUs`. |
| 20 | `listAMDGPUDirs()` + `readVRAMFromDir()`; `collectSysfs` no longer re-globs per GPU. |
| 21 | **Not fixed (intentional).** The two HTTP clients differ in timeout (5s vs 5min) and method (Get vs Do); a shared client isn't clearly right. Low value, per the audit itself. |
| 22 | `models.ShortModelName()` exported; api `shortenModelName` delegates. |
| 23 | `platformAppDirs()` shared by `DefaultDataDir` and `configSearchPaths`; the `/etc` candidate is still appended unconditionally on Linux (preserves the issue #61 fix). |

## Verified findings

### 1. Identical speculative-decoding flag emission block in preset.go and registry.go

`internal/models/preset.go:145-226 (writeConfigParams spec-type switch); internal/models/registry.go:229-322 (EffectiveFlagsFor spec-type switch)` · severity: high

The spec-type switch (draft, draft-mtp, ngram-mod, ngram-simple/ngram-map-k/ngram-map-k4v, ngram-cache) is duplicated across two files. `writeConfigParams` (preset.go, func starts line 100) writes INI lines to a `*strings.Builder`; `EffectiveFlagsFor` (registry.go, func starts line 179) appends `--flag value` pairs to a `[]string`. Both cover the identical case set and emit the same logical flag set per spec type (`--spec-type`, `--model-draft`, `--ctx-size-draft`, `--gpu-layers-draft`, `--device-draft`, `--n-cpu-moe-draft`, `--cache-type-{k,v}-draft`, `--spec-draft-n-{max,min}`, `--spec-draft-p-min`, `--spec-ngram-mod-n-{max,min}`). Only the output medium differs. The blocks are ~82 (preset) and ~94 (registry) lines — larger than the "~40 line" original estimate. Unify behind a shared helper returning structured flag data (`[]struct{Flag, Value string}`) that each caller formats. _(Original finding #7 reported this same duplication and is merged here.)_

### 2. Draft resource override block copied for two spec types in registry.go

`internal/models/registry.go:252-266 ("draft" case) and 278-292 ("draft-mtp" case)` · severity: high

Inside `EffectiveFlagsFor`, the 5-field draft-resource override sequence (DraftCtxSize, DraftGPULayers, DraftDevice, DraftCPUMoE, DraftKVCacheQuant) appears in both the `"draft"` case and the `"draft-mtp"` case:

    if c.DraftCtxSize > 0    { parts = append(parts, "--ctx-size-draft", ...) }
    if c.DraftGPULayers > 0  { parts = append(parts, "--gpu-layers-draft", ...) }
    if c.DraftDevice != ""   { ... }
    if c.DraftCPUMoE > 0     { ... }
    if c.DraftKVCacheQuant != "" { ... }

The `draft-mtp` copy is nested one level deeper (inside `if c.MtpPath != "" && !c.MtpDisabled`), so it is the same logic with extra indentation rather than byte-identical. Extract `appendDraftFlags(parts *[]string, c *ModelConfig)`.

### 3. routerKnown state-building block in handleDashboard vs routerKnownStates()

`internal/api/server.go:490-504 (handleDashboard inline block); internal/api/models.go:330-348 (routerKnownStates())` · severity: high

Both query `s.process.ListModels()` and iterate ID/Model/Aliases to build a name→status map, filtered by enabled registry models. ~~The inline block is a copy of routerKnownStates().~~ **Correction: not a literal copy.** The `handleDashboard` block builds *two* maps — `routerKnown map[string]bool` (name/model/alias → true) and `loadedState map[string]string` (only "loaded"/"loading") — whereas `routerKnownStates()` returns a single `map[string]string` of all name→status. The duplication is thematic (same source, same iteration), not verbatim; unifying would require `routerKnownStates()` to also return the `loadedState` sub-map.

### 4. Capabilities logic in openAIModel() vs buildCapabilities()

`internal/api/v1models.go:13-25 (openAIModel); internal/api/capabilities.go:37 + 49-51 (buildCapabilities)` · severity: high

Both derive the same three predicates: serving mode via `models.IsEmbeddingModel(...)`, `tools` via `m.SupportsTools`, and `vision` via `m.HasBuiltinVision || (cfg != nil && cfg.MmprojPath != "")`. ~~Same ~10 lines verbatim.~~ **Correction: same predicates, different output contract.** `openAIModel` appends strings (chat/embedding/tools/vision) to a `[]string`; `buildCapabilities` emits discrete boolean map keys (line 37 embedding check; lines 49-51 vision/tools). A new capability must be added in both, but they are not drop-in shareable — a shared helper would need to expose the predicates, with each caller formatting its own output. _(Closely related to #15, the openAIModel↔handleModelInfo duplication, which IS a true copy.)_

### 5. Broadcast pattern (subscribe/broadcast/unsubscribe) across three packages

`internal/builder/builder.go (SubscribeLogs 148-173, UnsubscribeLogs 176-182, broadcastLog 184-200); internal/process/manager.go (Subscribe 318-330, Unsubscribe 333-337, broadcast 346-361); internal/huggingface/downloader.go (broadcast 37-47, subscribe 49-62, unsubscribe 64-68)` · severity: high

All three implement the same mutex + `map[chan T]struct{}` + non-blocking `select`-send fan-out. Builder uses `logHistory map[string][]string` (ring size 2000, const at builder.go:48); Manager uses `logHistory []string` (ring size 500, const at manager.go:59). **Correction to original "ring buffer in all three":** the `download` struct has *no* ring buffer — it replays a single `lastStatus` value on subscribe (downloader.go:39, 54-59). So the ring-buffer history is shared by two of three; the broadcast/subscribe skeleton is shared by all three. Unify behind a generic `Broadcaster[T]` with optional history size. _(Original finding #22 reported the broadcast methods specifically and is merged here.)_

### 6. Option-override + CMake flag assembly loop in builder.Build vs api.effectiveCMakeFlags

`internal/builder/builder.go:252-267 (Build method); internal/api/build.go:102-125 (effectiveCMakeFlags)` · severity: high

Both iterate `ProfileOptions(profile)`/options and fold override-or-default into a flags map: for each opt, use `optionOverrides[opt.Flag]` if set, else `opt.Default`, then set the flag "ON". ~~Cited builder.go:205-222 (wrong — that region is DuplicateBuildError).~~ **Correction:** the real Build loop is 252-267. Not a literal copy: `Build` mutates a copy of `prof.CMakeFlags` in place with no string output; `effectiveCMakeFlags` builds a fresh map and returns a sorted `-D…` string. The enable-resolution logic is the duplicated part.

### 7. Download-progress HTML rendering block in two hf.go handlers

`internal/api/hf.go:176-198 (handleHFDownloadProgress); internal/api/hf.go:219-230 (handleHFActiveDownloads)` · severity: medium

Both handlers compute `pct`, `speedMB`, `downloadedGB`, `totalGB` with byte-identical arithmetic (compute at 176-182 and 219-225 respectively), then emit the same `<progress>`-bar HTML. The only semantic difference is SSE wrapping (download-progress) vs. raw HTML (active-downloads). Extract `renderDownloadProgress(status DownloadStatus) string`.

### 8. parseExportFormat and parseExportScope are near-identical switch normalizers

`internal/api/bench_export.go:229-239 (parseExportFormat); internal/api/bench_export.go:241-251 (parseExportScope)` · severity: medium

Both are three-case switches: empty→default value, valid value→pass-through, default→`fmt.Errorf`. They differ only in the query-param name (`"format"` vs `"scope"`), the two valid literals, and the error text. A shared `parseEnumParam(r, key, emptyDefault string, valid []string) (string, error)` would eliminate the divergence risk.

### 9. Active-build fallback in handleStartBenchmark duplicates startRouter (not EnsureBuildActive)

`internal/api/bench.go:267-272 (handleStartBenchmark); internal/api/service.go:403-415 (startRouter)` · severity: medium

~~Original finding #10 paired handleStartBenchmark with jobs_env.go EnsureBuildActive — REFUTED.~~ `EnsureBuildActive` (jobs_env.go:82-101) takes a `buildID` parameter, does a List()-scan-by-ID, validates `Status == BuildStatusSuccess`, and never calls `LatestSuccessfulBuild()`. **Correction:** the actual twin of `handleStartBenchmark`'s "check cfg.ActiveBuild, else fall back to `builder.LatestSuccessfulBuild()`" logic is `startRouter` in service.go:403-415 — and the code comment at bench.go:266 even says "the same way startRouter does." The `handleStartBenchmark` path lacks the `Status == BuildStatusSuccess` validation that startRouter/EnsureBuildActive have, so it could launch a failed build if ActiveBuild points to one. Centralize build resolution (e.g. `registry.ActiveBuildOrLatest()`).

### 10. builder.List() search-by-ID repeated across 5 sites

`internal/api/build.go:291-297 (handleBuildInfo); internal/api/service.go:404-409 (startRouter); internal/api/jobs_env.go:37-42 (CheckBuildRunnable); internal/api/jobs_env.go:89-95 (EnsureBuildActive); internal/api/bench.go:26-39 (builderResolver)` · severity: medium

Five sites run essentially identical `for _, b := range s.builder.List() { if b.ID == … }` linear searches (startRouter adds `&& b.Status == success`). ~~Original finding #11 listed a sixth site, bench_jobs.go handleJobForm (~278-283) — REFUTED as a match:~~ that loop iterates all builds filtering `b.Status != "success"` to populate a dropdown; it is a List() iteration but not a search-by-ID. A `(*Builder).Find(id) (*BuildResult, bool)` helper would eliminate the five duplicate loops.

### 11. `strings.ReplaceAll(modelID, "/", "--")` copy-pasted across api and huggingface

`internal/huggingface/downloader.go:105, 254 (modelID); internal/huggingface/downloader.go:106 (filename variant); internal/api/hf.go:313, 347 (modelID); internal/api/hf.go:369 (filename variant)` · severity: medium

The HF-repo→dirname sanitization `strings.ReplaceAll(modelID, "/", "--")` appears at downloader.go:105,254 and hf.go:313,347. The filename variant `strings.ReplaceAll(strings.TrimSuffix(filename, ".gguf"), "/", "--")` appears at downloader.go:106 and hf.go:369. ~~(Original cited 107, 255, 371 — corrected to 106, —, 369.)~~ A shared `SafeModelID(modelID string) string` helper would provide the single canonical definition.

### 12. IsEmbeddingModel double-call idiom repeated across api/ files

`internal/api/v1models.go:14; internal/api/capabilities.go:37; internal/api/models.go:17; internal/api/models.go:39 (negated); internal/api/models.go:175; internal/api/server.go:509; internal/api/service.go:589` · severity: medium

The expression `models.IsEmbeddingModel(m.ModelID) || models.IsEmbeddingModel(m.ID)` (checking both the HF repo ID and the internal registry ID, because legacy records may have only one populated) appears at all seven sites — models.go:39 is the negated `!… && !…` form, and service.go:589 uses the variable `model` rather than `m` but is otherwise identical. All cited lines are accurate against current source. A single `IsEmbeddingModelFor(m *Model) bool` helper would eliminate the copies. _(Original finding #21 reported a 6-site subset of this and is merged here.)_

### 13. domID and cssID are identical character-whitelisting functions

`internal/api/hf.go:239-248 (domID, package-level func); internal/api/server.go:129-138 (cssID, anonymous func in the template FuncMap)` · severity: medium

Both use `strings.Map` keeping `[a-z A-Z 0-9 - _]` and returning `'_'` for every other rune. The character sets are byte-for-byte identical (both explicitly permit `'-'`). `cssID` is registered under key `"cssID"` in `parseTemplates`; `domID` is a standalone func. Unify behind a single shared `makeIdentifier(s string)` helper. ~~Original finding #27 claimed "domID additionally allows '-' but cssID does not" — REFUTED: both permit '-' (server.go:132 and hf.go:242 both list `r == '-'`).~~ _(Original findings #26 and #27 reported this same pair and are merged here.)_

### 14. Model-filter loop duplicated in handleListModels and handleListEmbeddingModels

`internal/api/models.go:14-20 (handleListEmbeddingModels); internal/api/models.go:35-42 (handleListModels)` · severity: medium

Both call `s.registry.List()` then a for-range that builds an embedding/chat split. The loops are De Morgan negations of each other (line 17 `IsEmbeddingModel(ModelID) || IsEmbeddingModel(ID)` to collect embedding vs. line 39 `!… && !…` to collect non-embedding); otherwise structurally identical. Unify into a single filter helper.

### 15. Capabilities []string builder duplicated in handleModelInfo and openAIModel

`internal/api/models.go:173-185 (handleModelInfo); internal/api/v1models.go:13-25 (openAIModel)` · severity: medium

Both build a capabilities `[]string`: embedding/chat via `IsEmbeddingModel`, then `tools` via `SupportsTools`, then `vision` via `HasBuiltinVision || (cfg != nil && cfg.MmprojPath != "")`. This is a near-verbatim copy (unlike #4, which shares only the predicates). ~~Original cited models.go:160-171 and v1models.go:17-33 — corrected.~~ Extract `models.Capabilities(m *Model, cfg *ModelConfig) []string`.

### 16. Router model listing logic across handlePS, renderLoadedModelsJSON, renderLoadedModelsHTML

`internal/api/service.go:104-146 (handlePS); internal/api/service.go:239-296 (renderLoadedModelsJSON); internal/api/service.go:299-327 (renderLoadedModelsHTML)` · severity: medium

All three guard on `process.IsRunning()`, call `process.ListModels()`, and iterate per router model, matching each by ID/alias to registry model/config. ~~Original cited 75-120 / 167-213 / 227-255 — all wrong; corrected above.~~ Each builds a different payload (psModel with VRAM at 118-141; loadedModel with capabilities at 268-293; HTML fragment at 315-325), so this is a repeated skeleton rather than copy-paste bodies. A shared helper returning a unified `[]loadedModel` would collapse the common lookup.

### 17. Auto-load / router-state lookup near-duplicate

`internal/api/service.go:104-146 (handlePS, alias loop 118-141); internal/api/proxy.go:170-189 (lookupRouterState, loop 178-187)` · severity: medium

Both range over `process.ListModels()` and match on ID / Model / Aliases. `handlePS` resolves via `registry.Get` per alias; `lookupRouterState` compares against routerName/m.ID/PublicName and returns a `knownToRouter` bool alongside the state. ~~Original claimed the pattern appears "~3 times in proxy.go" — REFUTED: proxy.go has it once (lookupRouterState); the other ListModels+alias iterations are in service.go (renderLoadedModelsJSON 268-291, renderLoadedModelsHTML 315-325), i.e. finding #16.~~ Unify the alias-lookup as a helper on `process.Manager`.

### 18. JSON+Content-Type response idiom inlined instead of using respondJSON

`internal/api/respond.go:14-17 (respondJSON def); internal/api/proxy.go:41-49 (ErrorHandler inline); internal/api/proxy.go:214-224 (writeProxyError); internal/api/bench_export.go:49-55 (writeJSONExport); internal/api/v1models.go:83-85 (redundant Content-Type before respondJSON)` · severity: medium

`respondJSON` sets `Content-Type: application/json` and encodes. Four callers reimplement or partially duplicate it: the proxy ErrorHandler and `writeProxyError` set the header and call `json.NewEncoder(...).Encode` directly; `v1models.go:83` redundantly sets Content-Type immediately before calling `respondJSON` at 85 (which sets it again). `bench_export.go` `writeJSONExport` is a weaker candidate — it also sets `Content-Disposition` and uses `SetIndent`, so it can't simply delegate. A dedicated `writeError(w, status, msg)` helper would consolidate the error paths.

### 19. tensor-N parse prefix duplicated in gpu_assign.go

`internal/models/gpu_assign.go:119-136 (ResolveGPUAssign); internal/models/gpu_assign.go:235-247 (resolveModelGPUs)` · severity: medium

Both share the parse prefix: `strings.HasPrefix(assign, "tensor-")` + `fmt.Sscanf(assign, "tensor-%d", &n)` + `n > 0` guard. ~~Original claimed the `make([]int, n); for i := range gpus { gpus[i] = i }` slice construction is also shared — partially REFUTED:~~ that slice body appears only in `resolveModelGPUs` (241-244). `ResolveGPUAssign` instead builds a `[]string` tensor-split of length `numGPUs` and uses different overflow handling (`n >= numGPUs` → no split, vs `n > numGPUs` → clamp). Only the parse prefix is duplicated; extract `parseTensorAssign(assign string) (n int, ok bool)`.

### 20. readVRAMSysfs duplicates the sysfs card-enumeration loop from collectSysfs

`internal/monitor/rocm.go:106-147 (collectSysfs enumeration; glob 106, loop 108-147); internal/monitor/rocm.go:155-181 (readVRAMSysfs; glob 157, loop 159-179)` · severity: medium

Both call `filepath.Glob("/sys/class/drm/card[0-9]*/device/vendor")`, read the vendor file, filter for AMD (`"0x1002"`), and index-walk to the target GPU. Critically, `readVRAMSysfs(idx)` is **called from inside collectSysfs's own loop at line 123** (and from `collectROCmSMI` at line 90), so collectSysfs enumerates AMD cards and then re-enumerates them from scratch per GPU — redundant work. ~~Original finding #25 additionally claimed readVRAMSysfs "starts idx from 0 each call so returns wrong results for non-zero gpuIdx" — REFUTED:~~ the index-walk (`if idx != gpuIdx { idx++; continue }`) correctly reaches the Nth AMD GPU; the duplication is wasteful but not incorrect. Extract `listAMDGPUBuses() ([]string, error)` so enumeration runs once. _(Original finding #25 is merged here.)_

### 21. Connection-test HTTP client instantiated similarly in settings and benchmark runner

`internal/api/settings.go:121-123 (handleTestConnection); internal/benchmark/runner.go:211-212 (ensureModelLoaded)` · severity: low

Both create an inline `&http.Client{Timeout: …}`. ~~Original cited runner.go:219-223 and claimed a matching `client.Get`;~~ **correction:** settings.go uses `Timeout: 5 * time.Second` + `client.Get(url)` (lines 121, 123); runner.go uses `Timeout: 5 * time.Minute` + `client.Do(req)` on a POST (lines 211, 212). The shared part is only the `&http.Client{Timeout}` idiom; timeouts and methods differ. Note additional instances exist (runner.go:363 10min, downloader.go:305 30s, downloader.go:367 no timeout), so a single shared client is not obviously appropriate — this is a low-value cleanup.

### 22. Model-name shortening duplicated in two packages

`internal/api/jobs_env.go:184-192 (shortenModelName); internal/models/gpu_assign.go:264-273 (shortModelName)` · severity: medium

Both strip the org prefix via `strings.LastIndex(name, "/")`, then `TrimSuffix("-GGUF")` and `TrimSuffix("-gguf")` in that order — logically identical. ~~Original claimed jobs_env uses `strings.Index` (first) while gpu_assign uses `LastIndex` — REFUTED: both use `LastIndex` (jobs_env.go:186). (The `strings.Index` usage the auditor likely saw is in `Model.OrgAndBase`, preset.go:21, a different function.)~~ Unify behind `models.ShortModelName`.

### 23. OS-specific env-var lookups in DefaultDataDir and configSearchPaths

`internal/config/paths.go:14-30 (DefaultDataDir switch); internal/config/paths.go:66-87 (configSearchPaths switch)` · severity: low

Both switch on `runtime.GOOS` with the same three-arm shape (darwin → `Library/Application Support`; windows → `LOCALAPPDATA`; default → XDG env var then `UserHomeDir` fallback), differing in the env var consulted (`XDG_DATA_HOME` vs `XDG_CONFIG_HOME`) and per-arm return type (`string` vs `[]string`; the config version also appends a system `/etc` path). A looser structural parallel than a verbatim copy; a shared helper taking the divergent var as an argument could consolidate it.

## Refuted / mispaired (removed from the verified set)

Re-verification removed or corrected the following from the original report:

- **Original #10** (handleStartBenchmark ↔ EnsureBuildActive active-build resolution) — mispaired; the real twin is `startRouter`. Re-pointed as finding #9 above.
- **Original #7 = #1, #21 ⊂ #12, #22 ⊂ #4, #25 ⊂ #20 (rocm), #26/#27 = #13** — duplicate reports of the same duplication, merged.
- Sub-claims struck out in-place: #3 "literal copy" (it isn't), #4 "ring buffer in all three" (only two), #10/now-#9 pairing, #10/now-#10 sixth site (bench_jobs.go), #13 "'-' discrepancy", #16 "3× in proxy.go", #19 slice-body sharing, #20/#25 "wrong for non-zero idx" bug, #21 `client.Get` match, #22 `Index` vs `LastIndex`.

## Filtered out (not verified — from original report, unchanged)

_Reported by at least one run but not confirmed against the source. Listed for transparency._

- `internal/api/bench.go:271-278` — shortenModelName defined inline in bench.go, duplicated in jobs_env.go _(refuted)_
- `internal/api/hf.go:248-289` — handleIncompleteDownloads mirrors handleHFActiveDownloads structure _(refuted)_
- `internal/api/middleware.go:19-27` — API key check blocks duplicated verbatim in middleware and handler _(refuted)_
- `internal/api/models.go` — Router model name→status map rebuilt inline in handleDashboard _(refuted)_
- `internal/api/server.go` — Identical SVG icon constant groups copy-pasted in server.go and service.go _(refuted)_
- `internal/api/v1models.go` — Same embedding+capabilities block copy-pasted in v1models.go and capabilities.go _(refuted; the real pair is v1models.go↔models.go — see finding #15)_
- `internal/benchmark/runner.go:208-213` — Model load/unload HTTP POST pattern duplicated across runner, process manager, proxy _(refuted)_
- `internal/builder/builder.go:398-400` — `sendLog` defined as both a closure and a package-level function _(refuted)_
- `internal/models/gpu_assign.go:109-120` — parseGPURange body duplicated inside ResolveGPUAssign _(refuted)_
- `internal/models/gpu_assign.go` — Router-name lookup chain repeated across three locations _(refuted)_
- `internal/api/bench_export.go:175-195` — parseExportFormat / parseExportScope copied to bench_jobs.go _(refuted)_
- `internal/api/bench_jobs.go:39-47` — jobInTerminalState duplicated by switch in renderJobList _(refuted)_
- `internal/api/bench.go` — Three SVG icon constants duplicated verbatim _(refuted)_
- `internal/api/hf.go:505-510` — SVG icon string constants defined inline in both server.go and hf.go _(refuted)_
- `internal/api/jobs_env.go:184-191` — shortenModelName and autoDiscoveryName both extract suffix after final '/' _(refuted)_
- `internal/api/jobs_env.go:456-460` — newJobID definition copied into bench_jobs.go _(refuted)_
- `internal/api/models.go:334-342` — routerKnownStates() body near-identical to handleDashboard's inline loop _(refuted; but see finding #3 for the thematic overlap)_
- `internal/api/server.go:155-158` — safeShortSHA and shortSHA template func are the same logic _(refuted)_
- `internal/builder/detect.go` — GPU-name extraction loop copy-pasted in detectCUDA and monitor.nvidia.Collect _(refuted)_
- `internal/builder/profiles.go:1-30` — Flag equality/copying logic duplicated between builder and models _(refuted)_
- `internal/models/gpu_assign.go` — Single-char digit check + range parse idiom duplicated _(refuted)_
- `internal/models/gpu_assign.go` — GPU assignment value formatting duplicated across packages _(uncertain — not re-verified)_
- `internal/models/vram.go:183-194` — formatFloat and trimTrailingZeros implement the same rounding twice _(uncertain — not re-verified)_
- `internal/models/vram.go` — trimTrailingZeros private helper duplicated within same file _(uncertain — not re-verified)_
- `internal/models/gpu_assign.go:1-30` — Color array constant duplicated between GPU allocation and benchmark _(uncertain — not re-verified)_
- `internal/models/registry.go` — "model not found" error literal duplicated _(uncertain — not re-verified)_
- `internal/models/vram.go:117-136` — VRAMFitLabel duplicates a branch inside server.go's "vramFit" template func _(uncertain — not re-verified)_
