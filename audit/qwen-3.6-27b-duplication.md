# Code Duplication Audit

> Generated: 2026-06-24

## 1. Capabilities list construction (HIGH)

The same logic for building a `[]string` capabilities list from a `*models.Model` and `*models.ModelConfig` is duplicated in two places:

| Location | Function |
|---|---|
| `internal/api/v1models.go:12–25` | `openAIModel` |
| `internal/api/models.go:173–185` | `handleModelInfo` |

Both blocks do the identical sequence:
1. Check `IsEmbeddingModel` on `m.ModelID` and `m.ID`
2. Append `"embedding"` or `"chat"`
3. Append `"tools"` if `m.SupportsTools`
4. Append `"vision"` if `m.HasBuiltinVision || cfg.MmprojPath != ""`

**Recommendation:** Extract to a shared function, e.g. `func modelCapabilities(m *models.Model, cfg *models.ModelConfig) []string`.

---

## 2. `IsEmbeddingModel(m.ModelID) || IsEmbeddingModel(m.ID)` (MEDIUM)

This exact double-check expression is repeated **5 times**:

| Location | File:Line |
|---|---|
| 1 | `internal/api/v1models.go:14` |
| 2 | `internal/api/capabilities.go:37` |
| 3 | `internal/api/models.go:17` |
| 4 | `internal/api/models.go:175` |
| 5 | `internal/api/server.go:502` |

**Recommendation:** Extract to a helper, e.g. `func isEmbedding(m *models.Model) bool`, or add a method on `*models.Model` (e.g. `m.IsEmbedding()`).

---

## 3. `cssID` template function vs `domID` (MEDIUM)

Two identical string-sanitization functions exist:

| Location | Name |
|---|---|
| `internal/api/server.go:127–138` | `"cssID"` (template FuncMap) |
| `internal/api/hf.go:239–250` | `domID` (plain function) |

Both use `strings.Map` with the same switch logic: keep `[a-zA-Z0-9_-]`, replace everything else with `_`.

**Recommendation:** Define one function (e.g. `domID`) and reference it from the template FuncMap as `"cssID": domID`.

---

## 4. `shortSHA` template function vs `safeShortSHA` (LOW)

Nearly identical truncation logic:

| Location | Name | Behavior |
|---|---|---|
| `internal/api/server.go:155–161` | `"shortSHA"` (template) | Truncates to 12 chars |
| `internal/api/build.go:346–350` | `safeShortSHA` (plain function) | Truncates to 7 chars |

Both guard `len(s) < N` then return `s[:N]`. The only difference is the cutoff length (12 vs 7).

**Recommendation:** One parameterized function, e.g. `func shortenSHA(s string, n int) string`, with the two call sites passing their respective lengths.

---

## 5. Filter enabled models (MEDIUM)

The pattern "iterate registry, check config.Enabled, collect" is repeated in **4 places** with nearly identical code:

| Location | File:Line |
|---|---|
| 1 | `internal/api/bench_jobs.go:269–271` (`handleJobForm`) |
| 2 | `internal/api/bench.go:454–458` (`handleBenchmarkForm`) |
| 3 | `internal/api/server.go:501–506` (`handleDashboard`) |
| 4 | `internal/api/v1models.go:60–66` (`handleV1Models`) |

Each does:
```go
var enabled []*models.Model
for _, m := range s.registry.List() {
    if cfg, err := s.registry.GetConfig(m.ID); err == nil && cfg.Enabled {
        enabled = append(enabled, m)
    }
}
```

**Recommendation:** Add a method to the `Server` (or the `Registry`) like `func (s *Server) listEnabledModels() []*models.Model`.

---

## 6. GPU allocation map construction (MEDIUM)

Two places build the same `allModels` / `allConfigs` map from `s.registry.List()`:

| Location | File:Line |
|---|---|
| 1 | `internal/api/gpu_map.go:19–26` (`handleGPUMap`) |
| 2 | `internal/api/service.go:626–632` (`handleGetModelConfig`) |

Both iterate all models, call `GetConfig`, and populate a `map[string]*models.ModelConfig`. Both then pass these to `models.ComputeAllocations`.

**Recommendation:** Extract to `func (s *Server) buildGPUAllocationData() ([]*models.Model, map[string]*models.ModelConfig)`.

---

## 7. Builder list iteration for single-build lookup (LOW)

Three places iterate `s.builder.List()` looking for a single build by ID:

| Location | File:Line |
|---|---|
| 1 | `internal/api/build.go:291–298` (`handleBuildInfo`) |
| 2 | `internal/api/jobs_env.go:38–43` (`CheckBuildRunnable`) |
| 3 | `internal/api/jobs_env.go:95–101` (`EnsureBuildActive`) |

**Recommendation:** Add a `func (b *Builder) Get(id string) *BuildResult` method to the builder package.

---

## 8. FindOrphans / IncompleteRegistered iteration (LOW)

Two places iterate the same two registry methods to build lookup sets:

| Location | File:Line |
|---|---|
| 1 | `internal/api/models.go:434–443` (`renderModelList`) |
| 2 | `internal/api/service.go:494–504` (`handleModelEnable`) |

Both do:
```go
orphanSet := make(map[string]bool)
for _, m := range s.registry.FindOrphans() {
    orphanSet[m.ID] = true
}
incompleteSet := make(map[string]bool)
for _, m := range s.registry.IncompleteRegistered() {
    incompleteSet[m.ID] = true
}
```

**Recommendation:** Extract to `func (s *Server) modelProblemSets() (orphans, incomplete map[string]bool)`.

---

## 9. Content-Type JSON check (LOW)

The pattern `r.Header.Get("Content-Type") == "application/json"` appears in **4 handlers** to decide between JSON-body and form parsing:

| Location | File:Line |
|---|---|
| 1 | `internal/api/settings.go:51` |
| 2 | `internal/api/hf.go:110` |
| 3 | `internal/api/service.go:702` |
| 4 | `internal/api/build.go:184` |

**Recommendation:** The `isJSONContentType` function already exists in `proxy.go:130`. Reuse it in these 4 handlers instead of the inline string comparison.

---

## 10. `itoa` wrapper (TRIVIAL)

`internal/api/bench_export.go:197`:
```go
func itoa(n int) string { return strconv.Itoa(n) }
```

This is a zero-value wrapper around `strconv.Itoa`. It's used only within the same file (3 call sites). Not harmful, but adds no value.

**Recommendation:** Either remove and inline `strconv.Itoa`, or document why the alias exists (e.g. stylistic consistency with `ftoa`).

---

## 11. `ftoa` and `metricMean`/`metricStd` (LOW)

`internal/api/bench_export.go:200–214`:
```go
func ftoa(f float64) string {
    if f == 0 { return "0" }
    return strconv.FormatFloat(f, 'f', -1, 64)
}
func metricMean(m *benchmark.LlamaBenchyMetric) string {
    if m == nil { return "" }
    return ftoa(m.Mean)
}
func metricStd(m *benchmark.LlamaBenchyMetric) string {
    if m == nil { return "" }
    return ftoa(m.Std)
}
```

`metricMean` and `metricStd` are identical except for the field accessed. Used only within the same file.

**Recommendation:** Could merge into one generic helper, e.g. `func metricVal(m *LlamaBenchyMetric, val float64) string`, but the current code is clear enough. Low priority.

---

## 12. `isHTMX` branching pattern (INFO — not a bug)

Every API handler follows the same pattern:
```go
if isHTMX(r) {
    respondHTML(w)
    // render HTML
    return
}
respondJSON(w, data)
```

This is repeated in **40+ places** across 11 files. This is an intentional architectural pattern (content negotiation via HTMX header), not accidental duplication. However, if the pattern ever needs to change (e.g. adding a third format), it would be painful to update everywhere.

**Recommendation (future):** Consider a middleware or handler wrapper that does the branching, e.g. `func (s *Server) dualResponse(w http.ResponseWriter, r *http.Request, htmlFn, jsonFn func())`. Not urgent — the pattern is well-established and simple.

---

## Summary by Priority

| Priority | Count | Items |
|---|---|---|
| **HIGH** | 1 | #1 Capabilities list construction |
| **MEDIUM** | 4 | #2 IsEmbeddingModel check, #3 cssID/domID, #5 Filter enabled models, #6 GPU allocation map |
| **LOW** | 5 | #4 shortSHA, #7 Builder lookup, #8 Orphans/Incomplete, #9 Content-Type check, #11 ftoa helpers |
| **TRIVIAL** | 1 | #10 itoa wrapper |
| **INFO** | 1 | #12 isHTMX branching pattern |
