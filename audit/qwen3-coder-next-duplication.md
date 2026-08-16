# Code Duplication Audit Report

**Date:** 2026-06-24  
**Project:** llama-toolchest  
**Scope:** Go codebase under `internal/` directory  

---

## Executive Summary

This audit identifies significant code duplication patterns across the codebase. While the project demonstrates good structuring overall, several areas exhibit redundant logic, duplicated utilities, and patterns that could be consolidated for better maintainability.

### Key Findings

| Category | Severity | Files Affected | Lines of Code |
|----------|----------|----------------|---------------|
| VRAM Estimation Logic | Medium | 2 | ~10 lines |
| Model Name Trimming | Medium | 3 | ~30 lines |
| Benchmark Export Functions | Low | 2 | ~150 lines |
| Registry Pattern Loops | Low | Multiple | ~100+ lines |
| String Formatting Utilities | Low | 1 | ~20 lines |

---

## Detailed Findings

### 1. VRAM Estimation Duplication (Medium Severity)

**Issue:** Two separate implementations of VRAM estimation logic exist in different packages, both using the same formula but with slight variations.

**Files:**
- `internal/models/vram.go:18-21` - `EstimateVRAM()`
- `internal/huggingface/client.go:185-188` - `estimateVRAM()`

**Code Comparison:**

```go
// internal/models/vram.go
func EstimateVRAM(sizeBytes int64) float64 {
    return float64(sizeBytes)*1.1/(1024*1024*1024) + vramOverheadGB
    // vramOverheadGB = 0.2
}

// internal/huggingface/client.go
func estimateVRAM(sizeBytes int64) float64 {
    return float64(sizeBytes) * 1.1 / (1024 * 1024 * 1024)
    // No overhead added
}
```

**Impact:** Inconsistent calculations between HF API responses and internal model listings. The `EstimateVRAM` function adds 0.2GB overhead, while `estimateVRAM` does not.

**Recommendation:** Extract to a shared utility function in `internal/models/` and use consistently. If different behaviors are intentional, add comments explaining the rationale.

---

### 2. Model Name Trimming Duplication (Medium Severity)

**Issue:** Three different implementations of model ID/name shortening logic across the codebase.

**Files:**
- `internal/api/jobs_env.go:182-192` - `shortenModelName()`
- `internal/models/gpu_assign.go:263-272` - `shortModelName()`
- `internal/api/hf.go:369` - Inline trimming in `handleDownload()`

**Code Comparison:**

```go
// internal/api/jobs_env.go
func shortenModelName(modelID string) string {
    name := modelID
    if idx := strings.LastIndex(name, "/"); idx >= 0 {
        name = name[idx+1:]
    }
    name = strings.TrimSuffix(name, "-GGUF")
    name = strings.TrimSuffix(name, "-gguf")
    return name
}

// internal/models/gpu_assign.go
func shortModelName(modelID string) string {
    if idx := strings.LastIndex(modelID, "/"); idx >= 0 {
        modelID = modelID[idx+1:]
    }
    modelID = strings.TrimSuffix(modelID, "-GGUF")
    modelID = strings.TrimSuffix(modelID, "-gguf")
    return modelID
}
```

**Impact:** 
- Duplicated logic (75% identical code)
- Inconsistent naming (`shortenModelName` vs `shortModelName`)
- Potential for inconsistent behavior if logic diverges

**Recommendation:** Create a single `models.ShortenModelID()` function in `internal/models/gpu_assign.go` and refactor all callers to use it.

---

### 3. Benchmark Export Function Duplication (Low Severity)

**Issue:** `writeJSONExport` and `writeCSVExport` functions in `bench_export.go` are duplicated in purpose for per-job vs per-benchmark exports.

**Files:**
- `internal/api/bench_export.go` - Core export functions
- `internal/api/bench_jobs.go:435, 441` - Per-job export handlers
- `internal/api/bench.go:390, 396` - All-benchmarks export handler

**Current Structure:**
```go
// Both paths call the same functions but with different filenames:
// bench_jobs.go: writeJSONExport(w, fmt.Sprintf("job-%s.json", id), ...)
// bench.go: writeJSONExport(w, "benchmarks.json", ...)
```

**Impact:** Low - the functions themselves are not duplicated, but the export endpoints follow an identical pattern that could be abstracted.

**Recommendation:** Consider a single `handleExport` function with an `exportType` parameter that handles both "job" and "all" scopes.

---

### 4. Registry Pattern Loops (Low Severity)

**Issue:** The `Registry` struct in `internal/models/registry.go` uses a consistent iteration pattern across many methods:

```go
r.mu.RLock()
defer r.mu.RUnlock()
for _, m := range r.data.Models {
    // ... conditional logic
}
```

This pattern appears ~20 times in the file, sometimes with the same conditional logic (e.g., checking `GetConfig` and `cfg.Enabled`).

**Files Affected:**
- `internal/api/bench_jobs.go:269-270`
- `internal/api/models.go:93-95`
- `internal/api/gpu_map.go:19-22`
- `internal/api/bench.go:454-457`
- `internal/api/service.go:626-628`

**Code Pattern:**
```go
allModels := s.registry.List()
for _, m := range allModels {
    if cfg, err := s.registry.GetConfig(m.ID); err == nil && cfg.Enabled {
        // process enabled model
    }
}
```

**Impact:** Code repetition; minor performance impact from `List()` creating a copy vs direct iteration on `r.data.Models`.

**Recommendation:** Consider adding specialized methods to `Registry`:
- `Registry.EnabledModels() []*Model`
- `Registry.ConfiguredModels() map[string]*Model`

---

### 5. String Formatting Utilities (Low Severity)

**Issue:** Helper functions `itoa` and `ftoa` are defined locally within `bench_export.go` instead of being centralized.

**Files:**
- `internal/api/bench_export.go:205-218`

```go
func itoa(n int) string { return strconv.Itoa(n) }
func ftoa(f float64) string {
    if f == 0 {
        return "0"
    }
    return strconv.FormatFloat(f, 'f', -1, 64)
}
```

**Impact:** If other packages need similar utilities, they'll duplicate this pattern. The functions are simple but could be standardized.

**Recommendation:** Move to `internal/api/` shared utilities if needed, or use standard library directly with consistent formatting elsewhere.

---

### 6. File Suffix Trimming Pattern (Low Severity)

**Issue:** Multiple places use `strings.TrimSuffix` to remove file extensions, with inconsistent application.

**Files:**
- `internal/models/registry.go:25-26, 866-867, 1197-1198`
- `internal/models/gpu_assign.go:270-271`
- `internal/api/jobs_env.go:189-190`
- `internal/api/hf.go:369`
- `internal/huggingface/downloader.go:106`

**Pattern:**
```go
strings.TrimSuffix(strings.TrimSuffix(name, "-GGUF"), "-gguf")
strings.ReplaceAll(strings.TrimSuffix(filename, ".gguf"), "/", "--")
```

**Recommendation:** Consider a `models.TrimModelSuffixes(name string) string` helper.

---

## Duplicate Code Summary Table

| Function/Pattern | Occurrences | Location | Suggested Action |
|-----------------|-------------|----------|------------------|
| VRAM estimation (raw formula) | 2 | models/vram.go, huggingface/client.go | Consolidate |
| Model name trimming | 3+ | api/jobs_env.go, models/gpu_assign.go, api/hf.go | Centralize |
| Registry iteration | ~20 | models/registry.go | Add specialized methods |
| Export functions | 2 endpoints | api/bench_export.go, api/bench_jobs.go, api/bench.go | Consider unified handler |
| File extension trimming | 5+ | Multiple | Create helper function |
| ParseGGUFMeta | 4 calls | api/hf.go, models/registry.go (2x), benchmark | Already centralized |
| ParseQuant | 5 calls | api/hf.go, models/registry.go (2x), huggingface/client.go | Already centralized |

---

## Recommendations Priority

### High Priority
1. **Consolidate VRAM estimation** - The 0.2GB overhead difference may cause confusion in UI displays
2. **Unify model name trimming** - Prevents divergent behavior as codebase grows

### Medium Priority
3. **Registry helper methods** - Reduces code duplication and improves readability
4. **File suffix helper** - Makes intent clearer than inline TrimSuffix chains

### Low Priority
5. **Benchmark export unification** - Minor improvement to handler organization
6. **Standardize string formatting** - Only if utilities are needed elsewhere

---

## Testing Recommendations

After refactoring:
1. Verify VRAM estimates match between UI and HF browser
2. Confirm model names display consistently across all pages
3. Ensure benchmark exports produce identical CSV/JSON structure
4. Check registry iteration order remains stable (currently sorted by ModelID)

---

## Conclusion

The codebase has reasonable modularity with most duplicate patterns being isolated to specific packages. The most impactful improvements would come from:
1. Unifying VRAM calculation for consistent user-facing values
2. Centralizing model name formatting logic

The duplication found is generally straightforward to resolve without architectural changes.
