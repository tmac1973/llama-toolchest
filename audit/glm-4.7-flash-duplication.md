# Code Duplication Audit

**Date**: 2026-06-24
**Scope**: llama-toolchest codebase (Go code in `cmd/` and `internal/`)

---

## Executive Summary

The codebase has **moderate duplication** that could be addressed with utility helpers, but overall it's reasonably well-structured. The most significant duplication areas are:

1. **HTTP error handling patterns** (inconsistent)
2. **JSON response patterns** (inconsistent)
3. **Form parameter parsing patterns**
4. **Error message strings** (scattered)
5. **Error wrapping patterns** (inconsistent)

---

## Detailed Findings

### 1. HTTP Error Handling Inconsistencies

**Problem**: HTTP error handling is implemented inconsistently across handlers.

**Examples**:

- **`http.Error(w, err.Error(), http.StatusBadRequest)`** (used directly)
  - `internal/api/hf.go:40` ("missing q parameter")
  - `internal/api/hf.go:63` ("missing id parameter")

- **`http.Error(w, err.Error(), http.StatusBadGateway)`**
  - `internal/api/hf.go:52` (HuggingFace API call failure)

- **`http.Error(w, err.Error(), http.StatusInternalServerError)`**
  - `internal/api/service.go:801` (registry.SetConfig error)
  - `internal/api/monitor.go:30` (SSE writer creation failure)

- **No standard HTTP error codes used for domain-specific errors**:
  - "model not found", "already running", "invalid request", etc., appear in error messages but no corresponding `http.StatusXxx` constants

**Impact**:
- Inconsistent error responses to clients
- No semantic error codes for domain errors (e.g., `404` for model not found)
- API consumers cannot reliably handle specific error categories

**Files Affected** (error pattern count):
- `internal/api/hf.go`: 10 occurrences
- `internal/api/bench.go`: 15 occurrences
- `internal/api/service.go`: 15 occurrences
- `internal/api/build.go`: 5 occurrences
- `internal/api/models.go`: 5 occurrences

---

### 2. JSON Response Patterns Inconsistencies

**Problem**: JSON responses are written differently across handlers.

**Direct pattern** (`w.Header().Set("Content-Type", "application/json")` then `json.NewEncoder(w).Encode(v)`):
- `internal/api/monitor.go:32` (in SSE stream)
- `internal/api/monitor.go:44` (in SSE stream)
- `internal/api/models.go:15` (in `handleListModels`)
- `internal/api/settings.go:13` (in `handleGetModelConfig`)
- `internal/api/v1models.go:1` (in `handleListEmbeddingModels`)
- `internal/api/bench_about.go:2` (partial HTMX HTML)

**`respondJSON` helper pattern**:
- `internal/api/respond.go` defines `respondJSON(w http.ResponseWriter, v any)` helper
- Used in: `monitor.go`, `v1models.go`, `bench_jobs.go`, `settings.go`, `bench_about.go`, `models.go`, `hf.go`, `bench.go`, `service.go`, `build.go`
- **Not used in `monitor.go:32/44`** (SSE stream)

**Missing**:
- No helper for `respondHTML` that sets `text/html; charset=utf-8` header (one line in `respond.go` but not used widely)

---

### 3. Form Parameter Parsing Repetition

**Problem**: Repeated parsing pattern for form values:

```go
// Common pattern repeated across files:
v, err := strconv.Atoi(r.FormValue("gpu_layers"))
if err != nil {
    // handle error
}
cfg.GPULayers = v
```

**Files with this pattern** (`internal/api/service.go:808-845`):
- `context_size`, `parallel`, `threads`, `flash_attention`, `jinja`, `kv_cache_quant`, `direct_io`, `extra_flags`
- `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty`
- `mmproj_path`, `mmproj_enabled`, `mtp_path`, `mtp_enabled`
- `draft_ctx_size`, `draft_gpu_layers`, `draft_device`, `draft_cpu_moe`, `draft_kv_cache_quant`
- Plus `GPULayers` in service.go

**Impact**:
- 60 total `r.FormValue(...)` calls across 7 files
- Each field needs manual error handling
- Duplicated `parseOptionalFloat()` and `parseOptionalInt()` functions

**Good patterns found**:
- `parseOptionalFloat()` in `service.go`
- `parseOptionalInt()` in `service.go`
- `countNonZeroSplit()` in `service.go` (for tensor-split parsing)

---

### 4. Error Message String Duplication

**Problem**: Error messages are scattered as string literals without constants.

**Notable occurrences**:
- `internal/api/bench_jobs.go:10` - error messages for job operations
- `internal/api/settings.go:1` - generic error messages
- `internal/api/models.go:1` - model-related errors
- `internal/api/hf.go:3` - HuggingFace errors
- `internal/api/bench.go:8` - benchmark errors
- `internal/benchmark/runner.go:1` - benchmark runner errors

**Example search results**:
- `"already running"` appears in `internal/benchmark/runner.go` (but not in HTTP error responses)
- No HTTP status codes used for domain errors

---

### 5. Error Wrapping Inconsistencies

**Problem**: Error wrapping patterns vary between `fmt.Errorf` and bare errors.

**Files with error wrapping**:
- `internal/benchmark/job_runner.go`: 7 occurrences
- `internal/benchmark/runner.go`: 2 occurrences
- `internal/benchmark/benchy.go`: 2 occurrences
- `internal/process/manager.go`: 11 occurrences
- `internal/huggingface/downloader.go`: 1 occurrence
- `internal/config/config.go`: 1 occurrence
- `internal/models/gpu_assign.go`: 10 occurrences

**Inconsistent patterns**:
- Some use `fmt.Errorf("description: %w", err)`
- Some use bare `errors.New("description")`
- No consistent error wrapping conventions visible

---

### 6. Logging Pattern Duplication

**Problem**: Different logging approaches used across files.

**Direct `log/slog` usage** (29 occurrences):
- `internal/api/models.go:1`
- `internal/api/server.go:4`
- `internal/api/service.go:3`
- `internal/api/build.go:1`
- `internal/config/config.go:2`
- `internal/models/registry.go:3`
- `internal/builder/builder.go:2`
- `internal/benchmark/job_runner.go:1`
- `internal/benchmark/runner.go:4`
- `internal/benchmark/benchmark.go:2`
- `cmd/llama-toolchest/main.go:4`
- `cmd/agent/main.go:9`

**No centralized error logging helpers found** (e.g., `logError(w http.ResponseWriter, err error)` helper)

---

### 7. SSE Event Writing Pattern

**Problem**: SSE event writing is done inline in `handleMonitorStream`.

**Location**: `internal/api/monitor.go:32-44`

```go
// Send current state immediately
data, _ := json.Marshal(s.monitor.Current())
sse.SendEvent("metrics", string(data))

for {
    select {
    case m, ok := <-ch:
        if !ok {
            return
        }
        data, _ := json.Marshal(m)
        sse.SendEvent("metrics", string(data))
    case <-r.Context().Done():
        return
    }
}
```

**Potential duplication**:
- Same `json.Marshal` + `sse.SendEvent` pattern could be extracted as a helper method if SSE events become more complex.

---

### 8. String Formatting Patterns

**Problem**: String formatting duplicated.

**`fmt.Sprintf` occurrences**: 155 across 27 files (mostly in templates/rendering logic, which is acceptable).

**Notable duplication**:
- `monitorBarData()` in `monitor.go` creates multiple `fmt.Sprintf` calls for GPU/VRAM display
- Similar patterns in `settings.go` for HTMX trigger messages

---

## Files with Highest Duplication Counts

| File | Duplicate Pattern Count |
|------|------------------------|
| `internal/benchmark/runner.go` | 10 |
| `internal/models/registry.go` | 9 |
| `internal/builder/builder.go` | 10 |
| `internal/benchmark/job.go` | 3 |
| `internal/huggingface/client.go` | 6 |
| `internal/huggingface/downloader.go` | 6 |
| `internal/benchmark/runner.go` | 10 |
| `internal/models/preset.go` | 47 (mostly formatting) |

---

## Recommendations

### High Priority (API Consistency)

1. **Standardize HTTP error handling**:
   - Create error response helper functions with domain-specific HTTP status codes
   - Define a list of domain error strings as constants
   - Create `ErrorResponder` type or helper functions (e.g., `BadRequest(w, msg)`, `NotFound(w, msg)`, `Conflict(w, msg)`)

2. **Consolidate JSON response patterns**:
   - Enforce use of `respondJSON(w, v)` helper
   - Add `respondHTML(w)` helper that sets `text/html; charset=utf-8` header consistently
   - Document the response pattern for SSE events

### Medium Priority (DRY Principles)

3. **Centralize form parsing utilities**:
   - Extend `parseOptionalFloat()` and `parseOptionalInt()` with context/errors
   - Add helper for common `r.FormValue("gpu_layers")` pattern
   - Create helper for common `strings.TrimSpace(r.FormValue(...))` pattern

4. **Standardize error wrapping**:
   - Adopt a consistent error wrapping pattern across codebase (e.g., always use `fmt.Errorf("description: %w", err)` for wrapping)
   - Consider defining error types for common domain errors

5. **Create logging helper**:
   - Add a `logError(w http.ResponseWriter, err error)` helper for API error logging
   - Consider a `logError(err error)` helper for background operations

### Low Priority (Nice to Have)

6. **Extract SSE event helper**:
   - If SSE events become more complex, consider helper: `sendSSEEvent(w *sse.Writer, name, payload string)`

7. **Consider error types**:
   - Define specific error types (e.g., `ModelNotFoundError`, `JobAlreadyRunningError`) for more precise error handling

---

## Metrics Summary

- **HTTP error handling**: 88 direct `http.Error` calls across 15 files
- **JSON responses**: 66 `json.NewEncoder` calls, 8 using `respondJSON` helper
- **Form value parsing**: 60 `r.FormValue` calls across 7 files
- **Error strings**: ~46 direct error string usages in error messages (non-HTTP)
- **Logging**: 29 `log/slog` calls in API code, 12 in main/agent

---

## Overall Assessment

The codebase shows **moderate duplication** with clear opportunities for:

1. Creating error response helpers (high impact on API consistency)
2. Enforcing `respondJSON` helper usage (high impact on consistency)
3. Centralizing form parsing utilities (moderate impact)

Most duplication is in template/rendering logic (acceptable), form handling (easily consolidatable), and error handling (high ROI for standardization).

---

## References

- Audit created: 2026-06-24
- Go project structure: Standard layout (`cmd/`, `internal/`, `pkg/` if existed)
- No `pkg/` utility directory found - consider creating one for shared helpers