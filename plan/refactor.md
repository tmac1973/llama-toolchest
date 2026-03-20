# Refactoring Plan

Audit of the llamactl codebase (31 Go files, ~5,800 lines) covering bugs, code
duplication, design patterns, and the model config/restart architecture.

---

## 1. Bugs

### 1.1 CRITICAL: `handleUpdateModelConfig` Clobbers `Enabled` State

**File:** `internal/api/service.go:341-393`

When a user edits model config (GPU layers, context size, etc.), the handler
creates a **new** `ModelConfig` struct from scratch (line 344: `var cfg models.ModelConfig`).
The `Enabled` field is never read from the existing config, so it defaults to
`false`. After saving, the model silently becomes disabled.

In contrast, `handleModelEnable` (line 226) correctly does a read-modify-write:
```go
cfg, err := s.registry.GetConfig(id)  // read existing
cfg.Enabled = enabled                  // modify one field
s.registry.SetConfig(id, cfg)          // write back
```

**Impact:** Every config edit via the Configure dialog silently disables the
model. The auto-save (500ms debounce in model_config.html) makes this worse—any
slider adjustment triggers it.

**Fix:** Fetch existing config first, then overlay form values:
```go
cfg, err := s.registry.GetConfig(id)
if err != nil { ... }
// Then overwrite individual fields from form
cfg.GPULayers, _ = strconv.Atoi(r.FormValue("gpu_layers"))
// ...etc, leaving cfg.Enabled untouched
```

### 1.2 MINOR: Preset INI Not Regenerated on Model Delete

`handleDeleteModel` removes the model from the registry but does not call
`registry.WritePresetINI()`. The stale INI still references the deleted model.
Next router restart will fail to find the GGUF file.

**Fix:** Add `s.registry.WritePresetINI()` after delete.

---

## 2. Code Deduplication

### 2.1 Bytes-to-GB Conversion (HIGH — ~15 occurrences)

The expression `float64(bytes) / (1024 * 1024 * 1024)` appears in:
- `internal/api/models.go:46`
- `internal/api/service.go:284, 384`
- `internal/api/hf.go:123, 124, 166, 167`

A `divGB` template function already exists in `server.go:73` but isn't used
outside templates.

**Fix:** Add `func BytesToGB(b int64) float64` to `internal/models/vram.go` and
use it everywhere.

### 2.2 Download Progress Formatting (HIGH — 2 near-identical blocks)

`handleHFDownloadProgress` (hf.go:118-124) and `handleHFActiveDownloads`
(hf.go:160-172) duplicate the same pct/speed/GB calculations.

**Fix:** Extract a `DownloadMetrics` struct and `ComputeDownloadMetrics()` helper
on the `DownloadStatus` type.

### 2.3 HTMX Request Detection (HIGH — ~24 occurrences)

Every handler has:
```go
if r.Header.Get("HX-Request") == "true" {
    // render HTML partial
    return
}
// render JSON
```

**Fix:** Add `func isHTMX(r *http.Request) bool` helper. For handlers that
always branch, consider a `respondHTMXOrJSON(w, r, htmlFn, jsonFn)` wrapper.

### 2.4 SSE Streaming Boilerplate (MEDIUM — 3 handlers)

Build logs, service logs, and download progress all repeat the same
subscribe → loop → flush → done pattern (~20 lines each).

**Fix:** Generic SSE helper:
```go
func streamSSE(w http.ResponseWriter, r *http.Request, ch <-chan T, format func(T) (event, data string))
```

### 2.5 JSON Response Pattern (MEDIUM — ~22 occurrences)

`w.Header().Set("Content-Type", "application/json")` + `json.NewEncoder(w).Encode(data)`
repeated everywhere.

**Fix:** `func respondJSON(w http.ResponseWriter, v any)` helper.

### 2.6 URL Param + Not-Found Error (MEDIUM — ~12 occurrences)

`chi.URLParam(r, "id")` followed by a lookup and 404 error is repeated in
every handler that takes an ID.

**Fix:** `func (s *Server) requireModel(w http.ResponseWriter, r *http.Request) (*models.Model, bool)`
that returns the model or writes a 404 and returns false.

---

## 3. Design Pattern Improvements

### 3.1 Extract HTTP Response Helpers

Create `internal/api/respond.go` with:
- `respondJSON(w, data)`
- `respondError(w, status, msg)`
- `respondHTML(w, tmpl, data)`
- `isHTMX(r) bool`

Eliminates scattered Content-Type headers and error formatting.

### 3.2 Constants for Magic Strings

Status strings (`"running"`, `"stopped"`, `"building"`, etc.) are bare strings
in process/manager.go and builder/builder.go.

**Fix:** Define typed constants:
```go
type ProcessState string
const (
    StateStopped  ProcessState = "stopped"
    StateStarting ProcessState = "starting"
    StateRunning  ProcessState = "running"
    StateFailed   ProcessState = "failed"
)
```

Same for build statuses, query parameter names (`"keep_files"`, `"opt_"`
prefix), and default config values (999 GPU layers, 8192 context, etc.).

### 3.3 Move Inline HTML Out of Handlers

`internal/api/monitor.go:52-92` and `internal/api/service.go:284-295` build
HTML with `fmt.Fprintf`. These should be small template partials instead.
Keeps all markup in `web/templates/partials/`.

### 3.4 Structured Error Responses

Most handlers return plain-text errors via `http.Error()`. For API consumers
(agent CLI, external tools), structured JSON errors would be better:
```go
type APIError struct {
    Error   string `json:"error"`
    Code    string `json:"code,omitempty"`
}
```

### 3.5 Separate Config Domains

`config.Config` mixes server settings, credentials, and runtime state in one
struct. Consider splitting:
- `ServerConfig` — listen addr, data dir, log level
- `ServiceConfig` — llama port, active build, models max
- `SecretsConfig` — HF token, API key

Low priority—current struct is small enough that this is a nice-to-have.

---

## 4. Model Config Architecture: INI + Restart vs Hot Reload

### 4.1 Current Architecture

```
UI toggle/config change
  → registry.SetConfig()        (writes models.json)
  → registry.WritePresetINI()   (regenerates preset.ini)
  → process.LoadModel/UnloadModel()  (HTTP API to llama-server router)
```

The router reads `preset.ini` at startup. Individual models can be loaded and
unloaded at runtime via `POST /models/load` and `POST /models/unload` without
restarting the router process.

### 4.2 What Currently Works Without Restart

- **Enable/disable models** — unload via API, INI regenerated for next time
- **Load/unload models** — API calls to running router
- **Sampling parameter changes** — injected at proxy layer (service.go proxy),
  never touch the router at all

### 4.3 What Currently Requires Restart

- **Per-model parameter changes** (GPU layers, context size, KV cache quant) —
  INI is regenerated but the running model must be unloaded and reloaded for
  the router to pick up the new INI section
- **Binary change** (different llama.cpp build) — requires full process restart
- **Global settings** (models_max) — requires restart

### 4.4 Gap: No Unload+Reload on Config Change

`handleUpdateModelConfig` regenerates the INI but does **not** unload and
reload the model. The new settings don't take effect until the user manually
restarts the router or the model is otherwise reloaded.

**Fix:** After writing the new INI, if the model is currently loaded:
```go
if s.process.IsRunning() {
    s.process.UnloadModel(id)
    s.process.LoadModel(id)
}
```

This gives us hot config updates without a full router restart.

### 4.5 Re: Lemonade Server Comparison

Lemonade (llama-swap) uses a proxy that lazily loads/unloads models on demand
with configurable TTLs. Our current architecture with llama.cpp's native router
mode achieves similar behavior—the router itself handles multi-model scheduling.
The INI file is just the initial config; runtime management is API-driven.

The current approach is sound. The main improvement is wiring up the
unload+reload sequence on config changes (4.4 above) so users don't need to
manually restart.

---

## 5. Testing

The codebase has **zero tests**. While a full test suite is out of scope for
this refactor, the following are high-value targets if we want to add tests
later:

- GGUF parser (`internal/models/gguf.go`) — binary parsing with edge cases
- Preset INI generation (`internal/models/preset.go`) — easy to unit test
- VRAM estimation (`internal/models/vram.go`) — pure functions
- Config read-modify-write cycle — regression test for the Enabled bug

---

## 6. Proposed Execution Order

| Phase | Items | Risk | Effort | Status |
|-------|-------|------|--------|--------|
| **A** | Fix Enabled clobber bug (1.1), fix delete INI (1.2) | Low | Small | DONE |
| **B** | Add unload+reload on config change (4.4) | Medium | Small | DONE |
| **C** | Extract response helpers (2.3, 2.5, 3.1) | Low | Medium | DONE |
| **D** | Extract BytesToGB helper (2.1) | Low | Small | DONE |
| **E** | Extract SSE streaming helper (2.4) | Low | Medium | DONE |
| **F** | Move inline HTML to partials (3.3) | Low | Small | DONE |
| **G** | Add magic string constants (3.2) | Low | Small | DONE |
| **H** | Structured error responses (3.4) | Low | Medium | |
| **I** | Config domain separation (3.5) | Low | Optional | |

Phases A-D are complete. E-G are cleanup that can be done incrementally.
H-I are nice-to-haves.
