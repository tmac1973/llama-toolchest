# Refactoring Plan — Open Items

> Outstanding cleanup work carved out of the original full audit. The
> history (which items were done, plus the architecture discussion in
> §4 of the original) lives at `plan/archive/refactor-original.md`.

---

## 1. Code Deduplication

### 1.1 Download Progress Formatting

`handleHFDownloadProgress` (`internal/api/hf.go:118-124`) and
`handleHFActiveDownloads` (`internal/api/hf.go:160-172`) duplicate the
same pct/speed/GB calculations.

**Fix:** extract a `DownloadMetrics` struct and a
`ComputeDownloadMetrics()` method on `huggingface.DownloadStatus`.

### 1.2 URL Param + Not-Found Error

`chi.URLParam(r, "id")` followed by a registry lookup and 404 is
repeated in roughly a dozen handlers.

**Fix:** add a small helper:

```go
func (s *Server) requireModel(w http.ResponseWriter, r *http.Request) (*models.Model, bool)
```

…that returns the model or writes a 404 and returns `false`. Same idea
for builds (`requireBuild`).

---

## 2. Design Pattern Improvements

### 2.1 Structured Error Responses

Most handlers return plain-text errors via `http.Error()`. For API
consumers (the agent CLI, external tools), structured JSON would be
better:

```go
type APIError struct {
    Error string `json:"error"`
    Code  string `json:"code,omitempty"`
}
```

Pair with the existing `respondJSON` helper and add a `respondError`
that picks JSON vs HTML based on `isHTMX(r)`.

### 2.2 Separate Config Domains

`config.Config` mixes server settings, credentials, and runtime state in
one struct. Splitting could look like:

- `ServerConfig` — listen addr, data dir, models dir, log level
- `ServiceConfig` — llama port, active build, models max, auto start
- `SecretsConfig` — HF token, API key
- (or just leave as-is)

Low priority — the current struct is small enough that this is purely a
nice-to-have, and YAML round-tripping gets more complex with nested
structs.

---

## 3. Testing

The codebase has zero tests. A full suite is out of scope, but
high-value targets if/when we add some:

- **GGUF parser** (`internal/models/gguf.go`) — binary parsing with
  edge cases is exactly what unit tests are good at.
- **Preset INI generation** (`internal/models/preset.go`) — pure
  string manipulation, easy to test.
- **VRAM estimation** (`internal/models/vram.go`) — pure functions over
  numeric inputs.
- **Config read-modify-write cycle** — a regression test against the
  Enabled-clobber bug that bit us once already.
- **`config.ModelsPath()` / env-var override** — small but easy to
  break silently.

---

## 4. Suggested Order

| Item | Risk | Effort |
|---|---|---|
| 1.1 Download metrics dedup | Low | Small |
| 1.2 requireModel/requireBuild helper | Low | Medium (~12 call sites) |
| 2.1 Structured error responses | Low | Medium |
| 2.2 Config domain split | Low | Optional |
| 3 Add a few targeted unit tests | Low | Pick-and-choose |

None of these are blockers; tackle when touching the relevant code or
in a quiet stretch.
