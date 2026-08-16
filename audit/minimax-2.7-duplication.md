# Code Duplication Audit

Reviewed: `internal/models/`, `internal/api/`, `internal/benchmark/`, `internal/builder/`, `internal/monitor/`, `internal/process/`, `internal/config/`, `internal/huggingface/`

---

## High — Symlink-Resolution + Directory-Walk Duplicated in Registry

**Files:** `internal/models/registry.go`

Two independent code blocks handle identical symlink-aware directory traversal:

```go
// OrphanParts(), lines ~712–730
walkRoot := modelsDir
if resolved, err := filepath.EvalSymlinks(modelsDir); err == nil && resolved != modelsDir {
    walkRoot = resolved
}
// Build set of known file paths for fast lookup
r.mu.RLock()
registered := make(map[string]bool)
for _, m := range r.data.Models {
    for _, sh := range findShards(...) { registered[sh] = true }
}
r.mu.RUnlock()
filepath.Walk(walkRoot, func(path string, info os.FileInfo, err error) error {
    if walkRoot != modelsDir {
        if rel, relErr := filepath.Rel(walkRoot, path); relErr == nil {
            path = filepath.Join(modelsDir, rel)
        }
    }
    // ... filter by ".part"
})
```

```go
// ScanModels(), lines ~803–822 (nearly identical structure)
walkRoot := modelsDir
if resolved, err := filepath.EvalSymlinks(modelsDir); err == nil && resolved != modelsDir {
    walkRoot = resolved
    slog.Debug("scanning via resolved symlink", ...)
}
// Build set of known file paths for fast lookup
r.mu.RLock()
knownPaths := make(map[string]bool, len(r.data.Models))
for _, m := range r.data.Models { knownPaths[m.FilePath] = true }
r.mu.RUnlock()
filepath.Walk(walkRoot, func(path string, info os.FileInfo, err error) error {
    if walkRoot != modelsDir {
        if rel, relErr := filepath.Rel(walkRoot, path); relErr == nil {
            path = filepath.Join(modelsDir, rel)
        }
    }
    // ... filter by ".gguf" / ".part"
})
```

**What differs:** The walk-filter logic (`.part` vs `.gguf` + mmproj/shard skip checks) and the per-entry action. The surrounding infrastructure (symlink resolution, known-paths map, path remapping) is copy-pasted verbatim.

**Suggested fix:** Extract a helper that accepts the directory, a predicate over `os.FileInfo`, and a callback:

```go
func scanDir(dir string, knownPaths map[string]bool, pred func(os.FileInfo) bool, visit func(path string, info os.FileInfo)) int
```

---

## Medium — Near-Identical Directory-Scanning Functions for mmproj and MTP

**File:** `internal/models/registry.go`

`findMMProjInDir` (line ~506) and `findMTPInDir` (line ~965) are structurally identical:

```go
func findXXXInDir(dir string) string {
    entries, err := os.ReadDir(dir)
    if err != nil { return "" }
    for _, e := range entries {
        if e.IsDir() { continue }
        name := e.Name()
        if !strings.HasSuffix(strings.ToLower(name), ".gguf") { continue }
        if !predicate(name) { continue }
        path := filepath.Join(dir, name)
        if shouldParseMetadata && meta, err := ParseGGUFMeta(path); err == nil && predicateMeta(meta) {
            return path
        } else if !shouldParseMetadata {
            return path
        }
    }
    return ""
}
```

`findMMProjInDir` skips metadata parsing (filename pattern is sufficient); `findMTPInDir` needs to read GGUF headers because an MTP head is identified by its architecture field, not its filename. The surrounding callers (`FindMMProj`, `FindMTP`) also follow the same pattern: check subdir, check subdir/"MTP", check parent, check parent/"MTP".

**Suggested fix:** Factor into a generic finder:

```go
func findGGUFInDirs(baseDir string, subdirs []string, match func(name string) bool, checkMeta func(path string) bool) string
```

---

## Medium — GPU Enumeration Duplicated in ROCm Monitor

**File:** `internal/monitor/rocm.go`

Both `collectSysfs()` and `readVRAMSysfs()` independently enumerate AMD GPU cards:

```go
cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/vendor")
idx := 0
for _, vendorFile := range cards {
    vendor, _ := os.ReadFile(vendorFile)
    if strings.TrimSpace(string(vendor)) != "0x1002" { continue }
    if idx != gpuIdx { idx++; continue }  // <-- this is a linear search by index
    // ... use deviceDir ...
}
```

`collectSysfs` iterates all cards to collect utilization, temp, power — but not VRAM. `readVRARSysfs` iterates all cards again to read just the VRAM files. The GPU index parameter forces a linear walk on every call.

**Suggested fix:** Have `readVRAMSysfs` accept the absolute device directory path directly (already known from the `deviceDir` variable in `collectSysfs`), eliminating the need to re-enumerate cards each time.

---

## Low — `isNonFirstShard` and `findShards` Duplicated Across Packages

**Files:**
- `internal/models/registry.go` (lines ~1085–1118)
- `internal/api/bench_export.go` (lines ~210–237)

Both files define identical shard-helpers:

```go
var shardRe = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

func isNonFirstShard(filename string) bool { ... }
func findShards(dir, filename string) []string { ... }
```

The `api` package duplicates them because `bench_export.go` processes benchmark runs that carry model `FilePath`. It needs to resolve the full shard set for size calculation in the export, but avoids an import cycle.

**Suggested fix:** Move `isNonFirstShard` and `findShards` to `internal/models/` and have `bench_export.go` import the `models` package directly. This is the canonical location for shard-aware path logic (the `models` package already owns this concept). Note: an import cycle would occur if `api` imported `models` and vice versa — currently `api` imports `models` already (`github.com/tmlabonte/llamactl/internal/models` in `bench.go`), so adding the shard helpers is safe.

---

## Low — Thin Wrapper Helpers in bench_export.go

**File:** `internal/api/bench_export.go` (lines ~165–176)

```go
func itoa(n int) string { return strconv.Itoa(n) }
func ftoa(f float64) string { ... }
func metricMean(m *benchmark.LlamaBenchyMetric) string { ... }
func metricStd(m *benchmark.LlamaBenchyMetric) string { ... }
```

These add indirection over direct `strconv` calls and field access. They have nil-safe semantics (`metricMean`/`metricStd` return `""` for nil), which is useful, but `itoa` and `ftoa` could be inlined or renamed to avoid shadowing stdlib terminology.

---

## Summary Table

| Severity | Location | Issue |
|----------|----------|-------|
| High | `internal/models/registry.go` | Symlink-resolve + walk duplicated in `OrphanParts()` and `ScanModels()` (~30 lines each) |
| Medium | `internal/models/registry.go` | `findMMProjInDir` and `findMTPInDir` follow identical structure |
| Medium | `internal/monitor/rocm.go` | GPU card enumeration duplicated in `collectSysfs()` and `readVRAMSysfs()` |
| Low | `internal/api/bench_export.go` + `internal/models/registry.go` | `isNonFirstShard`/`findShards` copied verbatim across packages |
| Low | `internal/api/bench_export.go` | Thin `itoa`/`ftoa` wrapper helpers |

---

## Recommendations

1. **Extract `scanDirWithSymlinks` helper** — deduplicate the symlink-resolution + walk pattern in registry.go. Estimated reduction: ~40 lines, single source of truth for path remapping.

2. **Factor `findMMProjInDir`/`findMTPInDir` into generic finder** — consolidate the gguf-file search pattern. Estimated reduction: ~20 lines.

3. **Pass device path to `readVRAMSysfs`** — eliminate redundant GPU enumeration. Estimated reduction: ~15 lines.

4. **Move shard helpers to `models` package** — eliminate cross-package duplication. Low effort, removes duplicate code entirely from bench_export.go.

5. **Consider whether `itoa`/`ftoa` wrappers add enough value** to justify the extra indirection, or inline them with clear comments.