# Code Duplication Audit

## Overview
This document outlines potential code duplication issues found in the llama-toolchest codebase based on a review conducted on June 24, 2026.

## Findings

### 1. Helper Functions in `bench_export.go`
- **Location**: `internal/api/bench_export.go`
- **Functions**: 
  - `func itoa(n int) string`
  - `func ftoa(f float64) string`
  - `func metricMean(m *benchmark.LlamaBenchyMetric) string`
  - `func metricStd(m *benchmark.LlamaBenchyMetric) string`
- **Analysis**: 
  - These helper functions are defined only once in the codebase.
  - However, the `itoa` and `ftoa` functions are simple wrappers around `strconv.Itoa` and `strconv.FormatFloat` respectively.
  - Throughout the codebase, direct calls to `strconv.Itoa` and `strconv.FormatFloat` are used (e.g., in `internal/models/registry.go`, `internal/benchmark/benchy.go`), indicating that these helper functions are not being utilized elsewhere.
  - This represents a missed opportunity for centralization rather than active duplication.

### 2. Job Lookup Mechanisms
- **Location**: `internal/api/bench_export.go`
- **Definitions**:
  - `type jobLookup map[string]*benchmark.BenchmarkJob`
  - `func newJobLookup(jobs []*benchmark.BenchmarkJob) jobLookup`
- **Analysis**: 
  - These are defined only once and used within the same file.
  - No duplication found.

### 3. KV Scaling Computation
- **Location**: 
  - Definition: `internal/models/gguf.go` (`func computeKVScaling`)
  - Usage: Same file and test file `internal/models/vram_test.go`
- **Analysis**: 
  - The function is defined once in production code and used in its defining file.
  - Test usage in `vram_test.go` is appropriate for unit testing.
  - No duplication in production code.

### 4. GGUF Metadata Application
- **Location**: `internal/models/gguf.go`
- **Definition**: `func (meta *GGUFMeta) ApplyTo(m *Model)`
- **Analysis**: 
  - Defined once and used within the same file.
  - No duplication found.

## Recommendations
1. **Consider removing or using helper functions**: The `itoa` and `ftoa` helpers in `bench_export.go` are not used elsewhere. Either remove them (if redundant) or refactor other parts of the codebase to use them for consistent string conversion.
2. **Centralize metric helpers**: The `metricMean` and `metricStd` functions are currently only used in `bench_export.go`. If similar metric formatting is needed elsewhere, consider moving these to a shared utilities package.
3. **General**: While no significant active duplication was found in the areas investigated, maintain awareness of creating helper functions that are then not adopted across the codebase.

## Conclusion
The audit revealed no instances of identical or near-identical code blocks that constitute harmful duplication. The primary observation is the presence of underused helper functions that could promote consistency if adopted more broadly.