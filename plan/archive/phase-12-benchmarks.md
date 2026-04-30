# Phase 12: Benchmarks

Add a benchmarking system to measure, store, and compare inference
performance across models, configurations, and llama.cpp builds.

---

## Goals

1. **Measure** — run standardized benchmarks capturing tokens/sec for
   both prompt processing (prefill) and token generation (decode).
2. **Store** — persist results with full context: model, quant, config,
   build, GPU assignment, so runs are comparable.
3. **Compare** — side-by-side tables and charts for any combination of
   models, configs, and builds.
4. **Passive collection** — capture `timings` from every proxied
   chat completion response for ongoing performance visibility.

---

## Data Sources

### 1. llama-server `timings` field (API-based)

Every `/v1/chat/completions` response from llama.cpp includes:

```json
{
  "timings": {
    "prompt_n": 512,
    "prompt_ms": 245.3,
    "prompt_per_token_ms": 0.479,
    "prompt_per_second": 2087.4,
    "predicted_n": 128,
    "predicted_ms": 2150.7,
    "predicted_per_token_ms": 16.8,
    "predicted_per_second": 59.5
  }
}
```

This is our primary data source for benchmarks. We send a known prompt
with known parameters and capture the timings from the response.

### 2. llama-bench binary (raw inference)

`llama-bench` ships with every llama.cpp build. It benchmarks raw model
inference without server overhead — no HTTP, no sampling, no tokenization.
Outputs JSON with `avg_ts` (tokens/sec) for prompt processing and
token generation independently.

Useful as a ceiling/comparison point: "how fast *could* this model go
on this hardware?"

### 3. Passive collection from real usage

The proxy handler intercepts every chat completion response. We can
parse the `timings` field and store a rolling window of real-world
performance observations per model.

---

## Architecture

### New package: `internal/benchmark/`

```
internal/benchmark/
├── benchmark.go     — types, storage (load/save JSON)
├── runner.go        — benchmark execution (API-based + llama-bench)
└── stats.go         — aggregation, percentiles, comparison helpers
```

### Key Types

```go
// BenchmarkRun — one complete benchmark execution
type BenchmarkRun struct {
    ID          string    `json:"id"`           // uuid
    CreatedAt   time.Time `json:"created_at"`
    Status      string    `json:"status"`       // "running", "completed", "failed"
    Error       string    `json:"error,omitempty"`

    // What was tested
    ModelID     string    `json:"model_id"`
    ModelName   string    `json:"model_name"`   // human-readable
    Quant       string    `json:"quant"`
    SizeGB      float64   `json:"size_gb"`

    // Configuration snapshot (frozen at benchmark time)
    Config      ConfigSnapshot `json:"config"`

    // Which build of llama.cpp
    BuildID     string    `json:"build_id"`
    BuildRef    string    `json:"build_ref"`
    BuildProfile string   `json:"build_profile"` // "rocm", "cuda", "cpu"

    // Hardware context
    GPUs        []GPUSnapshot `json:"gpus"`

    // Benchmark parameters
    Preset      string    `json:"preset"`       // "quick", "standard", "thorough"
    PromptTokens []int    `json:"prompt_tokens"` // token counts tested, e.g. [128, 512, 2048]
    GenTokens   int       `json:"gen_tokens"`    // tokens to generate per test

    // Results
    Results     []BenchmarkResult `json:"results,omitempty"`
    Summary     *BenchmarkSummary `json:"summary,omitempty"`

    // Optional: raw llama-bench results
    LlamaBench  *LlamaBenchResult `json:"llama_bench,omitempty"`
}

// ConfigSnapshot — frozen copy of model config at benchmark time
type ConfigSnapshot struct {
    GPULayers      int    `json:"gpu_layers"`
    ContextSize    int    `json:"context_size"`
    GPUAssign      string `json:"gpu_assign"`
    TensorSplit    string `json:"tensor_split"`
    FlashAttention bool   `json:"flash_attention"`
    KVCacheQuant   string `json:"kv_cache_quant"`
    Threads        int    `json:"threads"`
    SpecType       string `json:"spec_type,omitempty"`
}

// GPUSnapshot — GPU hardware at benchmark time
type GPUSnapshot struct {
    Index      int    `json:"index"`
    Name       string `json:"name"`
    VRAMTotalMB int   `json:"vram_total_mb"`
}

// BenchmarkResult — one test point (one prompt size, one repetition)
type BenchmarkResult struct {
    PromptTokens      int     `json:"prompt_tokens"`
    GenTokens         int     `json:"gen_tokens"`
    Repetition        int     `json:"repetition"`
    PromptTokPerSec   float64 `json:"prompt_tok_per_sec"`
    GenTokPerSec      float64 `json:"gen_tok_per_sec"`
    TimeToFirstTokenMs float64 `json:"ttft_ms"`
    TotalMs           float64 `json:"total_ms"`
}

// BenchmarkSummary — aggregated stats across all results
type BenchmarkSummary struct {
    AvgPromptTokPerSec float64 `json:"avg_prompt_tok_per_sec"`
    AvgGenTokPerSec    float64 `json:"avg_gen_tok_per_sec"`
    AvgTTFTMs          float64 `json:"avg_ttft_ms"`
    MinGenTokPerSec    float64 `json:"min_gen_tok_per_sec"`
    MaxGenTokPerSec    float64 `json:"max_gen_tok_per_sec"`
}

// LlamaBenchResult — raw inference benchmark (no server overhead)
type LlamaBenchResult struct {
    PromptTokPerSec float64 `json:"pp_avg_ts"`
    GenTokPerSec    float64 `json:"tg_avg_ts"`
    PromptTokens    int     `json:"pp_tokens"`
    GenTokens       int     `json:"tg_tokens"`
    Repetitions     int     `json:"repetitions"`
}
```

### Storage

File: `{dataDir}/config/benchmarks.json` — array of `BenchmarkRun`.
Follows the same pattern as `builds.json` and `models.json`.

---

## Benchmark Presets

| Preset | Prompt sizes | Gen tokens | Repetitions | llama-bench | ~Duration |
|--------|-------------|------------|-------------|-------------|-----------|
| **Quick** | 256 | 128 | 1 | no | ~10s |
| **Standard** | 128, 512, 2048 | 128 | 3 | yes | ~2min |
| **Thorough** | 128, 512, 2048, 8192 | 256 | 5 | yes | ~10min |

The prompt text is a fixed, deterministic passage (e.g., first N tokens
of a public-domain text) to ensure reproducibility. We store the prompt
token count as reported by `timings.prompt_n`, not our estimate.

---

## Benchmark Execution (API-based)

### Runner flow

1. **Pre-load model** — call `PUT /api/models/{id}/activate` (the API
   route still exists, just not exposed in the UI). This triggers
   `/models/load` on the router, loading weights into VRAM. Wait for
   the model status to show "loaded" before proceeding.
2. **Warmup**: send one throwaway request (discarded) — handles JIT
   kernel compilation and any remaining first-request overhead.
3. For each prompt size in the preset:
   a. Build a prompt of approximately N tokens (repeat fixed text)
   b. For each repetition:
      - `POST /v1/chat/completions` with `stream: false`, `max_tokens: genTokens`
      - Parse `timings` from response
      - Record `BenchmarkResult`
   c. Log progress
4. If preset includes llama-bench:
   - Run `llama-bench -m <model_path> -p <pp> -n <tg> -r <reps> -o json -ngl <ngl> -fa <fa> -t <threads>`
   - Parse JSON output
5. Compute `BenchmarkSummary` (averages, min, max)
6. Save run

### Prompt construction

Use a fixed text block (stored as a Go constant). Repeat/truncate to
approximate the target token count. The actual token count comes back
in `timings.prompt_n` — we store that, not our approximation.

### llama-bench execution

The binary lives alongside `llama-server` in the build directory:
`{buildDir}/llama-bench`. We pass the same GPU/thread config the model
uses. Key flags:

```
llama-bench \
  -m /data/models/.../model.gguf \
  -p 512 -n 128 -r 5 \
  -ngl 999 -fa 1 -t 8 \
  -ts 1,1,0,0 \     # match tensor-split
  -o json
```

---

## Passive Timings Collection

### Proxy response capture

Modify `proxy.go` to wrap the response and parse `timings` from
non-streaming completions. For streaming responses, the final
`data: [DONE]` chunk in llama.cpp includes timings in a preceding
chunk — we can capture that too.

### Storage

```go
// TimingSample — one observed timing from real usage
type TimingSample struct {
    Timestamp       time.Time `json:"ts"`
    ModelID         string    `json:"model"`
    PromptTokens    int       `json:"prompt_n"`
    GenTokens       int       `json:"gen_n"`
    PromptTokPerSec float64   `json:"prompt_tps"`
    GenTokPerSec    float64   `json:"gen_tps"`
}
```

Store in a ring buffer (last 1000 samples per model) in memory.
Expose via API for the UI to show recent performance. Optionally
persist to `{dataDir}/config/timings.json` on shutdown.

This gives a "how is this model performing in practice?" view without
running explicit benchmarks.

---

## API Endpoints

### Benchmark management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/benchmarks` | List all benchmark runs (HTMX table or JSON) |
| `POST` | `/api/benchmarks` | Start a new benchmark run |
| `GET` | `/api/benchmarks/{id}` | Get a single run with full results |
| `DELETE` | `/api/benchmarks/{id}` | Delete a benchmark run |
| `GET` | `/api/benchmarks/{id}/progress` | SSE stream for live progress |
| `GET` | `/api/benchmarks/compare` | Compare selected runs (query: `?ids=a,b,c`) |

### Passive timings

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/timings` | Recent timing samples (all models) |
| `GET` | `/api/timings/{model_id}` | Recent timings for a specific model |

### Start benchmark form values

```
model_id:  "unsloth--DeepSeek-R1-0528-Qwen3-8B-GGUF--..."
preset:    "standard"       // quick, standard, thorough
```

The runner snapshots the current config, build, and GPU state
automatically — no need to specify those in the form.

---

## UI: Benchmarks Page

### Page layout: `/benchmarks`

```
┌─────────────────────────────────────────────────────┐
│  Run Benchmark                                       │
│  ┌────────────┐  ┌──────────┐                       │
│  │ Model ▼    │  │ Preset ▼ │  [Run Benchmark]      │
│  └────────────┘  └──────────┘                       │
│  (model dropdown: only loaded/enabled models)        │
│  (preset dropdown: Quick / Standard / Thorough)      │
├─────────────────────────────────────────────────────┤
│  Benchmark Results                          [Compare]│
│  ┌───┬────────────┬───────┬─────────┬───────┬──────┐│
│  │   │ Model      │ Quant │ PP t/s  │ TG t/s│ Build││
│  ├───┼────────────┼───────┼─────────┼───────┼──────┤│
│  │ ☐ │ Qwen3.5-27B│ Q4_K_M│ 1,842   │ 48.2  │ b8461││
│  │ ☐ │ Qwen3.5-27B│ Q8_0  │ 1,205   │ 31.7  │ b8461││
│  │ ☐ │ DeepSeek-R1│ Q8_0  │ 2,087   │ 59.5  │ b8461││
│  │ ☐ │ granite-4.0│ Q4_K_M│   892   │ 22.1  │ b8461││
│  └───┴────────────┴───────┴─────────┴───────┴──────┘│
│  (checkbox to select runs for comparison)            │
├─────────────────────────────────────────────────────┤
│  Live Performance (from passive timings)             │
│  DeepSeek-R1-8B: avg 58.3 t/s (last 25 requests)   │
│  Qwen3.5-27B:    avg 31.2 t/s (last 12 requests)   │
└─────────────────────────────────────────────────────┘
```

### Comparison view: `/benchmarks` with compare modal/section

When user selects 2+ runs and clicks "Compare":

```
┌─────────────────────────────────────────────────────┐
│  Comparison: 3 runs selected                        │
│                                                      │
│  Generation Speed (tokens/sec)                       │
│  ████████████████████████████████████  59.5  DR1-8B  │
│  ██████████████████████              48.2  Qwen-27B  │
│  ██████████████                      31.7  Qwen-Q8   │
│                                                      │
│  Prompt Processing (tokens/sec)                      │
│  ████████████████████████████████████ 2,087  DR1-8B  │
│  ████████████████████████████████    1,842  Qwen-27B  │
│  ████████████████████████           1,205  Qwen-Q8   │
│                                                      │
│  Details Table                                       │
│  ┌────────────┬────────┬────────┬────────┬─────────┐│
│  │            │ DR1-8B │Qwen-27B│Qwen-Q8 │         ││
│  ├────────────┼────────┼────────┼────────┤         ││
│  │ PP t/s     │ 2,087  │ 1,842  │ 1,205  │         ││
│  │ TG t/s     │  59.5  │  48.2  │  31.7  │         ││
│  │ TTFT (ms)  │   245  │   278  │   425  │         ││
│  │ Quant      │ Q8_0   │ Q4_K_M │ Q8_0   │         ││
│  │ Size       │ 8.1 GB │ 16.4 GB│ 28.1 GB│         ││
│  │ GPUs       │ 0      │ 0-1    │ all    │         ││
│  │ Context    │ 8192   │ 65536  │ 8192   │         ││
│  │ KV Quant   │ —      │ q8_0   │ —      │         ││
│  │ Flash Attn │ ✓      │ ✓      │ ✓      │         ││
│  │ Build      │ b8461  │ b8461  │ b8461  │         ││
│  │ llama-bench│ 62.1   │ 51.3   │ 34.2   │ TG t/s  ││
│  └────────────┴────────┴────────┴────────┴─────────┘│
└─────────────────────────────────────────────────────┘
```

Horizontal bar charts are CSS-only (same technique as GPU map).
No JS charting libraries needed.

### Benchmark detail view

Clicking a row expands to show:
- Per-prompt-size breakdown (table)
- Config snapshot
- llama-bench raw results (if run)
- Individual repetition data

### Live progress

While a benchmark is running, SSE stream updates a progress indicator:
"Running: 512 tokens, rep 2/3..." with a progress bar.

---

## Modified Files

### `internal/api/server.go`
- Add `/benchmarks` page route
- Register `/api/benchmarks/*` and `/api/timings/*` API routes
- Add `benchmarks.html` to page templates

### `internal/api/proxy.go`
- Wrap response to capture `timings` from non-streaming completions
- For streaming: capture timings from the final data chunk before `[DONE]`
- Store samples via benchmark store's `RecordTiming()` method

### `internal/api/server.go` (Server struct)
- Add `bench *benchmark.Store` field

### New files
- `internal/benchmark/benchmark.go` — Store type, load/save, CRUD
- `internal/benchmark/runner.go` — benchmark execution goroutine
- `internal/benchmark/stats.go` — aggregation and comparison helpers
- `internal/api/bench.go` — HTTP handlers for benchmark endpoints
- `web/templates/benchmarks.html` — page template
- `web/templates/partials/benchmark_run.html` — single run row/detail
- `web/templates/partials/benchmark_compare.html` — comparison view
- `web/templates/partials/benchmark_progress.html` — live progress partial
- `web/templates/partials/timings_summary.html` — passive timings display

---

## Implementation Order

1. **Types & storage** (`benchmark.go`) — `BenchmarkRun`, `TimingSample`,
   `Store` with JSON persistence. No dependencies.

2. **Runner** (`runner.go`) — API-based benchmark execution with preset
   support. Sends requests, parses timings, runs llama-bench.

3. **Stats** (`stats.go`) — summary computation, comparison data
   structures.

4. **API handlers** (`bench.go`) — CRUD endpoints, progress SSE,
   compare endpoint.

5. **Passive timings** (`proxy.go` modification) — response capture
   for non-streaming and streaming completions.

6. **Benchmarks page** (`benchmarks.html` + partials) — run form,
   results table, comparison view, live performance section.

7. **Server wiring** (`server.go`) — routes, page registration,
   store initialization.

8. **Nav link** — add "Benchmarks" to the layout nav.

---

## Verification

1. `go build ./...` — clean compile
2. Start server, load a model, run Quick benchmark — verify results stored
3. Run Standard benchmark — verify multiple prompt sizes, llama-bench included
4. Compare two runs — verify comparison table renders
5. Send chat completions — verify passive timings appear
6. Benchmark with different KV quant / flash attn — verify config snapshot differs
7. Benchmark same model on different builds — verify build info captured
8. Delete a benchmark run — verify removed from list
9. No models loaded — verify graceful error message

---

## Out of Scope (Future)

- Concurrency/throughput benchmarking (multiple simultaneous requests)
- Automated quant comparison (download + benchmark all quants of a model)
- Performance regression detection across builds
- Export/import benchmark results
- Benchmark scheduling (run overnight, etc.)
