# Capability Benchmarks — Project Overview

## Problem

The benchmark tab measures only speed. Nothing in the toolchest can answer
the questions its own features raise: how much quality does Q4_K_M give up
against UD-Q8_K_XL, does a quantized KV cache hurt reasoning, does a build
or flag change alter model output quality at all. Users step outside the
tool (or guess) for exactly the comparisons the tool's matrix machinery is
built to run.

llama.cpp already ships the measurement tool: `llama-perplexity`, compiled
by every build the toolchest makes (ninja builds all targets; the install
step just doesn't copy it). It implements perplexity, KL-divergence against
a reference model's logits, HellaSwag, Winogrande, and an MMLU-style
multiple-choice mode — with HellaSwag preprocessing deliberately matched to
EleutherAI's lm-evaluation-harness so scores are comparable to published
numbers.

## Goals

- Four evaluation modes in v1, as **capability presets** in the existing
  job matrix next to the performance presets: perplexity (wikitext-2),
  KL-divergence (vs a reference quant), HellaSwag, and Winogrande. One job
  can mix performance and capability presets — sweep KV cache quant and get
  tokens/sec and HellaSwag accuracy per cell from the same job.
- Two presets per mode: a **quick** variant sized for sweeps (HellaSwag
  400 tasks, Winogrande 400 tasks, perplexity over a bounded chunk count —
  the sizes llama.cpp's own documentation quotes for comparisons) and a
  **full** variant for publishable numbers, each with a duration hint in
  its label matching the existing preset naming style.
- KL-divergence reference logits: a KL cell names its reference model,
  defaulting to the largest installed quant of the same HF repo. A missing
  logits file is generated as its own visible job step, then cached in the
  data dir keyed by (reference model identity, dataset, chunk count,
  context size) for reuse across jobs. The multi-GB files get a disk-space guard before generation and a
  delete control in the UI.
- Datasets (a few MB total) download on first use from pinned URLs with
  SHA-256 verification, cached in the data dir — the HF tokenizer-cache
  pattern. Every dataset is redistributable (wikitext-2: CC BY-SA;
  HellaSwag: MIT; Winogrande: Apache-2.0) and its license is named in the
  help page.
- `llama-perplexity` joins `llama-server` in the build install step. Builds
  made before this change lack the binary; a capability cell on such a
  build fails with a message naming the fix (rebuild), never a silent skip.
- Scores land on `BenchmarkRun` as optional fields (perplexity ± error,
  KL-divergence statistics, task accuracy with confidence bounds, tasks
  run), render in
  the results/compare tables alongside the timing columns, and ride the
  existing export.
- Config fidelity: capability cells map the applicable subset of the
  model's `ModelConfig` (and job overrides) onto `llama-perplexity` flags —
  GPU layers, GPU placement/device, flash attention, KV cache quant,
  batch/ubatch, threads — so a swept parameter measurably reaches the
  evaluation. The recorded config snapshot tells the truth about what ran.

## Non-goals

- No MMLU-style multiple-choice mode in v1 — its dataset uses llama.cpp's
  own binary format with no clean public source; it can follow once the
  dataset story is settled.
- No generative (through-the-router) benchmarks — GSM8K/IFEval are a later
  tier with different machinery.
- No external harness integration (lm-evaluation-harness, simple-evals)
  beyond help-page documentation; no judge models; no execution of
  model-generated code.
- No custom or user-supplied datasets in v1; the pinned three
  (wikitext-2, HellaSwag, Winogrande — wikitext-2 serves both
  perplexity and KL-divergence) are the offering.
- Sampling parameters (temperature, top-p, …) are not applied to
  capability cells — the evaluations are deterministic log-likelihood
  scoring where sampling has no role. A job may still sweep them for its
  performance presets; capability cells record the load-time config only.

## Users & primary flow

Self-hosters comparing quants, builds, and settings on their own hardware.

Creating a job: Benchmarks → New Job → pick models (e.g. every installed
quant of one repo), a build, and presets — checking, say,
"hellaswag-quick" and "perplexity-quick" alongside "internal-standard".
For KL presets, a reference-model selector appears, prefilled with the
largest installed quant of each selected model's repo. The matrix count
and duration hints update as usual.

Running: capability cells load the model directly with `llama-perplexity`,
so the job runner stops the inference server for their duration (the
existing router-ownership machinery) and restores it afterward, exactly as
it already does around config changes. A KL cell missing its cached
reference logits generates them first as a distinct progress step.

Reading results: each cell row shows its scores — perplexity with its
error bound, HellaSwag/Winogrande accuracy with confidence interval and
task count, KL-divergence mean and top-token agreement — next to the
timing columns of performance cells from the same job. Compare and export
include the new columns.

## Constraints

- Go backend; the evaluation binary is invoked as a subprocess and its
  combined stdout+stderr is parsed (llama-perplexity prints final scores
  in stable, greppable lines, split across the two streams by its
  logger); no Python anywhere.
- All evaluations run at a fixed context size of 512 tokens — required
  by `llama-perplexity`, score-determining (it is the perplexity chunk
  size), and the value llama.cpp's published wikitext figures use, so
  scores stay comparable.
- Capability cells cannot run through the router: `llama-perplexity` loads
  the model itself. The inference server is therefore offline while a
  capability cell runs — the job queue already owns the router during jobs
  (`routerBusyWithJob` guards), and cleanup already restores it.
- VRAM exclusivity extends to the KL logits generation step, which loads
  the (large) reference model.
- The flag mapping is a subset: parameters meaningless to the evaluation
  (parallel slots, context-shift, sampling, speculative decoding) are
  excluded by listing, not silently dropped — the mapping function is the
  single source shared by execution and the recorded snapshot.
- Old builds lack `llama-perplexity`; detection is by file existence at
  cell start with an actionable error.
- Dataset downloads use plain `net/http` — these are small static files
  with pinned URLs, not HF-API objects, so the HF client's auth/resume
  machinery isn't needed; SHA-256 pins live next to the URLs in one
  table.
- Logits cache location: under the data dir with the other caches;
  size shown and deletable from the UI (the disk-space guard reuses
  `AvailableForDownload`'s pattern).
- Score fields are additive on `BenchmarkRun` (older records simply lack
  them — same pattern as every schema addition in this codebase).

## Success criteria

- A single job over four quants of one repo with "perplexity-quick" +
  "hellaswag-quick" produces a table where quality degradation across
  quants is visible per cell, exported with the same columns.
- A KL-divergence job on a Q4 quant with a Q8 reference: first run
  generates and caches the logits file (visible as its own step), second
  run reuses the cache and completes in a fraction of the time.
- A job mixing "internal-standard" with capability presets yields both
  t/s and accuracy per cell; the inference server is back up when the job
  ends, including on failure and cancel paths.
- Sweeping kv_cache_quant f16 vs q4_0 with "hellaswag-quick" shows the
  accuracy difference (or its absence) — the swept flag provably reaches
  `llama-perplexity`, verified by the recorded snapshot and a test
  asserting the flag mapping.
- A capability cell on a pre-existing build without `llama-perplexity`
  fails with a message that names rebuilding as the fix.
- Datasets download once, verify against pinned hashes, and are reused;
  a corrupted download is rejected with a clear error.
- `go build ./... && go test ./internal/...` passes; help page documents
  the modes, datasets, licenses, and the server-offline behavior.

## Decisions

- **Mode scope** → Perplexity, KL-divergence, HellaSwag, Winogrande in v1;
  MMLU-style multiple-choice deferred until its dataset story is cleaner.
- **Job shape** → Capability presets in the same job matrix as performance
  presets; one job can mix both; cells/retry/compare/export machinery
  reused.
- **KL reference logits** → Generated on demand as a visible job step,
  cached keyed by (reference model, dataset, chunk count, context
  size), user-selectable reference defaulting to the largest installed
  quant of the same repo; disk-space guard and delete control.
- **Datasets** → Download on first use from pinned URLs with SHA-256
  verification, cached in the data dir.
- **Run sizes** → Quick + full preset per mode, with duration hints in the
  labels.
