# Phase 04 — Results display, export, evaluation-data card, and docs

**Depends on:** 01, 02, 03 · **Enables:** nothing — final phase.

## Goal

Scores become visible and manageable: evaluation columns in the results
and compare tables, export coverage, the Evaluation Data card on the
Benchmarks tab (datasets + logits cache with delete), and help-page
documentation.

## Files touched

- `web/templates/benchmarks.html` / the results-table and compare
  partials (`web/templates/partials/benchmark_detail.html`,
  `benchmark_compare.html`, and the run-row rendering — exact partial
  confirmed at implementation from where timing columns render) — score
  columns.
- `internal/api/bench_export.go` — export columns.
- `internal/api/bench_jobs.go` or a new `internal/api/eval_data.go` —
  the Evaluation Data card's handlers.
- `internal/api/server.go` — route registration for the two new
  endpoints (all routes register there, e.g. `server.go:380`).
- `web/templates/partials/eval_data.html` (new) — the card.
- `web/templates/partials/bench_about.html` (and `internal/api/bench_about.go`
  only if the capability section needs data the template doesn't
  already get) — step 5's preset-table section and intro-copy fix.
- `web/templates/help.html` — Benchmarks section addition.
- Render tests following the existing partial-render pattern.

## Steps

1. Results columns, conditional: a job or run list containing any run
   with `Eval != nil` gains a "Score" column group; runs render their
   mode-appropriate value with its uncertainty and task/chunk count —
   `PPL 6.234 ±0.04 (100 chunks)` / `PPL 6.198 ±0.03 (full)` (the
   requested chunk cap from `EvalScores`; 0 renders as "full" — no
   actual-count parsing exists, phase 01 step 6),
   `HellaSwag 77.2% [75.9–78.5] (400)` (the
   asymmetric 95% confidence interval the tool reports, phase 01),
   `KLD 0.012 ±0.001 · same top token 97.4% (vs UD-Q8_K_XL)` (the
   top-token agreement the overview promises, from the `Same top p`
   statistic), `Winogrande 74.1% ±2.2 (400)` —
   em-dash in timing columns for capability runs and in the score
   column for performance runs. A KL cell skipped as the reference
   model (phase 03 step 4c) has no run at all — its `SkipReason`
   renders where cell status renders (the job's cell grid), styled as
   informational, not as an error. Tooltips explain each metric in
   plain language (lower perplexity is better; KL-divergence measures
   deviation from the reference, 0 = identical).
2. Compare view: the same conditional columns; comparing runs of
   different modes shows each row's own metric (the mode is part of the
   row identity, no cross-metric math).
3. Export: the CSV/JSON export gains the eval fields (mode, dataset,
   score, error, tasks/chunks, KL statistics, reference identity) —
   empty for performance runs, matching how per-mode columns already
   behave.
4. Evaluation Data card (Benchmarks tab, beneath the jobs list):
   - Datasets: name, license, size, state (not downloaded / verified),
     from phase 02's table + `Verify`.
   - KL reference logits: reference model + quant, dataset, chunks,
     size, age; per-row Delete (htmx post → refreshed card). Total size
     shown. Copy states these regenerate automatically when needed.
   - Handlers: `GET /api/benchmarks/eval-data` (card partial),
     `POST /api/benchmarks/eval-data/delete-logits` (form: the cache
     key fields). Deleting while a job runs is refused with the same
     busy message other job-conflicting actions use.
5. About-benchmarks modal (`bench_about.html`): its preset table's
   columns are performance-shaped (prompt/gen tokens, repetitions);
   capability presets get their own short section listing mode, task or
   chunk count, and what the score means — not em-dashes in columns
   that don't apply. The modal's intro copy ("Two source paths produce
   results, both via real HTTP requests through the running router")
   is also updated — capability evaluations load the model directly
   and do not go through the router, so the sentence becomes false as
   written.
6. Help page: a "Capability benchmarks" subsection under Benchmarks —
   what each mode measures, quick-vs-full guidance, dataset names and
   licenses (rendered from the phase 02 table's strings verbatim), the
   server-offline behavior during capability cells, KL reference
   selection and the logits cache's disk cost, and a plain-language note
   that sampling parameters do not affect these scores.
7. Render tests: run row with `Eval` set shows the score and em-dash
   timings; performance row unchanged; eval-data card renders both
   sections; export test asserts the new columns round-trip one
   capability run.

## Build gate

`go build ./... && go vet ./internal/... && go test ./internal/...`

## Test plan

- The render/export tests above.
- Manual end-to-end on the dev box (the overview's success criteria):
  four-quants job with `perplexity-quick` + `hellaswag-quick` → scores
  visible per cell and exported; KL job generates then reuses the base
  (visible step, then cache hit); mixed job yields t/s and accuracy
  side by side with the router restored afterward; kv_cache_quant sweep
  with `hellaswag-quick` shows per-value cells; logits delete from the
  card works and a re-run regenerates.

## Commit

`feat(benchmarks): capability score display, export, and evaluation data management`

## Rollback

Template/handler layer only; phases 01-03 remain fully functional (scores
in run JSON and export-by-API). No stored state beyond what 02 manages.
