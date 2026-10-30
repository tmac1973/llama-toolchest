# Phase 05 — Record events and usage in the server

**Depends on:** 03, 04 · **Enables:** Phase 06 renders real data; the JSON endpoint works from this phase on

## Goal

Connect the parser and the store to the running toolchest: a second subscriber on the router log, router starts with their reason, job tags, the build and GPUs of each instance, a WARN line per crash, usage from the proxy for every `/v1` model request, and JSON endpoints to read and delete the data. After this phase a crash on compute2 is saved and visible through `GET /api/service/notes`, with no UI yet.

## Files touched

- `internal/api/server.go` — `notes *enginenotes.Store`, `notesOpen atomic.Pointer[[]enginenotes.OpenInstance]` and `notesJob atomic.Pointer[jobRef]` fields; open the store in the constructor at `filepath.Join(cfg.DataDir, "engine-notes.json")`; `go s.watchEngineNotes()` next to `go s.watchRouterMemory()`; `s.notes.Close()` at the end of `Server.Shutdown`, after `s.process.Stop()`; register the routes.
- `internal/api/engine_notes.go` — new: `watchEngineNotes`, `recordOutcome`, `jobTag`, `notesSummary`, the HTTP handlers.
- `internal/api/engine_notes_test.go` — new.
- `internal/api/service.go` — in `startRouterWith`, after a successful `s.process.Start`, set `s.notesJob` and record a `router_start` event.
- `internal/api/proxy.go` — usage accounting in `ModifyResponse` and for failed requests.
- `internal/api/bench.go` — none; `captureTimings` stays as it is (usage is counted on its own path, so Live Performance does not change).

## Steps

1. **Watcher.** `watchEngineNotes` creates the `Tracker` as a local variable, subscribes with `s.process.Subscribe()`, and loops over lines, calling `tracker.Add(line, time.Now())`. For each outcome it calls `s.recordOutcome(o)` inline, in order. Everything `recordOutcome` reads is in memory and guarded by short locks (step 2), so the reader never falls behind (a slow subscriber loses lines, see `watchRouterMemory`). After any line that produced an outcome it stores `tracker.Open()` in `s.notesOpen`; HTTP handlers read that pointer, never the tracker, which only this goroutine touches.
2. **Enrichment** in `recordOutcome`:
   - Model: `s.findModelByAny(o.RouterName)` → `ModelID` and `PublicName()`. Unknown → `ModelID = RouterName`, `ModelName = RouterName`.
   - Config: `s.runningConfigFor(modelID)` if present (the config the router launched), else the registry config. `GPUs = models.GPUAssignLabel(cfg.GPUAssign)`, plus ", tensor" when `cfg.SplitMode == "tensor"`.
   - `Build = s.runningBuild()`.
   - Job: `jobTag()` returns `*s.notesJob` while `s.env.routerOwnedByJob()` is true, and empty otherwise. `jobRef` is `{ID, Name string}`. Both reads are in memory; the job store is never read on the log reader's path.
3. **Router starts.** In `startRouterWith`, after `s.process.Start` succeeds and `setRunningBuild` is called: when `opt.job` is true, read `s.jobs.Status()` (this runs on the job's goroutine, not the log reader) and store `{ID, Name}` in `s.notesJob`; otherwise store nil. Then `s.notes.Record(Event{Kind: "router_start", StartReason: "job" if opt.job else "user", Build: build.ID, JobID/JobName from s.notesJob})`. A start at toolchest boot counts as `"user"`.
   **Router stops and exits** are recorded from the tracker's `router_stop` and `router_exit` outcomes like any other outcome.
4. **WARN line.** For `crash` and `load_failed`, `slog.Warn("llama-server model crashed", "model", name, "exit_status", n, "build", b, "gpus", g, "job", jobName, "error", firstErrorLine)`; the message for `load_failed` is "llama-server model failed to load". `router_exit` logs `slog.Warn("llama-server router exited on its own", "error", o.RouterError)`.
5. **Usage.** In `proxy.ModifyResponse`, before the chat-only filter, for request paths ending in `/chat/completions`, `/completions`, `/embeddings` or `/rerank` (with or without the `/v1` prefix):
   - Skip when `s.env.routerOwnedByJob()` is true (job traffic stays out of usage; benchmark runs also bypass the proxy, so this covers a client request sent while a job runs).
   - Status not 200 → `s.notes.AddUsage(model, 0, 0, true)`. The model comes from a request-context value set by the proxy handler after `loadForProxy` resolved it (add `withProxyModel(ctx, modelID)` / `proxyModelFrom(ctx)`); no value → not counted.
   - Completions, JSON body: tokens from the existing `timings` parse (`prompt_n`, `predicted_n`). Streaming: wrap the existing `newSSETimingCapture` callback so the same final-timings event also calls `AddUsage`. Live Performance keeps getting its samples exactly as now.
   - Embeddings and rerank: read `usage.prompt_tokens` (or `usage.total_tokens` when `prompt_tokens` is absent) from the JSON body; generated tokens 0. No `usage` object, or a body that does not parse → one request with 0 tokens. Each response is counted exactly once.
   - A completion response that ends without timings counts as one request with 0 tokens. JSON: a completion body that parses but has no `timings` (or does not parse) → `AddUsage(model, 0, 0, false)`. Streaming: add an `onEnd(sawTimings bool)` callback to the SSE capture wrapper, called once from its `Close` (or on EOF, whichever is first); `sawTimings == false` → `AddUsage(model, 0, 0, false)`. The model for both comes from the request context value described in the failed-request bullet above.
6. **Endpoints**, under the existing `/service` route group:
   - `GET /api/service/notes?period=24h|7d|30d|all` → JSON `Summary` (default `24h`). HTMX rendering comes in Phase 06.
   - `POST /api/service/notes/delete` with form field `older_than=7d|30d|90d|all` → runs `DeleteOlderThan` or `DeleteAll`, answers JSON `{removed_events, removed_hours}`. Unknown value → 400.
7. **Shutdown.** `s.notes.Close()` runs after the router is stopped on toolchest shutdown, so the router's last lines are recorded first.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- `engine_notes_test.go`, with a fake process whose broadcast the test writes lines into (the pattern the memory-measured tests use): a crash fixture from Phase 03 produces a saved `crash` event with model ID, build and GPUs filled in.
- Job tag: with a fake job owning the router, the crash event carries the job ID and name, and the summary's "crashes in jobs" is 1.
- Router start reasons: `startRouterWith(routerOptions{job: true})` records `router_start` with reason `job`; `startRouter()` with `user`.
- Proxy usage: a JSON chat completion with timings adds one request and its tokens; an SSE completion adds one on the final chunk; an embeddings response adds `prompt_tokens`; a 500 response adds a failed request; a request while a job owns the router adds nothing.
- Delete endpoint: `older_than=30d` removes only older data; `older_than=bogus` → 400.
- Manual on compute2 (binary swapped in with `docker cp`, see the compute2 memory notes): load and unload a model, then `curl compute2:3000/api/service/notes?period=24h` shows one load and one unload; run a benchmark job and check the router starts are tagged `job`; `docker logs` shows a WARN line at the next real crash.

## Commit

`feat(server): record llama-server starts, crashes and model usage in engine notes`

## Rollback

Revert the commit. The watcher, the proxy hook and the endpoints go away; `engine-notes.json` stays on disk unused and is safe to delete. Phases 03 and 04 remain as unused code.

## As built

- `recordOutcome` uses the outcome's own time, so an event is dated by when its line arrived.
- `noteRouterStart(job, build)` in `engine_notes.go` holds the router-start record and the job lookup; `startRouterWith` calls it after `setRunningBuild`.
- The SSE capture now reports timings even when no chunk named a model. Live Performance still needs a model and skips such streams as before; usage counts them.
- A usage hour now counts when any part of it is in the period, not only when it starts there. Before this, the hour in which the record began (the first hour after "delete all") was left out of every summary. Fixed in `internal/enginenotes/summary.go`, with a test.
- The fake router in `proxy_test.go` has a `v1` handler field for tests that need `/v1` answers.
