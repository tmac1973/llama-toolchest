# Code Audit Report — 2026-10-27

Produced by the `/code-audit` skill (six parallel agents: standards,
duplication, logging, security, testing, dependencies). The
`saveBuilds`/`saveFlagPresetsLocked` finding was checked by hand against the
code. Items fixed after [CODE-AUDIT-2026-10-05.md](CODE-AUDIT-2026-10-05.md)
are not repeated.

**Project**: Go 1.26 web app (chi + html/template + htmx, `log/slog`), vanilla JS with node tests
**Scope**: full project (vendored minified assets, plan/audit/docs/dist excluded)
**Findings**: 61 total (8 high, 23 medium, 30 low)

Severity changes made when checking by hand:

- No login on `/api` was downgraded from High to Low. It was decided on
  2026-10-05 to keep the documented default; `--secure` covers it.
- No tests for `cmd/agent` was downgraded from High to Medium. Its `bash -c`
  tool can already reach any path, so the missing `..` check adds little
  risk.
- govulncheck not installed was downgraded from Medium to Low. It was run
  through `go run` (v1.9.0) and found no vulnerabilities.

Each item has a checkbox; tick it in the PR that fixes it.

---

## High Priority

### Standards
- [x] `saveBuilds` and `saveFlagPresetsLocked` ignore the errors from `MkdirAll` and `WriteFile` and do not use `atomicfile`, so a failed or interrupted write can lose or corrupt builds.json or the flag presets without any message — `internal/builder/builder.go:1201`, `internal/builder/flag_presets.go:81`

### Duplication
- [x] The image, video and music job handlers are three copies of one set (`*JobOr404`, list, delete, `*Enabled()`); only the store, the job name and the cancel function differ — `internal/api/gen_image_jobs.go:327` (and `gen_video_jobs.go:505`, `gen_music.go:692`) *(`jobRoutes[T]` in gen_jobs.go; one `genEnabled`.)*
- [x] The `restored()`, `withPosition()` and `state()` record methods are nearly the same in all three job types — `internal/api/gen_image_jobs.go:80` (and `gen_video_jobs.go:99`, `gen_music.go:289`) *(the shared fields are in an embedded `genJob`, with `jobState` and `restore`; each type keeps a short `restored` for its private part and a one-line `withPosition`.)*
- [x] The run-job code is copied three times (the `finish` closure, `ticket.Wait`/`Done`, the "Starting" update, `genReady`) — `internal/api/gen_image_jobs.go:143` (and `gen_video_jobs.go:394`, `gen_music.go:387`) *(the starting and generating updates share `startJob` and `genJob.start`. The `finish` closures stay separate: video cancels by a request flag and closes `done`, image removes late output files, and music clears its source audio.)*
- [x] The error handling at the end of each create handler is copied: the pickError mapping 4 times, the ErrQueueFull 429 reply 4 times, the 202 reply 3 times; `pickError` and `videoError` are the same struct *(Duplication + Standards)* — `internal/api/gen_images.go:168` (and `gen_image_jobs.go:359`, `gen_music.go:667`, `gen_video_jobs.go:142`)

### Testing
- [ ] The benchmark job HTTP handlers have almost no tests (create, update, cancel, retry, export, `decodeJobRequest`, `resolveSweeps`) — `internal/api/bench_jobs.go:271`
- [ ] The service and model handlers are untested beyond one stop request (start, restart, logs, `resolveRouterModel`, delete and scan models) — `internal/api/service.go`, `internal/api/models.go`
- [ ] The HuggingFace client (`Search`, `GetModel`, `populateFileSizes`) and the download and cancel handlers have no tests — `internal/huggingface/client.go:70`, `internal/api/hf.go`

## Medium Priority

### Standards
- [ ] Fields that Settings and restore change at runtime (ActiveBuild, the API key, the tokens, the external URL) are read without `cfgMu`, although a locked getter exists — `internal/api/server.go:1003`, `internal/api/middleware.go:13`, `internal/api/jobs_env.go:812`
- [x] Four functions build the same OpenAI-style error body (`writeProxyError`, `writeAuthError`, `writeVideoError` and `openAIError`) — `internal/api/proxy.go:242`, `internal/api/middleware.go:38` *(`writeVideoError` is gone and `writeProxyError` calls `openAIError`; `writeAuthError` keeps its own body, which has no code and type `auth_error`.)*
- [x] Best-effort cache writes drop their errors silently with `_ =`; the docs cache logs the same kind of failure — `internal/modelsource/meta_cache.go:105`, `internal/recommend/finalist.go:153`
- [ ] Some JS calls `fetch` directly without checking `r.ok`, so an error page is inserted into the dialog; the About dialog sets `loaded` before the request succeeds, so it never retries — `web/templates/benchmarks.html:368`, `web/templates/server.html:429`

### Duplication
- [x] `cancelImageJob`, `cancelMusic` and `cancelVideoJob` are the same function — `internal/api/gen_image_jobs.go:310` (and `gen_music.go:522`, `gen_video_jobs.go:489`) *(image and music use `cancelJob`; video keeps its own, since it only asks sd-server to stop.)*
- [x] The "not enabled on this server" 404 is written by hand 13 times — `internal/api/gen_images.go:149` and others
- [x] The code that reads an image request body is pasted twice, and the "128 MiB" text is hard-coded in 4 places — `internal/api/gen_images.go:153`, `internal/api/gen_image_jobs.go:341`
- [x] Three functions repeat the same code to read the model name from a request body — `internal/api/gen_evict.go:133`, `internal/api/proxy.go:138`, `internal/api/proxy.go:318`
- [x] The backend URL `http://127.0.0.1:%d` is built by hand in 5 places — `internal/generation/launch.go:98` and others
- [ ] The copy-to-clipboard fallback is written twice — `web/static/log-panel.js:66`, `web/templates/server.html:381`

### Logging
- [ ] When the router fails to stop before a restart, this is logged at Debug and the restart carries on; a llama-server left running may still hold the port — `internal/api/service.go:212`, `internal/api/helper_model.go:128`, `internal/api/helper_model.go:187`
- [ ] A failed SIGTERM ("terminate failed") is logged at Debug — `internal/process/supervisor.go:173`

### Security
- [x] State-changing endpoints (settings, restore, build delete) have no CSRF, Origin or Sec-Fetch-Site check — `internal/api/settings.go:166`, `internal/api/server.go:646` *(Go's `http.CrossOriginProtection` on every route except `/v1/` and the ACE-Step routes, which take the API key.)*
- [x] Extra runtime environment pairs are shown as full KEY=VALUE text and exported in every backup, so a token typed in as an env var is not masked — `internal/api/runtime_env_view.go:26`, `internal/backup/backup.go:143` *(values of names with TOKEN, SECRET, PASSWORD, PASSWD, KEY, CREDENTIAL or AUTH are masked on the pages and left out of backups made without `?secrets=1`; a masked line saved unchanged keeps its value, and a restore keeps the target's own.)*

### Testing
- [ ] `cmd/agent` has no tests: `executeTool` has no `..` check on file paths, `parseContentToolCalls` is untested, and the 8192-byte cut can split a multi-byte character — `cmd/agent/main.go:102`, `cmd/agent/main.go:330`, `cmd/agent/main.go:366`
- [ ] The builder pipeline (`runBuild`, `finishBuild`, the compiler finders, `filterReleaseTags`) and `detect.go` are untested, and tests cannot give it a fake host — `internal/builder/builder.go:531`, `internal/builder/detect.go:42`
- [ ] The benchmark helpers (`keepUnchangedCopies`, the `RecordTiming` cap, `LeastFreeCard`) and the recommend scoring helpers are untested — `internal/benchmark/benchmark.go:1154`, `internal/recommend/score.go`
- [ ] The settings handlers (`handleTestConnection`, `validateModelsDir`, gen-port conflicts) have no direct tests — `internal/api/settings.go`
- [ ] Some tests wait for goroutines with fixed 20–100 ms sleeps, so they can fail on a busy machine — `internal/recommend/recommend_test.go:448`, `internal/generation/queue_test.go:191`, `internal/api/gen_video_jobs_test.go:257`
- [ ] Some "nothing happened" checks run after a fixed sleep, so a late event slips past them — `internal/api/gen_image_jobs_test.go:228`, `internal/api/gen_music_test.go:296`
- [ ] A poll loop stops quietly at its deadline instead of failing the test — `internal/api/gen_video_jobs_test.go:207`
- [ ] The same poll-until helper is copied more than 20 times, with different timeouts — `internal/api/gen_image_jobs_test.go:31`, `internal/process/supervisor_test.go:52`
- [ ] Two packages each have a full fake of the benchmark job environment and of the router — `internal/autotune/runner_test.go:22`, `internal/benchmark/job_runner_test.go:25`

## Low Priority

### Standards
- [ ] One error is wrapped with `%v` instead of `%w` — `internal/api/jobs_env.go:277`
- [ ] Some errors are built with `errors.New("..." + err.Error())`, which loses the original error — `internal/api/gen_music.go:572`, `:593`, `:842`
- [ ] `handleBulkDeleteJobs` decodes JSON by hand instead of using `decodeJSONBody` — `internal/api/bench_jobs.go:1068`
- [ ] The request content type is checked three different ways — `internal/api/gen_music.go:570`, `internal/api/gen_image_jobs.go:338`
- [x] About 25 handlers ignore the error from `r.ParseForm()` — `internal/api/settings.go:152` and others
- [x] Errors from `MkdirAll` and `Chmod` are ignored on install and download paths — `internal/builder/builder.go:551`, `internal/huggingface/downloader.go:406`
- [ ] Log messages are worded three ways ("failed to X", "X failed", "could not X"), model IDs are logged as `id` in some places and `model` in others, and some messages are built by joining strings *(Standards + Logging)* — `internal/models/registry.go:1007`, `internal/process/supervisor.go:154`
- [ ] `log-panel.js` uses `const`/`let` and arrow functions, while every other script uses `var` and `function` — `web/static/log-panel.js:2`
- [ ] The field `runningBuild_` has a trailing underscore, which no other name in the code has — `internal/api/server.go:168`

### Duplication
- [x] There are two `setEnv` helpers, plus `setEnvDefault` — `internal/builder/rocminstall.go:359`, `internal/generation/launch.go:175` *(`setEnvDefault` stays in builder: it keeps an inherited value, which is a different rule.)*
- [x] There are two `copyFile` helpers, and the builder one has no atomic-write safety — `internal/builder/builder.go:1250`, `internal/api/gen_move.go:159`
- [x] Environment variables are read directly in several places outside the config layer; in `applyEnvOverrides` the port block is repeated 3 times and `LLAMA_PORT` skips the range check *(Duplication + Security)* — `internal/config/config.go:311`, `internal/api/server.go:950`, `internal/api/gen_instances.go:84` *(`ROCM_PATH` stays in builder, as it locates the ROCm tools, and `LLAMA_API_KEY` stays in the separate agent CLI; the runtime env view checks the process environment on purpose.)*
- [ ] The server log SSE handler repeats the live-tail code in log-panel.js — `web/templates/server.html:507`
- [x] rocminfo is run and parsed twice, and the magic number `1<<20` appears 3 times — `internal/builder/detect.go:108`, `internal/monitor/rocm.go:395` *(`1<<20` is now `musicJSONLimit`. rocminfo kept as is: the parsing is already shared, and the two callers cache differently on purpose, detection with a time limit and the monitor once.)*

### Logging
- [ ] Stop errors during a supervisor restart and at shutdown are logged at Debug — `internal/process/supervisor.go:209`, `internal/api/server.go:273`
- [ ] `ListModels` failures on a running router are logged at Debug only — `internal/api/service.go:272`, `:308`
- [ ] Expected refusals (a benchmark job holds the router) are logged at Info — `internal/api/service.go:170`
- [ ] The clone log writes the repo URL as-is, including any credentials in it — `internal/builder/builder.go:888`
- [ ] `cmd/agent` writes its diagnostics with `fmt.Fprintf(os.Stderr)` and prints tool results (which can be file contents) to the terminal — `cmd/agent/main.go:232`, `:312`

### Security
- [x] `/api` has no login, including `backup?secrets=1` and `PUT /api/settings` — `internal/api/server.go:692`, `internal/api/backup.go:20` *(decided 2026-10-05, left as is: the documented default; `--secure` puts the UI and `/api` behind a login.)*
- [ ] Child processes inherit the full service environment through `os.Environ()` — `internal/process/supervisor.go:80`, `internal/builder/builder.go:573`
- [ ] The agent CLI takes the API key as a command-line flag, so it shows in `ps` output and shell history — `cmd/agent/main.go:222`
- [ ] .gitignore has no entries for model or data folders, `*.gguf`, `*.p12`, `id_rsa*` or `.DS_Store` — `.gitignore`

### Testing
- [ ] Some tests build `&Server{}` literals instead of using `newTestServer`, so they will crash when a handler reads a new field — `internal/api/job_capability_test.go:237`, `internal/api/bench_config_test.go:230`
- [ ] Some tests compare exact HTML strings that include inline styles — `internal/api/build_render_test.go:36`, `:63`
- [ ] Some tests assert nothing outside `-race`, and one test calls the live HuggingFace API and treats a failure as 0 — `internal/api/ms_token_test.go:309`, `internal/models/gguf_derive_local_test.go:117`

### Dependencies
- [ ] govulncheck is not installed; v1.9.0 run through `go run` found no vulnerabilities — `golang.org/x/vuln@v1.9.0`
- [ ] A newer minor version of chi is available — `github.com/go-chi/chi/v5@v5.3.2 (v5.3.2 -> v5.5.0)`
- [ ] The versions of the files in web/static were not checked against their latest releases: htmx 2.0.11, plotly 3.7.0, scalar 1.73.1, pico 2.1.1 — `web/static/*`

---

# Implementation Plan

## Quick Wins (< 30 min each)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | Builds and flag presets ignore write errors | `internal/builder/builder.go:1201`, `flag_presets.go:81` | Use `atomicfile.Write` and log failures |
| 2 | Stop and SIGTERM failures logged at Debug | `service.go:212`, `helper_model.go:128`, `supervisor.go:173`, `:209`, `server.go:273` | Change to Warn |
| 3 | Refusals logged at Info | `service.go:170` and others | Change to Debug |
| 4 | Repo URL credentials in the clone log | `builder.go:888` | Remove userinfo with `url.Redacted()` |
| 5 | `%v` wrap and `errors.New` joins | `jobs_env.go:277`, `gen_music.go:572` | Use `fmt.Errorf("...: %w")` |
| 6 | Bulk delete decodes JSON by hand | `bench_jobs.go:1068` | Use `decodeJSONBody` |
| 7 | `runningBuild_` name | `server.go:168` | Rename to `curBuild` |
| 8 | Backend URL built in 5 places | `generation/launch.go:98` and others | Add an `Instance.BaseURL()` method |
| 9 | "128 MiB" and `1<<20` hard-coded | `gen_images.go`, `gen_music.go` | Derive the text from the constants |
| 10 | .gitignore gaps | `.gitignore` | Add `*.gguf`, models/data folders, `*.p12`, `id_rsa*`, `.DS_Store` |
| 11 | Silent poll loop | `gen_video_jobs_test.go:207` | `t.Fatal` at the deadline |
| 12 | chi minor update | `go.mod` | `go get chi/v5@v5.5.0` |
| 13 | govulncheck not installed | tooling | Install it or add it to CI |

## Medium Effort (30 min – 2 hours each)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | Error bodies and status-carrying errors | `gen_images.go`, `gen_video_jobs.go`, `proxy.go`, `middleware.go` | One `httpError` type and one `writeOpenAIError` |
| 2 | Create-handler error handling, "not enabled" 404, body read | `gen_images.go`, `gen_image_jobs.go`, `gen_video_jobs.go`, `gen_music.go` | Shared `writeQueueFull`, `writeAccepted`, `notEnabled`, `readLimitedBody` |
| 3 | Model-name parsing ×3 | `gen_evict.go`, `proxy.go` | `modelFromBody` helper |
| 4 | `cfgMu`-unsafe reads | `server.go`, `middleware.go`, `jobs_env.go`, `agentconfigs.go` | Use the locked getters or a config snapshot |
| 5 | Silent cache writes, ignored MkdirAll/Chmod/ParseForm | `meta_cache.go`, `finalist.go`, `builder.go`, `downloader.go`, handlers | Check the errors and log a warning |
| 6 | Env reads outside config, `applyEnvOverrides` repeats | `config.go:311`, `server.go:950`, `gen_instances.go:84`, `detect.go` | Port-override helper with a range check; move the reads into config |
| 7 | Duplicate `setEnv`, `copyFile`, rocminfo | `rocminstall.go`, `launch.go`, `builder.go`, `monitor/rocm.go` | Small shared helpers; use atomicfile |
| 8 | JS: fetch without `r.ok`, clipboard, live tail, `const`/`let` | `benchmarks.html`, `server.html`, `log-panel.js` | Use `apiCall` and `copyToClipboard`; match the `var` style |
| 9 | CSRF / Origin check | `internal/api/server.go` | Middleware that rejects cross-site Sec-Fetch-Site or Origin on unsafe methods |
| 10 | Runtime env values not masked | `runtime_env_view.go`, `backup/backup.go` | Mask values for names like TOKEN, KEY or SECRET; export them only with `?secrets=1` |
| 11 | Agent CLI: key flag, stderr output, no tests | `cmd/agent/main.go` | Prefer the env var, add path containment, test parsing and truncation without splitting characters |
| 12 | Test helpers: poll-until, fakeEnv, `&Server{}` literals | test files | Shared `eventually` helper, a `benchmarktest` package, use `newTestServer` |
| 13 | Fixed-sleep sync and negative checks | `recommend_test.go`, `queue_test.go`, `gen_*_test.go` | Sync with channels from the fakes |
| 14 | Fragile HTML matching, live HF call | `build_render_test.go`, `gguf_derive_local_test.go` | Check key substrings; fail on network error |
| 15 | Log wording and key names | across the code | Use "failed to X" and the `model` key, with constant messages and attributes |

## Complex (> 2 hours)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | Image, video and music job code copied three times (handlers, record methods, run-job, cancel) | `gen_image_jobs.go`, `gen_video_jobs.go`, `gen_music.go`, `gen_jobs.go` | Generic helpers over `jobStore[T]`, with the shared state logic in gen_jobs.go |
| 2 | Untested handlers: benchmark jobs, service and models, HF and downloads, settings | `bench_jobs.go`, `service.go`, `models.go`, `hf.go`, `settings.go`, `huggingface/client.go` | httptest-based handler tests, with error paths |
| 3 | Untested builder pipeline, benchmark helpers and recommend scoring | `builder.go`, `detect.go`, `benchmark.go`, `recommend/` | Let tests give a fake host and runner; add table tests |
| 4 | Child processes inherit the full environment | `supervisor.go`, `builder.go`, `evaluate.go`, `jobs_env.go`, `benchy.go` | Pass an allowlist of env vars (decide first whether this is worth the risk of breaking things) |

## Suggested Fix Order
1. **Data safety first.** Quick win 1 and Medium 5: lost writes and hidden errors can damage user data.
2. **Shared helpers.** Medium 1–3, 6 and 7: add the error, env and URL helpers before the larger cleanup.
3. **Job code refactor.** Complex 1, which builds on those helpers.
4. **Security.** Medium 9 and 10, then Complex 4 if wanted.
5. **Shared test helpers and sync fixes.** Medium 12–14, before writing new tests.
6. **New tests.** Complex 2 and 3, and Medium 11.
7. **Cosmetic.** Log wording, JS style, renames and dependency updates.

## Verification
- [ ] `go build ./...` and `go vet ./...` report 0 errors
- [ ] `go test -race ./...` passes, including the node JS tests
- [ ] New tests cover the handlers and packages that had none
- [ ] govulncheck shows no vulnerabilities
