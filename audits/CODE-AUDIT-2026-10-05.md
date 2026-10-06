# Code Audit Report — 2026-10-05

Produced by the `/code-audit` skill (six parallel agents: standards,
duplication, logging, security, testing, dependencies), with the highest
findings checked by hand against the code and against compute2. The
duplication findings exclude everything in
[AUDIT-2026-10-05.md](AUDIT-2026-10-05.md), which was handled in #234 and
#236.

**Project**: Go 1.25 web app (chi + html/template + htmx, `log/slog`), bash installer, Docker packaging
**Scope**: full project (vendored minified assets, plan/audit/docs/dist excluded)
**Findings**: 71 total (11 high, 27 medium, 33 low)

Severity changes made when checking by hand:

- The LD_LIBRARY_PATH finding is a confirmed bug: on compute2 the router
  runs with `LD_LIBRARY_PATH=/data/builds/<build>` only, without the
  container's `/usr/local/cuda/lib64`.
- The three "no authentication" findings were downgraded from High to Low.
  They describe the documented default install (README "no
  authentication" note, `docs/secure.md`), which `--secure` covers.
- `jobEnv.buildBackend` gives the same answer as `buildBackend` today,
  because every build profile's name equals its backend. It is a risk if
  that changes, not a current bug.

Each item has a checkbox; tick it in the PR that fixes it.

---

## High Priority

### Standards
- [x] htmx-driven handlers answer errors with a non-2xx `http.Error`, which htmx does not display, so the user sees nothing (HF search 502, download refused 507/400). Other handlers return a 200 banner, which is the stated contract — `internal/api/hf.go:84`, `internal/api/hf.go:239`, `internal/api/hf.go:250` (compare `internal/api/models_profiles.go:100`)
- [x] HTML built in Go puts the router failure text and the configured external URL into the page without escaping; neighbouring code escapes — `internal/api/service.go:127`, `internal/api/settings.go:192`

### Duplication
- [x] The code that sets LD_LIBRARY_PATH for a build exists 3 times. The process manager's copy appends a second entry instead of prepending, so the router loses the container's LD_LIBRARY_PATH (confirmed on compute2) — `internal/process/manager.go:476` (and `internal/api/jobs_env.go:85`, `internal/evaluate/evaluate.go:396`)
- [x] The "turn thinking off" logic exists twice with the same switch, both using literal strings instead of the `models.ReasoningToggle*` constants — `internal/benchmark/runner.go:545` (and `internal/llmcall/client.go:190`)
- [x] `canonicalSpecValue` reimplements `EncodeSpecValue` in the same file, keeping empty values the original drops — `internal/benchmark/sweep.go:455` (and `:422`)
- [x] `safeRepoDir` and `safeName` are the same function in two packages, and both redo `huggingface.SafeModelID` inline — `internal/modelsource/meta_cache.go:51` (and `internal/recommend/finalist.go:160`)

### Testing
- [x] `proxy.go` has no tests: sampling-default injection, the SSE timing reader, model auto-load and wait — `internal/api/proxy.go:122`, `:229`, `:286`
- [x] The production `jobEnv` is never exercised (both suites use a fake): `ApplyEphemeralConfig`, `configDiff`, `ResolveBuild`, `restartRouter` — `internal/api/jobs_env.go:247`, `:733`, `:802`
- [x] The GGUF parser has no tests for malformed input (bad magic, oversized array length, truncated header) — `internal/models/gguf.go:233`, `:287`
- [x] Registry logic with no test: `ResolveID`, `FindOrphans`, shard detection, `FindMMProj`/`FindMTP`, `removeEmptyDirs` (deletes from disk) — `internal/models/registry.go:632`, `:688`, `:934`

### Dependencies
- [x] Three chi `RealIP` advisories (X-Forwarded-For IP spoofing; govulncheck: `RealIP` is not called) — `github.com/go-chi/chi/v5@v5.2.5 (v5.2.5 -> v5.3.2)`

## Medium Priority

### Standards
- [x] The `/v1` 401 sets a JSON content type, then `http.Error` overwrites it with text/plain; the OpenAI-shaped error body is written by hand instead of with `writeProxyError` — `internal/api/middleware.go:19-29`
- [x] `handleRestore` builds JSON with `%q`, whose `\x..` escapes are not valid JSON, instead of `respondJSONStatus` — `internal/api/backup.go:56-58`
- [ ] No rule for when server HTML comes from templates and when it is built in Go — `internal/api/server.go:1066`, `internal/api/models.go:577`, `internal/api/build.go:208`
- [x] JS fetch calls handle failure differently: `cancelJob` and the batch delete ignore `r.ok`; job IDs are sometimes URL-encoded and sometimes not — `web/templates/benchmarks.html:174`, `:939`
- [x] `Registry.Get` and `GetConfig` return the live pointer from behind the lock; the benchmark and autotune stores return copies — `internal/models/registry.go:593`, `:704`

### Duplication
- [x] The monitor writes its own subscriber fan-out instead of using `broadcast.Broadcaster` — `internal/monitor/monitor.go:55-145`
- [x] `jobEnv.buildBackend` skips the profile lookup `buildBackend` does (same answer today) — `internal/api/jobs_env.go:495` (and `internal/api/build.go:473`)
- [x] Spec-mode names and labels are hard-coded in autotune and the template instead of coming from `models` — `internal/autotune/score.go:426`, `internal/autotune/plan.go:229`, `web/templates/partials/model_config.html:399`
- [x] "Sort the map keys, join `k=v`" is hand-written about 6 times — `internal/api/bench_export.go:66`, `internal/autotune/score.go:241`, `internal/builder/builder.go:1122`
- [x] Removing the org prefix and "-GGUF" suffix from model names is done 4 ways with different suffix lists — `internal/models/gpu_assign.go:613`, `internal/models/registry.go:23`, `internal/recommend/group.go:71` *(stage 2: `ShortModelName` now reuses `OrgAndBase`. `OrgAndBase` is left as is because `PublicName`, the `/v1` model name, is built from it; the `recommend` grouping strips a broader set on purpose.)*
- [ ] `handleSaveConfig` defines `atoiField`, then repeats its logic by hand for 5 draft fields — `internal/api/service.go:1063-1077`
- [x] The selection actions (compare and visualize) for runs and for job cells are duplicated — `web/templates/benchmarks.html:129` (and `:1032`)
- [ ] The installer decides user vs system scope in 3 places and has 2 `systemctl --user` wrappers — `scripts/lib/service.sh:30`, `scripts/lib/host.sh:32`, `setup.sh:1720`

### Logging
- [x] A runtime env var that gets ignored is logged with its value, so a token set there would be written to the log — `internal/process/manager.go:451`
- [x] The llama-benchy command line, which includes `--api-key`, is logged and stored ("EMPTY" today) — `internal/benchmark/benchy.go:103`, `:201`

### Security
- [x] The config file holding tokens and the API key is written with mode 0644 — `internal/api/settings.go:245`
- [x] The config write ignores errors, so a failed token change is lost silently — `internal/api/settings.go:245-251`
- [x] The settings page shows the inherited environment value of a configured runtime var (`Overridden: NAME=value`) — `internal/api/runtime_env_view.go:30`
- [x] llama-benchy's full stderr is stored and shown as the run's failure text, unredacted (the process has `HF_TOKEN`) — `internal/benchmark/benchy.go:207-221`

### Testing
- [x] Parsers for external tool output (`nvidia-smi` CSV, `ParseROCmGPUAgents` directly) have no tests — `internal/monitor/nvidia.go:25`, `internal/builder/rocminfo.go:22`
- [x] The llama-benchy path has no tests (argument building, shell quoting, summary) — `internal/benchmark/benchy.go:97`, `:127`, `:145`
- [x] Builder core is untested: rebuild decision (`hashFlags`/`flagsEqual`), build ranking, compiler lookup, `DetectBackends` — `internal/builder/builder.go:303`, `:1196` *(stage 4b: `hashFlags`, `flagsEqual`, ranking and `findNVCC` are tested. `findCUDAHostCompiler` only checks fixed `/usr/bin/g++-N` paths and `DetectBackends` runs the real tools, so neither can be tested without a production change.)*
- [x] Handlers with real branching have no handler tests: HF search and download, restore, `/v1/models`, build trigger and delete, service actions, `StreamLines` — `internal/api/hf.go`, `internal/api/backup.go`, `internal/api/sse.go:53`
- [ ] The installer libraries (`host.sh`, `migrate.sh`, `service.sh`) have no automated tests — `scripts/lib/*.sh`
- [x] About 15 test `Server` constructors and 31 inline `&Server{}` literals — `internal/api/models_profiles_test.go:26`, `internal/api/ms_token_test.go:67`
- [x] Template parsing for tests is copied 32 times across 18 files — `internal/api/ple_render_test.go:16`
- [x] `fakeEnv` and `newFakeRouter` are duplicated between the benchmark and autotune tests — `internal/benchmark/job_runner_test.go:23`, `internal/autotune/runner_test.go:23` *(stage 4: checked and left as is. The two fakes model different things — the benchmark one records every call and runs the evaluation machinery, the autotune one models a machine whose speed depends on the settings — and their routers differ the same way. What they share is about eight one-line "not used" stubs, which would not justify a new exported test package.)*

### Dependencies
- [x] Release CI builds with Go 1.25, which no longer gets security fixes; Go 1.27 is current — `.github/workflows/release.yml:68`

## Low Priority

### Standards
- [ ] An error is checked with a type assertion instead of `errors.As` — `internal/api/build.go:256`
- [ ] Mixed `==` / `os.IsNotExist` / `errors.Is` error checks — `cmd/llama-toolchest/main.go:67`, `internal/config/config.go:75`
- [ ] `_ = x` lines that keep dead values around — `internal/api/server.go:961-970`, `internal/benchmark/runner.go:244`, `internal/monitor/rocm.go:108`
- [ ] Receiver names differ within the same type — `internal/models/gguf.go:224`, `internal/models/specparams.go:80`
- [ ] Three `interface{}` uses remain; about 180 places use `any` — `internal/api/respond.go:51`, `internal/api/server.go:357`
- [ ] Mixed JS async styles and `var`/`const` — `web/templates/server.html:219`, `web/static/log-panel.js:67`
- [ ] HTTP-status error wording differs between clients; some error messages start with a capital — `internal/huggingface/client.go:82`, `internal/modelscope/client.go:149`
- [ ] About 31 `fmt.Errorf` calls with no format verbs — `internal/api/sse.go:18`, `internal/builder/builder.go:312`

### Duplication
- [x] `openJobForm` and `openJobEditForm` repeat the same steps — `web/templates/benchmarks.html:314-357`
- [ ] Small repeated helpers: `pluralS` and `plural`, `shortenModelName`, an 80-character truncation repeated twice that can split a UTF-8 character — `internal/api/bench_jobs.go:887-899`, `internal/api/autoconfig.go:512`

### Logging
- [x] Whole response bodies go into error messages (and from there into logs) with no size limit — `internal/benchmark/runner.go:712`, `internal/routerclient/routerclient.go:51`
- [x] A failed `SetHelperRole` save is logged at Debug — `internal/api/helper_model.go:51`
- [x] A failed cache write is logged at Debug — `internal/presets/fetcher.go:230`
- [ ] Requests refused because a job holds the router are logged at Info — `internal/api/service.go:172`
- [ ] The agent CLI echoes tool output, which could include file secrets, to the terminal — `cmd/agent/main.go:310`

### Security
- [ ] `/api`, including `backup?secrets=1` and `PUT /api/settings`, has no authentication (documented default; `--secure` covers it) — `internal/api/server.go:592`, `internal/api/backup.go:20`
- [ ] Raw `:8080` bypasses `api_key` (documented in `docs/secure.md`; `--secure` closes it) — `internal/process/manager.go:90`
- [x] The API key is compared with `!=`, not in constant time — `internal/api/middleware.go:26`
- [x] The settings form cannot clear a secret; a blank field keeps the old value — `internal/api/settings.go:98`
- [ ] Ad-hoc `os.Getenv` reads for keys in the agent CLI and the scraper; `-api-key` is visible in `ps` — `cmd/agent/main.go:200`, `scripts/scrape-sampling-presets/main.go:51`
- [ ] Env overrides read without validation (`EXTERNAL_URL`, `ROCM_BASE_IMAGE`, `ROCM_PATH`) — `internal/config/config.go:96`, `internal/builder/detect.go:24`
- [x] `.gitignore` gaps: `.env.*`, `*.pem`/`*.key`, `llama-toolchest-backup-*.json`, IDE dirs — `.gitignore:1-17`
- [ ] Local secret files (`config.yaml`, `.env`) are untracked and were never committed; keep them ignored — `config.yaml:6`

### Testing
- [x] The three JS-test wrappers repeat the same steps; pointer helpers are redefined per file — `internal/api/js_models_test.go:28`, `internal/api/capabilities_test.go:9`
- [ ] Fragile tests that check exact markup and formatted strings — `internal/api/restart_icon_test.go:37`, `internal/api/bench_eval_display_test.go:153`
- [ ] A test that cannot fail at run time (interface check); tests that wait with sleeps of up to 30 s — `internal/modelsource/iface_test.go:14`, `internal/autotune/runner_test.go:244`

### Dependencies
- [x] Windows-only advisory, not called — `golang.org/x/sys@v0.41.0 (v0.41.0 -> v0.48.0)`
- [x] Direct dependency patch update — `github.com/shirou/gopsutil/v4@v4.26.3 (v4.26.3 -> v4.26.9)`
- [x] Indirect dependencies outdated — `purego@v0.10.0 (-> v0.11.1)`, `go-sysconf@v0.3.16 (-> v0.4.0)`, `numcpus@v0.11.0 (-> v0.12.0)`, `go-ole@v1.2.6 (-> v1.3.0)` (updated in stage 1; go-ole in stage 5)
- [x] CUDA base image — `nvidia/cuda:12.8.1-devel-ubuntu24.04 (12.8 -> 13.x)` *(stage 5: deliberately kept on 12.8. CUDA 13 needs NVIDIA driver 580 or newer and drops Pascal and Volta GPUs. compute2 runs Debian 13's packaged driver, 550.163, which runs the 12.8 image through CUDA 12.x minor-version compatibility but cannot run CUDA 13. Revisit when the distributions ship a 580+ driver.)*
- [x] CPU base image — `debian:bookworm-slim (bookworm -> trixie-slim)`
- [x] Vendored htmx — `web/static/htmx.min.js@2.0.4 (-> latest 2.0.x)`
- [x] Vendored Pico CSS — `web/static/pico.min.css@2.0.6 (-> 2.1.x)`

---

# Implementation Plan

Done as one PR per stage, in the order under "Suggested Fix Order".

## Quick Wins (< 30 min each)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | Router drops the container's LD_LIBRARY_PATH | `internal/process/manager.go:476` | Prepend to an existing entry; share one helper with `jobs_env.go` and `evaluate.go` |
| 2 | Unescaped router error and external URL | `internal/api/service.go:127`, `internal/api/settings.go:192` | `html.EscapeString` |
| 3 | 401 sent as text/plain, hand-written | `internal/api/middleware.go:19-29` | `writeProxyError`; compare the key with `subtle.ConstantTimeCompare` |
| 4 | Restore JSON uses `%q` | `internal/api/backup.go:56` | `respondJSONStatus` |
| 5 | Env var value in log | `internal/process/manager.go:451` | Log the name only |
| 6 | `--api-key` in the stored benchy command | `internal/benchmark/benchy.go:103` | Mask the value in `FormatBenchyCommand` |
| 7 | Config written 0644, errors ignored | `internal/api/settings.go:245-251` | 0600 via `atomicfile`, return the error and show a banner |
| 8 | Unbounded bodies in errors | `internal/benchmark/runner.go:712`, `internal/routerclient/routerclient.go:51` | `io.LimitReader` (for example 4 KiB) |
| 9 | Log levels | `internal/api/helper_model.go:51`, `internal/presets/fetcher.go:230` | Debug → Warn |
| 10 | chi advisories | `go.mod` | chi v5.3.2, plus the gopsutil and x/sys patches |
| 11 | Release CI Go version | `.github/workflows/release.yml:68`, `go.mod` | Go 1.27 |
| 12 | `.gitignore` gaps | `.gitignore` | `.env.*`, `*.pem`, `*.key`, `llama-toolchest-backup-*.json`, `.vscode/`, `.idea/` |
| 13 | `atoiField` not reused | `internal/api/service.go:1063-1077` | Use the helper |
| 14 | Small style items | `internal/api/build.go:256`, `respond.go:51`, `server.go:357`, `server.go:961`, `runner.go:244`, `rocm.go:108` | `errors.As`, `any`, remove `_ = x` |
| 15 | UTF-8-unsafe truncation, plural helpers | `internal/api/bench_jobs.go:887` | One rune-safe `truncate`; one plural helper |
| 16 | A test that cannot fail | `internal/modelsource/iface_test.go:14` | `var _ modelsource.Client = ...` in source |

## Medium Effort (30 min - 2 hours each)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | htmx errors invisible | `internal/api/hf.go:84,239,250` + templates | 200 + banner for htmx requests, or one global `htmx:responseError` handler that shows the message |
| 2 | Thinking-off logic twice | `internal/benchmark/runner.go:545`, `internal/llmcall/client.go:190` | One helper in `models` using the `ReasoningToggle*` constants |
| 3 | `canonicalSpecValue` copy | `internal/benchmark/sweep.go:455` | Parse, then `EncodeSpecValue`; decide how empty values are handled |
| 4 | `safeRepoDir`/`safeName` | `internal/modelsource/meta_cache.go:51`, `internal/recommend/finalist.go:160` | One exported helper next to `SafeModelID` |
| 5 | `jobEnv.buildBackend` | `internal/api/jobs_env.go:495` | Call the package-level `buildBackend` |
| 6 | Sorted `k=v` joins, name trimming | listed sites | One `models.JoinSorted`; one name-trimming helper |
| 7 | Spec-mode literals | `internal/autotune/score.go:426`, `plan.go:229`, template | `models.SpecModeLabel` / `IsHeadBasedDraftMode`; a template func |
| 8 | Monitor fan-out | `internal/monitor/monitor.go` | `broadcast.Broadcaster` with history 1 |
| 9 | Settings shows inherited env values | `internal/api/runtime_env_view.go:30` | "Set in the service environment" without the value |
| 10 | Benchy stderr unredacted | `internal/benchmark/benchy.go:207` | Hide known token values and cap the length |
| 11 | JS fetch error handling, form and selection duplication | `web/templates/benchmarks.html` | One `apiCall()` that checks `r.ok`; share the selection and form-open code |
| 12 | Installer scope logic in 3 places | `scripts/lib/service.sh`, `scripts/lib/host.sh`, `setup.sh` | One scope function, one systemctl wrapper |
| 13 | Tests: parsers, benchy, nvidia, rocminfo | `internal/monitor`, `internal/benchmark`, `internal/builder` | Table tests with real tool output |
| 14 | Tests: GGUF malformed input | `internal/models/gguf.go` | Truncated, bad-magic and oversized-length test files |
| 15 | Tests: registry logic | `internal/models/registry.go` | `ResolveID`, `FindOrphans`, shards, `removeEmptyDirs` on a temp dir |
| 16 | Test fixtures duplicated | `internal/api/*_test.go`, benchmark and autotune tests | One `newTestServer(opts…)`, one template loader, a shared `benchmarktest` package |
| 17 | Error wording and `fmt.Errorf` constants | listed sites | One sweep |
| 18 | Base images and vendored JS/CSS | Dockerfiles, `web/static` | Bump CUDA, Debian, htmx, Pico; then a CUDA build check on compute2 |

## Complex (> 2 hours)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | `proxy.go` untested | `internal/api/proxy.go` | Tests for sampling injection, the split-chunk SSE reader, and auto-load with a fake router |
| 2 | `jobEnv` untested | `internal/api/jobs_env.go` | Tests on a real Server with a fake process manager: ephemeral config is restored, `configDiff`, `ResolveBuild` |
| 3 | Handler tests (HF, restore, `/v1/models`, build, service) | `internal/api/*.go` | Handler tests using the shared fixture from Medium #16 |
| 4 | Registry returns live pointers | `internal/models/registry.go:593,704` | Return copies; check every caller that mutates |
| 5 | Two HTML styles (Go-built vs template) | `internal/api/server.go`, `models.go`, `build.go` | Move the larger Go-built fragments to partials over time |
| 6 | Installer libraries untested | `scripts/lib/*.sh` | bats or plain-bash tests for the pure functions |
| 7 | Default install has no authentication (documented) | `internal/api/server.go`, `internal/process/manager.go` | Optional: when `api_key` is set, also require it for `/api` and secret export, and bind llama-server to loopback in that case |

## Suggested Fix Order
1. **Security and correctness quick wins** (Quick Wins 1–12): small, separate fixes, including the confirmed LD_LIBRARY_PATH bug and the chi advisories.
2. **Shared helpers and secret redaction** (Medium 2–10): unify the duplicated logic before writing tests against it, and stop the settings page and benchy failures from showing secrets.
3. **htmx error display** (Medium 1, 11): user-visible, best done as one consistent pattern.
4. **Test fixtures** (Medium 16), then the test gaps (Medium 13–15, Complex 1–3): the fixture makes the rest cheaper.
5. **Dependencies and images** (Medium 18): needs a compute2 build check.
6. **Larger refactors** (Complex 4–7), the installer cleanup (Medium 12) and cosmetic items (Quick Wins 13–16, Medium 17) last.

## Verification
- [ ] `go build ./...` and `go vet ./...` report 0 errors; `gofmt -l .` is empty
- [ ] `go test ./...` passes (including the node JS tests)
- [ ] `govulncheck ./...` is clean
- [ ] New tests cover proxy, jobEnv, GGUF malformed input, registry, the benchy and nvidia parsers
- [ ] compute2: the router's `LD_LIBRARY_PATH` keeps `/usr/local/cuda/lib64`; an htmx error (for example a failed HF search) shows a message
