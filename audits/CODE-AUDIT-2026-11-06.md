# Code Audit Report

**Project**: Go 1.26 web app (chi, yaml.v3, gopsutil) with Go html/templates, htmx and plain JS; Go `testing` tests next to the source, node tests in `web/jstest`
**Scope**: full project (minus dist/, bin/, docs/ and vendored minified JS)
**Findings**: 59 total (9 high, 17 medium, 33 low)

govulncheck v1.9.0 found no vulnerabilities. go.sum is committed and verified. Items marked won't-fix in `CODE-AUDIT-2026-10-27.md` were left out.

---

## High Priority

### Standards
- `loadBuilds` and `loadFlagPresetsLocked` throw away the `json.Unmarshal` error. A corrupt file loads as an empty list, and the next save overwrites it. The registry, benchmark and autotune stores go read-only in this case — `internal/builder/builder.go:1381` and `internal/builder/flag_presets.go:72`

### Duplication
- `forks.Manager` is a near line-for-line copy of `generation.Manager` (Stop, Status, IsRunning, IsActive, PID, WaitRunning, Subscribe, RecentLogs, Forget, StopAll, supervisor) — `internal/forks/manager.go:31-165` and `internal/generation/manager.go:48-191`
- `forks.Store` copies `generation.Store`: file shape, Load with the schema-version check, saveLocked, and the Instance CRUD — `internal/forks/store.go:44-83,135-203` and `internal/generation/store.go:46-84,174-250`
- The read-only JSON store guard (`readOnlyError`, `ReadOnlyReason`, `writableLocked`, the three-step load error sequence) is written three times — `internal/models/registry.go:561,1958-1982,2032-2047`, `internal/benchmark/benchmark.go:855-888,1503-1540`, `internal/autotune/record.go:192-275`
- The fork and gen instance HTTP handlers are parallel copies: delete (409 while active), logs, the instance lookup and the HX-Trigger change event — `internal/api/fork_instances.go:231-334` and `internal/api/gen_instances.go:714-724,827-870`
- The port-from-form logic is duplicated, and the copies already disagree: fork has an inline range check and its own error text, gen uses `store.CheckPort` and `ErrPortUsed` — `internal/api/fork_instances.go:139-171` and `internal/api/gen_instances.go:635-671`

### Testing
- `TestStartReplacesTerminalEntry` starts a real Hugging Face download with no fake provider. `run()` sends requests to huggingface.co before `Cancel` stops it. Sibling tests already use `RegisterProvider("fake", …)` — `internal/huggingface/downloader_test.go:45`
- The build flag preset handlers (save, delete, row render) have no tests. Untested: `opt_*` checkboxes to an on/off map, the empty tag/preset refusal, the "No saved set named" path — `internal/api/build_flag_presets.go:39,71`
- `handleGPUMap` and `gpuMapAllocations` have no tests. Untested: segment width %, used/free GB, `Overcommit`, a zero-VRAM card, the fallback when `len(room.Shared) != numGPUs` — `internal/api/gpu_map.go:11,88`

## Medium Priority

### Standards
- `Registry.Delete` ignores the `os.Remove` errors on the shards, then removes the entry. A failed remove leaves an untracked file on disk with no message — `internal/models/registry.go:800-801`
- `json.Unmarshal` errors on the ace-server status poll are ignored. A malformed reply polls until the context ends — `internal/api/gen_ace.go:67`
- Some template code uses `async/await` with raw `fetch` and its own error box. The project pattern is `.then()` with `apiCall` and the shared notice — `web/templates/models.html:200`, `web/templates/server.html:817,829,839`, `web/templates/benchmarks.html:973` (helper `web/static/errors.js:126`)
- Memory fields are named MB/GB but hold 1024-based values. Most code uses MiB/GiB names. One line does `RAMGiB = TotalMB / 1024` — `internal/monitor/monitor.go:23-24,108-109`, `internal/api/fork_instances.go:404`, `internal/api/gpu_map.go:41`
- `genStore`, `forkStore` and the fork build path reply with a bare 404 when the feature is off. Other gen routes use `writeNotEnabled`. The two store guards are also copies of each other (Standards, Duplication) — `internal/api/gen_models.go:54-68`, `internal/api/fork_models.go:36-50`, `internal/api/build.go:431`

### Standards / Logging
- 10 log messages say "could not X", against the agreed "failed to X" (68 uses). `main.go` also has "server error", "http shutdown" and "cleaning stale KL cache partials" on error paths — `internal/api/ngram_cache.go:29`, `internal/api/gen_video_jobs.go:450`, `internal/api/gen_prompt_cache.go:123`, `internal/api/gen_music.go:488`, `internal/api/reasoning_levels.go:68`, `internal/api/gen_vram.go:831`, `internal/api/hf.go:620`, `internal/benchmark/runner.go:374`, `internal/models/ngramcache.go:134`, `internal/enginenotes/store.go:174`, `cmd/llama-toolchest/main.go:69,89,109`

### Duplication
- The router URL `http://localhost:%d` is built by hand in 6 places. `process.Manager` has `routerBase()`. Fork instances build `http://127.0.0.1:%d` by hand, with no method like `generation.Instance.BaseURL()` (Standards, Duplication) — `internal/api/helper_model.go:88`, `internal/api/jobs_env.go:900`, `internal/api/reasoning_levels.go:88`, `internal/api/settings.go:420`, `internal/api/proxy.go:40`, `internal/api/fork_instances.go:471`, `internal/api/fork_proxy.go:184`
- The fork and gen auto-start loops are copies (router wait, loop, the same Warn/Info lines) — `internal/api/fork_instances.go:511-534` and `internal/api/gen_instances.go:874-899`
- `evictableGiB` is copied byte for byte from `models` into `api` — `internal/api/autoconfig.go:214-220` and `internal/models/fit.go:274-280`
- The mutex-guarded `SetToken`/`authToken`/`setAuth` code is written three times — `internal/huggingface/client.go:39-49,272-276`, `internal/modelscope/client.go:66-76,360-364`, `internal/presets/fetcher.go:55-65`
- Two different `cleanLogLine` functions strip the sd.cpp source suffix in different ways. A fix to one does not reach the other — `internal/api/gen_vram.go:1004-1018` and `internal/generation/logline.go:52-59`

### Testing
- Sleep-based synchronization. Fixed sleeps stand in for "goroutine reached X", and two are negative checks that pass when the event lands late — `internal/api/gen_evict_test.go:374`, `internal/api/proxy_test.go:665`, `internal/api/engine_notes_test.go:247`, `internal/enginenotes/store_test.go:241`
- `enrichModelPresets`, `backfillPresets` and `handleRefreshPresets` have no tests. `backfillPresets` sleeps 1–3 s per model with no hook, so it cannot be tested as written — `internal/api/presets.go:21,51,68`
- The `Downloader` tests never cancel mid-transfer, never call `Cancel` on an unknown ID, never test `PendingBytes`/`FreeBytes` or a dropped connection — `internal/huggingface/downloader.go:436,477,654`
- `web/static/log-panel.js` has no test file. Every other first-party static script has one — `web/static/log-panel.js:11`
- 9 hand-written GGUF byte writers repeat the same header/KV/tensor encoding. One builder in `internal/testutil` would replace them — `internal/models/gguf_mtp_test.go:16`, `gguf_draft_head_test.go:16`, `gguf_moe_test.go:15`, `gguf_sampling_test.go:18`, `hybrid_kv_test.go:16`, `gguf_ple_test.go:22`, `gguf_vocab_test.go:16`, `internal/modelsource/ple_probe_test.go:18`
- The JS tests use two styles. `errors_test.js` and `keep_in_view_test.js` define their own `El`/`document` stubs instead of `dom.js`. Four files use a hand-written `ok`/`fails` counter (one of them minified one-liners), the rest use `assert` (Testing, Standards) — `web/jstest/errors_test.js:7`, `keep_in_view_test.js:11`, `params_test.js:5`, `measure_needs_test.js:8`, `models_test.js:21`, `viz_test.js:10`

## Low Priority

### Standards
- The fork check `!s.forksOn() || s.forks == nil` is written inline 6 times. Gen has `genEnabled()` — `internal/api/fork_instances.go:422,512,537`, `fork_instances_ui.go:200`, `fork_proxy.go:51,125`
- The video and music handlers use `MaxBytesError` with a literal `32 << 20`. Image requests use `readImageBody` — `internal/api/gen_video_jobs.go:155,192`, `internal/api/gen_music.go:572`
- The file hash store does its own tmp-file write and rename instead of `atomicfile.Write` — `internal/filehash/store.go:137,141`
- Some Go error strings start with a capital letter and read as UI sentences — `internal/api/autoconfig.go:399`, `internal/api/gen_vram.go:1356,1367`, `internal/api/llm_room.go:263`
- `log-panel.js` puts its functions into the global scope. The other scripts use an IIFE and `window.X` — `web/static/log-panel.js:11,74,83`
- Copy-to-clipboard failures are dropped with `.catch(function(){})`. The htmx settings save failure replies 200 with inline HTML, bypassing errors.js — `web/templates/server.html:760`, `web/templates/partials/gen_set_details.html:53`, `web/static/log-panel.js:65`, `internal/api/settings.go:346`

### Logging
- A rejected `LLAMA_TOOLCHEST_EXTERNAL_URL` is logged raw. Userinfo would reach the log. `builder.go` has `redactURL` — `internal/config/config.go:458`
- The download-failure log writes `err.Error()` raw. A `*url.Error` after a CDN redirect can hold a presigned URL — `internal/huggingface/downloader.go:533`
- A GGUF read failure is logged at Info — `internal/models/registry.go:1039`
- One failed benchmark repetition inside the retry loop is logged at Error. Other per-cell failures use Warn — `internal/benchmark/runner.go:288`
- A log uses the key `name` for the model, where the agreed key is `model` — `internal/benchmark/runner.go:331,354`

### Security
- llama-benchy gets the API key as `--api-key` on the command line. Local users can read it via `ps` — `internal/benchmark/benchy.go:104`
- `?secrets=1` backup export has no `Cache-Control: no-store` — `internal/api/backup.go:34-37`
- `LLAMA_TOOLCHEST_ROCM_BASE_IMAGE` is copied into config with no check. Every other env override is checked — `internal/config/config.go:487`
- `config.Load` never tightens the config file mode. A 0644 file with `hf_token`/`api_key` stays 0644 until the first save — `internal/config/config.go:405`
- `.gitignore` ignores `/config.yaml` but not `llama-toolchest.yaml` — `.gitignore:24`
- `.gitignore` has no entries for `service-account*.json`, `*credentials*.json`, `.netrc`, `.npmrc`, `.pypirc`, `*.jks`, `*.keystore` — `.gitignore:7-14`
- `sdcpp-wan22-cuda-crash-report.md` is untracked and not ignored in the repo root — `sdcpp-wan22-cuda-crash-report.md`
- Test fixtures use credential-like literals (`hf_abc`, `sk-123`, a fake bcrypt hash) that secret scanners can flag — `scripts/test-lib.sh:498-499,921`

### Duplication
- `humanBytes` sits next to `evaluate.FormatBytes`, "the one formatter". `hf.go` divides by `1024*1024*1024` inline next to `models.BytesToGiB` — `internal/api/ngram_cache.go:65-74`, `internal/api/hf.go:237-238`
- MiB-to-GiB text has several helpers (`mibText`, `gibText`, two `gib` with different units) and inline divisions — `internal/api/gen_vram.go:437`, `internal/api/vram_margin.go:44`, `internal/api/memory_measured.go:331`, `internal/api/gen_instances.go:91-101`, `internal/recommend/group.go:207`
- `https://huggingface.co/` URLs are built by hand next to `Client.ModelURL`. `fileExists` is defined twice with different rules — `internal/api/recommend_view.go:140,149`, `internal/huggingface/downloader.go:198`, `internal/api/gen_set_form.go:304`, `internal/builder/rocminstall.go:274`

### Testing
- `threeCards()` is defined 4 times; `fmtSscan`, `writeFile` and `testBuilder` twice each — `internal/api/autoconfig_gpus_test.go:12`, `internal/autoconfig/verify_test.go:14`, `internal/models/embeddings_test.go:268`, `internal/api/autotune_test.go:118`, `internal/modelsource/ple_probe_test.go:110`
- A test compares exact HTML including an inline style and whitespace — `internal/api/service_badge_test.go:20`
- Four `TestParseROCmGPUNames*` tests should be one table test. The `llm_room` tests sleep 2 ms only to make timestamps differ — `internal/monitor/rocm_test.go:19,56,85,108`, `internal/api/llm_room_test.go:89,141`

### Dependencies
- gopsutil has a patch update — `github.com/shirou/gopsutil/v4@v4.26.9 (v4.26.9 -> v4.26.10)`
- Indirect deps of gopsutil are behind — `github.com/ebitengine/purego@v0.11.1 (-> v0.11.3)`, `golang.org/x/sys@v0.48.0 (-> v0.49.0)`, plan9stats and perfstat pseudo-versions
- yaml.v3 development moved to `go.yaml.in/yaml/v3` — `gopkg.in/yaml.v3@v3.0.1 (-> go.yaml.in/yaml/v3@v3.0.6)`
- CI tests with the go.mod version (1.26.x) but govulncheck and the release job use Go 1.27. Release binaries come from a different Go minor than the tested ones — `go.mod go directive`, `.github/workflows`
- Docs pin only `zensical==0.1.0`. Its Python dependencies drift — `docs/requirements.txt`
- Vendored JS versions are not recorded in headers (htmx 2.0.11 from its string; `htmx-sse.js` unknown, an old 1.x-style extension) — `web/static/htmx.min.js@2.0.11`, `web/static/htmx-sse.js`
- Base images are pinned by tag, not digest; `debian:trixie-slim` floats — `Dockerfile.cuda`, `Dockerfile.rocm`, `Dockerfile.rocm-next`, `Dockerfile.cpu`
- `actions/checkout@v4` in ci.yml and release.yml, `@v7` in docs.yml — `.github/workflows`

---

# Implementation Plan

## Quick Wins (< 30 min each)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | Builder drops Unmarshal errors | `internal/builder/builder.go:1381`, `flag_presets.go:72` | Log the error and refuse to save over the file (or move it aside as `.corrupt`) |
| 2 | Registry.Delete ignores remove errors | `internal/models/registry.go:800` | Return the error for the main file; ignore `ErrNotExist` and `.part` misses |
| 3 | ace-server poll ignores Unmarshal | `internal/api/gen_ace.go:67` | Return an error on a malformed status |
| 4 | Bare 404 when feature off | `gen_models.go:56`, `fork_models.go:38`, `build.go:431` | Use `writeNotEnabled` |
| 5 | Log wording and key drift | listed files | "failed to X"; `model` key |
| 6 | Log levels | `registry.go:1039`, `runner.go:288` | Warn for both |
| 7 | Raw URLs in logs | `config.go:458`, `downloader.go:533` | Use `redactURL`; strip query from `url.Error` |
| 8 | Backup no-store | `internal/api/backup.go:34` | Add `Cache-Control: no-store` |
| 9 | ROCM base image unchecked | `internal/config/config.go:487` | Check the image-reference format, warn on reject |
| 10 | Config file mode | `internal/config/config.go:405` | Warn or chmod 0600 when secrets are set and the mode is wider |
| 11 | .gitignore gaps and crash report | `.gitignore` | Add `llama-toolchest.yaml` and credential patterns; move or ignore the crash report |
| 12 | Test fixture literals | `scripts/test-lib.sh:498,921` | Use clear placeholders (`hf_TEST_PLACEHOLDER`) |
| 13 | Network in downloader test | `downloader_test.go:45` | Register the fake provider like the sibling tests |
| 14 | `forksEnabled()` helper, MaxBytes, filehash atomic write, capitalized errors | listed files | Add the helper; reuse `readImageBody`-style limit; use `atomicfile.Write`; lowercase |
| 15 | evictableGiB copy | `internal/api/autoconfig.go:214` | Export `models.EvictableGiB` |
| 16 | Dependency patch updates | `go.mod` | `go get -u` gopsutil and x/sys; `go mod tidy` |
| 17 | CI Go version and checkout drift | `.github/workflows` | One Go version source; one checkout major |
| 18 | Clipboard and settings error handling | `server.html:760`, `settings.go:346` | Show a notice on copy failure; reply with an error status |

## Medium Effort (30 min - 2 hours each)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | Router and fork base URLs by hand | `helper_model.go:88` and others | Export a router base URL getter; add `forks.Instance.BaseURL()` |
| 2 | Token holder x3 | `huggingface/client.go`, `modelscope/client.go`, `presets/fetcher.go` | Small embedded `tokenHolder` type |
| 3 | Two `cleanLogLine` | `gen_vram.go:1004`, `generation/logline.go:52` | One function in `generation`, used by both |
| 4 | Port-from-form divergence | `fork_instances.go:139`, `gen_instances.go:635` | One helper with `CheckPort`/`ErrPortUsed` semantics |
| 5 | Auto-start loops, instance handlers | `fork_instances.go`, `gen_instances.go` | Shared handler helpers over a small interface |
| 6 | Byte and GiB formatting | listed files | One formatter set in `evaluate`/`models`; replace inline divisions |
| 7 | MB/GB names | `internal/monitor/monitor.go:23` and users | Rename to MiB/GiB |
| 8 | async/await raw fetch in templates | `models.html:200`, `server.html:817`, `benchmarks.html:973` | Use `apiCall` and the shared notice |
| 9 | log-panel.js style and tests | `web/static/log-panel.js` | Wrap in an IIFE, export via `window`, add a jstest |
| 10 | JS test harness | `web/jstest/*` | Move stubs into `dom.js`; one assert style |
| 11 | Sleep-based sync tests | 4 test files | Use channels/hooks; inject the ticker in enginenotes |
| 12 | Missing tests | `build_flag_presets.go`, `gpu_map.go`, `presets.go`, `downloader.go` | Add handler and table tests; add a sleep hook to `backfillPresets` |
| 13 | Test helpers | GGUF writers, `threeCards`, etc. | GGUF builder and shared cards in `internal/testutil` |
| 14 | Fragile small tests | `service_badge_test.go:20`, `rocm_test.go`, `llm_room_test.go` | Contains-checks; table test; injected clock |
| 15 | benchy API key on argv | `internal/benchmark/benchy.go:104` | Pass via env var if llama-benchy reads one, else a 0600 temp file |
| 16 | yaml.v3 migration, docs pins, vendored JS versions, image digests | `go.mod`, `docs/requirements.txt`, `web/static`, Dockerfiles | Migrate import path; add a lock/constraints file; record versions; decide on digests |

## Complex (> 2 hours)
| # | Finding | File(s) | Fix |
|---|---------|---------|-----|
| 1 | forks.Manager / generation.Manager copy | `internal/forks/manager.go`, `internal/generation/manager.go` | Shared instance-set type in `process` (or generic), with `kind` as a parameter |
| 2 | forks.Store / generation.Store copy | `internal/forks/store.go`, `internal/generation/store.go` | Generic JSON instance store |
| 3 | Read-only store guard x3 | `registry.go`, `benchmark.go`, `autotune/record.go` | One `jsonstore` helper (load with schema check, read-only reason); builder stores (Quick Win 1) can use it too |

## Suggested Fix Order
1. Data safety: Quick Wins 1–3 first. A corrupt builds file can be overwritten today.
2. Shared helpers: the read-only store guard, router/fork base URLs, token holder, `cleanLogLine`, formatters. Later refactors build on them.
3. Refactors: the fork/gen Manager and Store merge, then the instance handlers, port logic and auto-start on top of it.
4. Tests: the network test fix, sleep-based sync, then the missing tests and shared test helpers.
5. Security and logging quick wins, .gitignore.
6. Cosmetic: naming (MiB/GiB), log wording, JS style, dependency and CI updates.

## Verification
- [ ] `go build ./...` and `go vet ./...` report 0 errors
- [ ] `go test ./...` and `make js-test` pass
- [ ] `go test ./internal/huggingface/` passes with the network off
- [ ] New tests cover `build_flag_presets.go`, `gpu_map.go`, `presets.go`, `log-panel.js`
- [ ] `govulncheck ./...` stays clean
