# Engine Notes — Project Overview

## Problem

llama-server crashes now and then on compute2, and the toolchest cannot say how often, on which model or why. The router's output lives only in a 500-line buffer in memory (`process.logHistorySize`) that is emptied on every router start, and benchmark jobs restart the router for every cell. None of that output reaches `docker logs`. A crash of one model instance does not stop the router: the router marks the model unloaded and loads it again on the next request, so from outside a crash looks like a slow request. On 2026-10-30 about twenty benchmark cells failed with "llama-server probably crashed", and by the time anyone looked, every crash line was gone.

There is also no record of how each model is used over time. Live Performance keeps recent speed samples in memory only, and nothing counts loads, time loaded, requests or tokens per model. A smaller problem on the same page: with several generation instances on the Server page, the log window shrinks to its 300 px minimum and there is no way to make it taller.

Tracked in issue #452.

## Goals

- Detect every start, ready, planned stop, failed load and crash of a llama-server model instance, and every start and unplanned exit of the router, from the router log as it arrives.
- For each crash, keep the time, model, port, exit status, build, GPUs and the first error lines, plus the last raw lines of that instance.
- Keep per-model stats: loads, failed loads, crashes, unloads, total time loaded, average load time, last used, requests served, prompt tokens and generated tokens.
- Save all of it to disk under `/data`, so it survives router restarts and container restarts.
- Tag events that happen during a benchmark or Autoconfigure job, and keep job traffic out of the usage totals.
- Show an "Engine notes" strip above the Server page log: quiet when the last 24 hours had no crash, amber when they had one.
- Behind a **Review** button, a dialog with a period picker (24 h, 7 days, 30 days, all), a per-model table, a crash list newest first, and a **Delete old data** control (older than 7, 30 or 90 days, or all).
- Write each crash to the toolchest's own log as a WARN line.
- Offer the same data as JSON for scripts.
- Make the Server page log window resizable by dragging a grip on its bottom edge. This browser remembers the height; a double-click on the grip returns to the default. With no saved height the log looks and sizes as it does today.

## Non-goals

- Generation instances (sd-server, ace-server). They run under their own supervisors and need their own watcher; that is later work. The saved file leaves room for them.
- A "this start" summary of context, slots, KV cache and load time, as in vllm-toolchest.
- Advice rules that suggest a setting change after a crash.
- Any change to how the router reloads a crashed model, and no automatic restart of anything.
- Notifications, nav-bar markers or webhooks.
- Automatic deletion of old data. Data is deleted only when someone asks.
- Per-model stats anywhere other than the Review dialog and the JSON endpoint.
- Speed history on disk. Live Performance stays as it is.

## Users & primary flow

The user runs the toolchest on a server such as compute2 and serves models to their own clients.

1. A model instance crashes during a request. The router prints the instance's error lines and `instance name=X exited with status N`.
2. Within one poll the strip above the Server log turns amber: "1 crash in the last 24 h — last: Qwen3.6-35B-A3B at 13:59". The toolchest log gets a WARN line with the same facts.
3. The user presses **Review**. The dialog opens on the last 24 hours. The crash list shows the model, time, exit status, build, GPUs, whether a job was running, and the error lines. A fold-out shows what llama-server wrote just before it died.
4. The user switches the period to 7 days to see whether this model crashes often, and reads the per-model table: loads, crashes, time loaded, requests and tokens.
5. A month later the user deletes data older than 30 days from the same dialog. The totals are recounted from what is left.
6. Separately, on a busy page, the user drags the log's grip down to make the log taller. The next visit opens at that height.

## Constraints

- Go server with chi, HTML templates and htmx; Pico CSS. The Server page is `web/templates/server.html`.
- The watcher reads the router's existing broadcast (`process.Manager.Subscribe`), as `watchRouterMemory` in `internal/api/memory_measured.go` does. Nothing new is launched, and the router is not polled for this.
- Line formats are those of llama.cpp v0.8.0 (`tools/server/server-models.cpp`): `spawning server instance with name=… on port …`, `stopping model instance name=…`, `instance name=… exited with status …`, and the `[port]`-prefixed child lines that `internal/memreport` already parses.
- Error lines for a crash come from the same code as `internal/benchmark/failure_log.go` (`serverErrorLines`), not a second copy.
- Usage counts come from the proxy's `ModifyResponse` hook in `internal/api/proxy.go`, the same place that feeds Live Performance (`captureTimings`), which stays unchanged. Benchmark runs talk to the router directly and do not pass through the proxy.
- Saved data goes under the configured data directory, written atomically with `internal/atomicfile`. Its size stays bounded: events are capped at 5,000, crash and failed-load records together at 100, and usage is kept as per-model hourly totals, not per request.
- Must work on Linux and Windows builds (`process/manager_windows.go` exists), and with ROCm and CUDA builds alike.
- UI text follows the project's tooltip and help conventions: short, about 80% ASD-STE100, no jargon.

## Success criteria

- A crash of a model instance on compute2 appears in the strip within one poll, and is still listed after a benchmark job has restarted the router and after the container has restarted.
- An unload, an eviction for a generation instance, a router restart and a stop that timed out and was killed are counted, but not as crashes.
- Each crash produces one WARN line in `docker logs` naming the model, exit status and first error line.
- A crash during a benchmark job is tagged with that job, and the strip and dialog say how many crashes were in jobs.
- Requests and tokens through `/v1` add to the model's usage; benchmark traffic does not.
- The period picker changes every figure in the dialog; the strip always reflects the last 24 hours.
- Delete old data with "older than 30 days" removes only older records, and the totals for "all" change to match.
- The log grip makes the log taller or shorter, the height is kept across page loads in the same browser, and a double-click restores the default. With no saved height the page looks as it does today.
- Unit tests feed recorded router output through the watcher: a clean load and unload, a crash with a CUDA error, a stop that timed out, a failed load, and two instances with interleaved lines.
- `make build`, `go vet ./...`, `go test ./...` and the JS tests pass.

## Decisions

- **Keep counts across toolchest restarts** → Yes, save to disk (JSON under `/data`, capped crash list, delete control).
- **Generation instances in scope** → No, later work.
- **Which model stats** → Lifecycle and usage: loads, failed loads, crashes, unloads, time loaded, average load time, last used, requests, prompt tokens, generated tokens.
- **How to delete old data** → A button in the Review dialog with "older than 7, 30, 90 days, or all"; totals recounted from what is left; no automatic deletion.
- **Time period shown** → A period picker in the dialog (24 h, 7 days, 30 days, all); the strip always shows the last 24 h.
- **Log window resize** → A drag grip on the bottom edge; height remembered by the browser; double-click restores the default.
- **Job runs** → Counted, tagged with the job; job tokens kept out of usage totals.
- **What else a crash does** → Also a WARN line in the toolchest's own log.
- **Where model stats appear** → Review dialog only, plus the JSON endpoint.
- **Storage** → One JSON file, `<data>/engine-notes.json`, rewritten atomically at most every 5 s.
- **Watcher structure** → A new package `internal/enginenotes` (parser and store) and its own subscriber on the router log in `internal/api`.
- **How the work lands** → One branch and PR per phase, each merged before the next starts.
- **Usage scope** → All `/v1` model requests: completions, embeddings and rerank; failed requests count with 0 tokens.
