# Phase 04 — Engine-notes store on disk

**Depends on:** 03 · **Enables:** Phase 05 records into it; Phase 06 shows its summaries

## Goal

A store that keeps lifecycle events and per-model usage in `<data>/engine-notes.json`, survives restarts, answers "what happened in this period" for the strip and the dialog, and deletes data older than a cut-off. It is tested with a temporary directory and an injected clock.

## Files touched

- `internal/enginenotes/store.go` — new: `Store`, `Event`, `Usage`, open/load/save.
- `internal/enginenotes/summary.go` — new: `Summary`, `ModelRow`, `Store.Summary`.
- `internal/enginenotes/store_test.go` — new.
- `internal/enginenotes/summary_test.go` — new.
- `internal/enginenotes/doc.go` — package comment now also allows `internal/atomicfile`.

## Steps

1. File format, version 1:
   ```json
   {
     "version": 1,
     "counting_since": "2026-11-01T09:12:00Z",
     "events": [ { …Event… } ],
     "usage": { "<model_id>": { "2026-11-01T13": { "requests": 4, "failed": 0, "prompt_tokens": 9120, "gen_tokens": 1830 } } }
   }
   ```
   `Event` has `Time`, `Source` (always `"llama-server"` for now; the field is how generation instances fit in later), `Kind` (an `OutcomeKind` from Phase 03, plus `"router_start"`), `ModelID`, `ModelName` (display name when recorded, so a deleted model still reads well), `RouterName`, `Port`, `ExitStatus`, `LoadSeconds`, `LoadedSeconds`, `Build`, `GPUs` (the launched config's GPU assignment label, e.g. "GPUs 1-2, tensor"), `StartReason` (`"user"` or `"job"`, on `router_start`), `JobID`, `JobName` (empty outside a job), `RouterError`, `ErrorLines`, `Tail`. Every field but `Time`, `Source` and `Kind` is `omitempty`.
   Usage is bucketed by UTC hour, key format `2006-01-02T15`.
2. Bounds: at most 100 events of kind `crash` or `load_failed` together, and at most 5,000 events in all. After each `Record`: while the crash and failed-load events are over 100, drop the oldest of them; then, while the total is over 5,000, drop the oldest event whose kind is neither `crash` nor `load_failed`. `Tail` is only kept on `crash` and `load_failed` events and is at most 30 lines; `ErrorLines` at most `serverlog.MaxErrorLines`.
3. `Open(path string, now func() time.Time) (*Store, error)`:
   - Missing file → empty store, `counting_since = now()`.
   - Unreadable JSON or an unknown `version` → rename the file to `engine-notes.json.bad-<unix time>`, log `slog.Warn` with the reason, start empty. A bad file never stops the toolchest.
4. Writes:
   - `Record(e Event)` appends under a mutex and marks the store dirty.
   - `AddUsage(modelID string, promptTokens, genTokens int, failed bool)` adds to the current hour's bucket and marks dirty.
   - A background goroutine started by `Open` saves at most every 5 s while dirty, with `atomicfile.Write` (mode 0644, as the other stores use). `Close()` stops it and saves once more. Phase 05 calls `Close` on server shutdown.
5. `Summary(window time.Duration, open []OpenInstance, resolve func(routerName string) (id, name string)) Summary`. `window = 0` means all saved data. The result:
   - `CountingSince`, `From` (the later of `now − window` and `CountingSince`), `To`.
   - Totals: router starts (user / job), router stops, router exits, loads, failed loads, crashes, crashes in jobs, unloads.
   - `Models []ModelRow`, one per model ID with any event or usage in the window, sorted by name: loads, failed loads, crashes, unloads, `LoadedSeconds` (sum of `LoadedSeconds` on events in the window, plus `now − ReadySince` for an open instance of that model, clipped to `From`), average `LoadSeconds`, `LastUsed` (the later of the start of the latest usage hour and the latest `load` event; crashes and unloads do not count as use), requests, failed requests, prompt tokens, generated tokens.
   - `Crashes []Event`, kinds `crash` and `load_failed`, newest first.
   - An event belongs to the window when its `Time` is in it; a usage bucket when its hour starts in it. A load that ended in the window counts all its loaded time there.
6. `DeleteOlderThan(cutoff time.Time)` removes events with `Time < cutoff` and usage buckets whose hour starts before `cutoff`, and saves at once. `DeleteAll()` empties events and usage, sets `counting_since = now()`, and saves at once. Both return what they removed (event count, bucket count) for the confirmation text.
7. All methods are safe for concurrent use.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- Round trip: record events and usage, `Close`, `Open` again from the same path → identical `Summary(0, …)`.
- Corrupt file → renamed to `.bad-…`, store opens empty, no error returned.
- Caps: 120 crashes recorded → 100 kept, the newest; 5,100 mixed events → 5,000 kept, crash events not dropped while other kinds exceed the cap.
- Windows with a fake clock: events at −2 h, −3 d and −40 d → 24 h shows one, 7 d two, 30 d two, all three.
- Time loaded: an open instance ready since −30 min adds 30 min to its model in the 24 h window.
- Usage buckets: two requests in the same hour sum in one bucket; one at −25 h is outside the 24 h window.
- `DeleteOlderThan(now − 30 d)` removes the −40 d event and its usage hour only; `Summary(0)` totals change to match. `DeleteAll` resets `counting_since`.
- Debounced save: ten `Record` calls within a second produce one write (count writes through a test hook on the save function).

## Commit

`feat(enginenotes): keep engine events and model usage in engine-notes.json`

## Rollback

Revert the commit. Nothing records into the store until Phase 05, so no file is written yet. After Phase 05, revert that phase first; a leftover `engine-notes.json` is harmless and can be deleted by hand.
