# Phase 02 — Unload reasons in Engine notes

**Depends on:** nothing · **Enables:** 03 (records "room for <model>"
reasons), 05 (the field test reads them)

## Goal
When toolchest unloads a model to make room, Engine notes says so. Today an
unload by toolchest and an unload by the user look the same: one count from
the router log. This phase adds a reason to the unload event, a total of
unloads to make room, and sets the reason in the place that already unloads
for room: `unloadForGen`. Phase 03 sets it for LLM-for-LLM unloads.

## Files touched
- `internal/enginenotes/store.go` — `Event.UnloadReason string
  json:"unload_reason,omitempty"`.
- `internal/enginenotes/summary.go` — `Totals.UnloadsForRoom` and
  `ModelRow.UnloadsForRoom`, counted for `KindUnload` events with a reason.
- `internal/enginenotes/summary_test.go` — counts with and without reasons.
- `internal/api/engine_notes.go` — a pending-reason map, `noteUnloadReason`,
  and `recordOutcome` takes the reason for a `KindUnload`.
- `internal/api/gen_evict.go` — `unloadForGen` calls `noteUnloadReason`
  before `UnloadModel`.
- `internal/api/engine_notes_test.go` — the reason reaches the event.
- `web/templates/partials/engine_notes.html` — show "To make room" next to
  Unloads, in the strip and the per-model table.

## Steps
1. Add `UnloadReason` to `enginenotes.Event`. Old saved events have none;
   no migration is needed.
2. In `Summary`, for `KindUnload` with `UnloadReason != ""`, increment
   `t.UnloadsForRoom` and `r.UnloadsForRoom` as well as the unload counts.
3. In `internal/api/engine_notes.go`, add to `Server` a
   `pendingUnloads map[string]pendingUnload` under its own mutex, where
   `pendingUnload` is `{reason string; at time.Time}`, keyed by router
   name. Add:

   ```go
   // noteUnloadReason keeps why toolchest is about to unload routerName,
   // so the unload event the router log reports next carries it.
   func (s *Server) noteUnloadReason(routerName, reason string)
   ```

4. In `recordOutcome`, for `o.Kind == enginenotes.KindUnload`, take the
   entry for `o.RouterName` and delete it. Use the reason only when the
   entry is younger than 2 minutes (`unloadReasonTTL`); an older entry is
   from an unload that never reported, and is dropped. Also drop entries
   older than the TTL each time the map is written, so it cannot grow.
5. In `unloadForGen`, before `s.process.UnloadModel(routerName)`, call
   `s.noteUnloadReason(routerName, "room for "+inst.Name)`.
6. In `engine_notes.html`, add a `dt`/`dd` pair "To make room" after
   Unloads (tooltip: "Unloads that toolchest made to free VRAM for another
   model or a generation backend."), and a column in the per-model table
   with the same tooltip.

## Build gate
```bash
go build ./... && go vet ./... && go test -race ./... && make js-test
```

## Test plan
- Summary: two unloads, one with a reason → `Unloads == 2`,
  `UnloadsForRoom == 1`, in totals and the model row.
- `recordOutcome`: after `noteUnloadReason("m", "room for x")`, a
  `KindUnload` outcome for router name `m` saves `UnloadReason ==
  "room for x"`; a second unload of `m` has none.
- An entry older than the TTL is not used.
- An existing `unloadForGen` test checks that the saved event has
  `"room for <instance name>"`.
- Template render test: the strip shows "To make room".

## Commit
`feat(engine-notes): record why toolchest unloaded a model`

## Rollback
Revert the commit. Saved events with `unload_reason` stay readable by the
old code, which ignores the unknown field.
