# Phase 01 — Track last use and list idle LLMs

**Depends on:** nothing · **Enables:** 03 (the unload order and the idle
check)

## Goal
Give toolchest what it needs to pick an LLM to unload: when each loaded LLM
was last used, whether it is idle now, and the planned memory it holds on
each GPU. Add the bookkeeping and one query function. Nothing calls the
query yet, so behavior does not change.

## Files touched
- `internal/api/gen_evict.go` — `llmUse` gets a `lastUsed` map, a `touch`
  method, a `lastUse` read and a non-waiting `tryEvict`.
- `internal/api/llm_room.go` (new) — `idleLLMsByLRU`, the list of loaded
  idle LLMs, oldest use first, with their planned MiB per GPU.
- `internal/api/llm_room_test.go` (new) — tests for the list and for
  `tryEvict`.
- `internal/api/gen_evict_test.go` — tests for `touch` and `begin`/`end`
  updating `lastUsed`.

## Steps
1. In `llmUse`, add `lastUsed map[string]time.Time`, created in
   `initLocked`. `begin` sets `lastUsed[id] = time.Now()` once it counts
   the request; the `end` func it returns sets it again, so a long chat
   counts as used until it ends.
2. Add `func (u *llmUse) touch(id string)`: sets `lastUsed[id]` to now,
   under `u.mu`. Phase 03 calls it when a model is loaded, so a model that
   was loaded by hand and never asked is not always first to go.
3. Add `func (u *llmUse) lastUse(id string) time.Time` (zero when unknown)
   and `func (u *llmUse) busy(id string) bool` (`u.n[id] > 0 ||
   u.evicting[id]`).
4. Add `func (u *llmUse) tryEvict(id string) (done func(), ok bool)`: under
   `u.mu`, when `u.n[id] == 0 && !u.evicting[id]`, sets `evicting[id]` and
   returns the same `done` as `evict`; otherwise returns `ok == false` at
   once. It never waits. A request that arrives later waits in `begin`, as
   it does for an unload for a backend.
5. In new `internal/api/llm_room.go`, add:

   ```go
   // idleLLM is a loaded LLM that no request uses now: its registry ID,
   // its last use and its planned memory per GPU.
   type idleLLM struct {
       id      string
       lastUse time.Time
       mib     map[int]int
   }

   // idleLLMsByLRU lists the LLMs the router has loaded and no request
   // uses, least recently used first, leaving out except. A model the
   // router is still loading is not idle.
   func (s *Server) idleLLMsByLRU(except string) []idleLLM
   ```

   Build it from `s.process.ListModels()` (status `loaded` only),
   `s.resolveRouterModel`, and `addLLMMiB` with
   `models.RoomOf(s.idleHardware())`, the same way `loadedLLMsOn` does.
   Leave out models where `s.llmUse.busy(id)` is true. Sort by `lastUse`,
   oldest first; for equal times, by registry ID, so the order is stable
   in tests.
6. Add `func (s *Server) loadedLLMCount() int`: the number of router models
   with status `loaded` or `loading` that resolve to a registry model.
   Phase 03 uses it for the Max Loaded limit.

## Build gate
```bash
go build ./... && go vet ./... && go test ./...
```

## Test plan
- `begin` then `end` sets `lastUsed`; a second model used later sorts after
  the first.
- `touch` moves a model to the end of the list.
- `tryEvict` on a model with a request in progress returns `ok == false`
  without blocking (test with a timeout); on an idle model it returns
  `ok == true`, and a `begin` for that model waits until `done` is called.
- `idleLLMsByLRU` with a fake router list (as in `gen_evict_test.go`):
  leaves out a busy model, a `loading` model and `except`; orders by last
  use; fills `mib` per GPU from the plan.
- `loadedLLMCount` counts `loaded` and `loading`, not `unloaded`.

## Commit
`feat(server): track last use of each LLM and list the idle ones`

## Rollback
Revert the commit. Nothing calls the new code yet, so it is safe to revert
alone.
