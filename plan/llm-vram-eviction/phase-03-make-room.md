# Phase 03 — Make room for an LLM by unloading idle LLMs

**Depends on:** 01 (`idleLLMsByLRU`, `tryEvict`, `touch`,
`loadedLLMCount`), 02 (`noteUnloadReason`) · **Enables:** 04 (turns this
mode on)

## Goal
Add the toolchest memory mode's load path: before an LLM load, check it
against VRAM on every GPU it uses and against Max Loaded; unload idle LLMs,
least recently used first, until it fits; only then stop generation
backends (GenEvict on) or refuse (GenEvict off); then check the real free
VRAM and unload more if it is short. Loads are kept apart by one lock held
until the router lists the new model as `loading`. The path runs only when
the router was started in toolchest mode. Nothing starts the router in that
mode until phase 04, so this phase changes no behavior outside tests.

## Files touched
- `internal/api/llm_room.go` — the mode type, `loadLLMWithRoom`,
  `unloadForLLM`, `llmShortGPUs`, `realFreeShortGPUs`.
- `internal/api/server.go` — `Server` fields `memoryMode` (atomic) and
  `llmLoadMu sync.Mutex`.
- `internal/api/proxy.go` — `ensureModelLoadedForRequest` calls
  `loadLLMWithRoom` in toolchest mode; `loadAndCountRequest`'s last attempt
  returns an error in toolchest mode (step 7).
- `internal/api/service.go` — the Server page load handler (around line
  356) does the same.
- `internal/api/gen_evict.go` — `llmReservationNeed` returns
  `replacing == false` in toolchest mode (step 4); `unloadForGen` shares
  its unload-and-wait body with `unloadForLLM` (step 2).
- `internal/models/fit.go` — export `FitMarginMiB(totalMiB int) int`.
- `internal/api/llm_room_test.go` — the scenarios in the test plan.

## Steps
1. **Mode.** In `llm_room.go` add:

   ```go
   // memoryMode is how the running router was told to share GPU memory
   // between LLMs, fixed at its start (startRouterWith).
   type memoryMode int32

   const (
       memoryCount     memoryMode = iota // --models-max decides, by count
       memoryRouter                      // --models-memory-margin: the router decides
       memoryToolchest                   // toolchest decides (generation backends or forks on)
   )
   ```

   `Server.memoryMode` is an `atomic.Int32`; add
   `func (s *Server) toolchestMemory() bool`. Nothing sets it to
   `memoryToolchest` in this phase except tests.
2. **Shared unload body.** Move the body of `unloadForGen` after the
   `llmUse.evict` call (state check, `savePromptCache`, the
   `noteUnloadReason` call from phase 02, `UnloadModel`, the wait for the
   router, `waitGPUFreed` per GPU) into
   `func (s *Server) unloadLLMLocked(ctx context.Context, m *models.Model, reason string, gpus []int) error`.
   `unloadForGen` calls it with `[]int{inst.GPU}` and
   `"room for "+inst.Name`. Behavior of `unloadForGen` is unchanged.
3. **`unloadForLLM`.** `func (s *Server) unloadForLLM(ctx, idle idleLLM, forName string) (bool, error)`:
   calls `s.llmUse.tryEvict(idle.id)`; when not ok (a request began since
   the list was made), returns `false, nil` and the caller takes the next
   one. Otherwise it `defer`s the `done` that `tryEvict` returned, so new
   requests for the model wait until the unload has finished, and calls
   `unloadLLMLocked` with the GPUs in `idle.mib` and
   reason `"room for "+forName`, logs at info `"unloaded LLM to make room
   for another LLM"` with `model` and `for`, and returns `true`. When
   `unloadLLMLocked` fails, it returns `false` and the error; the caller
   then stops and returns that error (wrapped as
   `"make room for <model>: %w"`), because a model that the router did not
   unload may still hold its memory.
4. **Planned check on every GPU.** `func (s *Server) llmShortGPUs(hw models.Hardware, need map[int]int) []int`:
   the same as `reservationShortGPUs` without the `v.Reserved == 0` skip:
   every GPU in `need` with more MiB than `models.CardBudgetMiB(g)`, where
   `g.ReservedMiB` is set from `s.genFitVRAMByGPU("")` (0 when none).
   `need` comes from `s.loadedLLMMiB(models.RoomOf(s.idleHardware()), m.ID, false)`
   and `hw` from `s.hardware()`, as `llmReservationNeed` does when it is
   not replacing. (In toolchest mode Max Loaded 1 is enforced by step 6.1,
   not by `replacing`.) For the same reason, `llmReservationNeed` returns
   `replacing == false` when `s.toolchestMemory()`: the router runs with
   `--models-max 0` and does not replace a model on its own, so
   `makeRoomForLLM` must not assume it does.
5. **Real free check.** Export `models.FitMarginMiB(totalMiB int) int`
   (`fitMarginGiB` in MiB). `func (s *Server) realFreeShortGPUs(m *models.Model) []int`:
   for each GPU `g` in `s.monitor.Current().GPU` that m's plan uses
   (`addLLMMiB` for m alone), it is short when

   `VRAMTotalMB − VRAMUsedMB < plan(m)[g] + Σ plan(models in "loading" state)[g] + Σ (Reserved − Using) of genFitVRAMByGPU[g] + FitMarginMiB(VRAMTotalMB)`.

   Models in `loading` state count with their full plan, because their
   memory may not be allocated yet. Without monitor data it returns nil.
6. **`loadLLMWithRoom`.**

   ```go
   // loadLLMWithRoom loads m in toolchest memory mode: it unloads idle
   // LLMs, least recently used first, until m fits under Max Loaded and
   // in VRAM; then, with eviction on, it stops generation backends; then
   // it checks the cards' real free memory. Loads run one at a time, from
   // the first check until the router lists m as loading. release ends
   // the hold on generation GPUs that makeRoomForLLM took; call it once
   // the load has finished.
   func (s *Server) loadLLMWithRoom(ctx context.Context, m *models.Model, routerName string) (release func(), err error)
   ```

   Order inside, all under `s.llmLoadMu` except where noted:
   1. **Max Loaded.** While `cfg.ModelsMax > 0` and `s.loadedLLMCount() >= cfg.ModelsMax`:
      take the first entry of `s.idleLLMsByLRU(m.ID)` and `unloadForLLM`
      it (a `false, nil` result: take the next entry). When the list is
      empty, return
      `fmt.Errorf("Max Loaded (%d) is reached and every loaded model is answering a request; try again when one is idle", n)`.
   2. **Planned VRAM.** While `llmShortGPUs` is not empty: take the first
      entry of `idleLLMsByLRU(m.ID)` that has `mib[g] > 0` on a short GPU
      `g`, and `unloadForLLM` it. Stop when none is left.
   3. **Backends.** If still short: unlock `llmLoadMu`, call the existing
      `s.makeRoomForLLM(ctx, m)` (which stops backends with GenEvict on and
      refuses with GenEvict off, as today), then lock again and go back to
      step 1, keeping the `release` it returned. When it returns an error,
      return that error. The lock is not held while `makeRoomForLLM` waits
      for a generation job, so other LLM loads are not held up behind it.
      Go back to step 1 at most once; if the model is still short after
      that, call `release` and return the planned fit error below.
      `makeRoomForLLM` can return no error and stop nothing, for example
      with GenEvict on and every backend already stopped; then the second
      pass through steps 1–2 finds the model still short and returns the
      planned fit error too.

      Planned fit error: for the lowest-index short GPU,
      `fmt.Errorf("%s does not fit: GPU %d has %s for models and needs %s; every other model on it is answering a request", m.PublicName(), g, fmtGiB(budget-others), fmtGiB(plan))`,
      where new `fmtGiB(mib int) string` in `llm_room.go` formats MiB as
      `"%.1f GiB"` (`gib` is taken by `memory_measured.go`). With a counting reservation
      on that GPU, return `reservationFitError(hw, need, true)` instead,
      so the message names the backend as it does today.
   4. **Real free VRAM.** While `realFreeShortGPUs(m)` is not empty: unload
      the first idle LLM on a short GPU, as in 2. When none is left, call
      `release` and return
      `fmt.Errorf("%s does not fit: GPU %d has %s free and needs %s", m.PublicName(), g, fmtGiB(free), fmtGiB(needed))`
      for the lowest-index short GPU `g`, with `free` and `needed` the two
      sides of step 5's comparison.
   5. **Load.** `s.prepareNgramCacheForLoad(m.ID)`, then
      `s.process.LoadModel(routerName)`. Then poll `lookupRouterState`
      every `autoLoadPollInterval` until the state is `loading` or `loaded`,
      for at most `llmUnloadTimeout`; then `s.llmUse.touch(m.ID)` and unlock.
      A timeout unlocks, calls `release` and returns the error.

   Every error return after `makeRoomForLLM` has returned calls `release`
   first. When `makeRoomForLLM` was not called, `release` is a no-op.
   `llmLoadMu` is held while an unload waits for the router and for the
   card to show the memory free (`waitGPUFreed`). That is on purpose: the
   next load's real free check must see the freed memory. An unload takes
   seconds; the long wait (a generation job) is outside the lock (6.3).
   `stopGenUntil` already waits for a stopped backend's memory to show
   free, so step 6.4 after 6.3 does not unload more because of it.
7. **Callers.** In `ensureModelLoadedForRequest` (`proxy.go`), in the
   `state != "loading"` branch: when `s.toolchestMemory()`, call
   `release, err := s.loadLLMWithRoom(ctx, regModel, routerName)` in place
   of `makeRoomForLLM` + `prepareNgramCacheForLoad` + `LoadModel`; keep
   `defer release()` and the wait for `loaded` after it. In the Server page
   load handler (`service.go`), do the same in place of its
   `makeRoomForLLM` call and its `LoadModel` call. Outside toolchest mode,
   both keep the current code.

   A request whose model is already `loaded` skips the lock. That is safe:
   `loadAndCountRequest` counts the request with `llmUse.begin` and then
   checks `modelLoaded` again; when an unload got in between, it ends the
   count and goes round again through `ensureModelLoadedForRequest`, which
   now takes the locked path. Once counted, a model is busy and
   `tryEvict` passes it over. One change there: today the last attempt
   (`loadAttempts`) keeps the count even when the model is not loaded, and
   the router would then autoload it with no VRAM check. In toolchest mode,
   the last attempt instead ends the count and returns
   `fmt.Errorf("%s was unloaded %d times before the request could start; try again", name, loadAttempts)`.
   Outside toolchest mode it stays as it is. Add a test for both.
8. Comments in `gen_evict.go` (the file header and `makeRoomForLLM`) say
   that in toolchest mode `loadLLMWithRoom` runs first and unloads idle
   LLMs before any backend is stopped.

## Build gate
```bash
go build ./... && go vet ./... && go test -race ./...
```

## Test plan
All with `s.memoryMode` set to `memoryToolchest`, `s.testHardware`, a fake
router list and fake monitor readings, as in `gen_evict_test.go`:
- **Fits beside:** chat (20 GiB) loaded on a 32 GiB card; an embedding
  model (1 GiB) loads and the chat model stays.
- **LRU:** chat A used, then embedding B used; a model C that fits only
  without one of them unloads A (older), not B. The saved Engine notes
  event for A has `"room for C"`.
- **Busy is kept:** A has a request in progress (`llmUse.begin`); C
  unloads B instead, even though A is older. With only A loaded and busy,
  C is refused.
- **Max Loaded:** `ModelsMax = 2`, A and B loaded and idle; loading C
  unloads the least recently used one, with no VRAM shortage.
- **LLMs before backends:** GenEvict on, a backend running with a 10 GiB
  reservation, an idle LLM in the way; loading C unloads the LLM and does
  not stop the backend. When unloading every idle LLM is not enough, the
  backend is stopped.
- **GenEvict off:** the same case refuses after the idle LLMs are unloaded,
  with the reservation named in the error.
- **No reservation:** GenEvict on, every backend stopped; a model larger
  than the card's budget minus loaded LLMs unloads idle LLMs before
  `LoadModel` is called (assert the fake router saw the unload first).
- **Real free check:** planned figures say it fits, but the monitor shows
  2 GiB less free; one more idle LLM is unloaded. A model in `loading`
  state counts with its full plan.
- **One at a time:** two goroutines load C and D at once on a card where
  only one fits beside A; the second check sees the first as `loading`,
  so A is unloaded once and both fit, or the second is refused; the fake
  router never has more planned memory than the budget.
- **Unload between check and count:** a request sees its model `loaded`;
  before its `begin`, another load unloads it; the request loads it again
  through `loadLLMWithRoom` and the fake router never autoloads it.
- **Retries run out:** in toolchest mode the 4th unload between check
  and count returns the error; outside it, the old behavior.
- **Unload fails:** the fake router refuses an unload; the load returns
  the wrapped error and `release` was called.
- **Max Loaded 1 in toolchest mode:** `makeRoomForLLM` does not treat the
  load as a replace (`replacing == false`).
- **Mode off:** with `memoryCount`, `ensureModelLoadedForRequest` uses the
  old path (existing tests pass unchanged).

## Commit
`feat(server): unload idle LLMs by VRAM before a load in toolchest memory mode`

## Rollback
Revert the commit. Nothing turns the mode on before phase 04, so revert is
safe alone; if phase 04 has landed, revert 04 first.
