# Keep Several Models Loaded With Generation Backends — Project Overview

## Problem
"Let the router keep several models loaded" (`router_memory`, #433) lets
llama-server's router keep as many LLMs loaded as fit in VRAM
(`--models-memory-margin`), and unload the least recently used idle ones
when a new model does not fit. It is refused while generation backends or
llama.cpp forks are on (`Config.HoldsVRAMOutsideRouter`,
`internal/api/router_memory.go`). The router only sees free VRAM, so it
cannot see toolchest's generation reservations and would load LLMs into
space that a stopped backend needs. With GenEvict on, toolchest and the
router would also both decide what to unload, from different views of the
GPUs.

So anyone who runs generation backends has only the Max Loaded count. With
Max Loaded at 1, a request for a small embedding or reranker model unloads
the chat model. With Max Loaded above 1, no one checks VRAM for LLM against
LLM: `makeRoomForLLM` (`internal/api/gen_evict.go`) only checks GPUs that
have a reservation that counts, and with GenEvict on and every backend
stopped, no reservation counts, so the router loads by count alone and can
run out of memory. When a load does not fit beside the reservations, it is
refused, or with eviction on it stops backends. It never unloads an idle
LLM to make room. Tracked in #539.

## Goals
- While generation backends or llama.cpp forks are on, let several LLMs stay
  loaded (in practice: a chat model plus small embedding and reranker models)
  as long as they fit in VRAM beside the generation reservations.
- When a new LLM does not fit, toolchest unloads the least recently used
  idle LLMs on the GPUs it needs until it fits, and only then (with GenEvict
  on) stops backends, as today.
- Toolchest is the only one that unloads in this mode: the router runs with
  `--models-max 0` and no `--models-memory-margin`.
- Check every LLM load against VRAM in this mode, also on GPUs with no
  reservation, so a load never relies on the count alone.
- Just before the load, also check the real free VRAM of the GPUs; if it is
  short, unload one more idle LLM, and repeat.
- Keep Max Loaded as an upper limit that toolchest enforces itself (LRU),
  with the same 1 → 0 change on turn-on that the router option makes.
- Log each unload and count it in Engine notes.
- One setting for both modes, shown as "Keep several models loaded".

## Non-goals
- No change while generation backends and forks are both off: the router
  option works there as it does today.
- No "keep loaded" flag on a model. Any idle LLM may be unloaded.
- No use of the VRAM measured at a model's last load; the fit check uses the
  planned figures plus the real free VRAM.
- No change to how a backend job makes room: it still unloads the LLMs on
  its GPU largest first, only until it fits (`makeRoomForGen`).
- No change to benchmarks and Autotune: they still load one model at a time.
- No toast or other new UI for unloads.
- No rename of the YAML key `router_memory`.

## Users & primary flow
A user runs toolchest with generation backends set up (for example
asmodean: one R9700 with image backends) and serves a chat model, an
embedding model and a reranker to a RAG client.

1. The user turns on "Keep several models loaded" on the Server page. The
   tooltip says that toolchest decides what to unload, because generation
   backends are on. Max Loaded goes from 1 to 0 if it was 1. The router
   restarts with `--models-max 0`.
2. The client sends a chat request. Toolchest checks that the chat model
   fits beside the reservations, and the router loads it.
3. The client sends an embedding request, then a rerank request. Each model
   fits beside the chat model, so each loads and the chat model stays.
4. The client asks for a second, large chat model that does not fit. Toolchest
   unloads the least recently used idle LLMs on its GPUs until it fits,
   logs each unload, and counts it in Engine notes. A model with a request in
   progress is never unloaded.
5. An image job starts. As today, with GenEvict on, it unloads the LLMs on
   its GPU largest first until its reservation fits, so the small models
   usually stay loaded.
6. A chat request comes for a model that does not fit beside the running
   backend. Toolchest unloads idle LLMs first; only if that is not enough does
   it stop the backend (GenEvict on), or refuse the load (GenEvict off).

## Constraints
- Go server in `internal/api`, config in `internal/config`, process launch in
  `internal/process/manager.go`, VRAM planning in `internal/models`.
- Build on the existing code: `makeRoomForLLM`, `llmUse`, `loadedLLMMiB`,
  `addLLMMiB`, `CardBudgetMiB`, `idleHardware`, `unloadForGen`, the router's
  `ListModels` / `UnloadModel`.
- llama.cpp router API only: toolchest can list, load and unload models; it
  cannot ask the router how much a model will use.
- Last-use times are not tracked today; they must be added (by registry ID),
  in memory only.
- Tooltips and help text: short, about 80% ASD-STE100. No slang.
- Must work with one GPU and with several; embedding and reranker models
  already go on one GPU by default (#425).
- Accepted, as today: when generation backends or forks are turned on
  while the router runs in router mode, the router keeps deciding by its
  margin until the next start, and cannot see the reservations in that
  time. The Server page asks for a restart.
- Accepted: LLM loads wait for each other while an unload frees memory
  (seconds), so the next check sees the true free VRAM.
- CI must pass: build, vet, test, js-test, installer-test, govulncheck.
- Never commit to `main`; one branch and PR per phase.

## Success criteria
- On asmodean with generation backends on and the option on, a chat model,
  an embedding model and a reranker are all loaded at the same time, and
  requests to each do not unload the others.
- A request for a model that does not fit unloads the least recently used
  idle LLM first; the log names the unloaded model and the one it made room
  for; Engine notes counts it.
- A model with a request in progress is not unloaded (unit test).
- With GenEvict on, a chat request with a backend in the way unloads idle
  LLMs before it stops the backend; the backend is stopped only when that is
  not enough (unit test, and seen on asmodean).
- With GenEvict on and every backend stopped, a load that is larger than the
  free VRAM is not sent to the router until enough idle LLMs are unloaded
  (unit test).
- Max Loaded 2 with the option on keeps at most two LLMs loaded, and the
  third load unloads the least recently used one (unit test).
- With generation backends and forks off, router launch flags and behavior
  are unchanged (existing tests pass).
- Turning generation backends on no longer turns the option off.

## Decisions
- **When toolchest's own unloading runs** → only with generation backends or
  forks on; with both off the router does it, as today. One setting.
- **Which LLMs may be unloaded** → any idle one, least recently used first.
  A model with a request in progress is never unloaded. No "keep loaded"
  flag.
- **Order with GenEvict on** → idle LLMs first; stop backends only if that
  is not enough.
- **Backend job making room** → keep largest first, as today.
- **Max Loaded** → toolchest enforces it too, by LRU; turning the option on
  changes 1 to 0; the router gets `--models-max 0`.
- **Fit data** → the planned memory per GPU, plus a real free-VRAM check just
  before the load that unloads one more idle LLM when short.
- **Visibility** → an info log line and an Engine notes count; no new UI.
- **Field test** → asmodean.
- **Setting name** → shown as "Keep several models loaded"; the YAML key
  `router_memory` stays; the tooltip says who decides.
- **Engine notes** → a reason on each unload toolchest makes (also for
  backends) and an "unloads to make room" total.
- **Two loads at once** → one lock from the first check until the router
  lists the new model as `loading`.
- **Real free check while another model loads** → subtract the full plan
  of every model in `loading` state; it may unload one too many, never too
  few.
- **Mode change while the router runs** → applies at the next router
  start; the Server page says "Restart the server to apply".
