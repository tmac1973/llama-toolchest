# Phase 05 — Field test on asmodean

**Depends on:** 01, 02, 03, 04 (all merged to `main`) · **Enables:**
closing #539

## Goal
Check the whole feature on a real machine with generation backends: the
single-R9700 asmodean box. Write the results to
`plan/llm-vram-eviction/field-test.md` and fix anything found in its own PR
before the issue is closed.

## Files touched
- `plan/llm-vram-eviction/field-test.md` (new) — steps, results, log
  excerpts, Engine notes figures.

## Steps
1. Build `main` for asmodean on ROCm 10 (`--rocm-image 10.1.0-full`) and
   swap the binary in as usual; keep the current one as
   `/usr/bin/llama-toolchest.orig`.
2. Set up: generation backends on, GenEvict on, one image backend; a chat
   model (about 18–20 GiB), an embedding model and a reranker enabled.
   Turn on "Keep several models loaded" and restart the server. Check the
   "starting router" log line says `memory=toolchest` and the router
   command line has `--models-max 0` and no `--models-memory-margin`.
3. **Three at once.** Send a chat request, an embedding request and a
   rerank request. All three stay loaded (Server page and `/models`).
4. **LRU.** Load a second chat model that does not fit beside the first.
   The least recently used idle model on the GPU is unloaded; the log line
   names it and the model it made room for; Engine notes shows it under
   "To make room".
5. **Busy model.** Start a long streaming chat on model A, then request a
   model that needs A's room. A is not unloaded while it streams.
6. **Image job.** Start an image job. The largest idle LLM is unloaded,
   the small models stay. Then send a chat request for the unloaded chat
   model: idle LLMs go before the backend is stopped.
7. **GenEvict off.** Turn GenEvict off and repeat 6's chat request; it is
   refused with an error that names the reservation, after idle LLMs are
   unloaded.
8. **Max Loaded 2.** Set Max Loaded to 2; a third model unloads the least
   recently used one.
9. **Mode change.** Turn generation backends off; the Server page says
   "Restart the server to apply". After a restart, the log says
   `memory=router` and the router has `--models-memory-margin`.
10. Note the VRAM shown by the monitor at each step next to the planned
    figures; note any out-of-memory load.
11. Put asmodean back on the build it ran before, or leave it on `main` and
    say which in the field test file.

## Build gate
```bash
go build ./... && go test -race ./...
```
(on the commit deployed; the field test itself is manual)

## Test plan
The steps above are the test plan. Each step's result goes in
`field-test.md` as pass or fail, with the log lines. A failure becomes a
fix PR with a unit test that reproduces it, before #539 is closed.

## Commit
`docs(plan): field test of keeping several models loaded with generation backends`

## Rollback
Documentation only. To roll back the deploy, copy
`/usr/bin/llama-toolchest.orig` back and restart the container.
