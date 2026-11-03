# Field test — keep several models loaded with generation backends on (#539)

**Date:** 2026-11-03 (UTC) · **Machine:** asmodean, one Radeon AI PRO R9700
(32 GB), container install · **Build:** toolchest `main` cb96bba (#545),
binary swap, version `2.50.0-field539`; the 2.50.0 release binary is kept as
`/usr/bin/llama-toolchest.orig` in the container · **llama.cpp build:**
`rdna-r206-rocm-llama-optimized` (older than b11930).

Settings: generation backends on, forks on, GenEvict on, "Keep several models
loaded" on. Max Loaded was 2 in the config; the option leaves any value other
than 1 alone, so it was set to 0 for steps 3–7 and back to 2 for step 8.
Generation backend: `ming-image` (11 GiB reservation) on GPU 0; seven more
image instances are enabled but stopped.

Models used, with the plan toolchest uses ("Estimated for the current
settings") and, where measured, what the card really showed:

| Model | Role | Plan | Real |
|---|---|---|---|
| ccore Qwen3.5-4B-3D-Coder Q8_0 (coder A) | chat | 13.8 GiB | — |
| ccore Qwen3.5-4B-GLB-3D-Coder Q8_0 (coder B) | chat | 9.7 GiB | — |
| gemma-4-12b-it UD-Q6_K_XL | chat | 12.6 GiB | — |
| Qwen3.6-35B-A3B-MTP UD-Q4_K_XL (big) | chat | 22.1 GiB | not loaded |
| embeddinggemma-2 Q8_0 (downloaded for the test) | embedding | 2.8 GiB | 1.6 GiB |
| Qwen3-Reranker-0.6B Q8_0 (ctx 32768, ubatch 32768) | rerank | 12.6 GiB | 6.1 GiB |

Room for models on the card is about 28.1 GiB, or about 17 GiB while ming
runs. The test driver (`~/scratch/field539.py`, `busy539.py`, `image539.py` on
asmodean) sends requests through the proxy on :3000 and reads the router's
`/models`, `rocm-smi` and the container log.

## Results

| Step | What | Result |
|---|---|---|
| 2 | Mode at start | **Pass.** `starting router … memory=toolchest`; router runs with `--models-max 0` and no `--models-memory-margin`. |
| 3 | Chat + embedding + reranker at once | **Pass** (with ming stopped). Coder B, embeddinggemma and the reranker loaded together (19.6 GiB used); a second round of requests unloaded nothing. With ming running they do not fit by plan (25.1 > 17 GiB), and each load unloaded the one before, as it should. |
| 4 | LRU | **Pass.** gemma did not fit beside the three; coder B (oldest) went first, then embeddinggemma; the reranker, used last, stayed. Two went because the plan was a few hundred MiB over after the first. The log names each model and the one it made room for. Engine notes: 21 unloads, 4 "to make room", per model as expected. |
| 5 | Busy model | **Pass.** While gemma streamed 600 tokens, a request for the big model unloaded the idle reranker, skipped gemma, and was refused: *"…does not fit: GPU 0 has 14.8 GiB for models and needs 22.1 GiB; every other model on it is answering a request"* (HTTP 507). The stream ended normally. See finding 2. |
| 6 | Image job, then chat with a backend in the way | **Pass.** The image job (24.8 s) unloaded the LLM with the largest plan on the GPU (the reranker, 12.6 GiB by plan); coder B and embeddinggemma stayed. A rerank request with ming running then unloaded the idle coder B (LRU) and did not stop ming. |
| 7 | GenEvict off | **Pass, with finding 2.** With eviction off every enabled instance's reservation counts (106.5 GiB over eight instances), so the big model was refused with the reservation message naming all of them. Both idle models were unloaded first, for nothing. |
| 8 | Max Loaded 2 | **Pass.** With embeddinggemma and the reranker loaded, coder B unloaded embeddinggemma (least recently used) by count, with VRAM to spare. |
| 9 | Mode change | **Pass, router mode not reachable here.** Turning generation backends and forks off showed "Restart the server to apply"; after the restart the router ran in `count` mode with `--models-max 2`, because this llama.cpp build is older than b11930 (the Server page says so). Turning them back on showed the notice again; after a restart the mode was `toolchest` again. The router mode itself was not tested on this machine. |
| 10 | Plan against real VRAM | See the table above and finding 1. No out-of-memory load. |
| 11 | Left as | asmodean stays on `2.50.0-field539` (main cb96bba) with the option on, Max Loaded 2, ming running. |

## Findings

1. **Encoder estimates are about twice the real use.** The reranker at ctx
   32768 with ubatch 32768 is planned at 12.6 GiB and uses 6.1 GiB;
   embeddinggemma-2 is planned at 2.8 GiB and uses 1.6 GiB. In this mode the
   plan decides what is unloaded, so small models are unloaded sooner than
   they need to be (step 3 with ming running, step 6, where the image job took
   the reranker instead of the larger chat model). Not part of #539; the
   estimate for embedding and reranking models should be checked against
   these figures.
2. **Unloads that cannot help still happen.** When a model will not fit even
   with every idle model gone (step 5: the other model is busy; step 7: the
   reservations alone are larger than the card), `makeRoomByUnloading` still
   unloads every idle model before it refuses. It could first check whether
   the load fits with all idle LLMs taken out, and refuse at once if not. The
   step 7 message also ends with "unload a model first" when there is nothing
   left to unload.
3. **Each unload waits about 6 s** for the card to show the memory free
   (`waitGPUFreed`). gemma's load with two unloads took 13.4 s in all. This
   is the price of the real free check seeing true figures; noted, no change
   proposed.
4. **Router mode was not exercised** because the active build lacks
   `--models-memory-margin`. The unit tests cover its flags.
