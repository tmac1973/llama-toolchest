# llama.cpp forks (experimental)

Some forks of llama.cpp are much faster for one model. llama-toolchest can build a short list of curated forks. Each one runs as a *fork instance*: its own llama-server, for the model the fork is made for, on one whole GPU. It runs next to the router and the generation backends. Now the list has one fork, **Avalon**. It runs Qwen3.8-Flash-Next on one RDNA3 GPU, with most experts in system RAM.

1. **Turn on** *llama.cpp forks (experimental)* in Settings (`/settings`). With eviction set to llama.cpp, toolchest then decides what to unload (see [Eviction](../gpu/eviction.md#modes)).
2. **Build** the fork on Builds (`/builds`). It builds the newest commit of the fork. The fork changes fast, so toolchest does not pin one commit. Build again to get the newest fixes.
3. **Download its model** on Models (`/models`), under *Fork model sets*. Avalon's model is about 90 GiB.
4. **Add an instance** on Server (`/server`), under *llama.cpp forks*. Choose the GPU. The dialog warns when the machine has too little RAM, no AVX2, or another GPU type. You can still start it.
5. **Start it**. Then send API requests with the instance name as `model`.

- A fork instance uses its **whole GPU**. It does not start while a router model or a generation instance is on that GPU, and the router cannot load a model there while it runs.
- Autotune, Autoconfigure and benchmarks do not cover fork instances.
- While the instance loads, requests get HTTP 503 with `Retry-After`. A large model can take several minutes.
- `/props` and `/slots` of a fork instance are on its own port, not on the toolchest port.

Known Avalon problems. Put the line in the instance's *Extra environment*:

- `GGML_CUDA_MOE_PIN=0`: the machine has less RAM than the model. Without it, the load can stop and not finish.
- `GGML_CUDA_MOE_CACHE_MODE=off`: gfx1201 GPUs, for example the Radeon AI PRO R9700. Without it, the first request can stop and not finish.
