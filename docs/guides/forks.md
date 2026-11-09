# llama.cpp forks (experimental)

Some forks of llama.cpp are much faster for one model. llama-toolchest can build a short list of curated forks. Each one runs as a *fork instance*: its own llama-server, for the model the fork is made for, on one whole GPU (or two, when the fork can use two). It runs next to the router and the generation backends. Now the list has one fork, **Avalon**. It runs Qwen3.8-Flash-Next on one RDNA3 GPU, with most experts in system RAM, or on two GPUs (see [Avalon on two GPUs](#two-gpus)).

1. **Turn on** *llama.cpp forks (experimental)* in Settings (`/settings`). With eviction set to llama.cpp, toolchest then decides what to unload (see [Eviction](../gpu/eviction.md#modes)).
2. **Build** the fork on Builds (`/builds`). It builds the newest commit of the fork. The fork changes fast, so toolchest does not pin one commit. Build again to get the newest fixes.
3. **Download its model** on Models (`/models`), under *Fork model sets*. Avalon's model is about 90 GiB.
4. **Add an instance** on Server (`/server`), under *llama.cpp forks*. Choose the GPU, and a *Second GPU* if you want the fork to use two. The dialog warns when the machine has too little RAM, no AVX2, or another GPU type. You can still start it.
5. **Start it**. Then send API requests with the instance name as `model`.

- A fork instance uses its **whole GPU**, or both whole GPUs on two. It does not start while a router model, a generation instance or another fork instance is on one of its GPUs, and the router cannot load a model there while it runs.
- Autotune, Autoconfigure and benchmarks do not cover fork instances.
- While the instance loads, requests get HTTP 503 with `Retry-After`. A large model can take several minutes.
- `/props` and `/slots` of a fork instance are on its own port, not on the toolchest port.
- A fork instance can be the [default model](settings.md#default-model). Pin it in Settings, or leave no pin and use it last. The Server page then marks its row with a *default* badge.

Known Avalon problems. Put the line in the instance's *Extra environment*:

- `GGML_CUDA_MOE_PIN=0`: the machine has less RAM than the model. Without it, the load can stop and not finish.
- `GGML_CUDA_MOE_CACHE_MODE=off`: gfx1201 GPUs, for example the Radeon AI PRO R9700. Without it, the first request can stop and not finish.

## Avalon on two GPUs { #two-gpus }

Avalon can run Qwen3.8-Flash-Next on two GPUs. Choose a *Second GPU* in the instance dialog. The Second GPU field shows only when the machine has two or more GPUs. Avalon then runs the model as a two-stage pipeline:

- Each GPU gets about half of the 48 layers, with its own part of the KV cache and its own expert cache.
- While the second GPU finishes one part of the prompt, the first GPU starts the next one. Each GPU copies the experts it needs over its own PCIe link. This makes long prompts much faster.
- Generation speed stays about the same as on one GPU.

The fork sees only the two GPUs of the instance, in the order you chose them. The first GPU gets the first layers. Both GPUs must be empty when the instance starts, and both stay reserved while it runs.

Tested on two RX 7900 XTX cards (ROCm) and on NVIDIA pairs. To change the pipeline, use these settings:

- `-ts 23,25` in *Extra arguments* moves the layer split between the GPUs.
- `LLAMA_STAGES=0` in *Extra environment* turns the pipeline off and uses a plain layer split.
