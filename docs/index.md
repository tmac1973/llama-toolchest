# llama-toolchest

llama-toolchest is a web app that runs [llama.cpp](https://github.com/ggml-org/llama.cpp) on your own GPUs. It builds llama.cpp from source, downloads models, plans what fits in VRAM, and serves an OpenAI-compatible API. It can also run image, video and music generation next to your LLMs. It runs on Linux, with NVIDIA CUDA, AMD ROCm, Vulkan or the CPU.

The Help page in the app gives short answers. These pages give the full detail.

- [Getting started](getting-started.md): from a new install to the first chat request.
- [How toolchest shares a GPU](gpu/index.md): eviction, the GPU memory margin, and generation reservations.
- [Manual install](manual-install.md) and [Secure install](secure.md).

The source code and the issue tracker are on [GitHub](https://github.com/tmac1973/llama-toolchest).
