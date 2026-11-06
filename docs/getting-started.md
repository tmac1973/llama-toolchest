# Getting started

1. **Build llama.cpp.** On Builds (`/builds`), pick the profile for your hardware (CUDA, ROCm, Vulkan, CPU) and start a build.
2. **Download a model.** On Download Models (`/models/browse`), *Find recommended LLMs* lists models that fit this machine, or search for a GGUF repo and pick a quant (`Q4_K_M` is a good default).
3. **Enable it.** On Models (`/models`), turn on the model's *On* switch.
4. **Autoconfigure it.** Press *Autoconfigure* on the model and say how long your conversations or documents are. Review the settings and press *Save and apply*. It calculates what fits in VRAM and test-loads it. For the advice of the publisher too, install the helper model on Settings (`/settings`) first.
5. **Start the server.** On Server (`/server`), press *Start*.
6. **Autotune it (optional).** Press *Autotune* on the model to measure faster speed settings. Then press *Use these settings* on the result you want. It takes a while, and the server cannot answer other requests while it runs.
7. **Send a request** to the OpenAI-compatible address in the *API Endpoint* card, with the model's public name in the `model` field.
