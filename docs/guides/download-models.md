# Download Models

Search **HuggingFace** or **ModelScope** for GGUF repos and download single quants. Downloads resume after an interruption. Gated models need a token for that host in Settings (`/settings`). The buttons above the search open *Find recommended LLMs*, *Get embedding models* and, with generation on, *Get generation models*.

## Recommended models { #recommended }

Checks about 40 popular and new models against this machine. It suggests the largest quant that fits at the chosen context size. The figures are estimates from the same planner as Autoconfigure. A model downloaded from a card starts with the settings of that card.

- **Sort**: *Best quality* (larger models first; below about 4 bits per weight counts for less), *Fastest* (least data read per token, so mixture-of-experts models do well), *Longest context*, *Newest*.
- **Bits per weight**: about 4.2 (IQ4_XS) or more keeps nearly all quality; below 3 is never suggested.
- **Experts in system memory**: a large MoE model can run with some experts in RAM, more slowly. Dense models that do not fit on the GPU are left out.
- The list is made again when the GPUs or the build change. After six hours, the page says that the list is old. *Refresh* makes it again.

## Vision and audio models { #vision-and-audio-models }

Many vision models need a separate **mmproj** (projector) file from the same repo. Download it from the file list too. For a model you already have, ⤓ next to its name on the Models page opens this file list. When it is in the model folder, Configure finds it. Models with built-in vision show *Vision: Built-in*.

Some publishers put the projector in a different repo. mradermacher's imatrix repos (`X-i1-GGUF`) have none; the static repo (`X-GGUF`) has it. For such a model, Configure finds the projector on Hugging Face and offers to download it into the model's folder. See [Projector (mmproj)](model-config.md#projector-mmproj).

Audio models use a projector file too. Toolchest reads from it what the model can do and shows badges: *vision* reads images, *audio* understands speech (transcription, live voice with `/v1/realtime`), *speech* speaks (`/tts`). A model that only speaks, such as Qwen3-TTS, cannot chat, so the agent configs leave it out. The audio routes need llama.cpp v0.8.0 or newer.
