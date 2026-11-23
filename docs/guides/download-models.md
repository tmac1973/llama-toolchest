# Download Models

Search **HuggingFace** or **ModelScope** for GGUF repos and download single quants. Downloads resume after an interruption. Gated models need a token for that host in Settings (`/settings`). The buttons above the search open *Find recommended LLMs*, *Get embedding models*, *Get reranking models*, *Get speech models* and, with generation on, *Get generation models*.

## Recommended models { #recommended }

Checks about 40 popular and new models against this machine. It suggests the largest quant that fits at the chosen context size. The figures are estimates from the same planner as Autoconfigure. A model downloaded from a card starts with the settings of that card.

- **Sort**: *Best quality* (larger models first; below about 4 bits per weight counts for less), *Fastest* (least data read per token, so mixture-of-experts models do well), *Longest context*, *Newest*.
- **Bits per weight**: about 4.2 (IQ4_XS) or more keeps nearly all quality; below 3 is never suggested.
- **Experts in system memory**: a large MoE model can run with some experts in RAM, more slowly. Dense models that do not fit on the GPU are left out. On a Mac this reads **Experts on the CPU**: the experts stay in the same memory, and the CPU runs them (see [Memory on a Mac](../macos.md#memory)).
- The list is made again when the GPUs or the build change. After six hours, the page says that the list is old. *Refresh* makes it again.

## Vision and audio models { #vision-and-audio-models }

Many vision models need a separate **mmproj** (projector) file from the same repo. Download it from the file list too. For a model you already have, ⤓ next to its name on the Models page opens this file list. When it is in the model folder, Configure finds it. Models with built-in vision show *Vision: Built-in*.

Some publishers put the projector in a different repo. mradermacher's imatrix repos (`X-i1-GGUF`) have none; the static repo (`X-GGUF`) has it. For such a model, Configure finds the projector on Hugging Face and offers to download it into the model's folder. See [Projector (mmproj)](model-config.md#projector-mmproj).

Audio models use a projector file too. Toolchest reads from it what the model can do and shows badges: *vision* reads images, *audio* understands speech (transcription, live voice with `/v1/realtime`), *speech* speaks (`/tts`). A model that only speaks, such as Qwen3-TTS, cannot chat, so the agent configs leave it out. VoxCPM2 and Fish S2 Pro keep the chat template of the text model they are built on, so llama-server reports them as chat models. Toolchest knows their audio generators and treats them as speech-only too. The audio routes need llama.cpp v0.8.0 or newer.

## Speech models { #speech-models }

*Get speech models* lists text-to-speech models that were checked with llama-server. *Download* gets two files into the same folder: the model and its projector. The projector holds the voice encoder, the audio code predictor and the decoder, so the model cannot speak without it. When both files are there, the model's Configure page shows the projector and the model has the *speech* badge.

| Model | Files | Size | VRAM | Needs |
| --- | --- | --- | --- | --- |
| Qwen3-TTS 1.7B Base | `qwen3-tts-1-7b-Q8_0.gguf` and `mmproj-qwen3-tts-1-7b-F16.gguf` from [moonbox-dev/Qwen3-TTS-12Hz-1.7B-Base-GGUF](https://huggingface.co/moonbox-dev/Qwen3-TTS-12Hz-1.7B-Base-GGUF) | 2.7 GB | 5.1 GiB | llama.cpp b12491 or newer |
| Kokoro 82M | `Kokoro-82M-F32.gguf` and `mmproj-Kokoro-82M-F16.gguf` from [ggml-org/Kokoro-82M-GGUF](https://huggingface.co/ggml-org/Kokoro-82M-GGUF) | 270 MB | 0.7 GiB | llama.cpp b12494 or newer |
| Chatterbox Turbo | `chatterbox-turbo-Q8_0.gguf` and `mmproj-chatterbox-turbo-F16.gguf` from [ggml-org/chatterbox-turbo-GGUF](https://huggingface.co/ggml-org/chatterbox-turbo-GGUF) | 1.0 GB | 2.1 GiB | llama.cpp b12159 or newer |
| ZONOS3 Mini | `ZONOS3-Mini-Q8_0.gguf` and `mmproj-ZONOS3-Mini-F32.gguf` from [ggml-org/ZONOS3-Mini-GGUF](https://huggingface.co/ggml-org/ZONOS3-Mini-GGUF) | 2.0 GB | 2.5 GiB | llama.cpp b12424 or newer |
| VoxCPM2 | `VoxCPM2-Q8_0.gguf` and `mmproj-VoxCPM2-BF16.gguf` from [ggml-org/VoxCPM2-GGUF](https://huggingface.co/ggml-org/VoxCPM2-GGUF) | 3.6 GB | 4.9 GiB | llama.cpp b12531 or newer |
| Fish Audio S2 Pro | `s2-pro-Q8_0.gguf` and `mmproj-s2-pro-BF16.gguf` from [ggml-org/s2-pro-GGUF](https://huggingface.co/ggml-org/s2-pro-GGUF) | 6.3 GB | 8.9 GiB | llama.cpp b12452 or newer |

The VRAM figures were measured on a Radeon AI PRO R9700 with each model's default settings (8192 context, 4 slots), the model loaded and requests answered. A speech model's audio generator needs more memory than its projector file, so toolchest adds the measured difference to its estimate, and its load checks keep enough room. If the active build is older than the model needs, the card says so; build a newer one on the [Builds](builds.md) page.

The models give a voice in different ways. The fields are in [Using the API](../api.md#speech-api).

- **Qwen3-TTS** copies a voice from a recording that you send with each request (`speaker_ref`). A clean recording of one speaker, without music, works best; about 5 seconds is enough. If you also send what is said in the recording (`speaker_ref_text`), the model continues from the recording, and the copy is closer. The recording must then be 30 seconds or shorter. This needs a projector with the reference encoder. The ggml-org projector (August 2026) does not have one, so the card downloads moonbox-dev's.
- **Kokoro** is small and fast. It has 54 voices of its own (`voice`, `af_heart` by default). The first letter of a voice is its language: `a` American English, `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `p` Brazilian Portuguese, `j` Japanese, `z` Mandarin. It turns only English text into sounds itself. For the other languages, send phonemes with `raw_prompt` (see the [llama.cpp notes on Kokoro](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/tts.md#kokoro)). It cannot copy a voice, and it takes `speed`.
- **Chatterbox Turbo** speaks English. It copies a voice from a `speaker_ref` recording, or uses its own voice when there is none.
- **ZONOS3 Mini** has 27 voices of its own (`voice`; llama-server's `GET /props` lists them), or copies one from a `speaker_ref` recording (the first 60 seconds are used). It takes `speed` and `loudness`. It writes out numbers, money and times itself in English, French, German, Spanish, Italian, Portuguese, Japanese and Chinese, when `lang` is set to the language of the text.
- **VoxCPM2** makes a voice from a description in brackets at the start of the text, such as `(A calm male narrator)The ship left at dawn.` It can also copy a voice from a `speaker_ref` recording, without a transcript. It takes `flow_steps` and `cfg_scale`. Very short texts can run long and repeat words; another `seed` helps.
- **Fish Audio S2 Pro** is the most expressive. Tags in brackets change how it speaks, such as `[laughing]`, `[whisper]` or `[pause]`, and speaker tags (`<|speaker:0|>`, `<|speaker:1|>`) give each person in a dialogue a voice. It copies a voice from a `speaker_ref` recording; with `speaker_ref_text` the copy is closer. Its license allows only research and non-commercial use.

## Two layouts in one repo { #mainline-copies }

Some repos hold two versions of each quant. Unsloth does this when its files need its own llama.cpp build. The files in the root need that build. Copies for mainline llama.cpp are in a `Mainline_Support` folder and end in `.gguf_file`, so other tools do not list them next to the root files.

The file list shows both versions. A copy for mainline llama.cpp has the label *mainline llama.cpp*. It downloads as a normal `.gguf` file, and *Update* on the Models page still finds it in the repo. A root file that has such a copy has the label *needs the publisher's llama.cpp*. llama-toolchest cannot build that llama.cpp, so download the mainline copy.

A new model can also need a newer llama.cpp than your build. If it loads but writes nonsense, rebuild llama.cpp on the Builds page and read the repo's model card.
