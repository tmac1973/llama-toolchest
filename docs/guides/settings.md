# Settings

## Auto-start { #auto-start }

Start llama-server when llama-toolchest starts.

## Runtime environment { #runtime-environment }

Environment variables for llama-server and every model, applied at the next restart. The list is short on purpose; most tuning tips have no effect. *Extra environment* takes anything else (`KEY=VALUE` per line, applied last). *Effective environment* shows the result, with values set outside toolchest (systemd, container), which win.

## Backup & Restore { #backup-restore }

One JSON file with settings, environment, flag sets and per-model configs. Configs match by repo and quant, so they restore on another machine. Secrets only if you tick the box. Restore merges and never deletes. Configs for models you do not have wait as *pending* until the download. Benchmark history is not included. Scripts: `GET /api/backup`, `POST /api/restore`.

## Eviction { #eviction }

The eviction mode, *Max loaded models*, and the switch that lets LLMs and generation backends take turns on a GPU. See [Eviction modes](../gpu/eviction.md) and [LLMs and generation backends on one GPU](../gpu/taking-turns.md#take-turns).

## GPU memory margin { #gpu-memory-margin }

VRAM on each GPU that toolchest does not plan models into, for llama.cpp's working memory and the driver. Set it for every GPU or for one GPU. See [GPU memory margin](../gpu/memory-margin.md).

## Generation backends { #generation-backends }

Turns on image, video and music generation. See [Image, video and music generation](generation.md).

## Server URL { #server-url }

The address for the API and chat links. Change it if you open the UI from another machine.

## API Key { #api-key }

Optional Bearer token for `/v1/*`.

## HuggingFace Token and ModelScope Token { #hf-token }

For gated models on each host.

## Models directory { #models-directory }

Where GGUFs and downloads go (default `<data dir>/models`). Restart after you change it.

## Theme { #theme }

Saved in this browser.
