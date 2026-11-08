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

## Default model { #default-model }

A model name that clients can send in place of a real one. Point a client at `default` once, and from then on you choose which chat model answers it here, with no change to the client.

- **Answer requests to the default model name** turns it on or off. It is on after an upgrade.
- **Name** is what clients send. Leave it blank for `default`. Letters, digits and `. _ - : /` are allowed. You can't save a name that a model, an alias or a fork instance already has.
- **Pinned model** is the chat model or fork instance that always answers. A pinned model that isn't loaded loads on the next request. A pinned fork instance that is stopped answers 503 until you start it.
- **No pin** sends each request to the loaded chat model, or running fork instance, that answered last. With nothing loaded, the chat model you used last loads, even after a restart.

The *Now* line under the form says where requests go at the moment and why. If you delete or disable the pinned model, or remove its fork instance, the pin is cleared. A fork pin stays while forks are switched off. The [API page](../api.md#default-model) covers which routes take the name and what clients see in `/v1/models`.

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
