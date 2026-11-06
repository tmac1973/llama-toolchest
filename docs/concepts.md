# Concepts

## Enabled { #enabled }

The server offers the model in `/v1/models` and can load it on request. Uses no VRAM. Enable as many as you like.

## Loaded { #loaded }

The model is in VRAM and ready. The router loads it on the first request, or when you press the load icon in the Server tab's Available Models card.

## Max loaded models { #max-loaded-models }

How many models can be loaded at once (Settings → Eviction). `0` means no limit. See [Eviction](gpu/eviction.md#modes) for what unloads when a new model needs room.

## Public name and aliases { #public-name-and-aliases }

The `id` in `/v1/models` is a short name from the repo and quant (e.g. `unsloth-Qwen3.5-4B.Q4_K_M`). The long registry ID and any aliases you add in *Configure → Aliases* work too.
