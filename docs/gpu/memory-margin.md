# GPU memory margin

## What it is { #what }

The margin is VRAM on each GPU that toolchest does not plan models or generation backends into. It is not empty space: llama.cpp uses it for working memory in real chats, and the driver uses some of it too. The estimate of a model does not count that use, so the margin holds room for it.

Set it in Settings → GPU memory margin (`/settings#vram-margin`), in GiB. The default and the smallest value is 1 GiB.

## Why at least 1 GiB { #at-least-1-gib }

The estimate of a model and a test load do not show all the memory that llama.cpp uses. Real chats process a prompt in parts, and llama.cpp then needs more working memory. Settings that passed a test load with about 500 MiB free ran out of memory on the first real chat. The 1 GiB also leaves room for the driver.

This bar shows what toolchest counts on one GPU when it plans a load:

<div class="tc-diagram" markdown>
--8<-- "assets/diagrams/vram-margin.svg"
</div>

- **Desktop and other programs** are counted apart, from what the GPU reports. Toolchest reads this with none of its own programs loaded.
- **The LLM** is its weights, its KV cache and its buffers, from the estimate.
- **The margin** is room for llama.cpp's working memory and the driver. A new model fits only in the room above it.

## One value or one per GPU { #per-gpu }

- **Every GPU:** one value for all cards.
- **One GPU:** a row for each detected GPU. A value there replaces the value for every GPU. Use it, for example, for a larger margin on the GPU that runs the desktop, where use can go up while a model runs. Leave the row blank to use the value for every GPU.

## What uses the margin { #users }

- Autoconfigure, when it plans a model.
- The fit labels on Download Models.
- Measure, for generation backends.
- The start check of a generation backend.
- Eviction by llama-toolchest, when it decides what must unload.
- Eviction by llama.cpp, with one value for all GPUs. See the next section.

## With eviction by llama.cpp { #router }

The llama-server router takes one margin for all GPUs. Toolchest gives it the largest margin of any GPU. So a larger margin on one GPU makes the router leave that much room on every GPU.

<div class="tc-diagram" markdown>
--8<-- "assets/diagrams/vram-margin-router.svg"
</div>

To leave more room on one GPU only, use eviction by llama-toolchest. It uses the margin of each GPU.

## When a change applies { #applies }

- For Autoconfigure, Measure, the fit labels and eviction by llama-toolchest: at the next plan, load or backend start.
- For eviction by llama.cpp: at the next router start, because the router reads the margin only when it starts.

A loaded model keeps its settings. Run Autoconfigure again to plan the model for a new margin.

## If a load fails out of memory { #oom }

Raise the margin of that GPU by 0.5 to 1 GiB, or give the model a smaller context.

## Related { #related }

- [How toolchest shares a GPU](index.md)
- [Eviction modes](eviction.md)
- [Generation reservations](reservations.md)
- [Settings](../guides/settings.md)
