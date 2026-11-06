# Eviction modes

## What eviction does { #what }

Eviction is how toolchest makes room on a GPU for a new model. When a model does not fit, something that is loaded must unload first. The eviction mode decides who chooses, and what unloads.

## Modes { #modes }

| Mode | Who decides | Generation models | What happens |
| --- | --- | --- | --- |
| **Off** | The llama-server router, by count | Yes | The router keeps up to *Max loaded models*, of any size. At the limit, the model used least recently unloads. On a GPU with a generation reservation, toolchest refuses an LLM that does not fit beside it (HTTP 507). Other GPUs are not checked: a model that does not fit there fails to load. |
| **llama.cpp** | The llama-server router, by VRAM | No | Idle models unload, least recently used first, until the new model fits. The router keeps the largest [GPU memory margin](memory-margin.md) free on each GPU. Needs llama.cpp b11930 or newer. An older build unloads by count, as in Off. |
| **llama-toolchest** | Toolchest, by VRAM | Yes | Idle LLMs unload, least recently used first, on the GPUs that are short, until the new model fits by its estimate. Generation reservations stay free, unless [take turns](taking-turns.md#take-turns) is on. Works with any build. |

The diagrams below show what each mode does with a new load.

### Off { #off }

```mermaid
flowchart TD
    A["Request for a model that is not loaded"] --> B{"A GPU of the model<br/>has a generation reservation?"}
    B -- yes --> C{"Model fits beside<br/>the reservations there?"}
    C -- no --> R["Refused: HTTP 507"]
    C -- yes --> D
    B -- no --> D{"Loaded models =<br/>Max loaded models?"}
    D -- yes --> E["Router unloads the model<br/>used least recently"]
    D -- no --> F["Router loads the model"]
    E --> F
    F --> G{"Fits in VRAM?"}
    G -- yes --> H["Loaded"]
    G -- no --> X["Load fails: out of memory"]
```

The router does not check VRAM in this mode. Set *Max loaded models* to a number of models that fits. `1` means one model at a time.

### llama.cpp { #llamacpp }

```mermaid
flowchart TD
    A["Request for a model that is not loaded"] --> B{"Fits in free VRAM,<br/>less the margin?"}
    B -- yes --> F["Router loads the model"]
    B -- no --> C{"An idle model is loaded?"}
    C -- yes --> D["Router unloads the idle model<br/>used least recently"]
    D --> B
    C -- no --> F
    F --> H["Loaded, or fails out of memory"]
```

The router gets one margin for all GPUs: the largest margin of any GPU. It reads the margin when it starts.

### llama-toolchest { #toolchest }

```mermaid
flowchart TD
    A["Request for a model that is not loaded"] --> N{"Can it fit at all,<br/>with every idle LLM gone?"}
    N -- no --> R["Refused: HTTP 507"]
    N -- yes --> B{"Loaded models =<br/>Max loaded models?"}
    B -- yes --> C{"An idle model is loaded?"}
    C -- no --> RB["Refused: all models are busy"]
    C -- yes --> D["Unload the idle LLM<br/>used least recently, on any GPU"]
    D --> B
    B -- no --> E{"Fits by its estimate<br/>on every GPU it uses?"}
    E -- no --> F["Unload the idle LLM used least recently<br/>that has memory on a short GPU"]
    F --> E
    E -- "yes, or nothing<br/>more can unload" --> G["Toolchest sends the load<br/>to llama-server"]
    G --> H["Loaded, or fails out of memory"]
```

!!! note "Since v2.51.0"
    Toolchest unloads idle LLMs by VRAM in this mode, and keeps several models loaded with generation backends on.

Toolchest plans with the estimate of each model, the [GPU memory margin](memory-margin.md) of each GPU, and the [generation reservations](reservations.md). It unloads only on the GPUs that are short, so models on other GPUs stay loaded.

If the generation reservations are still in the way, toolchest refuses the load with HTTP 507. With [take turns](taking-turns.md#take-turns) on, it stops idle backends instead.

When nothing more can unload, toolchest still sends the load. Then llama-server decides if the model fits.

## Rules in every mode { #always }

- A model that answers a request never unloads.
- *Max loaded models* is always an upper limit. `0` means no limit.
- Autotune and Autoconfigure test loads use one model at a time. Benchmarks do too, except in Off mode: there the benchmark router keeps up to *Max loaded models*.

## Small models stay { #small-models }

With llama.cpp or llama-toolchest, embedding and reranking models stay loaded beside a chat model while they fit.

A small model can unload a big one. This happens when the big model leaves too little room on a GPU that the small model uses:

```mermaid
sequenceDiagram
    participant C as Client
    participant T as Eviction
    participant B as Big chat model
    participant S as Embedding model
    C->>T: Request for the embedding model
    T->>T: The embedding model's GPU is short
    T->>B: Unload (idle, least recently used)
    T->>S: Load
    S-->>C: Answer
```

To keep both loaded, give the big model a smaller context. Then it leaves room for the small model.

## When llama.cpp gives way { #gives-way }

The router knows only its own models. It cannot see the VRAM of generation backends or llama.cpp forks. So the llama.cpp mode cannot run with them.

```mermaid
flowchart TD
    A["Saved mode: llama.cpp"] --> B{"Generation backends or<br/>llama.cpp forks turned on<br/>in Settings?"}
    B -- yes --> C["llama-toolchest decides<br/>The card shows llama.cpp as not available"]
    B -- no --> D["llama.cpp decides"]
```

This depends on the switches in Settings, not on running instances. Your llama.cpp choice stays saved. When generation backends and forks are off again, llama.cpp decides again.

## Split Mode tensor { #split-tensor }

The router cannot see the memory of a model with Split Mode `tensor`. So with llama.cpp, such a load can fail out of memory. Use llama-toolchest, or Split Mode `layer`.

## Changing the mode { #change }

!!! note "Since v2.52.0"
    One Eviction card in Settings holds all the unload settings.

Set the mode in Settings → Eviction (`/settings#eviction`). The Server tab shows the mode in effect.

A new mode applies at the next server start. The Eviction card has a *Restart server* button for this. The *take turns* switch applies at once.

## Related { #related }

- [How toolchest shares a GPU](index.md)
- [GPU memory margin](memory-margin.md)
- [Generation reservations](reservations.md)
- [LLMs and generation backends on one GPU](taking-turns.md)
- [Settings](../guides/settings.md)
