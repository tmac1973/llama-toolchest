# Model config (Configure)

## GPU Assignment { #gpu-assignment }

### GPU *N* / GPUs *i-j* / All GPUs { #gpu-n-gpus-i-j-all-gpus }

One GPU, or layers split across a range or all of them.

### Tensor Parallelism { #tensor-parallelism }

Experimental: splits each tensor over N GPUs. Needs a fast link between GPUs (and NCCL at build time on CUDA), flash attention on, and an `f16` KV cache. N must divide the attention head count of the model.

### Custom { #custom }

Your own `--tensor-split` string, passed through unchanged.

GPUs that are left out are hidden from the model with `--device`, so they stay free. Integrated GPUs show *(iGPU)*, and toolchest never chooses them for you. They need the *Include iGPU Build Target* build option.

## Key settings { #key-settings }

### GPU Layers { #gpu-layers }

Layers on the GPU; 999 means all.

### Context Size { #context-size }

Larger needs more memory for the KV cache. For an embedding model, Context Size is the longest input of one request; for a reranking model, the longest query plus document. Most embedding models, and all reranking models, read an input in one micro-batch, so toolchest raises the micro-batch to the same size. VRAM use grows fast: about 1 GiB at 8K and 7 GiB at 32K for EmbeddingGemma 2. The default is 8192, or the model's limit if it is lower.

### Embedding models and GPUs { #embedding-models-and-gpus }

At download, toolchest puts an embedding or reranking model on the one GPU with the most room, if it fits. Spread over several GPUs, each GPU needs its own working memory for the model. That can be many times the model's size.

### Reranking models { #reranking-models }

!!! note "Since v2.50.0"
    Toolchest can serve reranking models.

A reranker scores how well each document answers a query, for the reranking step of a RAG pipeline. Toolchest recognises one from its file or its name (*rerank*), lists it under *Reranking Models*, and starts it with `--reranking`. Clients send `POST /v1/rerank` (or `/rerank` at the root) with `model`, `query` and `documents`; the answer has one `relevance_score` per document. A reranker marked *check file* has no classification head in its file: llama.cpp serves it but its scores mean nothing, so use another conversion.

### Parallel Conversations and Shared Context { #parallel-conversations-and-shared-context }

With Shared Context on, all conversations use one context pool. A long conversation can use more than its part. If the pool fills, llama.cpp clears waiting conversations, and they read their prompt again. If that is not enough, a request fails. A per-conversation limit stops one conversation from taking the whole pool.

### Flash Attention { #flash-attention }

Usually on: faster, less memory.

### KV Cache Quant { #kv-cache-quant }

`q8_0` about halves KV memory at almost no quality cost; `q4_0` saves more, with some loss.

### Micro-batch (`-ub`) and Batch (`-b`) { #micro-batch-ub-and-batch-b }

Micro-batch usually has the largest effect on prompt speed; measure it with Autotune or a benchmark sweep. Batch must be ≥ micro-batch. Defaults: 512 and 2048.

A larger micro-batch also takes more GPU memory, mostly for the attention mask, which grows with micro-batch × context. With separate conversation slots, each mask covers only one slot's share of the context, so it is a lot smaller than with a shared pool (the default). On a layer split over several GPUs, llama.cpp keeps four copies of the mask so the GPUs can work on different micro-batches at the same time. It drops to one copy when four do not fit, and always runs one when experts stay in system memory (CPU MoE), when the MoE cache is on, or when some layers stay on the CPU, whether that is set in the profile or in Extra Flags. On CUDA builds the memory estimate follows all of this, so a large micro-batch at a long context is planned at what it really uses.

### Direct I/O { #direct-io }

Loads weights without the OS page cache. Use it when VRAM is larger than system RAM.

### Per-Layer Embeddings { #per-layer-embeddings }

Only for models with such a table (Gemma-3N, Gemma-4 E-series, Qwen4-Exp). *Auto* reads tables over 4 GiB from the file as needed. Has no effect with Direct I/O.

### Projector (mmproj) { #projector-mmproj }

Lets the model read images or audio, or speak. Toolchest finds it when it is in the model's folder, or in the folder above it.

Every chat model without built-in vision has this field. To use a projector from another folder, pick one of the projector files on disk from the field's list, or type its path. *Load Projector* off keeps the path but starts the model without it, to save VRAM.

When a model has no projector, Configure asks Hugging Face where one is. It looks in the model's own repository, then in the publisher's sibling repository (mradermacher's `X-i1-GGUF` keeps its projector in `X-GGUF`), then in other quantizations of the same base model, the same publisher's first. If it finds one, Configure shows *Download projector from* that repository, with a button for each file (Q8_0 first, then F16) and its size. The file is saved in the model's own folder and set as the projector when it is done; restart the model to load it. The answer is kept for a day. When Hugging Face cannot be reached, Configure shows nothing and asks again later.

*Load Projector* off starts the model without it and keeps the path. *Projector in RAM* keeps it in system memory instead of on the GPU (`--no-mmproj-offload`). Its VRAM then goes to the model, for more context or more layers on the GPU, but images and audio are read on the CPU, more slowly. The memory estimate counts it as system memory.

### Speculative Decoding { #speculative-decoding }

Predicts several tokens per step, with no change to the answer. A draft model gives the biggest gain; n-gram modes need no extra model. The **?** button explains each mode.

### N-gram Cache file { #n-gram-cache-file }

With llama.cpp b11758 or newer, toolchest keeps what N-gram Cache learns in `/data/cache/ngram/<model id>.bin`. llama-server reads the file at load and writes it at unload. Benchmarks and Autotune do not use it. A broken file makes llama-server stop at load, so toolchest deletes a broken file before a load. *Clear learned n-grams* deletes the file; a loaded model writes it again at unload.

### Sampling { #sampling }

Applies per request, with no restart. The *Preset* list shows defaults from the GGUF, the Unsloth docs and the `generation_config.json` of the base model. Empty fields still use the GGUF defaults. *↻ Refresh presets* looks again.

### System Prompt { #system-prompt }

!!! note "Since v2.50.0"
    A model can have a default system prompt.

Some fine-tuned models need a fixed system prompt. The model card then shows it. Paste it here. The server adds it to each chat request that has no system message. A request with its own system message keeps it. Applies per request, with no restart.

### Extra Flags { #extra-flags }

Any other llama-server flags, added as written.

!!! note

    Most changes need a **server restart**: press *Restart required* in the panel. Sampling changes do not.
