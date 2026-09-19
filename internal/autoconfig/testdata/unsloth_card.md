---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
tags:
- unsloth
---
<div align="center">
<a href="https://unsloth.ai"><img src="https://example.com/logo.png" width="200"/></a>
</div>

[![Discord](https://img.shields.io/badge/discord.svg)](https://discord.gg/x) [![Docs](https://img.shields.io/badge/docs.svg)](https://docs.unsloth.ai)

# Qwen3.5-9B GGUF

Unsloth Dynamic 2.0 quants for Qwen3.5-9B.

## Our company

Unsloth makes fine-tuning faster. Join our community.

## Recommended settings

For best results, use temperature=0.6, top_p=0.95, top_k=20, min_p=0.0 in thinking mode.
For non-thinking mode use temperature=0.7, top_p=0.8.

## Running with llama.cpp

```
./llama-server -m Qwen3.5-9B-Q4_K_M.gguf --jinja -c 32768
```

The model supports multi-token prediction (MTP) for speculative decoding with `--spec-type draft-mtp`.
