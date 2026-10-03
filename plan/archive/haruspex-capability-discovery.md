# Feature Request: Expose Model Capabilities for Client Auto-Discovery

> Driven by the Haruspex desktop client, which probes a toolchest server
> to auto-configure its inference settings. Today Haruspex can only
> auto-detect **context size** and **vision**; everything else (sampling,
> parallelism, reasoning mode) is either hardcoded by model-name string
> matching or left to manual entry. Toolchest's registry already holds
> almost all of this data — the work is mostly **exposing existing fields
> through the probe-facing endpoints in a stable, documented shape**, plus
> one genuinely new field (reasoning capability).
>
> Companion change lives in the `haruspex` repo (`src-tauri/src/inference.rs`,
> `src/lib/stores/settings.ts`). Once the endpoints below land, the client
> side gets wired up to consume them.

---

## 0. Background — how Haruspex probes today

Haruspex's probe (`try_llama_toolchest` in `src-tauri/src/inference.rs`)
does an N+1 walk:

1. `GET /api/service/status` — detection gate.
2. `GET /api/service/loaded-models` — enumerate enabled models + loaded state.
3. `GET /api/models/{id}/info` — **once per model**, to read context + vision.

It then parses `/info` defensively, trying **six** different key spellings
for context size (`context_size`, `n_ctx`, `ctx_size`, `max_context_length`,
`config.context_size`, `config.n_ctx`) and several for vision. That
tolerance exists only because there was never a contract. **Since we own
both sides, we can replace guessing with a documented schema.**

---

## 1. Goal

Let a client fully configure itself from a single probe with **no manual
entry and no model-name heuristics**. Concretely, after probing, Haruspex
should know, per model:

| Capability | Haruspex consumer today | Status in toolchest |
|---|---|---|
| Served context size | `getActiveContextSize()` → compaction threshold | ✅ exposed (`config.context_size`) |
| Vision / multimodal | `remoteVisionSupported`, vision checkbox | ✅ exposed (`capabilities: ["vision"]`) |
| Tool calling | gates the whole agentic tool loop | ✅ exposed (`capabilities: ["tools"]`) |
| **Parallel slot count** | `allowParallelInference` + concurrency cap | ⚠️ in registry (`cfg.Parallel`), **not exposed** |
| **Per-request context** | compaction threshold under `-np N` | ❌ must be derived (see §3) |
| **Recommended sampling** | hardcoded `SAMPLING_PROFILES` name-match | ⚠️ in registry + presets, **not exposed** |
| **Reasoning / thinking mode** | `getChatTemplateKwargs()` (`enable_thinking`) | ❌ no field anywhere |
| Max output tokens | `max_tokens` default | ❌ not present (nice-to-have) |

---

## 2. Deliverable A — enrich `/api/service/loaded-models` (kills the N+1)

**Today** (`renderLoadedModelsJSON`, `internal/api/service.go:239`) each
entry carries `id`, `public_name`, loaded state, and a sparse
`context_size`. Haruspex therefore has to fan out to `/info` per model.

**Ask:** fold the capability block (§4) into each loaded-models entry so a
client gets everything in **one** request. `/api/models/{id}/info` stays as
the detailed/per-model view, but the probe no longer needs it.

```jsonc
// GET /api/service/loaded-models
{
  "schema_version": 1,
  "models": [
    {
      "id": "unsloth--Qwen3.6-35B-A3B-GGUF--...",
      "public_name": "unsloth-Qwen3.6-35B-A3B.UD_Q8_K_XL",
      "loaded": true,
      "capabilities": { /* see §4 */ }
    }
  ]
}
```

If folding into loaded-models is undesirable, the fallback is a dedicated
batch endpoint `GET /api/models/capabilities` returning the same array.
Either way the goal is **one round-trip**, not one-per-model.

---

## 3. Two correctness items — please don't skip

These prevent real bugs in the client, not just missing UI niceties.

### 3.1 Served context vs. trained max
`/info` already does this right: `config.context_size` resolves `0 →
m.ContextLength` and reflects the **live** running value. Keep that
semantic and make it explicit in the schema doc: the field clients should
use for compaction is the **served runtime `n_ctx`**, never the model
card's trained max (`context_length`). Surfacing both is fine — just label
which is which.

### 3.2 Per-request context under parallelism
The registry comment on `Parallel` says it plainly: *">1 divides ctx_size
across slots."* So with `context_size=32768, parallel=4`, each request
effectively gets **8192** tokens of KV. Haruspex drives compaction off
this number — if it uses 32768 while the server hands each slot 8192,
requests blow the context and error.

**Ask:** expose `parallel` (currently missing from the `/info` config map —
`cfg.Parallel` exists in the struct but isn't written into `configMap` at
`internal/api/models.go:229`), **and** expose a precomputed
`context_per_request = context_size / max(parallel, 1)`. Precomputing it
server-side means every client gets the division right instead of each
re-deriving it (and some forgetting). This item and the parallel-slot
field in §4 should ship together.

---

## 4. The `capabilities` object (canonical schema)

One versioned object, documented field-by-field. Most map to existing
registry fields; only `reasoning` is new.

```jsonc
{
  "schema_version": 1,

  // --- context (see §3) ---
  "context_size": 32768,          // SERVED n_ctx (cfg.ContextSize, 0→trained max)
  "context_length": 131072,       // model's trained max (m.ContextLength), informational
  "parallel": 4,                  // cfg.Parallel; slot count (1 = no extra slots)
  "context_per_request": 8192,    // context_size / max(parallel,1)  ← clients compact on THIS

  // --- modalities / tools (already computed for the capabilities[] array) ---
  "vision": true,                 // m.HasBuiltinVision || cfg.MmprojPath != ""
  "tools": true,                  // m.SupportsTools
  "embedding": false,             // models.IsEmbeddingModel(...)

  // --- reasoning / thinking (NEW — see §5) ---
  "reasoning": {
    "supported": true,
    "default_enabled": true,
    "toggle": "chat_template_kwargs",   // or "reasoning_effort" | "none"
    "kwarg": "enable_thinking"          // key name when toggle == chat_template_kwargs
  },

  // --- recommended sampling (see §6) ---
  "sampling": {
    "source": "generation_config.json",     // SamplingPreset.Source
    "default": { "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0 },
    "presets": [
      { "name": "thinking",     "label": "Thinking",      "temperature": 0.6, "top_p": 0.95, "top_k": 20 },
      { "name": "non-thinking", "label": "Non-thinking",  "temperature": 0.7, "top_p": 0.8,  "top_k": 20 }
    ]
  },

  // --- generation limits (nice-to-have, §7) ---
  "max_output_tokens": null       // null = no server-imposed cap
}
```

Rules of the road:
- **Versioned.** `schema_version` at top level so the client can branch if
  the shape evolves. Bump on breaking changes only.
- **Omit vs. null.** Prefer explicit `null`/`false` over omitting keys, so
  the client can distinguish "known to be absent" from "server too old to
  report." (This is the opposite of Go's `omitempty` default — worth a
  deliberate choice here.)
- **Stable keys.** Once documented, treat these key names as API. The
  client will read them exactly, dropping its six-spelling fallback.

---

## 5. New field — reasoning / thinking capability

This is the one piece of genuinely new data. There's no reasoning/thinking
field in the registry today (only sampling **presets** happen to be *named*
"thinking"). Haruspex currently sends `enable_thinking` unconditionally via
`getChatTemplateKwargs()` — a Qwen-template assumption that's wrong for
non-Qwen models.

**Ask:** detect and expose, per model:
- `supported` — does the chat template / model expose a reasoning mode at all?
- `default_enabled` — is it on by default?
- `toggle` — the *mechanism*: `"chat_template_kwargs"` (Qwen `enable_thinking`),
  `"reasoning_effort"` (OpenAI-style), or `"none"`.
- `kwarg` — the exact key name when `toggle == "chat_template_kwargs"`.

Detection can likely piggyback on the same GGUF/chat-template inspection
that already populates `SupportsTools`. If full detection is too much for
v1, a per-model registry override (user sets it once) that defaults to
`{supported:false}` is an acceptable first cut — the contract matters more
than the auto-detection.

---

## 6. Recommended sampling — expose what's already there

The registry already carries both per-model sampling overrides
(`Temperature`, `TopP`, `TopK`, `MinP`, `PresencePenalty`, `RepeatPenalty`
at `internal/models/registry.go:132`) and a richer `SamplingPreset` type
(`internal/models/sampling_presets.go`) with thinking/non-thinking presets,
a `Source`, and a `SourceURL`. **None of it is exposed through `/info`.**

**Ask:** surface it under `capabilities.sampling` (§4). This lets Haruspex
delete its hardcoded `SAMPLING_PROFILES` table (in
`src/lib/stores/settings.ts`), which today picks parameters by
string-matching the model id against `"qwen3.5"` and falls back to Qwen
defaults for everything else. With real per-model values, any model gets
correct sampling.

- `default` = the resolved effective values (per-model override if set,
  else the model card's recommendation).
- `presets[]` = the `SamplingPreset` list as-is (the thinking/non-thinking
  pairing maps cleanly onto how Haruspex already splits sampling by
  thinking state).
- Include `source`/`source_url` so a client can show provenance.

---

## 7. Nice-to-have — `max_output_tokens`

Not in the registry today. If there's a known server-imposed completion cap
(e.g. from launch flags), expose it so clients can set a sane `max_tokens`
default. `null` when uncapped. Low priority — list it for completeness.

---

## 8. Explicitly out of scope

Haruspex doesn't behave differently on these, so don't spend effort
exposing them on its account (they may matter for other clients):

- Quant / param count / VRAM estimate / model size — fine as UI labels,
  not load-bearing. (Already in `/info` anyway.)
- Speculative decoding / draft / MTP config — internal to toolchest's
  serving; the client only sees the resulting tokens.
- Embedding dims / streaming / stop tokens — no RAG in Haruspex, streaming
  is universal, stop handling is server-side.
- Tool-call *parser format* (hermes/llama3/etc.) — the router normalizes to
  OpenAI tools, so a boolean `tools` is sufficient for the client. Expose
  the format only if cheap; the client won't branch on it initially.

---

## 9. Acceptance checklist

- [ ] `capabilities` object (§4) defined with `schema_version`, documented keys.
- [ ] `parallel` added to the `/info` config map (currently missing despite
      `cfg.Parallel` existing).
- [ ] `context_per_request` computed and exposed (§3.2).
- [ ] `capabilities.sampling` populated from registry overrides + `SamplingPreset` (§6).
- [ ] `capabilities.reasoning` populated, even if v1 is a manual override (§5).
- [ ] Capabilities reachable in **one** probe round-trip — folded into
      `/api/service/loaded-models` or a new batch `/api/models/capabilities` (§2).
- [ ] `/api/models/{id}/info` keeps returning the same data (detailed view)
      so nothing regresses for existing consumers.
- [ ] Schema documented (README / API docs) since it's now a contract.

---

## 10. Client-side follow-up (tracked in haruspex repo, for reference)

Once this lands, the haruspex changes are:
- `inference.rs`: extend `NormalizedModel` / `ProbeResult`, read the new
  `capabilities` object directly, delete the six-spelling context fallback
  in `parse_toolchest_model_info`, switch the probe to the single-round-trip
  endpoint.
- `settings.ts`: thread per-model sampling + reasoning toggle through
  `InferenceBackendConfig`; replace `SAMPLING_PROFILES` name-matching and the
  unconditional `enable_thinking` in `getChatTemplateKwargs()`; use
  `context_per_request` in `getActiveContextSize()`; auto-set
  `allowParallelInference` from `parallel > 1`.
