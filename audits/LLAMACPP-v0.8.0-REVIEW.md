# llama.cpp v0.8.0: what it means for llama-toolchest

Date: 2026-10-29
Release: https://github.com/ggml-org/llama.cpp/releases/tag/v0.8.0 (nightly b11960, ggml v0.28.0)
Code checked: `origin/main` at 1962790 (#417)

## Summary

**Nothing has to change for toolchest to build and run v0.8.0.** Every option
toolchest writes into the preset file still exists. The one that was removed
earlier (`--ctx-size-draft`) is already left out by version. The `v0.8.0` tag
matches toolchest's release tag pattern, and the architecture list still
parses: 166 names, including the new ones (`lfm3`, `deepseek41`, `flint`,
`sensenova_u1`, `beam`, `kolibri1`, `muse-spark`, `graniteswitch`).

The release adds features that toolchest does not use yet. The most useful
are the voice endpoints (`/tts`, `/v1/realtime`), the reasoning effort levels,
and a saved n-gram cache for speculative decoding. Toolchest also has one
existing weak point that this release makes more likely to show up:
`split-mode tensor` on architectures that do not support it.

Priorities: **P1** worth doing soon, **P2** useful, **P3** small or optional.

## Status

| Section | Issue | State |
|---|---|---|
| 3.1 tensor-split list | #418 | Done in #422 |
| 3.2 `/tts` | #419 | Done in #428 |
| 3.3 `/v1/realtime` | #420 | Done in #429 |
| 3.4 audio and TTS model types | #430 | Open |
| 3.5 reasoning effort levels | #431 | Open |
| 3.6 saved n-gram cache | #432 | Open |
| 3.7 router memory margin | #433 | Open |
| 3.8–3.10 P3 items | – | No issues yet |

The voice work was tested on compute2 with build b11946, which already has
`/tts` and `/v1/realtime`, not with a v0.8.0 build.

---

## 1. Checks: nothing breaks

| Area | Result |
|---|---|
| Release tag `v0.8.0` | Matches `releaseTagRE["v"]` (`internal/builder/builder.go:1036`). The b-number shown in the UI comes from the commit count, as before. |
| Architecture list | `src/llama-arch.cpp` still has the `LLM_ARCH_NAMES = {` table; `ParseArchNames` reads it unchanged. |
| Preset options | All present in v0.8.0's `common/arg.cpp`: `moe-cache-mib`, `reasoning-preserve`, `spec-type`, `spec-ngram-mod-*`, `spec-draft-*`, `n-cpu-moe-draft`, `model-draft`, `device-draft`, `cache-type-k-draft`, `split-mode`, `tensor-split`, `fit`, `mmproj`, `embeddings`, `models-max`, `models-preset`. |
| Speculative type names | Unchanged: `draft-simple`, `draft-mtp`, `draft-eagle3`, `draft-dflash`, `draft-dspark`, `ngram-cache`, `ngram-mod`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`. |
| "`--model` or `-hf` is required" (#31242) | Only the wording of an error message changed. Router mode (no `-m`) is not affected. |
| `--connect` (#28277) | New peer-to-peer tunnel. The build part is off by default (`LLAMA_CONNECT=OFF`), and the server only opens it when `--connect` or `LLAMA_ARG_CONNECT` is set. See 4.3. |

## 2. Fixes in this release that help toolchest directly

- **Router cancels the child request when the client disconnects** (#29707). Before this, a closed chat could keep generating on the GPU. This helps the eviction logic in `gen_evict.go`, which waits for running LLM requests.
- **The mmproj memory margin no longer grows on each sleep/resume** (#28233). Vision models could end up with less and less room after resuming.
- **HIP: bf16 WMMA build fix for gfx11/gfx12** (#30885). Fixes HIP builds for RDNA3 and RDNA4 (gfx1200/gfx1201, which includes the R9700 and RX 9070 XT) on newer LLVM. This is most relevant to the `rocm-next` image.
- **CUDA: graphs now run in chunks** (#30804), plus fused MoE top-k and gate/up+GLU. Expect 1–2 % faster generation. It changes how CUDA graphs launch, so it is worth one more test against the intermittent CUDA crashes on compute2. It is not a known fix for them.
- **SWA checkpoint fix** (#30666), and **`logprobs` now work with speculative decoding** (#27196).
- **MoE experts in system RAM use a better cache** (#30637): 1–5 % faster for models that use `moe-cache-mib`. No setting changes.
- **Grammar and JSON depth limits** (#29755, #30242). Very deep tool-call schemas are now rejected instead of crashing the server.

---

## 3. Suggested changes

### 3.1 [P1] `split-mode tensor` on architectures that refuse it

llama.cpp keeps a list of architectures that cannot use tensor split
(`llm_arch_supports_sm_tensor` in `src/llama-arch.cpp`). For these, the model
**fails to load** with "LLAMA_SPLIT_MODE_TENSOR not implemented for
architecture '...'". This release adds `lfm3`, `flint` and `mellum`
to the list (#31081). Already on it: `deepseek2`, `minimax-m2/m3`,
`nemotron_h_moe`, `glm-dsa`, `kimi-k3`, `laguna`, `qwen3tts` and more
(34 in total).

Toolchest does not know about this list:

- Autotune's placement stage tries `layer` and `tensor` on every model with
  2+ GPUs (`internal/autotune/plan.go:152`). For these models, every `tensor`
  cell fails and ends up in the "could not be measured" list.
- The GPU assignment dropdown offers `tensor-N` for every model
  (`internal/models/gpu_assign.go:159`). If the user picks it, the model does
  not load, and the only explanation is a log line.

**Suggestion:** read the list from the build's source, in the same way as
`archsFromCheckout`: parse the `case LLM_ARCH_...:` lines inside
`llm_arch_supports_sm_tensor`, then map them to names through
`LLM_ARCH_NAMES`. Store the result on the build. With it:
- autotune skips `tensor` for those architectures, as it already skips
  `tensor` when flash attention is off;
- the GPU dropdown greys out the tensor options, with a tooltip that explains
  why.

### 3.2 [P1] Text-to-speech: `/tts` is not reachable through toolchest

The new `POST /tts` (#26603) is at the server root, not under `/v1`. Toolchest
only forwards `/v1/*` (`internal/api/server.go:909-935`). Everything else
returns "page not found". TTS-capable models (Soprano, Qwen3-TTS, LFM3-Audio,
LFM3-Omni) can therefore load, but nobody can call them.

**Suggestion:** forward `POST /tts` through the same proxy handler, behind
`apiKeyAuth`. Because the body names the model, auto-load and request
counting work the same way as for chat. Multipart bodies (with
`speaker_ref`) already stream through without buffering. The upstream PR
says an OpenAI-style `/v1/audio/speech` is planned. When it comes, it will
pass through the existing `/v1` catch-all, but toolchest itself serves
`/v1/audio` (ACE-Step), so check that the routes do not clash.

### 3.3 [P1] Realtime voice: `/v1/realtime` passes through, but toolchest cannot see it

`/v1/realtime` (#30636, #31269, #31287) is a WebSocket session (OpenAI
realtime protocol) for live transcription and voice chat, now also with
tool calls. It already falls into the `/v1` catch-all, but:

1. **No auto-load and no request counting.** The proxy only does these for
   JSON POSTs (`internal/api/proxy.go:85`). A realtime session names its
   model in the query string (`?model=`) and lasts minutes. An image or video
   job can evict the model in the middle of a call, because `llmUse` does not
   know the session exists.
2. **WebSocket through the middleware has not been tested.** Go's reverse
   proxy supports the upgrade, but the global `middleware.Compress(5)` and
   `middleware.Logger` wrap the response writer. Both should allow taking
   over the connection, but this needs one real test.
3. **API keys from browsers.** Browsers cannot set an `Authorization` header
   on a WebSocket. OpenAI clients send the key in a `Sec-WebSocket-Protocol`
   value instead (`openai-insecure-api-key.<key>`). `apiKeyAuth` would have
   to accept that, or browser clients will not work while a key is set.

**Suggestion:** a small handler for `GET /v1/realtime` that reads `model`
from the query, auto-loads it, holds an `llmUse` count for the whole
connection, and then passes the connection to the proxy. Add a test with
`examples/realtime_client.py` from llama.cpp against an LFM3-Audio-2B GGUF.

### 3.4 [P2] Audio and TTS model types

Toolchest only tells chat and embedding models apart (by a name pattern), and
vision models (from GGUF metadata or a configured mmproj). There is no idea
of "speech-to-text", "text-to-speech" or "voice chat" yet. llama.cpp now
reports this:
- `mtmd_caps` has `gen_audio`, and the router marks TTS-capable models (#26603);
- `mtmd_helper_get_capabilities()` reads what an mmproj file can do (#31183).

**Suggestion:** read the capabilities from the router's model list or
`/props` once the model is loaded, and store them on the model record. Show
them as badges on the Models page and add them to `capabilities` in the
API, so clients like Haruspex can find a TTS or transcription model. This
also makes the "Test" buttons work for audio models: a short `/tts` call, or
a transcription of a bundled WAV.

### 3.5 [P2] Reasoning effort levels

`/props` now lists `chat_template_caps.reasoning_effort_levels`: the levels
the template really accepts, out of minimal / low / medium / high / xhigh /
max (#31271). Examples from the PR: Qwen3.8 accepts low/medium/xhigh,
DeepSeek-V4-Flash high/max, gpt-oss all six. Toolchest finds the reasoning
switch by searching the template text (`internal/models/reasoning.go:70`). It
can only turn reasoning off, and for `reasoning_effort` models it sends
`"none"`, which may not be a level the template accepts.

**Suggestion:**
- Add `effort_levels` to `ReasoningCapability`, filled from `/props` after the
  first load and saved on the model.
- Let a model's settings choose a default effort level (written as the
  request default), and show the levels in the chat test panel.
- In `ApplyThinkingOff`, use the lowest level the template reports when
  `none` is not accepted.

### 3.6 [P2] Saved n-gram cache for speculative decoding

With `ngram-cache`, the n-grams the model learns are now saved to the file
given in `--lookup-cache-dynamic` (#22055). They are written when the server
stops and read again at the next start, so the cache gets better over time
instead of starting empty after every load. Toolchest already offers
`ngram-cache` with no settings (`internal/autotune/plan.go:566`).

**Suggestion:** when `ngram-cache` is on, write
`lookup-cache-dynamic = /data/cache/ngram/<model-id>.bin` in the preset, and
add a "Clear learned n-grams" button. Note: llama.cpp stops with an error if
the file exists but cannot be read, so toolchest should create the folder and
check that it can write to it. Check with a test build whether router
children save the file when the router unloads them, and not only when the
whole server stops.

### 3.7 [P2] Router memory margin (`--models-memory-margin`)

The router can now unload the least recently used models until a new one fits,
keeping N MiB free on each device (#21231). Toolchest runs with
`--models-max 1` by default, and does its own VRAM checks (`fit.go`) and
evictions for generation backends.

**Suggestion:** do not turn this on by default. Toolchest's eviction and the
router would then both decide what to unload. It could become an option for
users who want several small models (embedding, reranker, TTS) loaded beside
one large LLM. The margin would come from `fitMarginGiB`, with `models-max`
raised, and only on builds that have the option (version-gated in
`launch.go`). This needs testing together with `gen_evict.go` before it is
offered.

### 3.8 [P3] Version gates for the new options

Every new option toolchest starts to write needs a `version*` constant in
`internal/models/launch.go`, because the router stops at the first option it
does not know. Use b11960 (the v0.8.0 nightly) or the exact commit counts:
- `--models-memory-margin`: #21231
- `--lookup-cache-dynamic` saving the file: #22055 (the option itself is
  older, but before this change the file was never written)

### 3.9 [P3] Smaller items

- **Rerank** accepts typed content input (#29625). Toolchest does not forward
  the root `/rerank` path either. `/v1/rerank` works through the catch-all.
- **Anthropic `/v1/messages`** now passes llama.cpp sampling options and
  `id_slot` (#30856). Toolchest only adds its per-model sampling defaults to
  `/chat/completions` (`proxy.go:102`). Add `/v1/messages` there so Claude
  Code-style clients get the same defaults.
- **Deferred tools and `tool_reference`** in the chat API (#31256). These
  pass through; nothing to do.
- **The built-in llama.cpp web UI can download models from HF Hub**
  (#29584). Toolchest does not show that UI, and should keep it that way:
  models downloaded there would bypass toolchest's model list.
- **New models** worth adding to the catalog or recommendations:
  LFM3-1.7B / LFM3-VL-2B (small, fast), LFM3-Audio-2B and LFM3-Omni-2B
  (voice, once 3.2–3.4 are done), DeepSeek V4.1 Flash, Nemotron-3.5-Embed.

### 3.10 [P3] `--connect` and the environment

`--connect` opens a peer-to-peer tunnel to the server from outside the
local network. It is off unless asked for, but it can also be turned on with
the `LLAMA_ARG_CONNECT` environment variable. Toolchest passes its own
environment to llama-server on purpose (#393). A one-line safety measure is
to remove `LLAMA_ARG_CONNECT` and `LLAMA_ARG_CONNECT_CODE` from the
environment of the router process, so the API key and Caddy setup cannot be
bypassed by accident. Also keep `LLAMA_CONNECT=OFF` in the build profiles
(it is the default), since turning it on downloads a prebuilt binary during
the build.

---

## 4. Suggested order

1. 3.1 tensor-split list (prevents failed loads and wasted autotune time)
2. 3.2 + 3.3 + 3.4 voice support, as one piece of work, tested with LFM3-Audio-2B
3. 3.5 reasoning effort levels
4. 3.6 saved n-gram cache
5. 3.7, 3.9, 3.10 as time allows

Before 2, build v0.8.0 on compute2 (CUDA) and asmodean (ROCm) and run the usual
smoke tests, to confirm the "nothing breaks" findings in section 1 on real
hardware.
