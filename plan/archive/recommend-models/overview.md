# Recommended Models — Project Overview

## Problem
The Download Models tab can only search HuggingFace by name. A user who does
not already know which model they want has no way to ask "what runs well on
this machine?". The answer depends on things most users cannot work out by
hand: how many GPUs they have and how much memory each has, how large each
quant of a model is, how much memory the KV cache needs at a given context,
and whether a mixture-of-experts (MoE) model can keep some experts in system
memory.

The tab's current Fit column is also rough. It multiplies the file size by
1.1 and compares the result with the first GPU's total memory
(`vramFit`, `internal/api/server.go:461`). It ignores context, other cards,
integrated GPUs and any safety margin. A file marked "fits" may only fit with
a small context.

vllm-toolchest solved the same problem with a "Find recommended models"
button (its `plan/recommend-models-overview.md` and phases 15–19). This
project copies its UX and its main design rules, adapted to llama.cpp and
GGUF.

## Goals
- **A "Find recommended models" button** on the Download Models tab. It
  builds a list of HuggingFace GGUF models that should run well on this
  machine, in four orders: **Best quality**, **Fastest**, **Longest
  context** and **Newest**.
- **A suggested quant for a chosen context.** A context selector
  (8K / 32K / 128K / model maximum) sits above the category buttons. For
  each model, the card shows the largest quant that runs fully on the GPUs
  at that context.
- **One card per base model.** Repos from different publishers (unsloth,
  bartowski, lmstudio-community, ggml-org, …) that quantize the same base
  model share one card. The card names the preferred publisher. The others
  are listed in Details.
- **One planner.** Every fit figure comes from `models.PlanFit`
  (`internal/models/fit.go:104`), the same planner Autoconfigure uses. The
  feed, the Fit column and Autoconfigure therefore always agree.
- **Fill in the config on download.** A model downloaded from the feed gets
  the planned context, KV cache type, GPU placement and expert offload as
  its starting config, with a visible note saying where they came from.
  This matches vllm-toolchest's phase 19.
- **A better Fit column.** The browse tab's Fit column uses the same planner
  and says what context each file holds.
- **Plain language.** Every figure on a card has a tooltip that says how to
  read it. The text is written for users who are not LLM experts and for
  readers whose first language is not English.

## Non-goals
- No model names in the code. The only curated data is a list of trusted
  publishers, used to raise a repo's rank and to choose between publishers.
  It never lowers a repo's rank.
- No quality scores from benchmarks or leaderboards. Quality is estimated
  from the parameter count and the bits per weight only.
- No grouping by use case (code, tool use, vision). The card marks vision
  support when the repo has an mmproj file, and nothing more.
- No automatic downloads. A card opens the existing Details panel, which
  keeps its disk-space, existing-copy and gated-repo checks.
- No ModelScope in the first version. The feed uses HuggingFace only. The
  search below it keeps both sources.
- No dense models with layers on the CPU. They are too slow to recommend.
  MoE models with experts in system memory are included and labelled as
  slower.
- No quants below about 3 bits per weight (IQ3_XXS and up).
- No change to how GPUs are detected. Machines with no detected GPU get a
  sentence saying no recommendations are possible, and search still works.
- No measured figures. Every figure on a card is a projection, because a
  model that has never run has nothing measured.
- No changes to vllm-toolchest.

## Users & primary flow
Users: people running llama-toolchest on their own GPU machine, many of them
not LLM experts.

1. The user opens Download Models. Above the search box is a collapsed
   section with a **Find recommended models** button and one sentence:
   "Models from Hugging Face that should run on this machine, with the
   quant and context each would get."
2. Clicking the button builds the list. The first build takes up to a
   minute; later ones use the cache and take a few seconds.
3. The feed shows:
   - a line naming the hardware the list was worked out for
     (e.g. "3× RTX A4000 · 48 GB VRAM · 64 GB RAM · build b6500"), and how
     long ago it was built;
   - the context selector (default 32K) and the four category buttons;
   - the first 8 cards, with "Show N more" below them;
   - a folded list of repos that could not be fully checked, each with the
     reason.
4. A card reads, for example:
   - **Qwen3-32B** · from unsloth · vision
   - Q5_K_M · 22.6 GB · 5.7 bits per weight
   - All on 2 GPUs · 32K context · f16 KV cache · 26.1 GB of 44 GB
   - At 128K: Q4_K_M with 8-bit KV cache
   - A **Details** button.
5. Changing the context selector or the category re-sorts the list
   immediately, with no new request to HuggingFace.
6. **Details** opens the existing file panel for the chosen repo, with the
   suggested file highlighted as "Suggested for 32K".
7. The user downloads the suggested file. When the download finishes, the
   model's config is filled in from the plan, and the model page shows
   "Settings suggested for this machine when this model was downloaded from
   the recommendations (32K context). Autoconfigure and Autotune can refine
   them."

## Architecture sketch
```
Download Models tab ──GET /api/recommend?intent=&ctx=──▶ recommend engine
                                                           │
       ┌───────────────────────────────────────────────────┤
       ▼                                                   ▼
 HuggingFace list queries                       per-finalist detail (cached on disk
 (filter=gguf, expand[]=gguf, …)                 per repo revision):
   → servable filter                               • file tree → quant sizes, shards, mmproj
   → group by base model                           • one remote GGUF header read
   → size filter (IQ3 weights vs memory)             (rangeReader + metadata-only parse)
   → coarse rank → ~40 finalist groups                       │
                                                             ▼
                                         models.PlanFit per (quant × context class)
                                                             │
                                                             ▼
                                   per class: suggested quant + plan; 4 orders × 4 classes
                                                             │
                                   in-memory pool keyed on the hardware profile, 6 h
```

Main parts:
- **Hardware profile.** `Server.hardware()` (`internal/api/autoconfig.go:66`)
  plus the active build and the architectures it supports. A change in any
  of these gives a new key, and the pool is rebuilt on the next click.
- **Remote GGUF metadata.** `internal/modelsource/remote.go` (`rangeReader`)
  and `models.ParseGGUFMetaFrom`, with a new mode that stops after the
  metadata section, so a probe costs a small read rather than the whole
  tokenizer.
- **Supported architectures.** Each build records the list of
  `general.architecture` names its llama.cpp source knows, read from
  `src/llama-arch.cpp`.
- **The engine** (`internal/recommend`, new): candidates, grouping, the quant
  picker, scoring, the cache and the endpoints.
- **The feed** (`web/templates/partials/recommend_feed.html`, new): htmx
  partials in the style of the existing browse page.
- **Seeding** in `onDownloadComplete` (`internal/api/hf.go:435`).

## Design decisions
- **One card per base model.** Grouped by the HuggingFace `base_model`
  field (`cardData.base_model`, the first entry when it is a list). When it
  is missing, the repo name with a trailing `-GGUF` removed is used.
  Publisher order: trusted publishers in list order first, then downloads.
- **The quant is chosen by context.** For each class, the largest quant that
  runs fully on the GPUs with at least about 4.2 bits per weight. If none
  fits, a lower-bit quant (3 to 4.2 bits) is tried, then MoE expert offload.
  Section "Quant picker" in phase 03 gives the exact order.
- **Bits per weight are measured, not read from the name.** They are worked
  out as `file size × 8 / parameter count`. This handles unsloth's "UD"
  quants and other mixed quants, whose names do not say their real size.
  The quant label is only shown, never used for decisions.
- **Switching category or context never calls HuggingFace.** All 16
  combinations (4 orders × 4 classes) are computed when the list is built.
- **Three outcomes for a repo.** It is verified (shown with a plan),
  unverified (listed with a reason), or dropped silently (not a text
  model, too small, too large, an MTP head or draft model).
- **Metadata-first.** `expand[]=gguf` in the HuggingFace list query returns
  the parameter count, architecture and trained context length of each
  repo, so most filtering needs no extra requests. Only finalists get a
  file tree and a header read.

## Phases
| # | Phase | Depends on |
|---|-------|------------|
| 01 | [Remote GGUF metadata and a better Fit column](phase-01-remote-metadata-and-fit-column.md) | — |
| 02 | [Architectures each build supports](phase-02-build-architectures.md) | — |
| 03 | [Recommendation engine](phase-03-recommend-engine.md) | 01, 02 |
| 04 | [The feed on Download Models](phase-04-recommend-feed.md) | 03 |
| 05 | [Fill in the config on download](phase-05-seed-on-download.md) | 03, 04 |

Phases 01 and 02 do not depend on each other and can be done in either
order. Each phase is one PR with a conventional commit title.

## Risks
- **HuggingFace rate limits.** A cold build makes about 20 list queries,
  then up to 2 requests and one ranged read per finalist: around 100
  requests. The client has no retry for 429 responses today. Phase 03 adds
  one retry that honours `Retry-After`, and a build that fails part way still
  shows what it has.
- **Estimates without the tensor table.** A metadata-only read does not
  give the expert layer span, the expert size or the embedding sizes. Phase
  01 says how each is filled in, and checks the result against full parses
  of local files.
- **Coefficients from one machine.** The VRAM estimator's coefficients come
  from one ROCm machine (`plan/ple-vram-findings.md`). The feed inherits
  any error in them. The margins in `PlanFit` (8% per card, at least 1 GiB)
  cover most of it.
- **Scoring weights are guesses at first.** vllm-toolchest's first live
  build found several ranking problems (huge models filling the list, an
  embedding model near the top of Fastest, CI test models). Phase 03 ends
  with a live build on compute2 and records any changes in an "As built"
  section.
