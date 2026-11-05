# Phase 03 — GPU memory: overview, eviction and margin

**Depends on:** Phases 01, 02, and PR #570 merged to `main` (outside this
plan; it rewrites the Help text this phase starts from) · **Enables:** Phase 04 (it links back to these
pages and reuses their diagrams' parts), Phase 07 (anchors to link to)

## Goal

Write the first three GPU memory pages, with diagrams: how toolchest shares a
GPU, the eviction modes, and the GPU memory margin. These are the topics that
PR #570 found hard to explain in Help. Start this phase after PR #570 merges,
from the Help text on `main` at that time.

## Files touched

- `docs/gpu/index.md` — How toolchest shares a GPU.
- `docs/gpu/eviction.md` — Eviction modes.
- `docs/gpu/memory-margin.md` — GPU memory margin.
- `docs/assets/diagrams/vram-margin.svg` — new.
- `docs/assets/diagrams/vram-margin-router.svg` — new.
- `docs/index.md` — remove the "Diagram test" admonition.
- `docs/assets/diagrams/_test-bar.svg` — delete.

## Steps

1. **Check the facts first.** For each claim in the `#eviction` and
   `#vram-margin` sections of `web/templates/help.html`, find the code that
   does it and note any difference. Places to read:
   `internal/config/config.go` (eviction mode, margin and their defaults),
   `internal/api/llm_room.go` and `internal/api/gpu_map.go` (what toolchest
   unloads and on which GPUs), `internal/api/gen_evict.go` (taking turns),
   `internal/models/fit.go` (the 1 GiB working memory and the margin in fit
   plans), and the router start code that passes the margin to llama-server
   (find it from the margin setting's uses). If the code and Help differ, the
   page follows the code, and the pull request description lists each
   difference so Help can be fixed in Phase 07.
2. **`docs/gpu/index.md`** — "How toolchest shares a GPU". Sections, each with
   an explicit id:
   - `## The parts { #parts }`: llama-server's router loads LLMs;
     sd-server and ace-server instances run generation; toolchest plans
     memory and starts and stops all of them. Mermaid `flowchart LR`:
     toolchest → router → LLM A, LLM B; toolchest → instance 1, instance 2;
     all of them → GPU 0 / GPU 1. A dotted edge shows that the router sees
     only its own LLMs' memory, not the instances'.
   - `## What fills a GPU { #what-fills }`: weights, KV cache, buffers,
     working memory, margin, desktop and other programs, reservations.
     Short list; links to the margin and reservations pages.
   - `## Where to go next { #next }`: one line per GPU memory page.
3. **`docs/gpu/eviction.md`** — "Eviction modes". Sections:
   - `## What eviction does { #what }`: two sentences.
   - `## Modes { #modes }`: the Off / llama.cpp / llama-toolchest table
     from Help (Mode, Who decides, What unloads). Then one Mermaid
     `flowchart TD` per mode, titled "Off", "llama.cpp", "llama-toolchest",
     each showing what happens to a new load: count check or VRAM check,
     which model unloads (least recently used, idle only, only on the short
     GPUs), and the end state (loads, fails out of memory, refused HTTP 507,
     or sent to llama-server to decide).
   - `## Rules in every mode { #always }`: a model that answers a request
     never unloads; Max loaded models is an upper limit; Benchmarks,
     Autotune and Autoconfigure load one model at a time.
   - `## Small models stay { #small-models }`: embedding and reranking
     models stay beside a chat model while they fit, and how a small model
     can unload a big one. Mermaid `sequenceDiagram`: request for the
     embedding model → planner finds the GPU short → big chat model unloads.
   - `## When llama.cpp gives way { #gives-way }`: with generation backends
     or forks on, llama-toolchest decides; the saved choice comes back when
     they are off. Mermaid `flowchart TD`: chosen mode llama.cpp → backends
     or forks on? → yes: llama-toolchest in effect, card shows llama.cpp not
     available → no: llama.cpp in effect.
   - `## Split Mode tensor { #split-tensor }`: why the router cannot see it,
     what to use.
   - `## Changing the mode { #change }`: Settings → Eviction, applies at
     the next server start, the Restart server button.
4. **`docs/gpu/memory-margin.md`** — "GPU memory margin". Sections:
   - `## What it is { #what }`: extra VRAM that toolchest keeps free;
     default 0.
   - `## The 1 GiB of working memory { #working-memory }`: why toolchest
     always keeps it (prompts processed in parts use more than a test load
     shows). Diagram `vram-margin.svg`: one 16 GiB bar, bottom to top:
     desktop and other programs, model weights, KV cache, buffers, 1 GiB
     working memory, margin, free. Labels on the right with sizes.
   - `## One value or one per GPU { #per-gpu }`: Every GPU and One GPU rows;
     use the per-GPU row for the desktop GPU.
   - `## What uses the margin { #users }`: Autoconfigure, fit labels,
     Measure, generation fit checks, eviction by llama-toolchest.
   - `## With eviction by llama.cpp { #router }`: the router takes one
     margin for all GPUs, the largest, plus 1 GiB. Diagram
     `vram-margin-router.svg`: three GPU bars with margins 0, 2 and 0.5 GiB
     as set, and next to them the same three bars as the router sees them,
     each with 2 GiB + 1 GiB.
   - `## When a change applies { #applies }`: next plan, load or router
     start; run Autoconfigure again.
   - `## If a load fails out of memory { #oom }`: raise that GPU's margin by
     0.5 to 1 GiB, or use a smaller context.
5. Every page ends with a "Related" list of links to the other GPU memory
   pages and to `guides/settings.md`.
6. Mark a feature with an admonition `!!! note "Since v2.x"` when
   `CHANGELOG.md` shows that it came in one of the last three minor releases
   (for example the per-GPU margin from #566). Count the three from the
   newest release on the day this phase starts, and name that release in the
   pull request description. Phases 04 and 05 use the same three releases.
7. Remove the Phase 01 test admonition and `_test-bar.svg`.

## Build gate

- `make docs` passes with no warnings.

## Test plan

- `make docs-serve`. Read each page top to bottom in light and dark mode. All
  Mermaid charts and both SVGs render, labels are readable, and nothing runs
  off the page at a 375 px wide window.
- Every claim on the three pages has a matching code reference in the pull
  request description (step 1). Someone who knows the code reviews the list.
- Open `…/gpu/eviction/#modes` and `…/gpu/memory-margin/#working-memory` by
  URL. The browser jumps to the heading.

## Commit

`docs(gpu): explain GPU sharing, eviction modes and the memory margin`

## Rollback

Revert the commit. The pages go back to stubs. Nothing in the app links to
them yet (Phase 06 adds the links), so nothing else breaks.
