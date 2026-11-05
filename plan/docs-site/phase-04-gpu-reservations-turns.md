# Phase 04 — GPU memory: reservations and taking turns

**Depends on:** Phases 01, 02, 03 · **Enables:** Phase 05 (the generation
guide links here), Phase 07 (anchors to link to)

## Goal

Write the two remaining GPU memory pages, with diagrams: generation
reservations, and how LLMs and generation backends take turns on one GPU.
Move the measured reservations table to its own reference page.

## Files touched

- `docs/gpu/reservations.md` — Generation reservations.
- `docs/gpu/taking-turns.md` — LLMs and generation backends on one GPU.
- `docs/reference/measured-reservations.md` — the table.
- `docs/assets/diagrams/vram-reservations.svg` — new.

## Steps

1. **Check the facts first**, as in Phase 03 step 1, for the
   `#generation-vram`, `#generation-sharing` and `#generation-evict`
   sections of `help.html`. Places to read: `internal/api/gen_evict.go`
   (stop order, largest first, waits for running jobs, lock),
   `internal/api/gen_image_jobs.go` (job queue: one per GPU, 16 waiting,
   HTTP 429), `internal/api/gen_prompt_cache.go` (prompt cache save and
   restore), `internal/api/llm_room.go` (HTTP 507 refusal), and the Measure
   code under `internal/generation/` (peak plus 10%). List differences in the
   pull request description.
2. **`docs/gpu/reservations.md`** — "Generation reservations". Sections:
   - `## What a reservation is { #what }`: the VRAM that one instance needs,
     measured once.
   - `## How Measure sets it { #measure }`: Mermaid `flowchart LR`: test
     generation → peak VRAM → plus 10% → reservation saved on the instance.
     Then the video sizes, the edit measurement, Measure inside a budget, and
     when Measure stops.
   - `## Planned, not held { #planned }`: the planner keeps reservations
     free, also for a stopped, enabled instance; an LLM load that would crowd
     one out gets HTTP 507. Diagram `vram-reservations.svg`: one 32 GiB bar:
     desktop and other programs, 1 GiB working memory, margin, reservation of
     instance 1, reservation of instance 2, and the room left for LLMs. Use
     the R9700 example from Help (Qwen-Image Turbo and ACE-Step XL leave
     about 8.6 GiB for an LLM). Recompute the parts from the reference table
     and the planner code before you draw it, and draw what the code gives.
   - `## Loads that toolchest cannot check { #unchecked }`: the chat page
     of llama-server on port 8080.
   - `## Text encoders on the CPU { #te-cpu }`: why catalog sets use
     `--backend te=cpu` and what it saves. Link to the reference table.
   - `## Measured again { #remeasure }`: "Measured with other files or
     settings".
3. **`docs/gpu/taking-turns.md`** — "LLMs and generation backends on one
   GPU". Sections:
   - `## Jobs and chat on one GPU { #sharing }`: they slow each other (the
     R9700 numbers); one job per GPU, up to 16 waiting, then HTTP 429; chat
     never waits; how to keep them on separate GPUs. Mermaid `flowchart LR`
     of the job queue: job → GPU busy? → queue (up to 16) → HTTP 429 when
     full.
   - `## Default: reservations always count { #default }`: switch off; an LLM
     that does not fit beside the reservations gets HTTP 507. Mermaid
     `flowchart TD`.
   - `## Taking turns { #take-turns }`: the switch "LLMs and backends can
     take turns on a GPU" and when it shows (generation backends on,
     llama-toolchest in effect). Two Mermaid `sequenceDiagram`s:
     1. Chat request for an LLM that does not fit → toolchest waits for a
        running job → stops idle backends on that GPU, largest first → LLM
        loads → reply.
     2. Job for a stopped backend → toolchest stops other backends → unloads
        idle LLMs on that GPU, largest first, after their replies finish →
        backend starts → job runs → LLM loads again after the job.
   - `## Lock an instance { #lock }`.
   - `## Keep the prompt cache { #prompt-cache }`: what it saves, the
     26,600-token example, restart the router after you turn it on.
   - `## What never stops backends { #never }`: Benchmarks, Autotune,
     Autoconfigure.
4. **`docs/reference/measured-reservations.md`**: the date, build versions and
   test notes line from Help; the full table as a Markdown table (same
   columns); the notes list under it. Markdown tables do not carry cell
   titles, so add each cell's `title` text to the end of that row's Notes
   cell, in the form "A4000: <title text>".
5. Mark recent features with `Since v2.x` notes by the Phase 03 step 6 rule.
6. Add a "Related" list at the end of each page, as in Phase 03.

## Build gate

- `make docs` passes with no warnings.

## Test plan

- `make docs-serve`. Read each page in light and dark mode and at 375 px
  width. All charts and the SVG render. The wide reference table scrolls
  sideways inside its box, not the whole page.
- The numbers in `vram-reservations.svg` add up to 32 GiB and match the
  reference table.
- The pull request description lists a code reference for each claim (step 1).

## Commit

`docs(gpu): explain generation reservations and taking turns on a GPU`

## Rollback

Revert the commit. The pages go back to stubs. Phase 03's pages link here, but
the stubs still exist, so the strict build still passes.
