# Phase 09 — Dashboard, Web UI Links and Help

**Depends on:** Phase 05 (instances), Phase 06 (reservations), Phase 07
(queue), Phase 08 (music jobs) · **Enables:** Phase 11 (end-to-end checks
use the dashboard)

## Goal

The dashboard shows the generation backends at a glance:
- each instance's state, GPU, reservation and current queue
- a link to each backend's own web UI, built the same way as "Open Chat UI →"
- a warning while a generation job shares a GPU with a loaded LLM, because
  chat is slower then
- a note that jobs started from a backend's web UI are not queued

The Help page explains the feature.

## Files touched

- `web/templates/partials/dashboard_cards.html`: a new "Generation backends"
  article, rendered only when `GenerationEnabled` is on.
- `web/templates/partials/gen_dashboard.html` (new): the instance rows, queue
  counts and warning banner.
- `internal/api/server.go`:
  - `dashboardCardsData` (L955) and `handleDashboard` (L991-1022) gain
    `GenInstances []genInstanceCard`.
  - Each instance's `WebUIURL` uses the host part of `ChatURL`:
    `scheme://host(ExternalURL):<public port>`.
  - The public port is `inst.Port - GenPortStart + publicStart`, where
    `publicStart` is `GenPublicPortStart`, or `GenPortStart` when that is 0
    (Phase 02).
- `web/templates/server.html`: include the same slowed-chat banner partial at
  the top of the page.
- `internal/api/gen_dashboard.go` (new): builds `genInstanceCard` values from
  the store, the manager, the queue snapshot and the router's loaded models.
  It also provides `GET /api/gen/queue`, an SSE snapshot stream that reuses the
  snapshot-diff pattern from `handleJobProgress` (`bench_jobs.go` L1154).
- `web/templates/help.html`: a "Generation backends" section.
- `internal/api/field_help.go`: tooltip text for every new field and metric.

## Steps

1. **Fill each card with:**
   - instance name, kind (Image or Music) and model set
   - state (stopped, starting, running, error)
   - GPU name
   - reservation, with the measured peak and date in the tooltip
   - jobs running and waiting on that GPU
   - "Open web UI →" when the build has a web UI (`HasWebUI`) and the instance
     is running
   - under the link, the note "Jobs started from the web UI skip the
     llama-toolchest queue."
2. **Show the slowed-chat warning.** While the queue snapshot shows a running
   job on GPU *i*, and the router has an LLM loaded with weights on GPU *i*
   (from `ComputeAllocations` for loaded models), show a banner on the
   dashboard and the Server page:

   > Chat is slower while image or music generation runs on GPU *i*. On
   > asmodean (R9700), chat dropped from about 45 to 11–20 tokens per second.

   The banner updates through the queue SSE.
3. **Add a Help section that covers:**
   - turning the feature on
   - building the backends
   - catalog and custom sets
   - measuring and reservations
   - the one-job-per-GPU rule
   - the image routes and both music route sets
   - web UI ports
   - **why the planner may refuse a combination that would fit in practice.**
     Example: on a 32 GB R9700, Qwen-Image + ACE-Step + Gemma 4 12B Q6 ran
     with only 1.8 GB spare. The planner keeps a max(1 GiB, 8%) safety margin
     and adds 10% to each measured peak, so it refuses that combination.
     A 4B LLM such as Qwen3-4B Q8_0 fits next to the two backends.
   - Ming-Image needs more than 17 GB, so it uses most of a 32 GB card. The
     Help text states what Phase 11 measured can run beside it.
   - that Vulkan builds work in host mode only

   Use plain technical language with no slang, and give a tooltip on every
   metric explaining how to read it.
4. **Leave the dashboard unchanged while the switch is off.**

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/api/...`

## Test plan

- **Render tests,** in the style of `build_render_test.go`:
  - With the switch off, the dashboard HTML has no generation section.
  - With it on and one running instance, the card shows the reservation and a
    web UI link whose host comes from `ExternalURL`.
  - With `GenPublicPortStart=9081`, an instance on port 8082 links to 9082.
  - The warning appears only when a job runs on a GPU that also holds a loaded
    LLM.
- **Manual on asmodean:**
  - Start Qwen-Image 2.1 Turbo, ACE-Step and Qwen3-4B Q8_0. That combination
    passes the fit planner on the 32 GB card (Phase 06).
  - Send an image request and watch the banner appear, then clear when the job
    finishes.
  - Open both web UI links.

## Commit

`feat(ui): show generation backends, queue and web UI links on the dashboard`

## Rollback

Revert the commit. This phase only adds UI and read-only endpoints, so a
revert is safe at any time.
