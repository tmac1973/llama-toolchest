# Phase 06 — Fork instances on the Server page

**Depends on:** 03 (fork builds to pick), 04 (model set state), 05 (instance API) · **Enables:** the user flow end to end except API routing (Phase 07)

## Goal

With the setting on, the Server page has a "Fork instances" card below the generation instances card. It lists each fork instance with its state, GPU, port and build, and has Start, Stop, Edit and Delete buttons and a log view. An Add or Edit dialog sets the instance's fields and shows hardware warnings. The GPU banner shows a fork's whole-GPU reservation.

## Files touched

- `web/templates/server.html` — the card and the dialog, shown only while forks are on.
- `web/templates/partials/fork_instances.html` — new: the card body (list), refreshed by htmx.
- `web/templates/partials/fork_instance_form.html` — new: the dialog body.
- `internal/api/fork_instances.go` — HTML handlers for the card and dialog, the `GET /api/forks/needs` handler; `ForksEnabled` in the Server page data.
- `internal/api/server.go` — routes for the card, the dialog and `/api/forks/needs`, all 404 while forks are off.
- `internal/api/gen_vram.go` — `genBannerGPUs` includes fork reservations through `genVRAM` (Phase 05); add the "(fork, whole GPU)" wording.
- `internal/api/fork_instances_ui_test.go` — new template tests.

## Steps

1. Card list row: name (the API model name), fork name, state (stopped / starting / running / failed, with the last error line for failed), GPU, port, build ID with the "untested" badge from Phase 03, and buttons. Start shows the busy-GPU message or the needs warnings from the response in the row. The list refreshes every 3 seconds while any instance is starting, like the generation card.
2. Log view: the same log component the generation instances use, one per instance, bound to `/api/forks/instances/{id}/logs`.
3. Dialog fields, each with a tooltip as a `title` attribute, as the generation instance dialog does:
   - Name — edits `Instance.ID`; read-only when editing an existing instance. Tooltip: "The model name clients use in API requests."
   - Fork — choice of curated forks. Shows the fork's notes and license.
   - Model set — read-only state for the chosen fork; when it is not complete, a link to the Models page and the Save button stays on.
   - Build — "Newest" (default) or a specific fork build; untested builds are marked.
   - GPU — one GPU. Tooltip: "The fork server uses this whole GPU. Router models and generation instances cannot use it while the fork server runs."
   - Port — the free port suggested; editable.
   - Extra arguments — "More llama-server options. They are added after the fork's own options."
   - Extra environment — one `NAME=value` per line. Tooltip: "Environment variables for this fork server only."
   - Start with toolchest — checkbox.
4. When the dialog opens and when Fork or GPU changes, call a small endpoint `GET /api/forks/needs?fork=&gpu=` that returns `CheckNeeds` sentences, and show them as a warning box above Save. Save stays enabled.
5. When no fork build exists, the dialog says "No Avalon build yet. Build it on the Builds page." with a link, and Save stays enabled (the start will fail with its own message until a build exists).
6. The GPU banner: a fork reservation reads "GPU 1: avalon-qwen (fork, whole GPU)".

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `fork_instances_ui_test.go`: the Server page has no fork card while forks are off; with forks on, the card lists instances with their states; the dialog shows the needs warnings from a fake `HostInfo`; untested builds carry the badge.
- Manual on asmodean: add an Avalon instance from the dialog, start it, watch the log fill and the state go to running, try to start a generation instance on the same GPU and see it refused, stop the fork instance, edit it, delete it.

## Commit

`feat(forks): manage fork instances on the Server page`

## Rollback

Revert the commit. The API from Phase 05 stays and still works from `curl`.

## As built (2026-11-01)

- The card refreshes every 5 seconds, like the generation card, not only while an instance starts.
- Start and Stop answer the refreshed table; the start's hardware warnings show under the state until the instance stops.
- "The GPU banner" in step 6 is the GPU map on the Server page: a running fork instance shows there as "<instance> (fork, whole GPU)", striped as reserved. The generation banner (jobs next to loaded LLMs) is unchanged.
- The needs box also takes the chosen build, to know the backend.
