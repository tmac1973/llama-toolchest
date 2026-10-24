# Phase 07 — Per-GPU Job Queue and OpenAI Image API

**Depends on:** Phase 05 (instances), Phase 06 (only instances with a
reservation are served) · **Enables:** Phase 08 (music jobs use the same
queue), Phase 09 (dashboard shows the queue)

## Goal

Serve `POST /v1/images/generations` and `POST /v1/images/edits` on port 3000.
Requests are routed by the `model` field to the right sd-server instance.

All generation work for a GPU goes through one **per-GPU queue**, so only one
image or music job runs on a GPU at a time. The asmodean test measured how much
jobs slow each other when they overlap: images 2.3x, songs 2.8x, chat 2.2–4x.

Chat is never held back by the queue. Image and music models appear in
`/v1/models` with a `kind` field.

## Files touched

- `internal/generation/queue.go` (new): `Queue`, with one FIFO per GPU index.
  - `Submit(ctx, gpu, job) (*Ticket, error)` and `Ticket.Wait(ctx)`.
  - `Ticket.Position()`, `Snapshot()` and `Subscribe()` for SSE.
- `internal/generation/queue_test.go` (new).
- `internal/api/gen_images.go` (new): the image handlers.
- `internal/api/server.go`: register `POST /v1/images/generations` and
  `POST /v1/images/edits` inside the `/v1` group (after `apiKeyAuth`, before
  the `/*` catch-all) and before `newProxyHandler()`.
- `internal/api/v1models.go`: append generation instances when the switch is
  on, and add `kind` to `meta` for LLMs as well.
- `internal/api/v1models_test.go`: update the expected shapes.
- `internal/generation/measure.go`: `Measure` also refuses while the GPU's
  queue has a running or waiting job (step 5).
- `internal/generation/route.go` (new): `PickInstance(model, kind)`, shared
  with Phase 08.

## Steps

1. **Queue rules:**
   - **One running job per GPU,** taken in FIFO order.
   - **At most 16 waiting jobs per GPU.** `Submit` returns `ErrQueueFull` when
     the limit is reached, and the handler turns that into HTTP 429 with
     `Retry-After: 30`.
   - **Waiting jobs can be cancelled.** If the client's context is cancelled
     (it disconnected), the waiting ticket is removed from the queue.
   - **Running jobs finish.** A disconnect does not interrupt a running sd job.
     sd-server has no cancel on the OpenAI route, so the job finishes and its
     result is discarded.
   - Each ticket records submitted, started and finished timestamps, plus the
     instance ID and kind.
   - The queue keeps the last 100 finished tickets per GPU in memory.
     `GET /api/gen/queue/history?gpu=<i>` returns them as JSON, with
     timestamps in RFC 3339 format and millisecond precision. This is how
     Phase 11 checks that jobs on one GPU never overlap.
2. **Image request flow:**
   1. Read the JSON body (or multipart for `/edits`) only far enough to find
      `model`. For multipart, parse the form with a 64 MiB memory limit and
      spill to temp files beyond that. Then rebuild the body.
   2. Pick an instance with `PickInstance(model, "sd")`. Only instances that
      are enabled, running, not `measuring`, and have `ReservedMiB > 0` are
      **eligible**.
      - **An instance ID matches:** use that instance.
      - **A model-set ID matches:** pick among the eligible instances of that
        set. Take the one whose GPU queue has the fewest running plus waiting
        jobs; on a tie, the lowest instance ID.
      - **`model` is empty:** if exactly one sd instance is eligible, use it.
      - **Nothing matches:** return 404 `model_not_found` in OpenAI error
        format.
   2a. For `/v1/images/edits`, if the picked instance's set does not have the
       `edit` capability stored in `ModelSet.Capabilities` (set in Phase 04),
       return 400 `invalid_request_error` with "model <id> does not support
       image edits".
   3. If matching instances exist but none is eligible, return 503 with the
      reason: "backend <id> is not running", "backend <id> is being
      measured" or "backend <id> has no VRAM reservation yet".
      - Instances do not start automatically on request; that would be
        load-on-demand, which is a non-goal.
   4. `Submit` to the queue for the instance's GPU. While waiting, the
      response has not started yet. There is no server-side timeout, which
      matches the proxy today.
   5. When the job starts, forward the request with a one-off
      `httputil.ReverseProxy` to `http://127.0.0.1:<port>` and the same path.
      Stream the response back. Mark the ticket finished when the response body
      has been copied.
3. **`/v1/models` entries.** Each running or enabled generation instance is
   listed as:

   ```
   {id: <instance id>, object: "model", owned_by: "llama-toolchest",
    meta: {kind: "image"|"music", model_set, recipe, capabilities, gpu,
           reserved_mib, state}}
   ```

   - While the switch is on, existing LLM entries gain `meta.kind: "text"`.
   - With the switch off, nothing is added, including `kind` on LLM entries,
     so the output is byte-identical to today.
4. **Keep catch-all routing clear.** The new routes come before `/*`, so
   unrelated `/v1/*` paths still go to the llama router unchanged.
5. **Close the Phase 06 gap.** `Measure` now checks
   `Queue.Snapshot()` for the instance's GPU and refuses while a job is running
   or waiting there.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/generation/... ./internal/api/...`

## Test plan

- **`queue_test.go`:**
  - Three jobs on one GPU run strictly one after another; the ticket
    timestamps do not overlap.
  - Jobs on two GPUs run at the same time.
  - The 17th waiting job gets `ErrQueueFull`.
  - A cancelled waiting ticket is removed and the queue moves on.
- **API tests with a fake sd-server** (an `httptest` server registered as a
  running instance):
  - The request is routed by `model`.
  - With two eligible instances of the same set, the one with the shorter
    queue is picked.
  - Instances that are measuring or have no reservation get a 503.
  - An unknown model gives a 404 in OpenAI format.
  - A stopped instance gives a 503.
  - Multipart `/edits` passes the image part through unchanged.
  - `/v1/chat/completions` still goes to the router.
- **`v1models_test.go`:**
  - With the switch off, the output is unchanged.
  - With it on, instances have `meta.kind` and `capabilities: ["alpha"]` for
    Ming-Image.
- **Manual on asmodean:**
  - Use `POST :3000/v1/images/generations` with `model: "ming-image"` and
    confirm the PNG has an alpha channel (`python3 -c` with PIL, or
    `file`/`identify` showing RGBA).
  - Send two image requests at once and confirm from the instance log
    timestamps that they ran one after the other.
  - Point Haruspex's image setting at `http://asmodean:3000/v1` and generate an
    image.

## Commit

`feat(api): add per-GPU generation queue and OpenAI image routes`

## Rollback

Revert the commit. Image requests then fall through to the llama router, which
returns 404. Nothing is persisted by the queue.
