# Phase 08 — Music Job API and ACE-Step Compatibility Routes

**Depends on:** Phase 05 (ace instances), Phase 06 (reservations), Phase 07
(per-GPU queue) · **Enables:** Phase 09 (dashboard shows music jobs), Phase 10
(Caddy must pass the compatibility routes)

## Goal

Give clients a way to make music on port 3000. There is no standard music API,
so there are two sets of routes, both feeding the same per-GPU queue:

1. **llama-toolchest's own job API** under `/v1/music/jobs`. It is plain REST,
   does not depend on any one model, and is meant for Haruspex and future music
   backends.
2. **ACE-Step compatibility routes** that copy the official ACE-Step 1.5 Python
   server API: `POST /release_task`, `POST /query_result`, `GET /v1/audio`.
   Tools already written for that server can point at llama-toolchest instead.

Both sets wrap acestep.cpp's own job API (`/lm`, `/synth`, `/job`), which
clients never see.

## Files touched

- `internal/generation/musicjobs.go` (new): `MusicJobs`, the job records,
  output files and cleanup, and `runAceJob`, which drives `/lm` → `/synth` →
  poll `/job` → fetch the multipart result → extract the audio.
- `internal/generation/musicjobs_test.go` (new).
- `internal/api/gen_music.go` (new): handlers for both route sets.
- `internal/api/server.go`:
  - Register `/v1/music/jobs*` and `GET /v1/audio` inside the `/v1` group,
    before `/*`.
  - Register `POST /release_task` and `POST /query_result` at the root,
    wrapped in `apiKeyAuth`.
  - All of them return 404 while the switch is off.

## Steps

1. **Define the job record:**

   | Field | Notes |
   |---|---|
   | `ID` | 16 hex characters |
   | `InstanceID` | |
   | `Status` | `queued`, `running`, `succeeded`, `failed`, `cancelled` |
   | `QueuePosition` | |
   | `Request` | The normalised request |
   | `Result` | Enriched metadata from `/lm`: caption, bpm, keyscale, duration, seed |
   | `AudioPath` | |
   | `BackendJobID` | ace-server's job ID for the step running now (`/lm` or `/synth`); empty while queued |
   | `Error` | |
   | `Submitted` | |
   | `Started` | |
   | `Finished` | |

   - Records live in memory.
   - Audio files are written to `<data>/gen-output/music/<id>.<format>`.
   - Jobs and files are deleted 24 hours after they finish, by a cleanup tick
     every 10 minutes.
   - At most 200 records are kept; the oldest finished ones are evicted first.
2. **Define the llama-toolchest routes:**

   **`POST /v1/music/jobs`**
   - Body: `{model, caption, lyrics, duration, bpm, keyscale, timesignature,
     vocal_language, seed, inference_steps, guidance_scale, output_format
     ("mp3"|"wav"), rewrite_caption (bool, default false)}`.
   - `rewrite_caption` maps to ace's `use_cot_caption`. It defaults to false
     because the asmodean test showed the rewrite can drift from the prompt:
     asking for a male vocal gave a female one.
   - Returns `202 {id, status, queue_position}`.

   **`GET /v1/music/jobs/{id}`** returns the record as JSON.

   **`GET /v1/music/jobs/{id}/audio`** streams the file with the right
   `Content-Type`. It returns 409 until the job has succeeded.

   **`DELETE /v1/music/jobs/{id}`**
   - A queued job is cancelled.
   - A running job is cancelled through ace-server's
     `POST /job?id=<BackendJobID>&cancel=1`.
   - Returns 204.

   **Model routing** uses `PickInstance(model, "ace")` from Phase 07, with the
   same eligibility rules and 404/503 status codes.
   - **On the `/v1/music/jobs` routes:**
     - An unknown `model` returns 404 in OpenAI error format,
       `{"error": {"code": "model_not_found", ...}}`.
     - An empty `model` falls back to the only eligible ace instance, as for
       images.
     - A 503 uses the same `{"error": {...}}` shape.
   - **On the ACE-Step compatibility routes:** official clients send their own
     model names, such as `acestep-v15-turbo`. So when `model` is empty or
     matches no instance or set ID:
     - If exactly one ace instance is eligible, use it.
     - Otherwise return 404 with
       `{"detail": "unknown model; available: <instance ids>"}`.
     - A 503 uses `{"detail": "<reason>"}`.
3. **Define the ACE-Step compatibility routes** (field names from the official
   ACE-Step 1.5 API docs):

   **`POST /release_task`**
   - Accepts JSON with `prompt` or `caption`, `lyrics`, `audio_duration`,
     `bpm`, `key_scale`, `time_signature`, `vocal_language`, `seed`,
     `use_random_seed`, `inference_steps`, `guidance_scale`, `audio_format`,
     `model`, `thinking`, and `batch_size`.
   - `batch_size` must be 1; larger values return 400 "batch_size > 1 is not
     supported".
   - `thinking` maps to `use_cot_caption`.
   - Unknown fields are ignored.

   **All successful compatibility responses** use the official wrapper:

   ```json
   {"data": <payload>, "code": 200, "error": null, "timestamp": <unix ms>, "extra": null}
   ```

   **Error responses** use the official error shape `{"detail": "<message>"}`,
   with status 400, 401, 404, 415, 429 (queue full) or 500.

   **`POST /release_task` payload:**

   ```json
   {"task_id": "<id>", "status": "queued", "queue_position": <n>}
   ```

   **`POST /query_result`**
   - Accepts `{"task_id_list": ["<id>", …]}`.
   - The payload is a list with one entry per task:
     `{"task_id": "<id>", "status": <0|1|2>, "result": "<JSON string>"}`.
     Status `0` is queued or running, `1` is succeeded, `2` is failed.
   - `result` is a JSON-encoded string holding a one-element list:

     ```json
     [{"file": "/v1/audio?path=<id>.<ext>", "wave": "", "status": <0|1|2>,
       "create_time": <unix s>,
       "metas": {"bpm": <int>, "duration": <seconds>, "genres": "",
                 "keyscale": "<string>", "timesignature": "<string>"},
       "prompt": "<caption used>", "lyrics": "<lyrics used>",
       "seed_value": "<seed>", "lm_model": "<lm file name without .gguf>",
       "dit_model": "<dit file name without .gguf>", "generation_info": "",
       "env": ""}]
     ```

   - `metas`, `prompt`, `lyrics` and `seed_value` come from the enriched
     `/lm` result.
   - `file` is empty until the job has succeeded.

   **`GET /v1/audio?path=<value>`**
   - The `path` value is URL-decoded and reduced to its basename.
   - Only a basename that matches `^[0-9a-f]{16}\.(mp3|wav)$` is accepted, so a
     path cannot reach any other file.
   - The file is served from `gen-output/music`.

   These shapes follow the official ACE-Step 1.5 `docs/en/API.md` on the `main`
   branch as of 2026-10-24. Put a comment with that date and the doc URL at the
   top of `gen_music.go`.
4. **Run each job through the GPU queue.** For every job, `runAceJob`:
   1. takes a ticket on the instance's GPU queue
   2. posts `/lm` and polls `/job` every 250 ms
   3. posts `/synth` with the enriched request and `output_format`, then polls
      again
   4. fetches `/job?id=N&result=1`, parses the `multipart/mixed` body and keeps
      the first part whose Content-Type starts with `audio/`
   5. writes the file and releases the ticket
5. **Handle backend failures.** A non-`done` status from ace-server fails the
   job with the backend's status text. If the instance stops mid-job, the job
   fails with "backend stopped".

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/generation/... ./internal/api/...`

## Test plan

- **`musicjobs_test.go` with a fake ace-server** (an `httptest` handler that
  implements `/lm`, `/synth`, `/job` and a multipart result):
  - Happy path: submit, then succeeded, then the audio bytes match.
  - Cancelling a queued job.
  - Cancelling a running job sends `cancel=1`.
  - A backend failure is reported.
  - Records and files expire after 24 h, tested with an injected clock.
- **Compatibility tests:**
  - `release_task` → `query_result` → `GET /v1/audio` round trip.
  - `batch_size: 2` is rejected.
  - A path traversal in `/v1/audio?path=../x` gets 400.
  - The root routes need the API key when one is set.
- **Queue test:** a music job and an image job on the same GPU do not overlap.
- **Manual on asmodean:**
  - With the ACE-Step XL SFT instance, `POST :3000/v1/music/jobs` with the
    lyrics from `~/scratch/acestep-test/req.json` gives a playable 60 s MP3.
  - Measured on 2026-10-24: about 17–19 s when the GPU is otherwise idle.
  - Run the same through `/release_task` and `/query_result` with `curl`.
  - Send one music and two image requests at once and confirm from the ticket
    timestamps in `GET /v1/music/jobs/{id}` and the instance logs that they ran
    one after another.

## Commit

`feat(api): add music job API with ACE-Step compatible routes`

## Rollback

Revert the commit. Files left in `<data>/gen-output/music` can be deleted. No
other state is persisted.
