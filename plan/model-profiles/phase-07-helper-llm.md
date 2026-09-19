# Phase 07 — Helper model setting and structured LLM calls

**Depends on:** 01 (writes to `models.json` go through the gate) · **Enables:** 08 (model card
advice uses the client), 09 (autoconfigure needs a helper model)

## Goal
Choose the helper model, offer the default one as a one-click download, and
add a small client that sends a chat request to any installed model through
the router and returns JSON checked against a schema. This is the first code
in the app that asks a model for something other than a benchmark.

## Files touched
- `internal/config/config.go`: `HelperModelID string` (`yaml:"helper_model_id"`).
- `internal/api/settings.go`, `internal/api/server.go` (`handleSettingsPage`),
  `internal/api/settings_page_render_test.go` (the mirror struct),
  `web/templates/settings.html`: an "Autoconfigure helper model" select
  listing installed models, with a "Download recommended (Qwen3.5-4B,
  ~3 GB)" button when none is set.
- `internal/api/helper_model.go` (new):
  - `POST /api/helper-model/download` starts the default download through the
    existing downloader and sets `HelperModelID` when it completes;
  - `GET /api/helper-model/status` reports whether a helper is set,
    installed, or downloading.
- `internal/llmcall/client.go` (new package):
  - `NewClient(routerURL func() string, proc *process.Manager, reg *models.Registry, busy func() string)`,
    where `busy` returns a plain-language reason the GPU is busy, or "";
    `server.go` passes a closure that checks `s.jobs`;
  - `Client.JSON(ctx, modelID, messages, schema, out any) error`.
- `internal/llmcall/client_test.go` (new).
- `internal/api/server.go`: routes, and building the `llmcall.Client`.

## Steps
1. **The default helper** is repo `unsloth/Qwen3.5-4B-GGUF`, quant `Q4_K_M`.
   - The download handler lists the repo's files with the existing
     `huggingface.Client.GetModel` and picks the file whose `ParseQuant`
     equals `Q4_K_M`. It does not hard-code a filename, which changes when a
     repo is re-uploaded.
   - If no such file exists, it returns the error "The recommended helper
     model is not available right now; choose an installed model instead."
2. **Settings.** When a model becomes the helper (chosen in Settings or set
   after the download), raise its `ContextSize` to 16,384 if it is lower,
   through `registry.SetConfig`, and show "Context raised to 16,384 so model
   cards fit" under the select. Add the field, following the pattern described in
   `internal/api/settings.go:56-120`: JSON pointer field, form field, page
   data, template select.
   - Tooltip: "A small model that reads model cards for Autoconfigure. It is
     loaded only when you run Autoconfigure, and unloaded afterwards."
3. **The LLM client.**
   - Loading: resolve `registry.RouterName(modelID)`, then POST `/models/load`
     and wait for the model to be loaded, as `Runner.ensureModelLoaded` does
     (`internal/benchmark/runner.go:274`). Extract that wait into an exported
     `process.Manager.LoadAndWait(ctx, name, timeout)` and use it from both
     places.
   - Request: `POST {RouterURL}/v1/chat/completions` with
     `{"model": name, "messages": …, "temperature": 0, "max_tokens": 2048,
     "response_format": {"type": "json_schema", "json_schema": {"name":
     "advice", "strict": true, "schema": schema}},
     "chat_template_kwargs": {"enable_thinking": false}}`.
   - Thinking is turned off with the same `ReasoningControl` logic the runner
     uses (`runner.go:411-437`), chosen from the helper model's detected
     reasoning capability.
   - Decode `choices[0].message.content` into `out`.
   - On a decode failure, retry once with the parse error appended as a user
     message. Then return an error.
   - `Client.Unload(ctx, modelID)` POSTs `/models/unload`.
4. **When the client refuses** (with plain-language errors):
   - if the router is not running;
   - if `busy()` returns a reason, e.g. a running job ("A benchmark is running; Autoconfigure
     needs the GPU. Try again when it finishes.").
5. `Client.LoadedOthers()` returns the names of other currently loaded
   models, so the caller can warn that they will be unloaded
   (`ModelsMax` defaults to 1).

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- `llmcall` tests with an `httptest` fake router (following `newFakeRouter`
  in `internal/benchmark/job_runner_test.go:340`):
  - the request body contains `response_format.json_schema` and
    `enable_thinking: false`;
  - valid JSON decodes;
  - invalid JSON triggers exactly one retry and then an error;
  - a running job is refused.
- Settings render test: the select lists installed models, and the download
  button shows only when `HelperModelID` is empty.
- Download handler test with a fake HF client listing three quants: picks the
  `Q4_K_M` file.
- Manual:
  - Click Download and check the model appears and becomes the helper.
  - Call a debug request through a Go test tag or `curl` against the running
    router with the same body, and confirm llama-server returns valid JSON
    for a small schema.

## Commit
`feat(settings): choose a helper model and call it for structured answers`

## Rollback
Revert the commit. The `helper_model_id` key stays in the YAML and older code
ignores it. A downloaded helper model remains an ordinary installed model.
