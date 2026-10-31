# Phase 07 — Route API requests to fork instances

**Depends on:** 02 (sampling defaults in the curated entry), 05 (running instances and their ports) · **Enables:** clients use a fork model through the toolchest port

## Goal

A POST to the toolchest's OpenAI-compatible API whose JSON `model` is a fork instance's name goes to that instance instead of the router. Requests to a stopped or loading instance get a clear 503. The fork's sampling defaults are added to chat requests that do not set them. Running fork instances appear in `/v1/models`. Everything else goes to the router as before.

## Files touched

- `internal/api/proxy.go` — fork check at the start of the POST wrapper; a reverse proxy per fork instance port.
- `internal/api/fork_proxy.go` — new: `forkForModel(id)`, the 503 answers, and `injectForkSamplingDefaults`.
- `internal/api/v1models.go` — `forkV1Models()` merged into the list and the single-model lookup.
- `internal/api/fork_proxy_test.go` — new tests.

## Steps

1. In the POST wrapper in `proxy.go`, for a JSON body, read the `model` field first (the body is already read whole there). When forks are on and `forkForModel(model)` finds a fork instance with that ID, handle the request in `serveFork` and return; the router path (`loadForProxy`, usage counting, router sampling defaults) does not run. Multipart requests are not checked: no fork recipe serves audio or speech.
2. `serveFork`:
   - Instance not running and not starting → 503 with OpenAI error JSON: `{"error":{"message":"Fork instance avalon-qwen is not running. Start it on the Server page.","type":"unavailable"}}`.
   - Starting (supervisor running, health not yet 200) → 503 with "Fork instance avalon-qwen is loading. Try again in a few minutes." and a `Retry-After: 30` header.
   - Running → for `/chat/completions` paths, add the fork's `SamplingDefaults` for each field the body does not set (same merge rule as `injectSamplingDefaults`), then forward with a `httputil.ReverseProxy` to `http://127.0.0.1:<port>`, keeping the path and query. Streaming responses must flush: set `FlushInterval: 50 * time.Millisecond`, as the router proxy does at `proxy.go:43`.
   - Cache one reverse proxy per port in a map under a mutex; drop the entry when the instance is deleted or its port changes.
3. `/v1/models`: `forkV1Models()` lists fork instances that are running or starting, each as `{"id": <instance id>, "object": "model", "owned_by": "fork:<fork id>", "meta": {"kind": "llm-fork", "fork": <fork id>, "state": "running"|"loading"}}`. Only while forks are on. The single-model lookup at `v1models.go:224` finds them too.
4. GET requests (for example `/props?model=`) are not routed to forks. Clients that need a fork server's `/props` call its port directly; the docs in Phase 08 say so.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- `fork_proxy_test.go`, with an `httptest` server standing in for the fork instance:
  - A chat POST with the fork model reaches the fake server with the same path, and the response streams back.
  - Sampling defaults are added only for missing fields; a client `temperature` is kept.
  - A stopped instance gives 503 "not running"; a starting one gives 503 "loading" with `Retry-After`.
  - A POST with a router model name still goes to the router proxy.
  - With forks off, a POST with a fork instance's name goes to the router (which answers unknown model).
  - `/v1/models` lists a running fork instance with `owned_by` `fork:avalon`, and not a stopped one.
- Manual on asmodean: with the Avalon instance running, send a streaming `/v1/chat/completions` to the toolchest port with `"model": "<instance name>"` and see tokens arrive; at the same time a router model on the other GPU (or CPU) answers its own requests.

## Commit

`feat(forks): route API requests to running fork instances`

## Rollback

Revert the commit. Fork instances keep running and answer on their own ports; only routing through the toolchest port goes away.
