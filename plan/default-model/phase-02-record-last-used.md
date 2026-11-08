# Phase 02 — Record which chat model was used last

**Depends on:** Phase 01 · **Enables:** Phase 03 (the unpinned rule and the
restart fallback)

## Goal

Know, at any moment, when each loaded chat model and each fork instance was
last used, and save the router chat model used last to the Phase 01 state
file. Router models already have an in-memory last-use time in `llmUse`.
Forks have none. This phase adds fork last-use times in memory. Forks are not
saved to the state file: a fork does not start on request, so after a restart
a saved fork could only answer 503. Request behavior does not change.

## Files touched

- `internal/api/fork_proxy.go` — `serveFork` records the fork's use.
- `internal/api/fork_use.go` (new) — the `forkUse` type: last-use time per
  fork instance ID.
- `internal/api/fork_use_test.go` (new) — tests for `forkUse`.
- `internal/api/server.go` — a `forkUse forkUse` field on `Server`.
- `internal/api/proxy.go` — `loadAndCountRequest` notes the router model in
  the state file when the request is a chat request for a chat model.
- `internal/api/default_model.go` (new) — `isChatPath(path string) bool` and
  `(s *Server) noteChatUse(id string)`.
- `internal/api/default_model_test.go` (new) — tests for `isChatPath` and
  `noteChatUse`.
- `internal/api/proxy.go` — `loadForProxy` passes the request path to
  `loadAndCountRequest` so it can tell a chat request from others.
- `internal/api/proxy_test.go`, `internal/api/llm_room_test.go`,
  `internal/api/gen_prompt_cache_test.go` — update the test calls of
  `loadAndCountRequest` to the new signature.

## Steps

1. Create `internal/api/default_model.go` with:

   ```go
   // chatPathSuffixes are the request paths the virtual model name
   // applies to: the chat and text completion APIs, with or without /v1.
   var chatPathSuffixes = []string{"/chat/completions", "/completions", "/messages", "/responses"}

   // isChatPath reports whether path is one of chatPathSuffixes.
   func isChatPath(path string) bool
   ```

   `isChatPath` matches by suffix, so `/v1/chat/completions` and
   `/chat/completions` both match, and `/v1/messages/count_tokens` and
   `/v1/embeddings` do not.
2. In the same file add `func (s *Server) noteChatUse(id string)`, which
   calls `s.defaultState.note(id, false, time.Now())`. The `Fork` field of
   the state file stays for a later use and is always false here.
3. Create `internal/api/fork_use.go`: `type forkUse struct { mu sync.Mutex;
   last map[string]time.Time; now func() time.Time }` with `note(id string)` and
   `lastUse(id string) time.Time`, using the same `now` test-clock pattern as
   `llmUse`.
4. In `serveFork`, after the state check passes (`process.StateRunning`) and
   before the request is forwarded, when `isChatPath(r.URL.Path)`:
   `s.forkUse.note(inst.ID)`.
5. Change `loadAndCountRequest(ctx, body)` (in `proxy.go`) to
   `loadAndCountRequest(ctx, path, body)`. Its one caller, `loadForProxy`,
   passes `r.URL.Path`; update the test callers too. Right after
   `s.llmUse.begin` succeeds, when `isChatPath(path)` and the registry
   model's `Kind() == models.KindChat` and `advertisedToClients(m, cfg)`,
   call `s.noteChatUse(id)`. Helper models and draft heads are never noted.

## Build gate

```bash
go build ./... && go vet ./... && go test ./internal/api/...
```

## Test plan

- `isChatPath`: true for `/v1/chat/completions`, `/chat/completions`,
  `/v1/completions`, `/v1/messages`, `/v1/responses`; false for
  `/v1/embeddings`, `/v1/rerank`, `/v1/audio/speech`, `/v1/messages/count_tokens`.
- `forkUse`: two notes with a test clock give the later time.
- Proxy test (existing fake router pattern): a chat request to a chat model
  updates `s.defaultState.get()` to that model with `Fork == false`; an
  `/v1/embeddings` request to an embedding model does not change it.
- Fork proxy test (existing `fork_proxy_test.go` pattern): a chat request to a
  running fork sets `s.forkUse.lastUse(id)` and leaves the state file as it
  was; a request to a stopped fork (503) sets nothing.

## Commit

`feat(proxy): record the last-used chat model and fork use times`

## Rollback

Revert the commit. It adds only bookkeeping; no response changes. The state
file it wrote stays and is harmless.
