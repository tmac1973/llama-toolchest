# Phase 03 — Resolve the virtual name in the proxy

**Depends on:** Phase 01, Phase 02 · **Enables:** Phase 04 (`/v1/models`
entry), Phase 05 (Settings card and badges)

## Goal

Make the virtual name work for chat requests. A chat request that names the
virtual name, or has no `model` field, gets its `model` field rewritten to a
real model before anything else in the proxy reads it. Fork routing, loading,
counting, sampling defaults and the system prompt then work with no change.
The same phase documents the behavior, since this is where users first see it.

## Files touched

- `internal/api/default_model.go` — the resolver, the clash check, the
  stale-pin cleanup and the body rewrite.
- `internal/api/default_model_test.go` — resolver and rewrite tests.
- `internal/api/proxy.go` — call the rewrite at the top of the JSON branch;
  refuse the virtual name on non-chat paths, in JSON and multipart requests.
- `internal/api/proxy_test.go` — end-to-end proxy tests with the fake router.
- `docs/api.md` — new section `## Default model name { #default-model }`.

## Steps

1. Add the target type and resolver to `internal/api/default_model.go`:

   ```go
   // defaultTarget is the real model the virtual name resolves to now.
   type defaultTarget struct {
       Name   string // what goes in the request's model field
       ID     string // registry ID or fork instance ID
       Fork   bool
       Reason string // "pinned", "loaded" or "last used"
   }

   // resolveDefaultModel picks the virtual name's target. ok is false
   // when there is none. It never loads a model and never writes the
   // config; clearStaleDefaultPin does the pin cleanup.
   func (s *Server) resolveDefaultModel() (t defaultTarget, ok bool)
   ```

   A model ID *qualifies* when `s.registry.Get(id)` finds it, its
   `Kind() == models.KindChat`, and `advertisedToClients` is true (so it is
   enabled). Its `Name` is `m.PublicName()`.

   Order:
   1. **Pin.** `cfg.DefaultModel.Pin` set: a model pin resolves when the
      model qualifies. A fork pin (prefix `config.DefaultModelForkPrefix`)
      resolves when `s.forksEnabled()` and `s.forks.Instance(id)` exists, with
      `Name = id`; the fork may be stopped. A pin that does not resolve is
      skipped, and resolution goes on to step 2.
   2. **Loaded.** Candidates are the router's models with status `loaded`
      that qualify (from `s.process.ListModels()`, matched to the registry as
      `lookupRouterStatus` does), with time `s.llmUse.lastUse(id)`, plus fork
      instances whose `s.forkMgr.Status(id).State == process.StateRunning`,
      with time `s.forkUse.lastUse(id)`. Pick the latest time. When the latest
      time is shared (zero after a start, before any request), pick the
      state file's model if it is among those tied, else the lowest `Name` in
      byte order, so the choice is stable.
   3. **Last used.** `s.defaultState.get()` when its model qualifies. Only
      router models are saved there (Phase 02), so this step always names a
      model the normal path can load.
   4. Otherwise `ok = false`.
2. Add `func (s *Server) clearStaleDefaultPin()`. It clears a pin whose model
   is gone for good, as the overview decided: a model pin whose model is
   deleted or disabled (does not qualify), or a fork pin whose instance no
   longer exists. A fork pin is kept while forks are switched off, because
   the instance still exists and works again when forks come back on. Under
   `cfgMu`, it checks the pin again, sets it to `""`, saves the config, and
   logs `slog.Info("default model pin cleared: model is gone", "pin", pin)`.
   It runs at the start of `rewriteDefaultModel` (step 4) and of the Phase 05
   card render, only when a pin is set, so a request to a real name never
   pays for it. `/v1/models` does not call it.
3. Add `func (s *Server) defaultNameClash() string`: the virtual name clashes
   when `s.registry.FindByAny(name)` finds a model (any model, enabled or
   not; `FindByAny` matches the registry ID, public name, router name and
   aliases) or `s.forks.Instance(name)` finds a fork instance. A fork
   instance name clashes while forks are off too, because switching forks on
   would make it shadow the virtual name. It returns a short reason, or `""`
   when there is no clash.

   Every call to `s.forks` in this phase and in Phase 05 first checks
   `s.forks != nil`. With no fork store, no fork name clashes, no fork pin
   resolves, and `clearStaleDefaultPin` keeps a fork pin (it cannot tell
   whether the instance exists). Add `func (s *Server)
   defaultModelActive() bool`: `cfg.DefaultModelOn() && s.defaultNameClash()
   == ""`.
4. Add `func (s *Server) rewriteDefaultModel(path string, body []byte) (out
   []byte, err *defaultModelError)`:
   - When `!s.defaultModelActive()` or `!isChatPath(path)`, return `body`
     unchanged.
   - Decode the body into `map[string]json.RawMessage`, so other values keep
     their exact bytes (no float64 rounding of large integers such as
     `seed`). If it does not decode, return `body` unchanged.
   - The request is for the virtual name when `model` is missing, is JSON
     `null`, is `""`, or equals `cfg.DefaultModelName()`. Otherwise return
     `body` unchanged.
   - Call `s.clearStaleDefaultPin()` when a pin is set, then resolve. With
     no target, return a `defaultModelError` with HTTP 503, code
     `no_default_model`, and the message "No model for \"default\": pin a
     model for it, or send a request to a model by its name first." (with the
     configured name).
   - Set `model` to the target's `Name`, marshal the map, and return it.
     Log at debug: `slog.Debug("default model resolved", "name", ..., "target",
     t.Name, "reason", t.Reason)`.
5. In `proxy.go`, in the JSON branch, right after the body is read and before
   `s.forkForModel(bodyModel(body))`:
   - Call `rewriteDefaultModel(r.URL.Path, body)`. On error, `writeProxyError`
     with its status and code and return. Otherwise use the returned body from
     here on (fork routing, `loadForProxy`, sampling, system prompt).
   - When `!isChatPath(r.URL.Path)` and `s.defaultModelActive()` and
     `bodyModel(body) == cfg.DefaultModelName()`, answer
     `writeProxyError(w, 400, "default_model_chat_only", "\"default\" names a
     chat model; send this request to a model by its name.")` (with the
     configured name) and return.
6. In the multipart branch, apply the same 400 when `multipartModel` returns
   the virtual name. A multipart request with no model field is left as it is
   today: every chat path takes JSON, so the missing-model rule of the
   overview needs only the JSON branch.
7. `docs/api.md`: add `## Default model name { #default-model }`. It says what
   the name is for, the resolution order, that a request with no `model`
   field counts as the name on chat paths, that responses name the real model,
   the 503 and 400 errors, and that a model with the same name turns the
   virtual name off. Give the config keys `default_model.name`,
   `default_model.pin` and `default_model.off` as the way to set them.

## Build gate

```bash
go build ./... && go vet ./... && go test ./internal/api/... && make docs
```

## Test plan

Unit tests for `resolveDefaultModel` with a fake registry, router and fork
manager:

- Pin to a chat model wins over a model loaded and used later.
- Pin to a fork instance resolves with `Fork == true`, even while the fork is
  stopped.
- Pin to a deleted model, a disabled model, an embedding model, or a removed
  fork instance: resolution goes on to step 2, and `clearStaleDefaultPin`
  clears the pin from config.
- Fork pin while forks are off: resolution goes on to step 2, and the pin is
  kept.
- `resolveDefaultModel` alone never changes the config.
- No pin, two loaded chat models: the one with the later `llmUse` time wins;
  `touch` on the other moves the result.
- No pin, two loaded chat models with zero times: the one in the state file
  wins; with an empty state file, the lower name wins.
- No pin, a running fork used after the loaded router model: the fork wins.
- No pin, nothing loaded, state file names a model: that model, reason "last
  used".
- No pin, nothing loaded, empty state: `ok == false`.
- Helper models, draft heads, embedding and reranker models are never
  candidates.

Rewrite and proxy tests:

- `{"model":"default"}` and a body with no `model` on `/v1/chat/completions`
  reach the fake router with the target's public name; `seed:
  12345678901234567` keeps its exact digits.
- `{"model":"default"}` on `/v1/embeddings` gets 400
  `default_model_chat_only`; a body with no model on `/v1/embeddings` goes
  through unchanged.
- A pinned model that is not loaded is loaded through the normal path (the
  fake router sees the load call).
- A pinned stopped fork gets the fork's normal 503.
- With `default_model.off: true`, `{"model":"default"}` goes upstream
  unchanged.
- A registry model whose alias is `default`: the request goes to that model,
  not the resolver.
- No target: 503 `no_default_model`.
- Sampling defaults and the system prompt of the resolved model are added.

## Commit

`feat(proxy): resolve the default model name for chat requests`

## Rollback

Revert the commit. Requests then go upstream as sent, as before. Phase 01–02
data stays and is harmless. If only the missing-model rule causes trouble,
the smallest fix is to drop the "missing or null or empty" case in step 4 and
keep the explicit name.
