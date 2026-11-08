# Phase 04 — List the virtual name in /v1/models

**Depends on:** Phase 03 · **Enables:** Phase 05 (the card shows the same
target the listing shows)

## Goal

Let clients find the virtual name. `/v1/models` lists it first, with meta that
marks it virtual and names its current target, and `/v1/models/{name}` answers
for it. The entry is left out when the feature is off, the name clashes, or
there is no target.

## Files touched

- `internal/api/v1models.go` — `defaultV1Model`, used by `handleV1Models` and
  `handleV1Model`.
- `internal/api/v1models_test.go` — listing and lookup tests.
- `web/apidocs/openapi.yaml` — describe the entry and its meta fields.
- `docs/api.md` — add the listing to the `#default-model` section.

## Steps

1. Add to `internal/api/v1models.go`:

   ```go
   // defaultV1Model is the /v1/models entry for the virtual model name, or
   // nil when it is off, clashes with a real name, or has no target.
   func (s *Server) defaultV1Model() map[string]any
   ```

   It returns `{"id": <name>, "object": "model", "created": 0, "owned_by":
   "llama-toolchest", "meta": {"kind": "text", "virtual": true, "target":
   <target Name>, "target_reason": <"pinned" | "loaded" | "last used">}}`.
   It calls `resolveDefaultModel`, which never loads a model or writes the
   config.
2. In `handleV1Models`, when `kind == "" || kind == "text"`, put the
   `defaultV1Model()` entry first in `data` when it is not nil.
3. In `handleV1Model`, before the registry lookup: when
   `s.defaultModelActive()` and the requested ID equals the configured name,
   answer with `defaultV1Model()`, or 404 like an unknown name when it is nil.
4. `web/apidocs/openapi.yaml`: in the model object schema, add the optional
   `meta.virtual` (boolean), `meta.target` (string) and `meta.target_reason`
   (string, enum `pinned`, `loaded`, `last used`) fields with a one-line
   description each. Keep `TestOpenAPISpecMatchesRouter` passing (no new
   routes).
5. `docs/api.md`: add a short paragraph and an example entry to the
   `#default-model` section.

## Build gate

```bash
go build ./... && go vet ./... && go test ./internal/api/... && make docs
```

## Test plan

- With a pin: the first entry is the virtual name, `meta.target` is the pinned
  model's public name, `target_reason` is `pinned`.
- `?kind=text` includes it; `?kind=image` does not.
- With `default_model.off: true`, or a real model aliased `default`, or no
  target: no virtual entry, and `/v1/models/default` returns 404 (or the real
  model, in the alias case).
- `/v1/models/default` with a target returns the same object as the list
  entry.
- Listing never triggers a model load (the fake router sees no load call).

## Commit

`feat(api): list the default model name in /v1/models`

## Rollback

Revert the commit. Requests to the virtual name keep working (Phase 03); only
the listing goes away.
