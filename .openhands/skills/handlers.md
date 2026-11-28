---
paths:
  - "internal/api/**"
  - "cmd/**"
---

HTTP handlers live in `internal/api`. Register routes in the existing mux setup
rather than adding a second server, and keep every handler's error path returning
the shared JSON error shape used by its neighbours - do not invent per-route error
formats. Anything rendered into `web/templates/` needs its JS in `web/static/` and
the matching docs page updated in the same pull request (see CLAUDE.md).

Tests are Go tests beside the code (`go test -race ./...`). Browser-side JS is
tested through the Go harness - `make js-test` runs the subset that needs node.
CGO is on: files touching llama.cpp bindings need gcc and build tags, so prefer
changing pure-Go code when a task can be done either way.
