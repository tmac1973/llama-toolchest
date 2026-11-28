# llama-toolchest

Self-hosted inference control plane: a single Go binary (`cmd/llama-toolchest`)
that supervises llama.cpp sidecars, plus a small agent (`cmd/agent`). HTML served
from `web/templates` with vanilla JS in `web/static`. CGO is on, so gcc and the
llama.cpp bindings are part of the build. Read `CLAUDE.md` for commit and docs
rules — those are enforced, not advisory.

## Commands

- build: `make build` (both binaries into `bin/`)
- check: `go vet ./...` then `go test -race ./...` — this is what CI runs
- browser-JS tests: `make js-test` (needs node; runs a named subset via `go test`)
- installer script tests: `make installer-test`
- docs site: `make docs` (Zensical in `docs/`, published to GitHub Pages)
- dev loop: `make dev`; docker variants: `make docker-compose-up`

## Rules

- Never commit or push `main` directly. Branch off `main`, open a pull request.
  A pre-tool-use hook refuses any push whose destination is `main` or `master`.
- No AI attribution in commits or PR bodies. Conventional Commits.
- Do not weaken or skip a test to get green; fix the code, or say why in the PR.
- A user-visible change needs its docs page updated in the same PR.
- `make build` here does not build llama.cpp sidecars — don't chase those binaries.

## How this repo is worked on

Work happens inside a Docker sandbox with the checkout bind-mounted at
`/workspace`. The shell's starting directory there is `/`, so `cd /workspace`
before anything else — `make`, git and relative paths all assume it. The gate is
`bash .openhands/quality.sh`; the stop hook runs it before any attempt may finish,
and a `pre_tool_use` hook refuses any push whose destination is `main` or `master`.
