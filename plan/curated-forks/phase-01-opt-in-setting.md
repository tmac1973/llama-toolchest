# Phase 01 — Opt-in setting

**Depends on:** nothing · **Enables:** every later phase checks this setting before it shows or runs anything

## Goal

Add the "llama.cpp forks (experimental)" switch to Settings. It is off by default. While it is on, router memory is turned off for the same reason as with generation: the router cannot see VRAM that is held outside it. With nothing behind the switch yet, turning it on only changes the router memory setting. Later phases hang their UI, routes and processes on it.

## Files touched

- `internal/config/config.go` — new field `ForksEnabled bool \`yaml:"forks_enabled,omitempty"\`` after `GenerationEnabled`, with a comment.
- `internal/api/server.go` — new `forksOn()` helper next to `generationOn()`.
- `internal/api/settings.go` — `ForksEnabled` in the GET view, the JSON PUT and the form PUT; `HX-Refresh` when it changes; a stop/start hook (empty in this phase).
- `internal/api/router_memory.go` — treat `ForksEnabled` like `GenerationEnabled` when it forces router memory off, and in the `Blocked` text.
- `internal/api/service.go` — `routerMemory := s.cfg.RouterMemory && !s.cfg.GenerationEnabled && !s.cfg.ForksEnabled`.
- `internal/api/backup.go` — refuse to restore `router_memory` on while forks are on, like the generation check.
- `web/templates/settings.html` — the switch and its tooltip, in its own form below the generation switch.
- `internal/api/forks_settings_test.go` — new tests.

## Steps

1. Add `ForksEnabled` to `config.Config`. Comment: "ForksEnabled turns on curated llama.cpp forks (see internal/forks). Off by default; while off, nothing about them is shown or run."
2. Add `func (s *Server) forksOn() bool { return s.cfg != nil && s.cfg.ForksEnabled }` in `server.go`.
3. In `settings.go`, read and write `forks_enabled` in all three handlers, the same way as `generation_enabled`. Keep `wasForksEnabled` before the change. After the change, call `s.forksSwitched(was, now)`. In this phase it is a method with an empty body and a comment that Phase 05 fills it. Send `HX-Refresh` when the value changed, so the page reloads with or without the fork sections.
4. In `router_memory.go`, change both `cfg.GenerationEnabled` checks to `cfg.GenerationEnabled || cfg.ForksEnabled`. The `Blocked` text becomes: "Turn off generation backends and llama.cpp forks in Settings to use this. The router cannot see their VRAM reservations."
5. Change `service.go:468` as listed above, and the backup check in `backup.go:130` the same way.
6. In `settings.html`, copy the generation block at lines 215–230: a `<form hx-put="/api/settings" hx-trigger="change" hx-target="#forks-status">` with a hidden `forks_touched` input and a `<label title="…">` around a `role="switch"` checkbox named `forks_enabled`. Label: "llama.cpp forks (experimental)". Tooltip (the label's `title`, as the generation switch does): "Run curated llama.cpp forks as separate servers. Each fork server uses one whole GPU. This turns off router memory." The form PUT handler treats `forks_touched` like `generation_touched`.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `forks_settings_test.go`: a JSON PUT with `forks_enabled: true` saves it and the GET returns it; a form PUT does the same; the response has `HX-Refresh` only when the value changed.
- With `ForksEnabled` on and `RouterMemory` on in the config, `startRouterWith` passes no router memory option, and `routerMemoryInfo().Blocked` is set.
- A backup restore with `router_memory: true` is refused while forks are on.
- Manual: the switch appears in Settings, survives a restart, and turning it on reloads the page.

## Commit

`feat(forks): add the llama.cpp forks opt-in setting`

## Rollback

Revert the commit. A saved `forks_enabled: true` in the config is then an unknown key that the YAML loader ignores. Safe to leave on its own: nothing else reads the setting yet.
