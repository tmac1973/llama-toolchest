# Phase 04 — Turn on toolchest mode with generation backends or forks

**Depends on:** 03 (`memoryMode`, `loadLLMWithRoom`) · **Enables:** 05
(the field test)

## Goal
Let the user keep "Keep several models loaded" on while generation backends
or llama.cpp forks are on. The router is then started in toolchest mode
(`--models-max 0`, no `--models-memory-margin`) and phase 03's load path
runs. Turning generation backends or forks on no longer turns the option
off. The mode is fixed at each router start. Labels, tooltips, help text,
README and config comments say who decides in each case.

## Files touched
- `internal/api/router_memory.go` — `routerLimits` returns the mode;
  `routerMemoryInfo`, `applyRouterMemoryLocked` and
  `errRouterMemoryWithGeneration` change.
- `internal/api/service.go` — `startRouterWith` sets `s.memoryMode`.
- `internal/api/settings.go` — no refusal with generation or forks on.
- `internal/api/backup.go` — restore keeps the option with generation on.
- `internal/config/config.go` — the `RouterMemory` comment.
- `web/templates/server.html` — label, tooltip, margin line, restart note.
- `web/templates/settings.html` — the generation and forks tooltips.
- `web/templates/help.html` — the help entry.
- `README.md` — the `router_memory` comment and the forks section (line
  294).
- `internal/api/router_memory_test.go`, `internal/api/gen_settings_test.go`,
  `internal/api/forks_settings_test.go`, `internal/api/settings_handlers_test.go`,
  `internal/api/settings_page_render_test.go` — updated and new tests; the
  backup restore test goes in `internal/api/router_memory_test.go`.

## Steps
1. **`routerLimits`.** Change it to
   `routerLimits(modelsMax int, routerMemory, holdsVRAMOutside, job, buildHasOption bool, marginMiB int) (max, margin int, mode memoryMode)`:
   - not `routerMemory` → `modelsMax, 0, memoryCount`
   - `job` → `1, 0, memoryCount` (benchmarks and Autotune, as today)
   - `holdsVRAMOutside` → `0, 0, memoryToolchest` (toolchest enforces Max
     Loaded itself, phase 03 step 6.1). Works on any build.
   - not `buildHasOption` → `modelsMax, 0, memoryCount`
   - otherwise → `modelsMax, marginMiB, memoryRouter`
2. **`startRouterWith`.** Drop the `&& !s.cfg.HoldsVRAMOutsideRouter()`
   from `routerMemory`, read `holdsVRAMOutside := s.cfg.HoldsVRAMOutsideRouter()`
   under the same lock, pass both to `routerLimits`, and store the mode in
   `s.memoryMode` right after `s.process.Start` succeeds. On a stop, set it
   to `memoryCount`. Log the mode in the "starting router" line
   (`"memory", "count"|"router"|"toolchest"`).
3. **Settings.** In `settings.go`, remove the refusal at line 141 and the
   "Settings not saved" message at line 198 that use
   `errRouterMemoryWithGeneration`; remove the error var. In
   `applyRouterMemoryLocked`, keep the 1 → 0 change for Max Loaded, and
   remove the block that turns the option off when
   `HoldsVRAMOutsideRouter()`. In `backup.go`, remove the
   `!(*in.RouterMemory && s.cfg.HoldsVRAMOutsideRouter())` condition.
4. **`routerMemoryInfo`.** Replace `Blocked` with:
   - `Toolchest bool` — `cfg.HoldsVRAMOutsideRouter()`.
   - `Restart bool` — the router is running and the mode it was started in
     (`s.memoryMode`) differs from the mode a start now would get.
   - `Margin` — only when not `Toolchest`.
   - `Note` (build older than b11930) — only when not `Toolchest`, because
     toolchest mode does not need the router option.
5. **Server page** (`server.html`, around line 241):
   - Label: "Keep several models loaded".
   - Tooltip when `Toolchest`: "Keeps several models loaded while they fit
     in VRAM. Toolchest decides, because generation backends or forks are
     on. When a model does not fit, it unloads the idle models used least
     recently. Applies at the next server start."
   - Tooltip otherwise: "Keeps several models loaded while they fit in
     VRAM. The llama-server router decides. When a model does not fit, it
     unloads the idle models used least recently. Applies at the next
     server start." plus `Note` when set.
   - Under the switch, when on and `Toolchest`: `<small>Toolchest decides,
     with the fit margin of each GPU.</small>`; when on and not
     `Toolchest`: the margin line as today.
   - When `Restart`: `<small>Restart the server to apply.</small>`.
   - The checkbox is no longer `disabled`.
6. **Settings page** (`settings.html`, line 222 and the forks switch):
   remove "Turning this on turns off 'Let the router keep several models
   loaded'". New tooltip on both: "With 'Keep several models loaded' on,
   toolchest decides what to unload. Applies at the next server start."
7. **Help** (`help.html`, line 112): rename the entry to "Keep several
   models loaded". Text, short and about 80% ASD-STE100: what it does; that
   the router decides with generation backends and forks off (needs
   b11930), and toolchest decides with either on (any build); that
   toolchest unloads idle LLMs first and stops backends only when that is
   not enough (GenEvict on); that a model with a request in progress is
   never unloaded; that Max Loaded still sets an upper limit; the existing
   note about a small model unloading a big one; benchmarks and Autotune
   load one model at a time.
8. **README and config.** In `README.md`, change the `router_memory`
   comment to: "Keep several models loaded and unload the least recently
   used idle ones by GPU memory: the router decides (llama.cpp b11930+),
   or toolchest with generation backends or forks on". In the forks section
   (line 294), remove "This also turns off 'let the router keep several
   models loaded'". Update the `RouterMemory` comment in `config.go` to
   describe both modes and say one decides, never both.

## Build gate
```bash
go build ./... && go vet ./... && go test -race ./... && make js-test
```

## Test plan
- `TestRouterLimits`: one case per row of step 1, including toolchest mode
  on a build without the option.
- `startRouterWith` with generation on and the option on: process gets
  `ModelsMax 0`, `ModelsMemoryMarginMiB 0`; `s.memoryMode` is toolchest.
  With generation off: unchanged from today.
- Settings: turning the option on with generation on is saved (no 400);
  turning generation on keeps the option on.
- Backup restore with generation on keeps `router_memory: true`.
- Server page render: label text; toolchest tooltip with generation on;
  router tooltip and margin line with it off; "Restart the server to apply"
  when the running mode differs.
- Update `TestRouterMemorySetting` and `TestRouterMemoryServerPage` for the
  removed refusal.

## Commit
`feat(server): keep several models loaded with generation backends on`

## Rollback
Revert the commit. The router then starts in count mode again with
generation on, and phase 03's path stops running. A config saved with
`router_memory: true` and generation on is harmless to the old code: the
old `applyRouterMemoryLocked` turns it off on the next settings save.
