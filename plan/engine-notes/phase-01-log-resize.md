# Phase 01 — Resizable Server log

**Depends on:** nothing · **Enables:** a taller log on busy Server pages; the strip in Phase 06 sits above a log whose height the user controls

## Goal

Let the user make the Server page log taller or shorter by dragging a grip on its bottom edge. The browser remembers the height. A double-click on the grip clears it. With no saved height the log sizes exactly as today (it fills the space left in `.server-page`, with a 300 px minimum). This phase has no link to the rest of the feature and ships on its own.

## Files touched

- `web/templates/server.html` — add the grip element after `<pre id="service-logs">` inside `.log-panel`, and its CSS; a `.log-section.sized` rule for a saved height.
- `web/static/log-resize.js` — new: drag, save, restore and reset logic.
- `web/templates/layout.html` — add `<script src="/static/log-resize.js" defer></script>` next to `log-panel.js`.
- `web/jstest/log_resize_test.js` — new: tests for the height logic.
- `internal/api/js_log_resize_test.go` — new: `TestLogResizeJS`, built like `js_live_stream_test.go`.
- `Makefile` — add `TestLogResizeJS` to the `js-test` run list.

## Steps

1. In `server.html`, add `<div class="log-grip" role="separator" aria-orientation="horizontal" title="Drag to change the log height. Double-click for the default size."></div>` as the last child of `.log-panel`.
2. CSS: `.log-grip` is 8 px tall, full width, `cursor: ns-resize`, with a centred 40 × 3 px bar in `var(--pico-muted-border-color)`; `touch-action: none`. Add `.server-page .log-section.sized { flex: 0 0 auto; }` so a saved height replaces the flex fill. The existing `min-height: 300px` stays and also bounds dragging.
3. `log-resize.js` exposes `window.LogResize = { clampHeight, init }`:
   - `clampHeight(px, viewportPx)` returns `px` clamped to `[300, viewportPx - 120]` and rounded to a whole pixel. Pure, so it is testable.
   - `init(section, grip, storage)`: on load read `localStorage["llama-toolchest.serverLogHeight"]` inside try/catch. When it holds a number, set `section.style.height` to the clamped value and add the `sized` class.
   - Drag uses pointer events: `pointerdown` on the grip calls `setPointerCapture`, records the start Y and the section's current `offsetHeight`; `pointermove` sets the height to `start + dy`, clamped, and adds `sized`; `pointerup` saves the height (try/catch).
   - `dblclick` on the grip removes the saved key, clears `section.style.height` and removes `sized`, so the default flex sizing returns.
   - On window `resize`, re-clamp a saved height so the log never goes taller than the viewport allows.
4. Call `LogResize.init(document.querySelector('.server-page .log-section'), document.querySelector('.log-grip'), window.localStorage)` on DOMContentLoaded, only when both elements exist.
5. Live tail keeps working: `log-panel.js` scrolls the `<pre>`, whose `flex: 1` fills the resized article, so no change there.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `web/jstest/log_resize_test.js`: `clampHeight` keeps 300 as the minimum, caps at viewport − 120, rounds; `init` with a stored "520" sets the height and the `sized` class; with nothing stored, or a storage that throws, it leaves the section untouched; a double-click clears the key and the class (use `web/jstest/dom.js` for the fake DOM).
- Manual, on a Server page with three generation instances: the log opens at today's size; drag it to about 700 px; reload and the height is kept; double-click the grip and the default size returns; Live tail still follows new lines at both sizes; the page still scrolls to the generation cards.

## Commit

`feat(ui): let the Server log be resized by dragging its bottom edge`

## Rollback

Revert the commit. Only the template, one new script and its test change. A height already saved in a browser is then ignored, since nothing reads the key.
