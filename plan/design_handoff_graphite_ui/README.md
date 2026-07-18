# Handoff: Llama Toolchest UI refresh ("Graphite")

## Overview
This is a visual restyle of the Llama Toolchest web UI. The goal was to tighten up spacing,
hierarchy and density and move the look away from stock Pico CSS — while keeping the exact same
structure, information architecture, and server-rendered (Go `html/template` + htmx + Pico) setup.
**No structural, behavioral, or backend changes.** Same routes, same htmx wiring, same partials.

It also fixes one specific usability complaint: on the **Models** tab it was hard to tell where one
model's expanded config panel ended and the next model began. The new design makes the active
(expanded) card unmistakable — see "The model-config fix" below.

## About the design files
The files in this bundle are **design references created in HTML** (Design Component prototypes) —
they show the intended look and behavior, they are **not** production code to copy verbatim.

**The target codebase already exists** and this is NOT a greenfield build. The task is to apply this
visual design to the real repository:

> **tmac1973/llama-toolchest** — a single Go binary that serves server-rendered HTML with
> **htmx** + **Pico CSS**, no JS build step. UI lives in `web/templates/` (+ `web/templates/partials/`)
> and `web/static/`.

Almost all of the restyle can be achieved by:
1. Overriding Pico's CSS custom properties + adding a small amount of scoped CSS in the `<style>`
   block of **`web/templates/layout.html`** (this is where the app already keeps its custom CSS and
   its alternate themes), and
2. A handful of small, surgical edits to individual templates/partials for the few places that need
   markup-level changes (chiefly the Models config panel).

Keep everything server-rendered. Do not introduce a bundler, React, or a CSS framework swap. Pico stays.

## Fidelity
**High-fidelity.** Colors, typography, spacing, and radii below are final and exact. Recreate them
faithfully. Where a value isn't listed, keep the app's current behavior/value.

---

## Design tokens (the core of the change)

The current app defines its palette through Pico's variables (default dark theme) plus three custom
themes in `layout.html` (`terminal-green`, `terminal-amber`, `cyberpunk`). "Graphite" should become
the **new default dark appearance**. Implement it by overriding Pico's variables on
`:root` / `[data-theme="dark"]` (or add it as a `data-custom-theme="graphite"` alongside the existing
custom themes and make it the default — your call; making it the default dark theme is recommended).

### Color tokens

| Purpose | Value | Maps to Pico var |
|---|---|---|
| App background | `#181a1e` | `--pico-background-color` |
| Sidebar background | `#141619` | (custom, see nav CSS) |
| Card / panel background | `#1e2126` | `--pico-card-background-color` |
| Elevated control background (inputs, secondary btn) | `#23272d` | `--pico-form-element-background-color` |
| Inset / recessed background (config panel, log, progress track) | `#141619` | `--pico-card-sectioning-background-color` |
| Border (default) | `#2b2f36` | `--pico-muted-border-color`, `--pico-card-border-color`, `--pico-form-element-border-color` |
| Border (subtle, inner dividers) | `#23262c` | (custom) |
| Body text | `#c3c7cf` | `--pico-color` |
| Muted text / labels | `#828892` | `--pico-muted-color` |
| Headings / emphasized text | `#e9ebef` | `--pico-h1..h6-color` |
| **Accent** (primary, active nav, links, switches, bar fill) | `#9aa6d4` | `--pico-primary`, `--pico-primary-background` |
| Accent soft (active-nav bg, pill bg, accent-tinted surfaces) | `#262a3a` | (custom) |
| Accent hover (link/nav hover text) | `#c0c9ea` | `--pico-primary-hover` |
| Warning text ("restart required") | `#cba76a` | (custom) |
| Success (running, "Fits", success badge) | `#7fb490` | `--pico-ins-color` |
| Danger (failed, delete, orphan/incomplete) | `#d98a8a` | `--pico-del-color` |
| `kbd` background | `#262a31` | `--pico-code-kbd-background-color` |
| `kbd` text | `#aeb6c6` | `--pico-code-kbd-color` |
| Second GPU-segment color (allocation map) | `#c58fb8` | (custom, per-model palette) |

Notes:
- The accent is a **muted periwinkle** (`#9aa6d4`), used sparingly. On accent-filled buttons the text
  is near-black `#14161c` for contrast. Keep saturation low — do not brighten it.
- Status/badge fills use translucent tints of their color: e.g. running = `rgba(127,180,144,.16)` bg +
  `#7fb490` text; failed = `rgba(217,138,138,.16)` bg + `#d98a8a` text; "running" job =
  `rgba(154,166,212,.16)` bg + `#9aa6d4` text.

### Typography
- **Sans (UI):** `'IBM Plex Sans', system-ui, sans-serif` — load from Google Fonts (weights 400/500/600/700).
  This replaces Pico's default system stack. Add the `<link>` in `layout.html <head>`.
- **Mono (code, model names, metrics, kbd, log, SHAs):** `'IBM Plex Mono', ui-monospace, monospace`
  (weights 400/500/600). Applied to: `code`, `kbd`, `pre`, model public names, VRAM/size numbers,
  benchmark metrics, git SHAs, the API endpoint.
- Base font-size 14px. Page title (`h1`) 20px/600/`-0.01em`. Card section header ("card-h") 11.5px/600,
  uppercase, `letter-spacing:.06em`, muted. Table header 10.5px/600 uppercase `letter-spacing:.05em` muted.
  Sub-labels 12.5px muted.

### Shape & spacing
- Frame/border radius scale: **outer cards `12px`**, inner nested config card `9px`, inputs/selects `8px`,
  small buttons `8px`, normal buttons `9px`, pills `20px`, `kbd` `4px`, progress bars `4–6px`.
- Sidebar width **224px** (currently 220px). Sidebar padding 16px.
- Main content padding **22px 26px**.
- Card padding **15px 17px**.
- Standard gap between stacked cards **14px**; between model cards **12px**.
- Inputs: padding `8px 10px`, 13px text. Buttons: `7px 14px` (normal), `5px 11px` (small).
- Progress bars: height 7px, track = inset color, fill = accent.

### Switches (the On toggle, Flash Attention, etc.)
32×18px pill, radius 9px. On = accent bg with a white 14px knob at right (2px inset). Off = border-color
bg with a muted-color knob at left. (Pico's native `role="switch"` styled via
`--pico-switch-*` vars will do — set `--pico-switch-checked-background-color: #9aa6d4`.)

---

## Screens / views

All screens keep their existing routes, partials, and htmx behavior. The shell (sidebar + main) is
`web/templates/layout.html`. Reference the prototype `Llama Toolchest — Graphite.dc.html` — its
sidebar switches tabs client-side only for demo purposes; in the real app each nav item is a normal
page link as it is today.

### Shell (`layout.html`)
- **Sidebar** (`nav.sidebar`): 224px, bg `#141619`, right border `#2b2f36`. Brand block gets a 28px
  rounded-square badge (radius 9px, bg `#262a3a`, 1px accent border, accent "L" glyph) to the left of
  the "Llama Toolchest" title. Nav links: 8px 11px padding, radius 9px. Hover: bg `#1e2126`. Active
  (`aria-current="page"`): bg `#262a3a`, text `#c0c9ea`, weight 600 (currently it's a solid accent
  block — soften it to this tinted style).
- **Monitor bar** (`partials/monitor_bar.html`): keep as is but ensure progress tracks use the inset
  color and fills use the accent. GPU/CPU/RAM label rows 11.5px muted, values right-aligned.
- Main: padding 22px 26px, `overflow-y:auto`, `max-height:100vh` (unchanged).

### Server (`server.html` + dashboard handler in `internal/api/server.go`)
- Page title "Server" (20px) + sub "Router running · N model loaded · uptime …".
- Top row: 3 cards (Server controls / Available Models / Inventory) in a 3-col grid, then a full-width
  **API Endpoint** card, then **Live Performance** table card, then **Server Logs** card.
- Server card: status pill (running = success tint), Build `<select>`, Max Loaded `<input>`, then
  Start (primary) / Stop (ghost) / Restart (default) buttons.
- Available Models: each row = play glyph (accent) + mono public name + optional `loaded`/`loading`
  pill. NOTE: this markup is generated in Go (`handleDashboard` in `internal/api/server.go`), not a
  template — update the inline HTML strings there to match (pill = accent-soft bg, mono name).
- Logs pane: bg `#101215`, 1px border, radius 8px, mono 11.5px, `white-space:pre-wrap`.

### Builds (`builds.html`, `partials/build_card.html`, `partials/build_log.html`)
- New Build card: Profile + Git Ref selects on a 2-col grid; Git Ref row keeps the inline "Release
  Notes" link + "Refresh Tags" button. Build Tag input. Build options render as toggle switches / the
  existing `#build-options` content. Start Build = primary button.
- Compiled Builds: the existing table. `build_card.html` rows — wrap the ID in `kbd`, the SHA in mono
  `code` (muted), Status as a pill (`success` = success tint, `failed` = danger tint), date muted,
  actions as small ghost buttons right-aligned.

### Models (`models.html`, `partials/model_card.html`, `partials/model_config.html`)
This is the most important screen. See "The model-config fix" below for the key change.
- Header: "Models" title + sub-count, "Scan for Models" button top-right.
- GPU Allocation Map (`partials/gpu_map.html`): card with per-GPU labeled segmented bars (10px tall,
  radius 6px, inset track, 2px gaps between segments), plus a legend row of colored chips. Keep the
  per-model color assignment; first accent `#9aa6d4`, second `#c58fb8`, etc.
- Column header row + model cards use the existing grid template
  `2.5rem minmax(0,1fr) 5rem 5rem 5rem` + an actions column (~11rem). Keep it.
- Card row: On switch, model name (headings color, 500) + capability pills (`tools`/`vision` = accent
  pill; `missing`/`incomplete` keep danger styling), quant `kbd`, VRAM (mono), size (mono muted),
  Configure + Remove buttons.
- Disabled/orphan cards: `opacity:.7`. Incomplete cards keep a danger left-border accent.
- Embedding Models section below a divider, same card style.

### Search HF (`models_browse.html`, `partials/hf_results.html`, `partials/hf_files.html`)
- Search input (flex:1) + primary Search button in a row.
- Each result = card: repo name (headings, 600) + HF ↗ link + license note; downloads/likes line
  (12px muted); Files button top-right. Expanded files = table (File mono / Quant kbd / Size / VRAM Est.
  / Fit / action). "Fits" = success text, "Too Large" = danger; Download = primary small button,
  "Downloaded" = success text + View link.

### Benchmarks (`benchmarks.html`, `partials/job_list.html`, `partials/job_detail.html`, `partials/benchmark_form.html`, `partials/job_form.html`)
- "Benchmark Jobs" card with `?` (ghost) and `+ New Batch Job` buttons.
- Job rows = collapsible (`details`): caret + name (headings) + status pill + "N/M cells" progress +
  created/finished meta (muted, right). Expanded shows the detail table (checkbox / model·build·preset
  mono / status pill / PP t/s / TG t/s / TTFT), plus Compare / Export CSV buttons.
- Status pill colors: completed/done = success tint, running = accent tint, failed = danger tint,
  pending = outline. (The existing `.status-*` classes in `benchmarks.html` — re-point their colors to
  these tokens.)
- Quick Benchmark card: Model select + Preset select + Run (primary), same as today.

### Settings (`settings.html`)
- Stack of cards (max-width ~720px): Startup (switch), Server URL, OpenAI-Compatible Proxy (endpoint in
  a code/log block + Test Connection), API Key, HF Token, Theme, Storage.
- **Theme card:** add a "Graphite" button as the first/active option alongside the existing Dark, Light,
  Terminal Green, Terminal Amber, Cyberpunk buttons (these call `setTheme(...)` — wire "Graphite"
  accordingly, and make it the default in the `layout.html` init script).

### Help (`help.html`)
- Keep all content. Restyle only: TOC becomes a chip row (card bg, 10px radius); `h2` section headers
  get a top border divider; callouts use the accent left-border + accent-soft bg. Code inline = kbd
  colors, mono.

---

## The model-config fix (priority)

Problem today (`partials/model_card.html` + `model_config.html`): the Configure form is injected into
`.model-card-config` with only a thin top border and the page background, no owner label — so stacked
open configs blur together.

Required change (implemented in the Graphite prototype's Models tab — copy it):
1. **When a card is expanded, make the card itself stand out:** set its border to the accent color
   (`#9aa6d4`). (Add a class like `.model-card.is-open` toggled when the config is populated, or style
   `.model-card:has(.model-card-config:not(:empty))`.)
2. **Recess + nest the config:** `.model-card-config` gets the inset background (`#141619`) and a small
   (6px) padding, and the form sits inside a **nested card** (bg `#1e2126`, 1px `#262a3a` border,
   radius 9px, padding 14px 16px). The nested surface visually "belongs to" the row above it.
3. **Add an owner header inside the panel:** a header line reading
   `Configure — <model id>` (13px, heading color; the "— model id" part muted) with a right-aligned
   `Restart required` note in the warning color (`#cba76a`), separated by a 1px bottom divider
   (`#23262c`). This is the single most important addition — it names which model the config belongs to.
4. Config fields keep the existing 2-column `.grid` layout; inputs use the elevated bg `#23272d`.

The prototype shows the exact spacing, and the (superseded) baseline `1a` in
`Llama Toolchest UI (directions 1a-1d).dc.html` shows the "before" for comparison.

---

## Interactions & behavior
No behavioral changes. All existing htmx attributes, SSE log streams, auto-save on config change,
filter boxes, collapse/expand, modals, and the theme switcher stay exactly as they are. This is a
**CSS/markup-styling pass only**. Hover states: nav link hover bg `#1e2126`; button hover
`border-color:#9aa6d4`; link hover text `#c0c9ea`.

## State management
None added. Prototype-only `tab` state exists purely to demo the tabs in a single HTML file; ignore it —
the real app already routes per page.

## Assets
- Google Fonts: **IBM Plex Sans** + **IBM Plex Mono** (add `<link>` to `layout.html`; or self-host into
  `web/static/` to keep the no-external-deps property — recommended for an offline/LAN tool).
- No image assets. The brand "L" badge is a CSS box, not an image. Play/copy/check icons in the Server
  dashboard are the existing inline SVGs in `internal/api/server.go` — keep them, they inherit
  `currentColor`.

## Files
Design references in this bundle:
- `Llama Toolchest — Graphite.dc.html` — the full target design, all 7 tabs (open in a browser).
- `Llama Toolchest UI (directions 1a-1d).dc.html` — the Models screen: `1a` current/baseline vs the
  three explored directions; `1d` was chosen and is the basis for the Graphite app.

Real repo files to edit (grounded in the current source):
- `web/templates/layout.html` — **primary target**: font `<link>`, Pico variable overrides, sidebar/nav
  CSS, the shared `.model-card*` CSS, default-theme init. Most of the restyle lives here.
- `web/templates/server.html`, `builds.html`, `models.html`, `models_browse.html`, `benchmarks.html`,
  `settings.html`, `help.html` — small per-page CSS/markup tweaks as described.
- `web/templates/partials/model_card.html`, `model_config.html` — the config-fix markup.
- `web/templates/partials/build_card.html`, `gpu_map.html`, `hf_results.html`, `hf_files.html`,
  `job_list.html`, `job_detail.html`, `service_status.html`, `monitor_bar.html`, `timings_summary.html`,
  `embedding_presets.html` — pill/kbd/mono/table styling touch-ups.
- `internal/api/server.go` — `handleDashboard` emits the Server tab's Available Models / Inventory /
  API Endpoint cards as Go string literals; update those inline HTML strings to match.

## Suggested order of work
1. `layout.html`: fonts + Pico var overrides + sidebar/nav + switch vars. (≈70% of the visual change,
   applies everywhere automatically.)
2. Models config fix (`model_card.html` + `model_config.html` + the `.model-card` CSS in `layout.html`).
3. Table/pill/kbd touch-ups across partials.
4. `handleDashboard` inline HTML in `server.go`.
5. Add the "Graphite" theme button in `settings.html` + make it default.
