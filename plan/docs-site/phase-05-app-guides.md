# Phase 05 — The rest of Help: guides, API and troubleshooting

**Depends on:** Phases 01, 02, 03, 04 · **Enables:** Phase 07 (every page that
the app will link to exists, so Help can shrink)

## Goal

Move every remaining section of `help.html` to the site, so that each section
of today's Help has a home. Text needs no diagrams here; it gains room for
detail that Help had to cut.

## Files touched

- `docs/getting-started.md` — Quickstart (`#quickstart`).
- `docs/concepts.md` — Core concepts (`#concepts`).
- `docs/guides/server.md` — Server tab (`#server`, `#engine-notes`).
- `docs/guides/builds.md` — Builds (`#builds`).
- `docs/guides/models.md` — Models (`#models`).
- `docs/guides/autoconfigure.md` — Autoconfigure and Autotune (`#profiles`).
- `docs/guides/model-config.md` — Model config (`#model-config`).
- `docs/guides/download-models.md` — Download Models (`#search-hf`,
  `#recommended`).
- `docs/guides/benchmarks.md` — Benchmarks (`#benchmarks`).
- `docs/guides/generation.md` — Image, video and music generation
  (`#generation`, the numbered setup steps, sets, custom sets, Measure steps).
- `docs/guides/forks.md` — llama.cpp forks (`#forks`).
- `docs/guides/settings.md` — Settings (`#settings`).
- `docs/api.md` — Using the API (`#api`) and the generation API
  (`#generation-api`).
- `docs/troubleshooting.md` — Troubleshooting (`#troubleshooting`).

## Steps

1. For each file above, copy the matching Help section into Markdown. Keep the
   wording, which is already in the target style. Turn the HTML tables into
   Markdown tables, `<dl>` lists into `**Term**` paragraphs, and inline SVG
   icons into the icon's name in words (for example "the load icon").
2. Give every `##` heading an explicit id (`{ #id }`): its text in lower
   case, with words joined by `-` and other characters removed (for example
   `## Custom sets { #custom-sets }`). These three are required as named,
   because Phase 07 links to them:
   - `guides/server.md`: `## Engine notes { #engine-notes }`.
   - `guides/download-models.md`: `## Recommended models { #recommended }`.
   - `api.md`: `## Image, video and music API { #generation-api }`.
   The Help anchors `#generation-vram`, `#generation-sharing` and
   `#generation-evict` do not become ids on `guides/generation.md`: that
   content lives on the Phase 04 pages, which own it.
3. Replace each in-Help link (`href="#eviction"`) with a relative site link
   to the page and anchor that the table in Phase 07 step 3 gives for that
   Help anchor. The table is fixed in this plan; Phase 05 only reads it. Write
   the path relative to the page's own folder: `../gpu/eviction.md#modes`
   from `guides/`, `gpu/eviction.md#modes` from `api.md` or
   `troubleshooting.md`. Replace links to app pages (`/settings`,
   `/models`) with the page name in words and its path in code, for example
   "Settings (`/settings`)", because the site cannot know the user's app
   address.
4. On `docs/guides/generation.md`, cut the reservation and taking-turns
   details that Phase 04 wrote; link to those pages instead.
5. On `docs/api.md`, state that the full API reference is in the app at
   `/docs/api` on the user's own install.
6. Add `Since v2.x` notes by the Phase 03 step 6 rule.
7. Check every number and default value on each page against the code, as
   in Phase 03 step 1. List differences in the pull request
   description.

## Build gate

- `make docs` passes with no warnings. No page from the content map is still
  a stub.

## Test plan

- Make a checklist of every `<h2>`, `<h3>` and `<dt id>` in `help.html`. Each
  one has a section on the site. Put the checklist in the pull request
  description.
- `rg "not written yet" docs/` finds nothing.
- Read three pages end to end in the local preview, in light and dark mode.

## Commit

`docs(site): move the remaining Help sections to the docs site`

## Rollback

Revert the commit. The pages go back to stubs. The app does not link to them
yet, so the app is not affected.
