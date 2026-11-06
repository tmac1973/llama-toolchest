# Phase 07 — Short in-app Help

**Depends on:** Phases 03, 04, 05, 06 · **Enables:** the end state: a short
Help page that links out to the site

## Goal

Make `web/templates/help.html` short. Quickstart and Core concepts stay in
full. Every other section shrinks to 2–4 sentences that say what the feature
does, plus a "More in the docs" link built with `docsURL`. Every anchor that
Help has today still exists, so links from Settings and other pages keep
working, also with no internet access.

## Files touched

- `web/templates/help.html` — shortened.
- `web/templates/layout.html` — add a "Docs" link that opens the site, next
  to the Help item in the navigation (line with `href="/help"`).
- `CLAUDE.md` — add a short "Documentation" section to the repo
  instructions: Help text in the app is short; full text goes in `docs/`;
  when a change alters what a user sees, update the matching docs page in the
  same pull request.
- `internal/api/*_test.go` — update any test that checks for text in
  `help.html` that this phase removes. Find them with
  `rg -l 'help.html|/help' internal --glob '*_test.go'`.

## Steps

1. List every `id="…"` in `help.html` before the change (save the list in the
   pull request description).
2. Keep the page header, the table of contents, `#quickstart` and
   `#concepts` as they are.
3. For each other section, write a short summary in the same style:
   2–4 sentences, 80% ASD-STE100. Keep its `<h2>`/`<h3>`/`<dt>` with the
   same `id`. End the section with
   `<p class="help-more"><a href="{{docsURL "…"}}" target="_blank" rel="noopener">More in the docs</a></p>`.
   The map from Help anchor to docs ref:

   | Help anchor | docsURL ref |
   | --- | --- |
   | `#eviction` | `gpu/eviction#modes` |
   | `#vram-margin` | `gpu/memory-margin#what` |
| `#eviction-reservations` (added by #570) | `gpu/reservations#what` |
| `#eviction-turns` (added by #570) | `gpu/taking-turns#take-turns` |
   | `#server`, `#engine-notes` | `guides/server`, `guides/server#engine-notes` |
   | `#builds` | `guides/builds` |
   | `#models` | `guides/models` |
   | `#profiles` | `guides/autoconfigure` |
   | `#model-config` | `guides/model-config` |
   | `#search-hf`, `#recommended` | `guides/download-models`, `guides/download-models#recommended` |
   | `#benchmarks` | `guides/benchmarks` |
   | `#generation` | `guides/generation` |
   | `#generation-vram` | `gpu/reservations#what` |
   | `#generation-sharing` | `gpu/taking-turns#sharing` |
   | `#generation-evict` | `gpu/taking-turns#take-turns` |
   | `#generation-api` | `api#generation-api` |
   | `#forks` | `guides/forks` |
   | `#settings` | `guides/settings` |
   | `#api` | `api` |
   | `#troubleshooting` | `troubleshooting` |

   Every anchor in this table is created by Phases 03–05. Other ids in Help
   (any `<dt id>` or `<h3 id>` not in the table) keep their element and id,
   and are covered by the "More in the docs" link of the section around
   them; they get no link of their own. The text under each of them shrinks
   to one sentence.
4. The Measured reservations table leaves Help. `#generation-vram` keeps two
   sentences (Measure records each instance's peak VRAM plus 10%; the planner
   keeps that much free) and two links: `gpu/reservations#what` and
   `reference/measured-reservations`.
5. Apply the differences between Help and the code that Phases 03–05 listed
   (in the descriptions of PRs #575 and #576), so the short text matches the
   code.
6. Add the "Docs" navigation link in `layout.html`:
   `<a href="{{docsURL ""}}" target="_blank" rel="noopener">Docs</a>`.
   Phase 06 made `docsURL ""` return the base URL.
7. Add a CSS rule for `.help-more` in the page's existing style block:
   `font-size: 0.9em; margin-top: 0.25rem;`.
8. List every `id="…"` again and compare with step 1. They must match.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test -race ./...` (includes `TestDocsLinksExist` over the new links)
- `make js-test`
- `make docs`
- `wc -l web/templates/help.html` is under 150.

## Test plan

- Run the app (`make dev`), open `/help`. Every section reads as a summary
  and has a working "More in the docs" link that opens the right heading on
  the site in a new tab.
- From Settings, click the Eviction card's Help link and the GPU memory margin
  card's Help link. They land on `/help#eviction` and `/help#vram-margin`,
  and the summary is there.
- Open `/help` with the network cut off. The page loads and all in-app anchors
  work; only the docs links fail.
- The anchor lists from steps 1 and 8 match.

## Commit

`docs(help): shorten in-app Help and link each section to the docs site`

## Rollback

Revert the commit. Help comes back in full, and the docs site stays as it is.
Phase 06's `docsURL` stays and has no uses, which is safe.
