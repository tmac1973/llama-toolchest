# Documentation site — Project Overview

## Problem

The in-app Help page (`web/templates/help.html`) has grown to about 500 lines.
It explains every tab, and it also tries to explain how toolchest shares GPU
memory: eviction modes, generation reservations, the GPU memory margin, and how
LLMs and generation backends take turns on a GPU. Those mechanics are hard to
explain in text alone. PR #570 rewrote the eviction and margin sections, and
they are still dense tables and bullet lists. A reader needs diagrams to see who
unloads what, and why.

The app has no place for long, illustrated documentation. Help must stay a
page inside the app, so it must stay short. The two other docs files
(`docs/manual-install.md`, `docs/secure.md`) are only reachable from GitHub.

## Goals

- Publish a documentation site at `https://tmac1973.github.io/llama-toolchest/`,
  built with Zensical from Markdown in `docs/`.
- Explain the four GPU-sharing topics with diagrams: eviction modes, generation
  reservations, the GPU memory margin, and how LLMs and generation backends take
  turns on a GPU.
- Move all of Help's long content to the site: every tab, the measured
  reservations table, the API overview and troubleshooting.
- Make the in-app Help page short. Keep Quickstart and Core concepts. Cut every
  other section to 2–4 sentences with a "More in the docs" link.
- Keep every Help anchor that exists today (`#eviction`, `#vram-margin`,
  `#generation-evict` and the others), so links from Settings and other pages
  keep working, also offline.
- Build and deploy the site with GitHub Actions on every merge to `main`.
  Check every docs pull request with a strict build that fails on broken links.
- Make app-to-site links safe: one base URL in Go, and a Go test that fails when
  a linked page or anchor does not exist in `docs/`.

## Non-goals

- No contributor or developer documentation (architecture, building from
  source, adding a backend). Developer notes stay in the repo (`plan/`,
  `audits/`, code comments).
- No public API reference on the site. The full reference stays in the app at
  `/docs/api`. The site has a short API overview that links to it.
- No versioned site and no version picker. The site shows `main`.
- No screenshots of the app.
- No custom domain.
- No move of the README's install and setup text. README stays as it is, except
  for a link to the site. (Phase 07 adds a short docs rule to `CLAUDE.md`, the
  repo instructions, not to the README.)
- No change to how the app works. The app changes are limited to Help text,
  one template function with its tests and test fixtures, updates to existing
  tests that check Help text, a "Docs" link in the navigation, and one CSS
  rule for the "More in the docs" links.

## Users & primary flow

The reader is a person who installs toolchest and runs models on their own
GPUs. Many readers have English as a second language.

1. The user is on a Settings card (for example Eviction) and wants to know what
   a choice does.
2. They click the card's Help link and land on a short Help section in the app,
   for example `/help#eviction`. It says in a few sentences what the setting
   does.
3. For more, they click "More in the docs". The site opens on the matching page
   and heading, for example `…/gpu/eviction/#modes`.
4. The page explains the mechanism with a diagram, then the details, then what
   to do when something goes wrong.
5. The site's navigation and search let them find related pages, such as the
   GPU memory margin or taking turns.

## Constraints

- Generator: Zensical, installed with pip and pinned to one version in
  `docs/requirements.txt`. Config in `zensical.toml` at the repo root. Source
  folder `docs/` (Zensical's default), output folder `site/` (git-ignored).
- Diagrams: Mermaid code blocks in the Markdown for flows, decisions and
  sequences (Zensical runs Mermaid and follows light and dark mode). SVG files in
  `docs/assets/diagrams/` only for VRAM layouts (stacked bars of one GPU's
  memory). Both are plain text in git. SVGs use `currentColor` and CSS variables
  so they work in light and dark mode.
- Existing files keep their paths: `docs/manual-install.md` and
  `docs/secure.md` become site pages at the same paths, because `README.md`,
  `setup.sh`, `Caddyfile.template` and `docker-compose.secure.yml` link to them.
- CI: `.github/workflows/ci.yml` already ignores `docs/**`, `**.md` and
  `plan/**`. It stays as it is. A new `.github/workflows/docs.yml` builds the
  site, and (from Phase 06) also runs the Go test that checks app links into
  `docs/`, because a docs-only pull request does not run `ci.yml`.
- GitHub Pages must be set to "GitHub Actions" as its source in the repo
  settings. Only the user does this; it is a manual step in Phase 02.
- Writing style: about 80% ASD-STE100, as in Help and tooltips. One idea per
  sentence, short sentences, active voice, one word for one meaning. Product,
  code and domain terms stay as they are. No slang.
- Content must match the code on `main`. Phase 03 starts only after PR #570
  is merged, and the GPU-sharing pages start from the Help text it leaves, and check each claim against the code
  (`internal/api/gen_evict.go`, `internal/api/llm_room.go`,
  `internal/api/gpu_map.go`, `internal/models/fit.go`, `internal/config/config.go`).
- Repo rules: Conventional Commits with no AI attribution; every change on a
  feature branch; the user decides what merges to `main`.

## Success criteria

- `zensical build --strict` passes locally and in `docs.yml` with no warnings.
- A push to `main` that touches `docs/**`, `zensical.toml` or (from Phase 06)
  `web/templates/**` deploys the site,
  and `https://tmac1973.github.io/llama-toolchest/` shows it.
- A pull request with a broken link or anchor in `docs/` fails the docs check.
- The four GPU-sharing pages each have at least one diagram that shows the
  mechanism, and they render in light and dark mode.
- Every section of today's `help.html` has a home on the site.
- `help.html` is under 150 lines. Every anchor it has today still exists.
- Every "More in the docs" link in the app is built with the `docsURL` template
  function, and `go test ./...` fails if its page or anchor is missing.
- `go build ./...`, `go vet ./...`, `go test -race ./...` and `make js-test`
  still pass.

## Decisions

- **Audience** → users of the app only. No contributor docs.
- **Generator** → the Material for MkDocs family; then **build tool** →
  Zensical (the newer tool from the same team), not the maintenance-mode
  mkdocs-material.
- **Diagrams** → Mermaid in the Markdown; hand-made SVG files only for VRAM
  layouts that Mermaid cannot draw well.
- **Source folder** → `docs/`, with `manual-install.md` and `secure.md` moved
  into the site at their current paths.
- **In-app Help** → a short page with the same anchors: Quickstart and Core
  concepts stay; other sections shrink to 2–4 sentences with a "More in the
  docs" link.
- **Versions** → one site, built from `main`. Pages mark new features with
  "Since v2.x".
- **PR checks** → a new `docs.yml`: strict build on pull requests that touch the
  docs, build and deploy on push to `main`. `ci.yml` is not changed.
- **Site address** → `tmac1973.github.io/llama-toolchest`, no custom domain.
- **App links** → a `docsURL` template function with one base URL, and a Go test
  that checks every linked page and anchor against `docs/`.
- **First scope** → all of Help moves to the site in this plan, the four
  GPU-sharing topics first.
- **Writing style** → the same as Help: about 80% ASD-STE100.
- **Screenshots** → none for now.
