# Phase 01 — Site skeleton

**Depends on:** nothing · **Enables:** Phase 02 (the workflow builds this
config), Phases 03–05 (pages go into this structure)

## Goal

Set up a Zensical site in `docs/` that builds locally with `--strict` and no
warnings. It has a home page, the full navigation of the content map with stub
pages, the two existing docs files as pages, Mermaid on, and one test diagram
of each kind (Mermaid and SVG) to prove that both render in light and dark
mode. Nothing is published yet.

## Files touched

- `zensical.toml` — new. Site config at the repo root.
- `docs/requirements.txt` — new. One line: `zensical==0.1.0`.
- `docs/index.md` — new. Home page: what toolchest is (3–4 sentences, from the
  README intro), and links to `getting-started.md`, `gpu/index.md`,
  `manual-install.md` and `secure.md`.
- `docs/manual-install.md`, `docs/secure.md` — kept at the same paths. Add a
  front-matter `title`. Fix relative links that point outside `docs/` (for
  example to `../README.md`) to full GitHub URLs, so the strict build passes.
- Stub pages for every page in the content map below. Each stub has its title
  and one sentence: "This page is not written yet." Phases 03–05 fill them.
- `docs/assets/diagrams/_test-bar.svg`, `docs/assets/stylesheets/diagrams.css`
  — new. The SVG test bar and the CSS variables the SVGs use.
- `Makefile` — add `docs` (strict build) and `docs-serve` (local preview)
  targets.
- `.gitignore` — add `/site/` and `/.venv-docs/`.
- `README.md` — add one line near the top: "Documentation:
  https://tmac1973.github.io/llama-toolchest/".

## Content map

The navigation in `zensical.toml`, in this order. Paths are under `docs/`.

- **Home** — `index.md`
- **Getting started** — `getting-started.md` (Quickstart, in full)
- **Concepts** — `concepts.md` (enabled, loaded, max loaded models, public
  names and aliases)
- **GPU memory**
  - `gpu/index.md` — How toolchest shares a GPU (overview of all parts)
  - `gpu/eviction.md` — Eviction modes
  - `gpu/memory-margin.md` — GPU memory margin
  - `gpu/reservations.md` — Generation reservations
  - `gpu/taking-turns.md` — LLMs and generation backends on one GPU
- **Using the app**
  - `guides/server.md` — Server tab
  - `guides/builds.md` — Builds
  - `guides/models.md` — Models
  - `guides/autoconfigure.md` — Autoconfigure and Autotune
  - `guides/model-config.md` — Model config (Configure)
  - `guides/download-models.md` — Download Models and recommended models
  - `guides/benchmarks.md` — Benchmarks
  - `guides/generation.md` — Image, video and music generation (setup, sets,
    custom sets, Measure)
  - `guides/forks.md` — llama.cpp forks (experimental)
  - `guides/settings.md` — Settings
- **API** — `api.md` (using the API, the generation API overview, link to the
  in-app `/docs/api` reference)
- **Install**
  - `manual-install.md` — Manual install
  - `secure.md` — Secure install
- **Reference**
  - `reference/measured-reservations.md` — Measured reservations table
- **Troubleshooting** — `troubleshooting.md`

## Steps

1. Create a virtual environment for local work: `python3 -m venv .venv-docs`
   and `.venv-docs/bin/pip install -r docs/requirements.txt`.
2. Write `zensical.toml`:
   - `[project]`: `site_name = "llama-toolchest"`,
     `site_url = "https://tmac1973.github.io/llama-toolchest/"`,
     `repo_url = "https://github.com/tmac1973/llama-toolchest"`,
     `edit_uri = "edit/main/docs/"`, `docs_dir = "docs"`,
     `site_dir = "site"`, `strict = true`, and the `nav` from the content map.
   - `[project.validation]`: `invalid_links = true`,
     `invalid_link_anchors = true`.
   - Theme: the default Zensical theme with a light and a dark palette and a
     toggle; features `navigation.sections`, `navigation.top`, `search.highlight`,
     `content.action.edit`.
   - Markdown extensions: `attr_list` (for explicit heading ids such as
     `## Modes { #modes }`), `admonition`, `pymdownx.details`, `tables`,
     `toc` with `permalink = true`, `pymdownx.superfences` with the
     `mermaid` custom fence as the Zensical diagrams page shows, and
     `pymdownx.snippets` with `base_path = ["docs"]` (for the SVG includes
     in step 4).
   - `extra_css = ["assets/stylesheets/diagrams.css"]`.
3. Write `docs/assets/stylesheets/diagrams.css`. Define these variables on
   `[data-md-color-scheme="default"]` and `[data-md-color-scheme="slate"]`:
   `--tc-llm` (model weights), `--tc-llm-kv` (KV cache, a lighter shade of
   the same hue), `--tc-llm-buf` (buffers, the lightest shade of that hue),
   `--tc-gen` (generation reservation), `--tc-margin`, `--tc-working`
   (1 GiB working memory), `--tc-other` (desktop and other programs),
   `--tc-free`, and `--tc-text`. Pick colors that have at least 3:1 contrast
   with the page background in both schemes. A bar that shows an LLM as one
   part uses `--tc-llm`.
4. SVG rule for every diagram in this plan: draw with `<rect>` and `<text>`
   only; use `fill="var(--tc-…)"` for parts and `fill="currentColor"` for
   text; include `<title>` and a `viewBox`; no fixed width. Embed SVGs inline
   in the page with Zensical's file include, not with `<img>`, because CSS
   variables do not reach an SVG loaded by `<img>`. The include line is
   `--8<-- "assets/diagrams/name.svg"` (the snippets extension from step 2).
   If the Phase 01 test shows that Zensical does not run `pymdownx.snippets`,
   paste the SVG markup into the page instead, and keep the `.svg` file as the
   source to copy from; record the result in the pull request. Phases 03 and
   04 use whichever method this test proved.
5. Write `docs/assets/diagrams/_test-bar.svg`: one 24 GiB bar with one part
   for each fill variable from step 3. Put it, and a small Mermaid flowchart, on `docs/index.md` inside an
   admonition titled "Diagram test". Phase 03 removes this admonition and the
   test SVG.
6. Write `docs/index.md` and every stub page from the content map.
7. Add the front-matter titles to `docs/manual-install.md` and
   `docs/secure.md`, and fix any link that leaves `docs/`.
8. Add Makefile targets:
   - `docs:` runs `.venv-docs/bin/zensical build --clean --strict`.
   - `docs-serve:` runs `.venv-docs/bin/zensical serve`.
   - `docs-venv:` creates `.venv-docs` and installs `docs/requirements.txt`;
     `docs` and `docs-serve` depend on it through the file
     `.venv-docs/bin/zensical`.
9. Add `/site/` and `/.venv-docs/` to `.gitignore`. Add the docs line to
   `README.md`.

## Build gate

- `make docs` passes with no warnings.
- `go build ./...` still passes (no Go change, but the gate is cheap).

## Test plan

- `make docs-serve`, open the local address. Check that the navigation matches
  the content map, that search finds "secure", and that the Mermaid chart and
  the SVG bar on the home page render.
- Switch the palette toggle. The Mermaid chart and the SVG bar change colors
  and stay readable in both schemes.
- Add a link to `missing.md#x` on a stub page and run `make docs`. The build
  fails. Remove the link.
- Open `docs/secure.md` on GitHub. It still renders there (front matter shows
  as a small table, which is fine).

## Commit

`docs(site): add Zensical docs site skeleton in docs/`

## Rollback

Revert the commit. It adds new files and touches only `Makefile`,
`.gitignore` and one README line. The front matter added to
`docs/manual-install.md` and `docs/secure.md` is harmless if left.
