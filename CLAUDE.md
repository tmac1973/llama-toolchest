# Repository instructions

## Commits

Never add AI attribution to commits: no `Co-Authored-By: Claude <noreply@anthropic.com>`
trailer, no "Generated with Claude Code" or similar lines. Commit messages are plain
Conventional Commits with no attribution of any kind.

Always work on a branch — never commit directly to `main`, and never push `main`
ourselves. Create a feature branch off `main` for every change, including small
documentation fixes; the user decides when things land on `main`.

## Documentation

The Help page in the app (`web/templates/help.html`) stays short: a few sentences per
section and a "More in the docs" link built with `docsURL`. The full text, with diagrams,
is the docs site in `docs/` (Zensical, published to GitHub Pages; build it with
`make docs`). When a change alters what a user sees or how something works, update the
matching docs page in the same pull request. Keep the Help anchors that other pages link
to, and give every docs heading the app links to an explicit id (`## Modes { #modes }`).
