---
paths:
  - "web/templates/help.html"
  - "docs/**"
---

The in-app Help page stays short - a few sentences per section plus a "More in the
docs" link built with `docsURL`. Full prose belongs in `docs/` (Zensical, published
to GitHub Pages, built with `make docs`). When you change what a user sees or how
something works, update the matching docs page in the same pull request. Keep the
Help anchors other pages link to, and give every docs heading the app links to an
explicit id (`## Modes { #modes }`).
