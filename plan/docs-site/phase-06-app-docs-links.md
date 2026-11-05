# Phase 06 — App links to the docs, with a check

**Depends on:** Phases 01, 02 · **Enables:** Phase 07 (Help uses
`docsURL` for its "More in the docs" links)

## Goal

Give the app one safe way to link to the docs site: a `docsURL` template
function that builds the full URL from one base constant, and a Go test that
fails when a linked page or anchor does not exist in `docs/`. The test runs in
the normal `go test ./...` of `ci.yml`, and `docs.yml` runs it too, because a
docs-only pull request does not run `ci.yml`. So a pull request that renames a
docs heading or deletes a page fails if the app still links to it.

## Files touched

- `internal/api/docs_links.go` — new. The base URL constant and the
  `docsURL` function.
- `internal/api/server.go` — add `"docsURL": docsURL` to `templateFuncs()`.
- `internal/api/docs_links_test.go` — new. Unit test of `docsURL`, and the
  link check over all templates.
- `internal/api/sampling_presets_render_test.go` — read `testFuncMap`. If it
  lists functions by hand instead of calling `templateFuncs()`, add
  `docsURL` to it, so templates that use the function still parse in tests.
- `internal/api/testdata/docs_links/` — new fixture folder (step 4).
- `.github/workflows/docs.yml` — add a `links` job (step 6), and add
  `web/templates/**` and `internal/api/docs_links*.go` to both `paths`
  lists.

## Steps

1. In `internal/api/docs_links.go`:
   - `const docsBaseURL = "https://tmac1973.github.io/llama-toolchest/"`.
   - `func docsURL(ref string) string`. `ref` is `page`, optionally followed
     by `#anchor`. `page` is the path under `docs/` without `.md`, for
     example `gpu/eviction`. An `index.md` page is written as its folder with
     a trailing slash, for example `gpu/`. The function splits `ref` at the
     first `#` and builds the URL in this order:
     1. Empty `page`: `docsBaseURL`.
     2. `page` ends in `/`: `docsBaseURL + page`.
     3. Otherwise: `docsBaseURL + page + "/"` (Zensical's directory URLs).
     4. If there is an anchor, append `"#" + anchor`.
     Examples: `""` → base; `gpu/` → `base + "gpu/"`; `gpu/#parts` →
     `base + "gpu/#parts"`; `gpu/eviction#modes` →
     `base + "gpu/eviction/#modes"`. It does no other rewriting. A comment says that the test in `docs_links_test.go` checks
     every use.
2. Register it in `templateFuncs()` next to the other helpers, with a short
   comment in the style of the ones around it.
3. In `internal/api/docs_links_test.go`:
   - `TestDocsURL`: table test of the forms in step 1.
   - `TestDocsLinksExist`: walk `../../web/templates/` for `*.html`; find
     every `docsURL "…"` call with a regular expression; for each `ref`,
     check that `../../docs/<page>.md` exists, or `../../docs/<page>index.md`
     when `page` ends in `/`; if `ref` has an anchor, check that the file has a
     heading line that ends with `{ #anchor }`. Only explicit ids count:
     every heading the app links to must have one. The test fails with the
     template file, the `ref`, and what is missing.
   - The test skips with a clear message if `../../docs` does not exist, so
     the package still tests from a module cache. In the repo it always
     exists.
4. No template uses `docsURL` in this phase; the first uses come in Phase 07.
   So the checker is also tested on a fixture. Put the check in a function
   `checkDocsLinks(templatesDir, docsDir string) []string` that returns one
   message per bad link. `TestDocsLinksExist` calls it on the real folders.
   `TestCheckDocsLinksFixture` calls it on
   `internal/api/testdata/docs_links/templates/sample.html` (two `docsURL`
   calls: `"gpu/eviction#modes"` and `"gpu/eviction#nope"`) and
   `internal/api/testdata/docs_links/docs/gpu/eviction.md` (one heading
   `## Modes { #modes }`), and expects exactly one message, for `#nope`.
5. The empty ref `""` is valid: `docsURL ""` returns `docsBaseURL`, and the
   checker accepts it without a file check. Phase 07 uses it for the "Docs"
   navigation link.
6. In `.github/workflows/docs.yml`, add a job `links` that runs on every event
   of the workflow: `actions/checkout@v7`, `actions/setup-go@v5` with
   `go-version-file: go.mod` and `cache: true` (as in `ci.yml`), then
   `go test ./internal/api -run 'TestDocsURL|TestDocsLinksExist|TestCheckDocsLinksFixture'`.
   Make `deploy` need both `build` and `links`.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test -race ./...`
- `make js-test`
- `make docs`

## Test plan

- `TestDocsURL` covers the empty ref, a page, a page with an anchor, and a
  folder index.
- The fixture test reports the missing anchor and nothing else.
- `TestDocsLinksExist` passes on the real templates (no uses yet, so it
  checks nothing and passes).
- Temporarily add `{{docsURL "gpu/eviction#nope"}}` to `help.html`. The test
  fails and names the file and the anchor. Remove it.

## Commit

`feat(ui): add docsURL template function and check app links to the docs`

## Rollback

Revert the commit. This also removes the `links` job from `docs.yml`. No
template uses `docsURL` until Phase 07, so the revert is
safe on its own. If Phase 07 is merged, revert Phase 07 first.
