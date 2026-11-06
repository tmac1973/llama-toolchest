# Phase 02 — Build and deploy workflow

**Depends on:** Phase 01 · **Enables:** every later docs change is checked on
its pull request and published on merge

## Goal

Add `.github/workflows/docs.yml`. On a pull request to `main` that touches the
docs, it runs a strict build, so broken links and anchors fail the check. On a
push to `main` that touches the docs, it builds and deploys the site to GitHub
Pages. After this phase, the skeleton from Phase 01 is live at
`https://tmac1973.github.io/llama-toolchest/`.

## Files touched

- `.github/workflows/docs.yml` — new.
- `.github/workflows/ci.yml` — no change. Its `paths-ignore` already skips
  `docs/**` and `**.md`, so a docs-only pull request runs only `docs.yml`.
  A pull request that touches code and docs runs both.

## Steps

1. Write `.github/workflows/docs.yml`, in the comment style of `ci.yml` (a
   short paragraph on top that says what it does and why):
   - `name: docs`.
   - Triggers: `pull_request` to `main` and `push` to `main`, both with
     `paths: ["docs/**", "zensical.toml", ".github/workflows/docs.yml"]`;
     and `workflow_dispatch` for a manual redeploy.
   - Top-level `permissions: contents: read`.
   - `concurrency: group: docs-${{ github.ref }}`, `cancel-in-progress: true`.
   - Job `build` (all events): `actions/checkout@v7`,
     `actions/setup-python@v6` with `python-version: "3.13"`,
     `pip install -r docs/requirements.txt`, `zensical build --clean --strict`.
     On `push` and `workflow_dispatch` only, it also runs
     `actions/configure-pages@v6` before the build and
     `actions/upload-pages-artifact@v5` with `path: site` after it.
   - Job `deploy`: `needs: build`,
     `if: github.event_name != 'pull_request'`,
     `permissions: pages: write, id-token: write`,
     `environment: name: github-pages, url: ${{ steps.deployment.outputs.page_url }}`,
     one step `actions/deploy-pages@v5` with `id: deployment`.
   - No pip cache: the Zensical docs advise against caching for now.
2. Open the pull request for this phase. The `build` job runs on it, because
   the workflow file itself is in `paths`.
3. Manual step for the user, before or right after the merge: in the GitHub
   repo, Settings → Pages → Build and deployment → Source: **GitHub Actions**.
   The plan does not change repo settings; only the user does this.
4. After the merge, check the `deploy` job, then the site address. If the
   first deploy ran before the Pages source was set and failed, run the
   workflow again with **Run workflow** (`workflow_dispatch`).

## Build gate

- The `docs / build` check passes on the pull request.
- `make docs` passes locally.

## Test plan

- On the pull request: the `docs` workflow runs `build` and skips `deploy`.
  `ci` does not run if only the workflow and docs changed.
- On a throwaway branch (not merged), add a broken link to a stub page and
  push. The `docs / build` check fails with the link in the log. Delete the
  branch.
- After merge: `https://tmac1973.github.io/llama-toolchest/` shows the home
  page with the test diagrams, and the "Edit this page" link opens the file on
  GitHub.

## Commit

`ci(docs): build the docs site on pull requests and deploy it to Pages`

## Rollback

Revert the commit. The site stays online with its last version until the user
turns Pages off in the repo settings (Settings → Pages). No other workflow
depends on `docs.yml`.
