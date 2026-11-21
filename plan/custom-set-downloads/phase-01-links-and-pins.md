# Phase 01 — Hugging Face links, revision pins and recipe repos

**Depends on:** nothing · **Enables:** Phase 02 (reads links from the form, pins revisions, uses the new client calls), Phase 03 (suggests the recipe repo, writes links)

## Goal

Add the building blocks that the later phases use. None of them changes what a user sees:

- a parser and a writer for Hugging Face file links,
- `Revision` and `Error` fields on `generation.Component`,
- the recipe's `model_url` as `Recipe.ModelURL`, with the repo ID it names,
- two Hugging Face client calls: resolve a branch or tag to a commit, and list a repo's files at a revision.

This phase can ship alone. Old sets load unchanged, and the new JSON fields are empty for them.

## Files touched

- `internal/generation/types.go`: add `Component.Revision` (`json:"revision,omitempty"`) and `Component.Error` (`json:"error,omitempty"`), with doc comments.
- `internal/generation/hflink.go` (new): `ParseHFLink` and `HFLink`.
- `internal/generation/hflink_test.go` (new): table tests for both.
- `internal/generation/recipebook.go`: read `model_url` into `Recipe.ModelURL`; add `(*Recipe).ModelRepo()`.
- `internal/generation/recipebook_test.go`: test `ModelURL` and `ModelRepo` on snapshot recipes.
- `internal/huggingface/client.go`: add `RepoCommit` and `RepoFilesAt`; `RepoFiles` calls `RepoFilesAt(ctx, modelID, "main")`.
- `internal/huggingface/client_test.go`: tests for both new calls against an `httptest` server.

## Steps

1. In `types.go`, add to `Component`, after `Path`:
   - `Revision string \`json:"revision,omitempty"\``: the commit that `Repo`/`File` are downloaded from. It is empty for catalog sets and for sets saved before this change; empty means the repo's default branch.
   - `Error string \`json:"error,omitempty"\``: why the last downloaded file for this slot was refused. It is empty when there is no error.
2. In `hflink.go`, add `ParseHFLink(s string) (repo, rev, file string, ok bool)`:
   - Trim spaces. The value must start with `https://huggingface.co/` (also accept `http://` and the `www.` host). Otherwise return `ok=false`.
   - Drop the query and the fragment. URL-decode the path.
   - Split the path into segments. Find the first segment equal to `blob` or `resolve`. The segments before it are the repo ID (1 or 2 segments; any other count is `ok=false`). The segment after it is `rev`. The remaining segments, joined with `/`, are `file`. A missing rev or file is `ok=false`.
   - A link to a repo page (no `blob`/`resolve`) is `ok=false`.
3. In `hflink.go`, add `HFLink(repo, rev, file string) string`. It returns `https://huggingface.co/<repo>/blob/<rev>/<file>`, with each path segment escaped by `url.PathEscape`. An empty `rev` writes `main`. `ParseHFLink(HFLink(r, v, f))` returns `r, v, f`.
4. Tests for steps 2 and 3:
   - `blob` and `resolve` links.
   - A one-segment repo (`gpt2`).
   - A file in a subfolder (`split_files/vae/x.safetensors`).
   - A query string (`?download=true`).
   - A name with a space (`%20`).
   - A repo page link, a local path and an empty string, which are all `ok=false`.
   - A round trip.
5. In `recipebook.go`, add `ModelURL string` to `Recipe`, and `ModelURL string \`json:"model_url"\`` to the file struct. Copy it in the parser. Add `(*Recipe).ModelRepo() string`. It returns the repo ID when `ModelURL` is `https://huggingface.co/<org>/<name>` (with an optional trailing `/` or `/tree/<rev>`). Otherwise it returns `""`.
6. Recipe tests:
   - `flux2-klein-4b` gives `black-forest-labs/FLUX.2-klein-4B`.
   - A recipe without `model_url` gives `""`.
   - Count the snapshot recipes with a non-empty `ModelRepo()`, and fail if fewer than 100 (today 104 of 108). This finds a parser break.
7. In `client.go`, add `RepoCommit(ctx, modelID, rev string) (string, error)`:
   - It calls `GET <apiURL>/models/<modelID>/revision/<rev>` and returns the response's `sha`.
   - Errors go through the same handling as `GetModel`, so a gated or missing repo gets the message from `errors.go` (#657).
   - When `rev` is already a 40-character hex string, return it without a request.
8. In `client.go`, add `RepoFilesAt(ctx, modelID, rev string) ([]modelsource.RepoFile, error)`. It is `RepoFiles` with `tree/<rev>` in place of `tree/main`. Make `RepoFiles` call it with `"main"`. Change the private `tree` helper to take the revision. Callers of `RepoFiles` (`model_update.go`) see no change.
9. Client tests use an `httptest` server that answers `/api/models/org/repo/revision/main` and `/api/models/org/repo/tree/<sha>`. Test:
   - the sha,
   - the 40-hex shortcut (no request),
   - a 403 gated answer, which gives the #657 message,
   - the tree at a commit.

## Build gate

```bash
go build ./...
go vet ./...
go test ./internal/generation/ ./internal/huggingface/
go test ./...
```

## Test plan

- The unit tests above.
- `go test ./internal/api/` still passes: no handler uses the new code yet.
- A test in `internal/generation/` loads a `generation.json` fixture with a catalog set and a custom set, saves it again, and checks that no `revision` or `error` keys appear.

## Commit

`feat(generation): parse Hugging Face file links and pin repo revisions`

## Rollback

Revert the commit. Nothing uses the new code yet, and old saved data has no new fields, so a revert is safe at any time before Phase 02.
