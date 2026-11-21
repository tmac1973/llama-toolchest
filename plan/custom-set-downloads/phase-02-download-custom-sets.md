# Phase 02 — Download, check, resume and delete custom set files

**Depends on:** Phase 01 (`ParseHFLink`, `HFLink`, `Component.Revision`/`Error`, `RepoCommit`, `RepoFilesAt`) · **Enables:** Phase 03 (the Browse dialog and the size line reuse `resolveHFComponents`), Phase 04 (copy fills links that this phase downloads)

## Goal

A custom set's file field takes a pasted Hugging Face file link. **Add set** and **Save changes** do these things:

- resolve each link to a pinned commit and a size,
- refuse the set when the files do not fit on the disk,
- save the set, and
- download the missing files the same way catalog sets do.

Each finished file passes the model-file check, or it is deleted and the error shows on the set's row. An incomplete custom set can be resumed. Deleting a custom set deletes its downloaded files unless another set uses them.

After this phase, the whole feature works with pasted links. Phase 03 adds the Browse dialog.

## Files touched

- `internal/api/gen_set_form.go`:
  - `formComponents` reads links.
  - `setFromForm` resolves them and checks the space.
  - The edit form shows a downloaded file as its link.
  - The field hint text changes.
- `internal/api/gen_set_resolve.go` (new): `resolveHFComponents` and the free-space check.
- `internal/api/gen_models.go`:
  - `startSetDownloads` is taken out of `handleGenCatalogDownload`.
  - `handleGenCreateSet` and `updateSet` start downloads.
  - New `handleGenSetDownload` (resume).
  - `genFileDone` runs the file check.
  - `genSetRow.FileError`.
  - Delete covers downloaded custom files.
- `internal/api/server.go`: route `POST /api/gen/models/sets/{id}/download`.
- `web/templates/partials/gen_model_sets.html`: Resume for custom sets, and "Incomplete (wrong file)".
- `web/templates/partials/gen_set_form.html`: the file field note and placeholder.
- `internal/api/gen_set_resolve_test.go` (new), `internal/api/gen_models_test.go`, `internal/api/gen_set_form_test.go`: tests.
- `docs/guides/generation.md`: the "Use your own files" item, plus a new section `## Download files for a custom set { #custom-set-downloads }`.
- `web/templates/help.html`: one sentence in the generation section, linking to `#custom-set-downloads` with `docsURL`.

## Steps

1. **Read links in `formComponents`.**
   - When `ParseHFLink(value)` is ok, add `Component{Slot, Repo, File, Revision: rev}` with no `Path`.
   - Otherwise keep today's local-path behaviour.
   - The `files` list form ("slot = value") takes links the same way.
2. **Resolve links (`gen_set_resolve.go`).** Add `func (s *Server) resolveHFComponents(ctx context.Context, comps []generation.Component) ([]generation.Component, error)`. For each component with a `Repo`:
   - **Commit:** `Revision = RepoCommit(ctx, repo, rev)`, so a branch or tag becomes a commit.
   - **Size:** `RepoFilesAt(ctx, repo, Revision)` gives the files. Find `File`, and set `Bytes` to its size. When the file is not in the list, return the error "<slot>: <file> is not in <repo> at <rev>".
   - **Reuse:** when a regular file of that size already exists at `huggingface.FilePath(generation.ModelsDir(s.cfg), Repo, File)`, set `Path` to it.
   - **Cache:** cache file lists in a small map on `Server` (`hfRepoCache`, with a mutex), keyed by `repo@rev`, for 10 minutes. Here the key is always a commit, whose list never changes. Phase 03 also stores `repo@main` lists in it; a list up to 10 minutes old is fine for browsing, and Add set always resolves to a commit again.
   - **Time limit:** all Hugging Face calls of one resolve share a 20-second context. On a timeout, return "Hugging Face did not answer in time; try again".
   - Errors from the client keep their #657 text.
3. **Check the space.** Add `func (s *Server) checkDownloadSpace(comps []generation.Component) error`.
   - Sum `Bytes` of the components with a `Repo` and no `Path`.
   - When `s.downloader.AvailableForDownload()` is ≥ 0 and smaller than the sum, return "The files to download need %.1f GB, and %.1f GB is free in the models folder."
   - `setFromForm` returns this error with `http.StatusInsufficientStorage` (507) as its status, the same way it returns 400 for its other checks.
4. **`setFromForm` rules.**
   - Call `resolveHFComponents` (on the request context), then `checkDownloadSpace`.
   - Run the existing local checks (exists, regular file, `CheckFileFormat`) only on components with no `Repo`. A `Repo` component that already has a `Path` (reused) also gets `CheckFileFormat`.
   - Set `set.Complete` to true only when every component has a `Path`.
   - The LoRA-folder rule applies only to local LoRA files. LoRA slots refuse links with "LoRA files must be local files" (#527 stays out).
5. **Share the download start.** Move the download loop of `handleGenCatalogDownload` into `func (s *Server) startSetDownloads(set generation.ModelSet) error`:
   - For each component with a `Repo` and no `Path`, call `StartToAt(SourceHuggingFace, Repo, Revision, File, root, Bytes, onDone)`. An empty `Revision` is the default branch, the same as `StartTo` today.
   - Keep the "already on disk, take it as it is" branch.
   - Record the IDs in `genDownloads`.
   - `handleGenCatalogDownload` calls it; catalog behaviour does not change.
6. **Add and edit.**
   - `handleGenCreateSet` calls `startSetDownloads` after `AddModelSet` when the set is not complete, and sets the `HX-Trigger: dlPanelRefresh` header.
   - `updateSet` refuses an edit while any of the set's downloads is `downloading` (409, "wait for the download to finish, or delete the set"). After saving, it calls `startSetDownloads`.
   - An edit that replaces a downloaded file does not delete the old file. The user deletes the set to clean up. This is the same rule as for local files.
7. **The edit form.** In `genSetFilledForm`, a component with a `Repo` shows `HFLink(Repo, Revision, File)` in its field for an edit (`EditID != ""`), whether or not the file is on disk. So saving the edit keeps it a downloaded file. Copies are unchanged in this phase (Phase 04).
8. **Check each finished file in `genFileDone`.** After the size check, run `generation.CheckFileFormat(slot, path)`.
   - On an error: remove the file, set `c.Path = ""` and `c.Error = err.Error()`, and log a warning.
   - On success: clear `c.Error`.
   - The existing loop then sets `set.Complete` from whether every component has a `Path`, so a refused file keeps the set incomplete and a passed last file completes it.
   - This runs for catalog sets too. A catalog file that fails is equally wrong.
9. **Resume.** Add `handleGenSetDownload` on `POST /api/gen/models/sets/{id}/download`:
   - It refuses a set that is complete (409), a catalog set (400, "use the catalog download"), and a set while `genMove.busy()`.
   - It clears every component's `Error`, saves the set, and calls `startSetDownloads`. It answers with `handleGenModels`.
   - The resume uses the stored `Revision`, so it gets the pinned commit.
10. **The set row.**
    - Add `FileError string` to `genSetRow`: the first non-empty `Component.Error`.
    - In `gen_model_sets.html`, the incomplete state shows "Incomplete (wrong file)" when `FileError` is set, with the error in the tooltip.
    - The Resume button posts to `/api/gen/models/catalog/{{.CatalogID}}/download` for catalog sets, and to `/api/gen/models/sets/{{.ID}}/download` for custom sets.
    - Hide Resume for a custom set whose missing files are all local (no `Repo`). Nothing can be downloaded for it.
11. **Delete.** In `handleGenDeleteSet`, apply the existing file-removal loop to every component with a `Repo`, for catalog and custom sets. Components with no `Repo` (local files) are skipped. The `sharedFiles` check stays. Cancel the running downloads first, as today.
12. **The form text.**
    - The field placeholder (when there is no hint): "A file in the models folder, or a Hugging Face file link".
    - The note under the fields: "Type part of a file name to see matching files in the models folder (`…`), or paste a Hugging Face file link to download the file. Files elsewhere must be under that folder. In a container, mount their folder under it."
13. **Docs.**
    - In `docs/guides/generation.md`, add `## Download files for a custom set { #custom-set-downloads }`. It covers the link form, the pinned commit, the space check, the file check, Resume, what Delete removes, and gated repos (link to the Settings token section).
    - Add one sentence to the "Use your own files" item that links to it.
    - In `help.html`, add one sentence with a `docsURL` link.

## Build gate

```bash
go build ./...
go vet ./...
go test ./...
make js-test
make docs
```

## Test plan

Automated, with an `httptest` Hugging Face server and the real `huggingface.Downloader` pointed at it, as `fork_models_test.go` does:

- `formComponents` sorts a mix of links and local paths into `Repo`/`Path` components.
- `resolveHFComponents`:
  - pins `main` to the sha,
  - sets `Bytes`,
  - reuses a file that is already on disk,
  - refuses a file that is not in the repo,
  - keeps the #657 message on a 403.
- `checkDownloadSpace` refuses with both sizes in the message, and does not count reused files.
- Create a set with one link: the set is saved incomplete, and one download starts with the pinned revision.
- `genFileDone` with a file that starts with `<!DOCTYPE html>`: the file is removed, `Error` is set, and the row shows "Incomplete (wrong file)". With a valid safetensors header: the set is complete, and `Error` is clear.
- Resume clears the errors and starts downloads. It refuses a complete set and a catalog set.
- Delete a custom set with a downloaded file, a file shared with a catalog set, and a local file: only the first is removed.
- An edit while a download is running is refused. The edit form shows the link for a downloaded component.
- Existing catalog download tests still pass.

Manual (dev instance on a free port, not :3000): paste `https://huggingface.co/stabilityai/sd-turbo/blob/main/sd_turbo.safetensors` into an `sd-turbo` custom set. Watch the row's progress, and check that the set becomes complete.

## Commit

`feat(generation): download Hugging Face files for custom sets`

## Rollback

Revert the commit. Sets saved with links keep `Repo`/`Revision` in `generation.json`. After the revert:
- a complete set still works, because it has `Path`;
- an incomplete one shows as Incomplete with no Resume; the user deletes it;
- downloaded files stay on disk until removed by hand.

The extracted `startSetDownloads` returns to the catalog handler with the revert.
