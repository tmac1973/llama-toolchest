# Phase 04 — Copies fill missing files with their Hugging Face links

**Depends on:** Phase 01 (`HFLink`), Phase 02 (links in fields are downloaded on Add set), Phase 03 (the manual test swaps a file with Browse) · **Enables:** Phase 05 (the field test copies a catalog set that is not on disk)

## Goal

A copy of a set fills each missing file with its Hugging Face link, so Add set downloads it. This works for a catalog set from the catalog card, a downloaded catalog set, and a custom set with downloaded files. Today such a field stays empty with only a file-name hint. A file that is on disk keeps its local path, as today. A copy is the easy way to change one file of a catalog set, for example a different quant.

## Files touched

- `internal/api/gen_set_form.go`: `genSetFilledForm` fills links for missing files on copies.
- `internal/api/gen_set_form_test.go`: tests.
- `docs/guides/generation.md`: one sentence about copies in `#custom-set-downloads`.

## Steps

1. In `genSetFilledForm`, for a copy (`EditID == ""`), handle a component with `c.Repo != ""` like this:
   - **On disk:** when `c.Path` is set, or `huggingface.FilePath(root, c.Repo, c.File)` exists, set the local path (today's behaviour).
   - **Not on disk:** set `HFLink(c.Repo, c.Revision, c.File)`. A catalog component has no `Revision`, so its link uses `main`. A custom set component keeps its pinned commit.
   - **Hint:** do not set `hints[c.Slot]` for that slot. The field has a value now.
2. The copy note above the form ("Copied from X. Change the files or settings, then save.") gets this sentence when any field holds a link: "Files that are not on this machine will download when you add the set."
3. Docs: one sentence in `#custom-set-downloads`. It says that a copy downloads the files that are not on disk, and that the user can swap one file before adding.

## Build gate

```bash
go build ./...
go vet ./...
go test ./...
make docs
```

## Test plan

Automated:

- Copy a catalog entry with no files on disk. Every slot holds `https://huggingface.co/<repo>/blob/main/<file>`, there are no hints, and the extra sentence shows.
- Copy a catalog entry with one file on disk. That slot holds the local path, and the others hold links.
- Copy a custom set with a pinned component that is not on disk. The link uses the pinned commit.
- The edit form is unchanged from Phase 02 (links for downloaded components).

Manual, on a dev instance: copy a catalog set that is not downloaded, change one file with Browse, add it, and watch only the needed files download.

## Commit

`feat(generation): fill copied sets' missing files with Hugging Face links`

## Rollback

Revert the commit. Copies go back to empty fields with hints. No saved data changes in this phase.
