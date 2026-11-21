# Phase 02 — Copy recipe files into each sd build

**Depends on:** nothing · **Enables:** 03 (each build has its own recipes to read)

## Goal

When toolchest builds stable-diffusion.cpp, it copies the checkout's `examples/server/recipes/*.json` into the build's folder, next to `sd-server`: `builds/<id>/recipes/`. Each build then keeps the recipes of the commit it was built from, even after the shared checkout moves on. Nothing reads these folders yet; phase 03 does.

## Files touched

- `internal/builder/project.go` — new `Project` field `Extras []ExtraFiles` and type `ExtraFiles {Src, Dst, Glob string}`; the sd project sets `Extras: []ExtraFiles{{Src: "examples/server/recipes", Dst: "recipes", Glob: "*.json"}}`.
- `internal/builder/builder.go` — new function `copyExtras(srcDir, outDir string, extras []ExtraFiles, sendLog func(string)) error`; `runBuild` calls it after the optional binaries are installed into `outDir`.
- `internal/builder/extras_test.go` — new: tests `copyExtras` on temporary folders (no cmake needed).

## Steps

1. Add the `ExtraFiles` type and the `Extras` field, with a doc comment: files from the checkout that a build keeps next to its binaries, because the shared checkout moves to another commit on the next build.
2. Set `Extras` on the sd project only. llama.cpp and acestep.cpp get none.
3. Write `copyExtras`. For each entry:
   - `matches, _ := filepath.Glob(filepath.Join(srcDir, e.Src, e.Glob))`.
   - No matches: `sendLog("    No " + e.Src + " in this checkout; the build uses the built-in recipe list")` and go on. This is not a build failure: old sd.cpp commits have no recipe folder.
   - Else create `filepath.Join(outDir, e.Dst)` (mode 0o755) and copy each match with `atomicfile.Copy`, the same helper the binaries use. A copy error is returned as `copy <file>: <err>`.
   - `sendLog(fmt.Sprintf("    Copied %d files to %s", n, e.Dst))`.
   - It returns the first copy error. In `runBuild`, call it after the "Not installed" loop for optional binaries; on an error, `finishBuild(result, BuildStatusFailed, "install failed: "+err.Error())` and return, like a binary copy error.
4. Rebuilding with the same build ID (force) replaces the folder: remove `filepath.Join(outDir, e.Dst)` before the copy, so a recipe that upstream deleted does not stay.
5. A build that is deleted removes its folder as today; `recipes/` is inside it, so nothing new is needed.

## Build gate

```
go build ./...
go vet ./...
go test ./internal/builder/...
```

## Test plan

- Unit (`copyExtras`): a fake sd checkout with `examples/server/recipes/a.json`, `b.json` and `README.md` builds into a folder with `recipes/a.json` and `recipes/b.json` only.
- Unit: a fake checkout without the folder builds successfully, and the log has the "No examples/server/recipes" line.
- Unit: a forced rebuild after `b.json` is removed from the checkout leaves only `recipes/a.json`.
- Manual (this machine, container or host mode): build the newest sd.cpp; `ls ~/.local/share/llama-toolchest/builds/sd-*/recipes | wc -l` gives the same count as the checkout's recipe folder.

## Commit

`feat(builder): keep stable-diffusion.cpp recipe files with each build`

## Rollback

Revert the commit. Builds made with it keep a harmless `recipes/` folder that nothing reads until phase 03; it can stay or be deleted by hand.
