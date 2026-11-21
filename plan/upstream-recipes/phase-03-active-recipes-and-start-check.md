# Phase 03 — Active recipes from the newest build, and the start check

**Depends on:** 01 (recipe book, registry, `LoadRecipes`), 02 (builds have `recipes/` folders) · **Enables:** 04 and 05 (the form shows the active book's recipes), 06 (field test)

## Goal

Make the newest sd build's recipes the active list, with the embedded snapshot as the fallback. Check every sd and video instance start against the recipes of the build it runs on. Warn in Details when a set's recipe is not in the active list. After this phase, a new upstream recipe works with the next sd build, without a toolchest release.

## Files touched

- `internal/generation/recipebook.go` — `RecipesForBuild(build builder.BuildResult) *RecipeBook` with a cache by build ID; `ForgetBuildRecipes(id string)` (drops one cache entry); `CheckRecipeForBuild(set ModelSet, build builder.BuildResult) error`.
- `internal/generation/launch.go` — `LaunchSpec` calls `CheckRecipeForBuild` for `KindSD` and `KindVideo` sets before it builds the arguments.
- `internal/builder/builder.go` — new field `OnChange func()` on `Builder`, called (in a new goroutine, outside `b.mu`) at the end of `finishBuild` and after a successful `Delete`.
- `internal/api/server.go` — after `builder.NewBuilder`, set `bld.OnChange = s.refreshRecipes` and call `s.refreshRecipes()` once at start.
- `internal/api/gen_recipes.go` — new: `refreshRecipes` (picks the newest runnable sd build and swaps the registry).
- `internal/generation/details.go` — `SetDetails` gets `RecipeWarning string`, filled in `fill`.
- `web/templates/partials/gen_set_details.html` — show `RecipeWarning` at the top of the dialog when set.
- `internal/api/gen_set_form.go`, `web/templates/partials/gen_set_form.html` — copy and edit keep a recipe the active list does not know, with a warning (step 7).
- `internal/generation/recipebook_test.go`, `internal/generation/launch_test.go`, `internal/generation/details_test.go`, `internal/api/gen_recipes_test.go`, `internal/api/gen_set_form_test.go` — tests.
- `docs/guides/generation.md` — the custom set section says which recipe list the form uses, and what happens when a build does not know a set's recipe. Heading id `{ #recipes }` on a new short subsection "Where recipes come from".
- `web/templates/help.html` — one sentence in the generation section about where recipes come from, with a `docsURL` link to `#recipes`.

## Steps

1. `RecipesForBuild(build)`: the folder is `filepath.Join(filepath.Dir(build.BinaryPath), "recipes")`. If it does not exist, or `LoadRecipes` returns an error, return `SnapshotRecipes()` (log once per build ID at `slog.Info`: "build <id> has no recipe files; using the built-in list"). Else return the loaded book with `Source = build.ID`. Cache the result in a `sync.Map` keyed by build ID; a build ID is never reused for other files except by a forced rebuild, so `refreshRecipes` clears the cache entry of the build that just finished (it calls `ForgetBuildRecipes(id)`).
2. `CheckRecipeForBuild(set, build)` for `KindSD` and `KindVideo`:
   - the recipe is missing from `RecipesForBuild(build)` → `build <build.ID> does not know recipe <recipe>, which <set.Name> uses. Choose a newer stable-diffusion.cpp build for the instance, or build one on the Builds page.`
   - the recipe is there but unavailable → `recipe <recipe> cannot be used: <reason>`.
   - the recipe is there but of the other kind → the same text `ValidateComponents` uses for a kind mismatch.
   - else `nil`. Other kinds always pass.
3. Call `CheckRecipeForBuild` at the start of the `KindSD, KindVideo` case in `LaunchSpec`, before `LoraDir`. Measure and normal starts both go through `LaunchSpec`, so both are checked. The error reaches the user the same way other `LaunchSpec` errors do today (the instance's error text).
4. `Builder.OnChange`: add the field with a doc comment ("called after a build finishes or is deleted; may be nil"). At the end of `finishBuild` (after `saveBuilds`) and at the end of a successful `Delete`, `if f := b.OnChange; f != nil { go f() }`.
5. `refreshRecipes` in `internal/api/gen_recipes.go`:
   - `builds := s.genBuildsFor(generation.KindSD)` (newest first, runnable on this machine's GPUs).
   - none → `generation.SetActiveRecipes(generation.SnapshotRecipes())`.
   - else `generation.ForgetBuildRecipes(builds[0].ID)` and `generation.SetActiveRecipes(generation.RecipesForBuild(builds[0]))`.
   - `slog.Info("recipe list", "source", book.Source, "recipes", count)`.
   It must be safe to run at the same time as itself (the registry is an atomic pointer; the cache is a `sync.Map`).
6. Details: in `SetDetails.fill`, for an sd or video set or catalog entry, if `ActiveRecipes().Get(recipe)` is missing, set `RecipeWarning` to `The active recipe list (<Source>) does not know recipe <recipe>. The set is kept. An instance starts only on a build that knows this recipe.` If the recipe is unavailable, set it to `Recipe <recipe> cannot be used: <reason>`. Show it in `gen_set_details.html` as a warning line above the slots, with the same class the dialog uses for other warnings.
7. Copy and edit of such a set still open. Today `genSetFormFrom` (`internal/api/gen_set_form.go:78`) replaces a recipe that does not fit the kind with the kind's first recipe. Change it: when the form is filled from a set or catalog entry (copy, `CopiedFrom != ""`) or edits a set (`EditID != ""`) and `ActiveRecipes().Get(recipe)` fails, keep the recipe, add it as the selected first option of the recipe select with the text `<recipe> (not in the recipe list)`, build the slot fields from the set's own components (label from `SlotLabel`, all marked required), and set a new `genSetForm.RecipeWarning` field to the Details warning text, shown above the slot fields in `gen_set_form.html`. A new set (no copy, no edit) keeps today's rule. Saving such a set is refused by `ValidateComponents` with its "unknown recipe" text, which is correct: the active list cannot validate it.8. Docs and Help (see Files touched). The "Where recipes come from" subsection says: each stable-diffusion.cpp build keeps the recipe files of its commit; the form uses the newest build's list; with no build, a list built into llama-toolchest; an instance starts only on a build that knows its set's recipe.

## Build gate

```
go build ./...
go vet ./...
go test ./internal/generation/... ./internal/builder/... ./internal/api/...
make js-test
make docs
```

## Test plan

- `RecipesForBuild`: a build whose folder has `recipes/` with two fixture files gives a book of two recipes with `Source` = build ID; a build without the folder gives the snapshot; a second call hits the cache; `ForgetBuildRecipes` makes the next call read the folder again.
- `CheckRecipeForBuild`: known and available → nil; unknown → the "does not know recipe" text; unavailable → the reason; ace and upscale sets → nil.
- `LaunchSpec`: an sd set whose recipe the build's folder lacks returns the error and no config.
- `refreshRecipes`: with a fake builder that has two sd builds, the active book comes from the newer one; with no sd build, the snapshot; a CUDA build on an AMD machine is skipped (through `genBuildsFor`).
- `OnChange` is called after `finishBuild` and `Delete` (a test with a channel).
- Details: a set with a recipe the active book lacks has the warning; a normal set has none.
- Form (`internal/api/gen_set_form_test.go`): copy and edit of a set with an unknown recipe keep the recipe, show its component slots and the warning; a new set with an unknown recipe value still falls back to the kind's first recipe.
- Manual: with a new sd build from phase 02, the server log shows `recipe list source=sd-master-…`; a set saved earlier still starts; delete the new build and the log shows the older build or the snapshot.

## Commit

`feat(generation): use the newest sd build's recipes and check them at start`

## Rollback

Revert the commit. The registry stays on the snapshot from phase 01, and starts are no longer checked against the build. Nothing stored changes.
