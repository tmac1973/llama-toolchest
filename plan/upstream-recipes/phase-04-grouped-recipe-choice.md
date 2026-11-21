# Phase 04 — Grouped recipe choice with reasons

**Depends on:** 01 (recipe names, descriptions, `Unavailable` reasons), 03 (the active book the form reads) · **Enables:** 05 (the prefill hooks into the same recipe change), 06 (field test)

## Goal

With about 100 recipes, the plain recipe select of the custom set form is hard to use, and recipes that toolchest cannot run are missing without a word. This phase groups the select, shows each recipe's upstream name, shows the chosen recipe's description, and lists the recipes toolchest cannot run as disabled options with their reason. The `/api/gen/models` JSON gets the same reasons.

## Files touched

- `internal/api/gen_set_form.go` — `genSetForm.Recipes []string` becomes `RecipeGroups []genRecipeGroup`; new `RecipeDescription string`.
- `web/templates/partials/gen_set_form.html` — the recipe `<select>` uses `<optgroup>`s; the description line under it.
- `internal/generation/recipebook.go` — `RecipeBook.CatalogIDs(kind)` and `OtherIDs(kind)` (the two halves of `IDs`), used for the groups.
- `internal/api/gen_models.go` — `genModelsData` gets `UnavailableRecipes map[string]string` (recipe ID → reason, both kinds).
- `internal/api/gen_set_form_test.go`, `internal/api/gen_models_test.go` — tests.
- `docs/guides/generation.md` — the "Use your own files" item says how the list is grouped and that greyed recipes show why.
- `web/templates/help.html` — no change (the phase 03 sentence covers it).

## Steps

1. Define `genRecipeGroup {Label string; Options []genRecipeOption}` and `genRecipeOption {ID, Text, Reason string; Disabled bool}`. `Text` is `<upstream name> (<id>)`, or just `<id>` when the name is empty. For a disabled option, `Text` is `<upstream name> (<id>) — <reason>`.
2. In `genSetFormFrom`, build the groups for the form's kind from `ActiveRecipes()`:
   - "Used by the catalog" — `CatalogIDs(kind)`, in catalog order;
   - "Other recipes" — `OtherIDs(kind)`, A–Z by ID;
   - "Not available in llama-toolchest" — `Unavailable(kind)`, disabled, A–Z.
   An empty group is left out. The phase 03 case (a copied or edited set with an unknown recipe) adds its option before the first group, as phase 03 does.
3. `RecipeDescription` is the chosen recipe's upstream `Description`, empty when there is none. Show it as a `<small>` line under the select. Long descriptions are shown in full; they are one or two sentences upstream.
4. The select keeps its `hx-get="/api/gen/models/sets/form/fields"` and `hx-include`, so a change still redraws the form. A disabled option cannot be chosen, so the server never gets one from the select; `ValidateComponents` still refuses one sent by hand (phase 01).
5. `/api/gen/models`: keep `recipes` and `video_recipes` as they are (usable IDs, same order), and add `unavailable_recipes` with every unavailable recipe of both kinds and its reason. Document the field in the doc comment on `genModelsData`.
6. The recipe tooltip on the label (`gen_set_form.html:79`) gets one more sentence: "Greyed recipes need something llama-toolchest cannot send or pass."
7. Docs: update the "Use your own files" item (see Files touched).

## Build gate

```
go build ./...
go vet ./...
go test ./internal/api/... ./internal/generation/...
make js-test
make docs
```

## Test plan

- Form, image kind: three groups in the right order; `z-image-turbo` (a catalog recipe) is in the first group; `pixart-alpha-1024` in the second; `sd35-large` disabled in the third, its text holds the `clip_g` reason.
- Form, video kind: `wan22-t2v` in the first group; `animatediff-sd15` disabled with the `motion_module` reason; `wan21-vace` disabled with the overlay reason.
- Form: the description of the chosen recipe is shown; a recipe with no description shows no line.
- `/api/gen/models`: `recipes` is unchanged from phase 03; `unavailable_recipes` has `sd35-large` and `wan22-s2v` with reasons.
- Manual: open the form in the browser, check the three groups, that a disabled option cannot be chosen, and that the description changes with the recipe.

## Commit

`feat(ui): group the custom set recipe list and show why recipes are not available`

## Rollback

Revert the commit. The form goes back to the plain list of usable recipes from phase 03. API clients that read `unavailable_recipes` lose the field; `recipes` and `video_recipes` never changed.
