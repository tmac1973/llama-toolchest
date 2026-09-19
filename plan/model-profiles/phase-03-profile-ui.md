# Phase 03 — Profile bar in the model config panel

**Depends on:** 02 · **Enables:** 09 and 13 (their review and results screens
link to restoring a profile from here); completes the saved-profiles feature
(PR 1 together with 01, 02, 04)

## Goal
Let the user save, restore and delete profiles from the model config panel.
The bar sits **outside** the autosaving form: the form posts
`hx-include="closest form"` on every change, so a select inside it would
autosave while the user browses, and the name box would be posted with each
save.

## Files touched
- `internal/models/profiles.go`: `ValidateProfileConfig(cfg ModelConfig) error`
  (the file and field checks of step 3), so that Phase 12 can reuse them
  without importing `internal/api`.
- `internal/api/models_profiles.go` (new): handlers for save, apply and
  delete, plus a shared re-render helper.
- `internal/api/models_profiles_test.go` (new).
- `internal/api/server.go`: routes.
- `internal/api/service.go`:
  - `handleGetModelConfig` passes the profile list, the active profile and
    the edited flag;
  - the panel is wrapped in a container that the handlers can re-render.
- `web/templates/partials/model_config.html`:
  - wrap the existing `<form id="model-config-form">` in
    `<div id="model-config-panel-{{.ModelID}}">`;
  - add `{{template "profile_bar" .}}` above the form.
- `web/templates/partials/profile_bar.html` (new).
- `internal/api/profile_bar_render_test.go` (new).

## Steps
1. **Routes** (all POST, because htmx sends included fields in the body and
   `ParseForm` ignores a DELETE body):
   - `POST /api/models/{id}/profiles`, form `name`: save the live config as a
     profile, with source "user" and the current `s.cfg.ActiveBuild`.
   - `POST /api/models/{id}/profiles/apply`, form `name`: restore.
   - `POST /api/models/{id}/profiles/delete`, form `name`: delete.
2. **One re-render helper for every handler.**
   `renderConfigPanel(w, r, id, banner{Kind: ok|warning|error, Text})`
   re-renders the whole panel (profile bar + form). Errors appear as a banner
   and never replace the form.
3. **Restore runs the same checks as a config save.** These checks live in
   `models.ValidateProfileConfig`, which the handler calls:
   - It calls the existing `ValidateBatchSizes`, `ValidateFlashAttention` and
     `ValidateSpec` on the profile config.
   - It checks that `MmprojPath`, `MtpPath` and `DraftModelPath` exist on
     disk.
   - On failure it refuses and names the setting and the reason. A profile
     that points at a deleted draft model would otherwise fail minutes into a
     load.
   - A profile saved on a different build restores with a warning banner:
     "Saved on build X; the active build is Y. Speculative decoding methods
     and flags may behave differently."
4. **After a restore:**
   - rewrite the preset INI;
   - mark the model dirty, the same way `handleUpdateModelConfig` does (reuse
     its post-save code by extracting it into `afterConfigChange(id)`), so the
     existing "Restart required" marker appears.
5. **Profile bar layout:**
   - A `<select>` of profiles. Each option reads "Name — saved 2026-09-19,
     build b1234" and gains "(autoconfig)" or "(autotune)" when `Source` is
     not "user".
   - **Restore** needs no confirmation. **Delete** uses
     `hx-confirm="Delete profile NAME? The live config is not changed."`.
   - A name input, pre-filled with the active profile, and a **Save as
     profile** button. Saving over an existing name shows "Replaced profile
     NAME" in the banner.
   - A status line with a tooltip:
     - "Running settings from profile NAME", or
     - "From profile NAME — edited since", or
     - "Not saved as a profile".
   - Each control has a `title` tooltip in plain language, e.g. Restore:
     "Replace the settings below with the ones saved in this profile. Takes
     effect the next time the server restarts."
6. **Read-only registry:** the bar is disabled, and the Phase 01 banner
   explains why.
7. The select, input and buttons use `hx-include` of the bar's own fields
   only, and `hx-target="#model-config-panel-{{.ModelID}}"`.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- **Render test:** the panel with two profiles, one active and edited. Assert
  that `profile_bar` markup appears **before** `<form id="model-config-form"`
  and that no `name="profile` field appears inside the form (substring check
  between the form tags).
- **Handler tests:**
  - save → apply → delete through `httptest`;
  - posting the config form's own fields leaves profiles unchanged;
  - restore refuses a profile whose `DraftModelPath` does not exist;
  - restore of a profile saved on another build returns a warning banner;
  - with a read-only registry, save returns the error banner.
- **Manual (`make dev`):**
  - Save "baseline", change the context size (autosave), and check the
    status reads "edited since".
  - Restore "baseline" and check the field is back and "Restart required"
    shows.
  - Delete it.

## Commit
`feat(models): save and restore config profiles from the config panel`

## Rollback
Revert the commit. The stored profiles from Phase 02 remain in `models.json`
and are simply not shown.

## As implemented

- **Template split.** `model_config` is now the whole panel: the header,
  the read-only banner, `profile_bar`, then `model_config_form` (the form
  itself). The form's autosave still swaps only the form (`hx-target="this"`).
  The PUT response is `model_config_autosave`: the form plus
  `profile_status`, marked `hx-swap-oob`. "Changed since" therefore stays
  current, and the whole bar is not re-rendered, which would clear a profile
  name being typed.
- **Header and banner moved.** The header and the read-only banner moved
  out of the form into the panel. The card styling moved from
  `.model-card-config > form` to `.model-card-config > .model-config-panel`
  in `layout.html`.
- **Refactors.**
  - `configPanelData(id)` builds the panel data; `handleGetModelConfig` and
    the profile handlers share it.
  - `afterConfigChange` holds the post-save steps (preset INI, dirty mark,
    VRAM trigger), extracted from `handleUpdateModelConfig`.
- **Delete confirmation.** It reads "Delete the selected profile? The
  settings below are not changed." It does not name the profile, because
  `hx-confirm` text is fixed when the page renders.
- **Status wording.**
  - "Matches profile X."
  - "Based on profile X, changed since."
  - "These settings are not saved as a profile."
- **Still to check by hand:** the layout of the bar in a real browser. The
  tests check the markup and the handler behavior only.
