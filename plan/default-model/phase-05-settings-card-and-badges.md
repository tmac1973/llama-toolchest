# Phase 05 — Settings card, Models page badge, Help and docs

**Depends on:** Phase 01, Phase 03, Phase 04 · **Enables:** nothing (last
phase)

## Goal

Give the user control without editing config: a Settings card with the on/off
switch, the name and the pin, which also shows where the name points now and
why. Mark that model on the Models page and on its fork instance row. Add the
short Help entry and the full docs text.

## Files touched

- `web/templates/partials/default_model_card.html` (new) — the card.
- `web/templates/settings.html` — load the card after the Eviction and GPU
  memory margin cards, with `id="default-model"`.
- `internal/api/default_model_card.go` (new) — `defaultModelView`,
  `defaultModelInfo()`, `handleDefaultModelCard`.
- `internal/api/default_model_card_test.go` (new) — render and save tests.
- `internal/api/server.go` — route `r.Get("/default-model",
  s.handleDefaultModelCard)` under `/api/settings`.
- `internal/api/settings.go` — accept `default_model_on`,
  `default_model_name` and `default_model_pin` in both the JSON and the form
  update paths; return them from `handleGetSettings`.
- `web/templates/models.html` — a `default` badge on the target model's card.
- `internal/api/models.go` — pass the target ID to the Models page view.
- `internal/api/fork_instances_ui.go` and
  `web/templates/partials/fork_instances.html` — a `default` badge on the
  target fork instance row.
- `web/templates/help.html` — `<h3 id="default-model">` under Settings.
- `docs/guides/settings.md` — `## Default model { #default-model }`.
- `docs/guides/models.md` — new section `## default badge { #default-badge }`.
- `docs/guides/forks.md` — one sentence on the badge on a fork instance row.

## Steps

1. `internal/api/default_model_card.go`:

   ```go
   type defaultModelView struct {
       On      bool
       Name    string
       Pin     string           // saved pin, "" for none
       Options []defaultPinOption // enabled chat models, then fork instances
       Target  string           // current target Name, "" for none
       Reason  string           // "pinned", "loaded" or "last used"
       Clash   string           // non-empty: why the name is off
       Error   string           // a refused save, shown in the card
   }
   type defaultPinOption struct{ Value, Label string }
   ```

   `defaultModelInfo()` calls `clearStaleDefaultPin()` when a pin is set, then
   builds the view from `cfgCopy()`, `resolveDefaultModel()` and
   `defaultNameClash()`. Options list
   every model with `Kind() == KindChat` that passes `advertisedToClients`
   (value = registry ID, label = public name), then, when forks are on, every
   fork instance (value = `fork:` + ID, label = `ID (fork)`), each group
   sorted by label. While forks are off, a saved fork pin is still listed,
   as one option labelled `ID (fork, forks are off)`, so the select shows it
   and a save of another field sends it back unchanged. The first option is "No pin (loaded model used last)"
   with value `""`.
2. `handleDefaultModelCard` renders the partial, as `handleEvictionCard`
   does.
3. `settings.go`: in the JSON update struct add `DefaultModelOn *bool`,
   `DefaultModelName *string`, `DefaultModelPin *string`; in the form path
   read `default_model_on` (`"on"`), `default_model_name` and
   `default_model_pin` when the form carries the card's hidden marker field
   `default_model_form=1` (the card is its own form, like the Eviction radios).
   The name is trimmed first. An empty name is saved as `""`, which means
   `default`. A non-empty name is saved trimmed. Validation, in this order,
   before anything is saved:
   - `config.ValidDefaultModelName(name)`, only for a non-empty name.
   - The name, or `default` when the name is empty, must not clash, by the
     same rule as `defaultNameClash` (Phase 03).
   - A non-empty pin must be one of the card's option values.
   A refusal re-renders the card with `Error` set and HTTP 200 (HTMX swap),
   or answers 400 with the message on the JSON path. `On` is stored as
   `Off = !on`.
4. The card template: a switch "Answer requests to the default model name", a
   text field for the name (placeholder `default`), a select for the pin,
   and one status line. By `Reason`: `pinned` → "Now: <target> (pinned)";
   `loaded` → "Now: <target> (loaded, used last)"; `last used` → "Now:
   <target> (used last; loads on the next request)". With no target: "Now:
   no model. Pin one, or send a request to a model by its name first." When `Clash` is set,
   a warning line says the name is off because a model or fork instance has
   the same name. Tooltips follow the repo's short ASD-STE100 style. Include a
   "More in the docs" link with `{{docsURL "guides/settings#default-model"}}`.
   The form uses `hx-put="/api/settings"`, `hx-trigger="change"`, and swaps
   the whole card (`hx-target="#default-model"`, `hx-swap="outerHTML"`).
5. Models page: add `DefaultTargetID string` to the page view, set from
   `resolveDefaultModel()` when `!t.Fork`, and show a small `default` badge
   next to that model's name, with the tooltip "Requests to "<name>" go to
   this model." The fork instance row gets the same badge when `t.Fork` and
   the ID matches.
6. `help.html`: under `<h2 id="settings">`, add `<h3 id="default-model">Default
   model name</h3>` with two or three sentences and a "More in the docs" link
   built with `docsURL "guides/settings#default-model"`.
7. `docs/guides/settings.md`: `## Default model { #default-model }` with the
   card fields, the resolution order, what clears a pin, the clash rule, and
   a link to `../api.md#default-model`. `docs/guides/models.md`: add
   `## default badge { #default-badge }`, two sentences on what the badge
   means and a link to `settings.md#default-model`. `docs/guides/forks.md`:
   one sentence that the same badge marks a fork instance.

## Build gate

```bash
go build ./... && go vet ./... && go test ./internal/api/... && make js-test && make docs
```

## Test plan

- Render: the card shows the pin options in order, the status line for each
  reason, and the clash warning.
- Save: valid name and pin are saved; a name with a space, a name equal to a
  model alias, and an unknown pin are each refused with the card's error line
  and nothing saved.
- Turning the switch off sets `default_model.off: true`, and `/v1/models`
  drops the entry.
- A pin to a model that is then deleted shows "No pin" after the next card
  load.
- Models page: the badge sits on the resolved model and moves when the pin
  changes; a fork pin puts it on the fork instance row instead.
- Manual browser check on a local run: change the pin, send a chat request to
  `default` with curl, see the pinned model answer and the badge move.
- Help link and docs anchor resolve (`docs_links_test.go`).

## Commit

`feat(ui): add the Default model card and badge`

## Rollback

Revert the commit. The feature keeps working from config (Phases 01–04); only
the card, badge and docs pages go away.
