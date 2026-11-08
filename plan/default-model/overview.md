# Default model name — Project Overview

## Problem

Every client that talks to llama-toolchest (Open WebUI, IDE plugins, scripts,
Haruspex) stores a real model name. When the user switches to a different chat
model, each client must be edited. A client that still names the old model
either loads it again (evicting the new one) or fails if the old model was
removed.

Some simple clients send no `model` field at all. Today the router decides what
happens to those requests, and llama-toolchest does none of its own loading,
counting, sampling defaults or system prompt for them.

## Goals

- Add one virtual model name, `default` unless the user renames it, beside the
  real model names. Real names and their static aliases keep working unchanged.
- Resolve the virtual name on each request to one real chat model or fork
  instance, in this order:
  1. The pinned target, if the user pinned one.
  2. Else the loaded chat model (router model or running fork) that was used
     most recently.
  3. Else the router chat model used most recently, remembered across
     restarts, which is then loaded with the normal room checks.
  4. Else a clear error.
- A pinned router model that is not loaded loads as its real name would. A
  pinned fork that is stopped answers 503, as forks do now.
- Treat a chat request with a missing or empty `model` field the same as the
  virtual name.
- List the virtual name in `/v1/models` as its own entry, first, with meta that
  marks it as virtual and names its current target. Leave it out when there is
  nothing to resolve to.
- Leave the response's `model` field as the real model that answered.
- Give the user a Settings card with an on/off switch, the name, and a pin
  dropdown. Show a badge on the Models page for the model the name points to now.
- On by default after upgrade.

## Non-goals

- No virtual names for embedding, reranker, image, video, music or upscale
  models. They keep their real names only.
- No "best model" ranking by size or score, and no load balancing across models.
- No rewriting of the `model` field in responses or SSE chunks.
- No change to how real names, static aliases or fork names resolve.
- No more than one virtual name.
- No per-client or per-API-key targets.

## Users & primary flow

The user runs llama-toolchest with several chat models and points clients at it.

1. After upgrade, the Settings card shows the virtual name on, named `default`,
   with no pin.
2. The user sets each client's model to `default` once.
3. With no pin, `default` answers with the chat model the user used last. The
   Models page badge shows which one.
4. The user pins a model (router model or fork) in the Settings card. From then
   on every `default` request goes to that model, loading it if needed.
5. To switch, the user changes the pin. No client changes.
6. A client calls `/v1/models`, sees `default` first, and its meta names the
   current target.

## Constraints

- Go server; the proxy in `internal/api/proxy.go` reads the `model` field once
  at the top of the handler. Fork routing, loading, request counting, sampling
  defaults and the system prompt all key off that name, so resolution must
  happen before all of them.
- The proxy handles JSON bodies and multipart forms. Chat requests are JSON.
- Router mode can have several models loaded. Eviction modes (Off, llama.cpp,
  llama-toolchest) and the VRAM room checks must apply to a model loaded
  through the virtual name exactly as to one loaded by its real name.
- Fork instances do not auto-load on request.
- The virtual name must never shadow a real model ID, public name, router name,
  static alias or fork instance name. A clash turns the virtual name off until
  renamed, and Settings refuses a clashing name.
- Settings live in the existing config and must survive backup export/import.
- The Help page entry stays short and links to a docs page with the full text.

## Success criteria

- A chat request to `default` with a pin is served by the pinned model,
  including after it was unloaded.
- With no pin and two loaded chat models, `default` goes to the one used most
  recently; using the other by its real name moves `default` to it.
- After a restart with nothing loaded and no pin, `default` loads the model that
  was used last before the restart.
- A request with no `model` field behaves exactly like one naming `default`.
- `/v1/models` lists `default` first with its target in meta, and hides it when
  there is no target or the feature is off.
- A response to a `default` request names the real model.
- Sampling defaults, system prompt, engine-notes counting and timings use the
  resolved model.
- Setting the name to an existing model name or alias is refused with a message.
- Help and docs describe the feature.

## Decisions

- **Main problem to solve** → A stable name for clients, so changing models needs no client edits.
- **How the name resolves** → Pinned model; else the most recently used loaded chat model.
- **No pin and nothing loaded** → Load the most recently used chat model (remembered across restarts); error if there is no history.
- **Name** → `default`, renameable in Settings; clashing names refused.
- **Kinds covered** → Chat LLMs plus fork instances.
- **Running forks in the unpinned rule** → Yes, they count as loaded, same as router models.
- **Pinned router model not loaded** → Load it as its real name would; a stopped pinned fork answers 503.
- **Response `model` field** → The real model that answered; no rewriting.
- **`/v1/models` listing** → Own entry, listed first, meta marks it virtual and names the target; hidden with no target.
- **Default state** → On by default; a Settings switch turns it off.
- **UI** → Settings card (switch, name, pin) plus a badge on the Models page.
- **Requests with no `model` field** → Resolve like the virtual name.
- **Where the last-used model is kept** → A small state file, `<data>/default-model.json`, written only when the model changes. Router models only: a fork does not start on request.
- **How the proxy applies it** → Rewrite the body's `model` field before fork routing and loading.
- **Paths** → Chat paths only; the virtual name on other paths gets a 400.
- **Pin whose model is gone** → Clear the pin when the model is deleted or disabled, or its fork instance is removed. A fork pin is kept while forks are switched off.
- **Delivery** → A GitHub issue, a plan PR, then one PR per phase.
