# How Generation Sets Run, and Better Custom Sets — Project Overview

## Problem

Generation model sets are a black box. The catalog shows a name, a size, a licence and a tooltip. Users cannot see which file goes in which slot, which flags sd-server or ace-server gets, or why. The reasons for the flags (for example, why almost every set has `--backend te=cpu`, or why LTX-2.3 uses `--temporal-tiling` and not `--vae-tiling`) are only in Go comments in `internal/generation/catalog.go`.

The custom set form ("Add a custom set from files on disk", `web/templates/partials/gen_model_sets.html`) works on real hardware for all four kinds. A test on asmodean on 2026-11-01 made an image set, a video set with LoRAs, a music set and an upscaler set, and all four ran. But the form is hard to use: the files go in one free-text box as `slot = /path` lines, and the slot names are only in a tooltip. A set cannot be edited, only deleted and added again. The test also found five gaps:

- A host path in a container gives a raw `stat ... no such file or directory` error.
- Wrong files are accepted. The failure shows only at Measure, as `exit status 1`.
- Custom sets give less information in `/v1/models`: no `scale` for an upscaler, and no `max_frames`, `max_size` or `size_multiple` for a video set.
- When a catalog set is deleted, its files are compared with custom set files as path text, so a custom set that reaches the same file by a different path can lose it.
- There is no route to edit a set.

Tracked in issue #515.

## Goals

- A "How this set runs" dialog for every set on this machine and every catalog entry. It shows the recipe, each slot and its file, the flags the set runs with (`ModelSet.RunArgs` for a set; the entry's flags for a catalog entry), a short reason for each flag, and the full command line. The parts that toolchest adds for each instance are shown as markers, for example `<port>`.
- The same data as JSON for API clients, on the same routes, for requests that are not htmx requests.
- A reason for each flag of each catalog entry, as data in `internal/generation/catalog.go`, not as Go comments only.
- "Copy as custom set" from any set or catalog entry. It opens the custom set form with the recipe, flags, test sizes and limits filled in. The files are filled in when they are on this machine; otherwise each file field shows the expected file name as a hint.
- A custom set form with one text field for each slot of the chosen recipe. Each field shows the slot's label and whether it is required. A search list suggests matching model files under the models folder. Recipes are filtered by the chosen backend.
- Edit a custom set: its files, flags, test sizes, limits, upscale factor and the base DiT checkbox. The name and ID do not change.
- A file format check when a custom set is added or edited: each file must be GGUF, safetensors, a PyTorch `.pth` file, or JSON for a tokenizer.
- When any generation instance fails to start or measure, the error shows the backend's last error line from its log, not only `exit status 1`.
- A clear message for a path that does not exist: it says where the models folder is, for example `/data/models` in a container.
- Custom sets give the same `/v1/models` information as catalog sets: `size_multiple` from the recipe; optional max frames, max width and max height for video sets; an upscale factor for upscaler sets.
- Deleting a catalog set never deletes a file that a custom set uses, also when the custom set reaches it by a different path.

## Non-goals

- Editing catalog sets. A catalog set always runs with the current catalog flags. To change one, the user copies it.
- Editing recipes, or adding new recipes in the UI.
- Reading recipes from the stable-diffusion.cpp checkout's `examples/server/recipes/*.json`. The Go recipe data stays the source. This is a later issue.
- LoRAs for image sets. This is a later issue.
- Renaming a set, or changing its ID.
- A folder browser dialog.
- Searching for files outside the models folder, or a setting for extra search folders.
- Checking that a file is the right model for its slot (for example, a VAE in the `llm` slot). The format check only checks the file type.

## Users & primary flow

The user runs llama-toolchest with generation turned on. Many users run it in a container. Some already have model files on disk, for example from ComfyUI.

**Look at how a set runs**

1. On the Download Models tab, the user opens the generation catalog and clicks "Details" on "FLUX.3 Image Klein".
2. A dialog shows the recipe `flux3-klein` and a table of slots: `diffusion_model`, `llm` and `vae`, with the file each one gets and where it will be saved.
3. The dialog shows a table of flags, each with a short reason, for example: `--backend te=cpu` — "Runs the text model on the CPU. This saves about 5 GiB of VRAM and adds a second or two."
4. The dialog shows the full sd-server command, with `<port>` and `<vram limit>` where toolchest adds values for each instance.

The Models tab has the same "Details" button on each row of "On this machine".

**Make a custom set from files already on disk**

1. In the Details dialog of a catalog entry, the user clicks "Copy as custom set".
2. The custom set form opens in a dialog on the same tab. The backend, recipe, flags, test sizes and video limits are filled in. Each slot has its own field. Each empty field shows the catalog file name as a hint.
3. The user types in a slot field. A list shows matching files under the models folder. The user picks one for each slot, or types a full path.
4. The user gives the set a name and clicks "Save". On the Download Models tab, a note then links to the Models tab. Toolchest checks the slots, that the files exist, and the file formats. A wrong path says where the models folder is.
5. The user makes an instance and measures it, as for a catalog set.

**Edit a custom set**

1. On the Models tab, the user clicks "Edit" on a custom set.
2. The same form opens with the set's values. The name, backend and recipe cannot be changed.
3. If an instance of the set is running or measuring, saving is refused with "stop <instance> first".
4. Otherwise the set is saved. If the files or flags changed, the instance shows its measurement as stale. The reservation stays until the user measures again.

## Constraints

- Go backend in `internal/`; HTML templates with htmx in `web/templates/`. Use the same patterns as the existing generation code (`internal/generation/`, `internal/api/gen_*.go`).
- Routes that serve both HTML and JSON give HTML to htmx requests (`HX-Request` header) and JSON to all others, as `/api/gen/models` does.
- Container and host installs must both work. In a container, the models folder is `/data/models`, and toolchest cannot see host paths outside the mounts.
- The models folder can hold many large files. The file search must not read file contents, and it must answer quickly.
- A custom set's ID is used by API clients, so it cannot change.
- Sets saved before this work must still load and work without changes.
- All UI text and tooltips follow the project's writing rules: short, ASD-STE100 style, no slang.

## Success criteria

- Every catalog entry and every set on this machine has a Details dialog. It shows the recipe, slots and files, flags with reasons, and a command line that matches what an instance runs, except for the marked parts.
- Every flag of every catalog entry has a reason. A test fails when a catalog flag has no reason.
- The Details routes return JSON to non-htmx requests.
- "Copy as custom set" from a downloaded catalog set, then "Add set" with no changes, gives a set that measures and generates like the catalog set. A test on asmodean shows this for one image set and one video set.
- The custom set form shows one field per slot of the chosen recipe, offers only recipes for the chosen backend, and suggests files from the models folder.
- A custom set can be edited. Editing is refused while one of its instances runs or measures. After a change to the files or flags, the instance's measurement shows as stale.
- A non-model file is refused when the set is added. A start or Measure failure shows the backend's last error line.
- A missing path in a container gives a message that names `/data/models`.
- A custom upscaler set shows `scale`, and a custom video set shows `size_multiple`, and `max_frames` and `max_size` when they are set.
- A test shows that deleting a catalog set keeps a file that a custom set uses through a different path.
- `go build`, `go vet`, `go test ./...` and the JS tests pass in CI.

## Decisions

- **Where the Details dialog opens** → on each set in "On this machine" (Models tab) and on each catalog entry (Download Models tab), so users can look before a download.
- **How flag reasons are shown** → one short reason per flag, stored as data with each catalog entry's flags.
- **Command line** → the full command with real file paths; the parts that toolchest adds per instance are shown as markers such as `<port>`. For a catalog entry that is not downloaded, the paths are where the files will be saved.
- **What can be copied** → any set on this machine and any catalog entry. Files that are not on this machine become hints in empty fields.
- **What can be edited** → custom sets only: files, flags, test sizes, limits, upscale factor and the base DiT checkbox. Not the name, ID, backend or recipe. Catalog sets are copied, not edited.
- **How files are chosen** → one text field per slot, with a search list of model files under the models folder; typed full paths still work.
- **Search area** → the models folder only. The help text says how to make other files visible there.
- **Wrong files** → a file format check at add and edit, and the backend's last error line in start and Measure errors for all instances.
- **Missing information for custom sets** → `size_multiple` from the recipe data; optional max frames, width and height, and an upscale factor (default 4) as form fields, filled in on copy.
- **JSON API** → yes: the Details routes give JSON to non-htmx requests, as the other gen model routes do.
- **Editing a set that has instances** → refused while an instance runs or measures; when stopped, the edit saves and the measurement shows as stale; the reservation stays.
