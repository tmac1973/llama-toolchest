# Custom sets with Hugging Face files — Project Overview

## Problem

llama-toolchest downloads generation models only as curated catalog sets. A model that is not in the catalog can be used only as a custom set made from files that are already on disk. The user must download the files by hand and put them under the models folder. In a container, the user must also mount the folder. Since #528, the custom set form offers every stable-diffusion.cpp recipe (112 in the newest build), but the catalog has files for only a few of them. So most recipes can be chosen, but their files cannot be downloaded from the app.

The parts are nearly there. A `generation.Component` already has `Repo` and `File` fields next to `Path`. `handleGenCatalogDownload` already downloads a set's components with `downloader.StartTo`, tracks them in `genDownloads`, and shows the progress on the set's row. `modelsource.Source.RepoFiles` lists the files of a repo. 104 of the 108 snapshot recipes have a `model_url` that points to the model's Hugging Face repo, but the recipe parser does not read it yet. The gap is the form: a file field takes only a local path.

## Goals

- In the custom set form, each file field can take a Hugging Face file in place of a local path, for every backend (sd, video, ACE-Step, upscale).
- A **Browse** button on a file field opens a dialog with:
  - the recipe's `model_url` repo,
  - the exact files that catalog sets use for the same slot (sd and video: same recipe; ACE-Step and upscale: same backend), and
  - a box for any other repo.
  It lists the repo's files with their sizes.
- The list shows loadable files only (`.safetensors`, `.gguf`, `.ckpt`, `.pt`, `.pth`; `.json` for the `tokenizer` slot). A **Show all files** switch shows the rest. A repo in the Diffusers layout (it has a root `model_index.json`) gets a note that sd-server needs one combined file.
- A file field also takes a pasted Hugging Face file link (`https://huggingface.co/<org>/<repo>/blob/<rev>/<path>` or `/resolve/`).
- The form shows the total download size. **Add set** is refused when the files do not fit in the free space, with the same check catalog downloads use. Files already on disk do not count.
- **Add set** saves the set at once, pins each repo to its current commit, and starts the downloads. The set's row shows progress like a catalog set. Instances refuse the set until all files are there. An incomplete set can be resumed.
- When a file finishes, toolchest runs the same model-file check as for a local file (`generation.CheckFileFormat`). A file that fails the check is deleted and marks the set with an error on its row. Catalog downloads get this check too.
- One set can mix local files and downloaded files.
- Copying a catalog set fills a missing file's field with that catalog file. A file on disk keeps its local path.
- Deleting a custom set deletes the files toolchest downloaded for it, unless another set uses them. Files given as local paths are never deleted.
- The Help page and the docs (`docs/guides/generation.md`) describe the new flow.

## Non-goals

- No ModelScope or other sources. Hugging Face only.
- No image LoRAs (#527 stays set aside).
- No conversion of Diffusers folders into one file, and no download of whole folders.
- No Hugging Face search inside the Browse dialog. The user types or pastes a repo name.
- No check of a remote file's first bytes before saving.
- No new download page for generation files outside the custom set form.
- No other change to how catalog sets download. The only change is the model-file check on each finished file.

## Users & primary flow

The user runs llama-toolchest and wants a generation model that is not in the catalog.

1. On **Models**, the user presses **Add a custom set**, then chooses the backend and the recipe, for example `sd-turbo`.
2. Next to **Diffusion model**, the user presses **Browse**. The dialog opens on the recipe's repo (`stabilityai/sd-turbo`). It lists the loadable files with their sizes.
3. The user picks `sd_turbo.safetensors`. The field now shows the Hugging Face file and its size, in place of a local path.
4. For a slot whose file comes from another repo, Browse offers the files of catalog sets with the same recipe. The user can also type any repo, or paste a file link into the field.
5. The form shows the total download size. The user presses **Add set**.
6. Toolchest saves the set, records each repo's commit, and starts the downloads. The set's row on Models shows the progress.
7. When all files are there and pass the model-file check, the user makes an instance, measures it and uses it, as for any set.

Other flows:

- **Copy a catalog set:** the files that are not on disk get the catalog file. The user can change one of them, for example to a different quant, and add the set.
- **Edit a set:** a file field can change between a local file and a Hugging Face file. A new Hugging Face file downloads after **Save changes**.
- **Delete a set:** the downloaded files that no other set uses are deleted.

## Constraints

- Go, chi, htmx partials, Pico CSS and `go:embed` templates, as in the rest of the app. No new dependencies.
- The form is redrawn through `/api/gen/models/sets/form/fields` with `hx-include="closest form"`. New fields must survive that redraw and the copy and edit forms (`from_set=1`).
- Downloads use the existing `huggingface.Downloader`:
  - Files go to `huggingface.FilePath(generation.ModelsDir(cfg), repo, file)`, the same place as catalog files. So a file that a catalog set or another custom set already has is used again.
  - A gated repo uses the Hugging Face token and the error messages from #657.
- A revision pin needs a new `Revision` field on `generation.Component`. Downloads use the existing `Downloader.StartToAt` (fork models already use it). Sets saved before this change have no revision and keep working.
- The saved set file must stay readable by older releases. New JSON fields are `omitempty`, and nothing is renamed.
- Toolchest can run in a container. The files must land under the models folder, which is mounted.
- The repo file list calls Hugging Face. It must time out and show an error in the dialog, never block the form.
- The repo rules apply: a branch and PR per phase, Conventional Commits with no AI attribution, and docs in the same PR as the change.
- Tooltips and Help use short sentences (about 80% ASD-STE100). Help stays short and links to the docs.

## Success criteria

- On a test machine, the user makes a custom `sd-turbo` set from Hugging Face in the form alone. The file downloads, passes the check, measures, and makes an image.
- A custom set with one local file and one downloaded file works.
- A copy of a catalog set that is not on disk downloads all its files when the user adds it.
- Add set is refused, with the sizes in the message, when the files do not fit on the disk.
- A file that fails the model-file check (for example, an HTML error page or a JSON file in a model slot) shows an error on the set's row, and no instance can start.
- After a restart of toolchest during a download, the set shows as incomplete and can be resumed. The resume uses the pinned commit.
- Deleting the set removes its downloaded files, keeps the files that another set uses, and never removes a local file.
- Sets saved before this change (catalog and custom) load and work as before.
- `go build ./...`, `go vet ./...`, `go test ./...`, the JS tests and the CI checks pass.

## Decisions

- **Where a download starts** → in the custom set form. A file field takes a Hugging Face file, and Add set downloads it like a catalog set.
- **How a file is chosen** → Browse opens the repo's file list. The field also takes a pasted Hugging Face file link.
- **Which repos Browse offers** → the recipe's `model_url`, plus the exact files that catalog sets use for the same slot. A text box takes any other repo.
- **Which files the list shows** → loadable files with sizes, a **Show all files** switch, and a note on Diffusers repos.
- **When the set saves and files are checked** → save at once and start the downloads. Check each file with the model-file check when it finishes. A set can mix local and downloaded files.
- **Delete** → delete the downloaded files unless another set uses them. Never delete local files.
- **Backends** → all of them (sd, video, ACE-Step, upscale). Only sd and video have recipe repos. The others suggest catalog files of the same backend and slot.
- **Sources** → Hugging Face only.
- **Disk space** → show the total download size, and refuse Add set when the files do not fit.
- **Revision** → pin each repo to its commit at add time, and download that commit.
- **Field value** → a Hugging Face link in the same `slot_<slot>` field as a local path.
- **A file that fails the check** → delete it, store the error on the component, and keep the set incomplete.
- **Phases** → five: links and pins; download, check, resume and delete; Browse and size; copies; field test.
- **Copy of a catalog set** → a missing file's field gets the catalog's Hugging Face file. A file on disk keeps its local path.
