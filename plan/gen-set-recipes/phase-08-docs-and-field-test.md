# Phase 08 — Help text and field test

**Depends on:** Phases 01–07 · **Enables:** closing issue #515

## Goal

Update the help page for the new dialogs and the custom set flow, then test the whole flow on real hardware in a container, the way users run toolchest. The test repeats the 2026-11-01 custom set test through the new UI, and checks each of the five gaps from that test.

## Files touched

- `web/templates/help.html` — the generation section: Details, Copy as custom set, the custom set form, Edit, and how to make files visible in a container.
- `README.md` — one line in the generation part: each set shows how it runs, and custom sets can start from a catalog set.
- `plan/gen-set-recipes/field-test.md` (new) — the results.

## Steps

1. **Help.** In `help.html`, in the generation section near "Download a model set":
   - Replace "You can also add a custom set from your own files." with a short list: open **Details** on any set to see its files, settings and command; **Copy as custom set** to start a custom set from it; the custom set form shows one field per file and suggests files from the models folder; **Edit** changes a custom set when its instance is stopped.
   - Add: in a container, the models folder is `/data/models`; files elsewhere must be put under it, or their folder mounted there.
   - Change the `--backend te=cpu` paragraph to point to the Details dialog for the reasons of each flag; keep its numbers.
   Use the project's writing rules (short sentences, ASD-STE100 style).
2. **README.** Add the one line from Files touched.
3. **Build a test binary** from `main` after Phases 01–07 merge, and deploy it to asmodean with the binary swap in the asmodean notes (build on the ROCm 10 image there; `docker cp` the binary into the `llama-toolchest` container and restart). Ask the user before the deploy.
4. **Field test**, on asmodean, recording each result in `field-test.md`:
   1. Details on `z-image-turbo` (downloaded), `wan22-i2v-a14b-lightning` (downloaded) and `ace-step-turbo` (not downloaded on asmodean): the tables, the "will be saved to" paths for the missing files, and the markers. Start the `qwen-image-2-1-turbo-edit` instance and check that its `ps` command line matches its Details command, except for the markers.
   2. Copy `z-image-turbo` from the Download Models tab; save with no changes; make an instance; Measure; make one 1024×1024 image. Compare with the 2026-11-01 custom set test of the same files: 7.3 GiB peak, 8.5 GiB reservation, about 22 s per image.
   3. Copy `wan22-i2v-a14b-lightning`; save; Measure at Short; make one Short video from a start image. Compare with 2026-11-01: 24.6 GiB peak, about 217 s per video. Check that the frames are sharp (the LoRAs work).
   4. Make a custom upscaler set from scratch with scale 4; `/v1/models` shows `scale: 4`; upscale one image.
   5. The custom video set in `/v1/models` shows `size_multiple`, `max_frames` and `max_size`.
   6. Edit the image set's `--steps` with its instance running: refused. Stop it; edit: saved; the instance shows a stale measurement.
   7. Type a host path (`/home/tim/llm/...`): the message names `/data/models` and the container.
   8. Put a text file in a slot: refused at save. Then make a set with a real but wrong model file (the upscaler file as the diffusion model; it passes the format check); Measure fails with the `get sd version from file failed` line in the error.
5. The delete fix from Phase 01 is not tested on asmodean: the test would delete a catalog set that the live install uses. Its unit tests (a symlink and a `..` path) cover it.
6. Remove every test set and instance, and start again any instance that Measure stopped. Leave asmodean as it was found.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

The field test in step 4. Every item has a pass or fail line in `field-test.md`, with the times and peaks. A fail is fixed in this phase's PR, and the item is run again.

## Commit

`docs(gen): explain set details, copy and edit; add the custom set field test`

## Rollback

Revert the commit. It changes help text and adds a results file only.
