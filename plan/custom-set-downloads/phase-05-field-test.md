# Phase 05 — Field test on asmodean

**Depends on:** Phases 01–04 merged to `main` · **Enables:** closing the issue for this work

## Goal

Check the whole feature on a real machine with real Hugging Face downloads, and write the results to `plan/custom-set-downloads/field-test.md`. The machine is asmodean: one R9700, the container on the ROCm 10.1 image, and the models folder `/home/tim/llm` mounted as `/data/models`.

## Files touched

- `plan/custom-set-downloads/field-test.md` (new): a results table, notes, and what is left on the machine.

## Steps

1. **Deploy.** Build `main` and copy the binary into the container with `docker cp` to `/usr/bin/llama-toolchest.new`. Then `mv` it into place and `docker restart`. Keep the running release as `/usr/bin/llama-toolchest.orig`. Record the version string.
2. **Pasted link.** Delete the existing `sd-turbo-test` set and its instance. Make a new `sd-turbo` custom set by pasting `https://huggingface.co/stabilityai/sd-turbo/blob/main/sd_turbo.safetensors`.
   - Check that `/home/tim/llm/gen-test/sd_turbo.safetensors` is not reused, because it is not at the repo path.
   - Check the progress on the row, the pinned commit in `generation.json`, the file check, the measurement, and one image.
3. **Browse.** Make a `flux2-klein-4b` custom set with Browse only:
   - Diffusion model: from the recipe's repo.
   - LLM: from the suggested catalog file.
   - VAE: from the suggested catalog file.
   Note which files were already on disk (the klein catalog set is installed) and were reused without a download. Measure it and make one edit.
4. **Mixed set.** Make a set with one local file (the existing Ming diffusion model path) and one downloaded file. Check that it completes and runs.
5. **Copy.** Copy a catalog set that is not downloaded, swap one file for a different quant with Browse, and add it. Check that only the missing files download. Then delete the set.
6. **Space check.** Run `df -h /home/tim/llm` and note the free space. Paste a link to a single file larger than it (the free space minus the downloader's safety margin is what counts). Check that the size line shows the refusal first, and that Add set is refused with both sizes. When no Hugging Face file is larger than the free space, mark the step "Skipped" with the free space in the notes; the Phase 02 tests cover the refusal.
7. **Wrong file.** Paste a link to a repo's `README.md` in a model slot. Check that the file downloads, is refused, and is deleted, and that the row shows "Incomplete (wrong file)" with the reason. Edit the set to a correct file and Resume.
8. **Restart during a download.** Start a large download (more than 5 GB), restart the container half way, and check that the set shows Incomplete. Press Resume, and check that it resumes the `.part` file from the pinned commit.
9. **Delete.** Delete the sets of steps 2–8, one at a time. Check on disk that:
   - downloaded files are removed,
   - files shared with the klein catalog set stay,
   - the local Ming file stays.
10. **Gated repo.** Browse a gated repo without accepting its terms. Check that the dialog shows the #657 message.
11. **Old sets.** Check that Ming, klein, ti2v and the sd-turbo-style sets from before the deploy still load, start and generate.
12. Write `field-test.md` with a row per step (Pass/Fail and notes), any notes, and a "Left on asmodean" section. Open a PR for it.

## Build gate

```bash
make docs
```

The other gates ran in Phases 01–04. This phase adds only a markdown file.

## Test plan

The steps above are the test. A step that fails gets a fix PR before this write-up is merged, and the step is run again.

## Commit

`docs(plan): field test results for custom set downloads`

## Rollback

On asmodean, move `/usr/bin/llama-toolchest.orig` back into place and restart the container. Revert the write-up PR if needed. It changes no code.
