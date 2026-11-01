# Field test — set details, copy and edit (#515)

**Date:** 2026-11-01 · **Host:** asmodean (1× Radeon AI PRO R9700, 32 GB; container install) · **Build:** main `b11ae00` (Phases 01–07 merged) · **sd.cpp:** `sd-master-1015-e41fbcf-rocm`

The flow ran through the UI where the plan asks for it (copy and save), and through the same HTTP routes the UI uses for the rest. Before the test, `qwen-image-edit-2511` and `real-esrgan` were running; both run again after it. Every test set and instance was deleted, and the shared model files stayed.

The comparison numbers are from the custom set test of the same files on the same host earlier that day, before any phase of this plan.

| # | Check | Result | Notes |
|---|---|---|---|
| 1 | Details of `z-image-turbo`, `wan22-i2v-a14b-lightning` (downloaded) and `ace-step-turbo` (not downloaded) | pass | 3 and 6 files present; ACE-Step turbo shows only its DiT as missing (the other three files are shared). Every flag and every toolchest flag has a reason. |
| 1a | Details command against a running process | pass | `real-esrgan`: the command with `<listen address>`, `<port>` and `<instance folder>` filled in is identical to the process's `/proc` command line. |
| 1b | Details JSON of a set with no flags | **fail → fixed** | `ace-step-turbo` gave `"flags": null`. Now `[]` (fix and test in this PR). |
| 2 | Copy `z-image-turbo` on the Download Models tab, save unchanged, measure, one 1024×1024 image | pass | Flags and 3 files filled in; the save showed "ft zimage is added." Peak **7.4 GiB** (before: 7.3), reservation **8.5 GiB** (same), image in **21.8 s** (before: 22 s). |
| 3 | Copy `wan22-i2v-a14b-lightning` on the Models tab, save unchanged, measure Short, one Short video from a start image | pass | All 6 files filled in, both LoRAs included. Peak **25.6 GiB** (before: 24.6; Real-ESRGAN, about 1 GiB, was loaded this time and Measure reads the whole GPU). Video in **217 s** (before: 217 s), frames sharp, so the LoRAs applied. |
| 4 | Custom upscaler from scratch, factor 4 | pass | Measure 0.8 GiB, reservation 1.0 GiB; 1024² → **4096²**; `/v1/models` shows `scale: 4`. |
| 5 | `/v1/models` of the copied video set | pass | `max_frames 81`, `max_size 1280x720`, `size_multiple 16`, capabilities `start_image`, `end_image`. |
| 6 | Edit the copied image set's `--steps` while its instance runs; stop; edit again | pass | Running: HTTP 409 "stop ft-zimage first: a set cannot change while its instance runs". Stopped: saved, `--steps 12`, instance `measured_stale: true`, reservation kept at 8.5 GiB. |
| 7 | Host path in a container | pass | "… does not exist. The models folder is /data/models. Toolchest runs in a container and cannot see host folders. Put the file under the models folder, or mount its folder there." |
| 8 | JSON file in a model slot | pass | Refused at save: "… is not a model file (GGUF, safetensors or PyTorch)". |
| 8a | Real model file in the wrong slot (the ESRGAN file as the diffusion model) | pass | Passes the format check, as expected. Measure fails with "the instance did not start: sd-server ft-wrong failed to start: exit status 1. **Last log line: [E] get sd version from file failed: ''**". |

Not run here, as planned: deleting a catalog set to test that a custom set's file is kept when it is reached by another path. That would delete a catalog set the live install uses; the Phase 01 unit tests cover a symlink and a `..` path.

Earlier phase checks on the same host, in the phase PRs: the Details dialog at 375 px width (#522); file suggestions, the in-dialog error notice and the htmx target bug (#523); copy from both tabs (#524); the stale mark for a set with no flags (#525).
