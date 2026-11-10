# Autoconfigure and Autotune

A **profile** is a named copy of a model's settings. The profile bar at the top of Configure saves and restores them. Profiles survive re-downloading the model and are included in backups.

|  | Autoconfigure | Autotune |
| --- | --- | --- |
| **Does** | Suggests settings that fit | Measures many settings and keeps the fastest |
| **Sets** | Context size, KV cache type, GPU layers, expert offload, GPU placement, sampling | Batch sizes, flash attention, split mode, threads, speculative decoding |
| **Asks** | How long your conversations or documents are | What you use the model for (speculative decoding pays off most when answers repeat the prompt, as in code editing) |
| **Result** | A reviewed list of settings, each with a reason; *Save as profile* or *Save and apply* | Up to three profiles: fastest generation, fastest prompt reading, fastest full answer |
| **While it runs** | Test-loads the settings (restarts the server, unloads other models) | Restarts the server for each setting; stop any time and continue later |

- **Run Autoconfigure first.** Autotune keeps the context, memory and sampling settings, so it makes a badly set-up model faster, not correct.
- **Test load.** The memory estimate adds all GPUs together, but llama.cpp does not fill them evenly. If the test load runs out of memory, Autoconfigure tries these, in order: fewer layers on the full GPU, a smaller prompt batch, a smaller context. These settings show *test load*.
- **GPUs.** On a machine with more than one GPU, select the GPUs that the model can use. Autoconfigure fits the model only on those GPUs. Clear a GPU to keep it for image, video or music models. Autotune then measures the model on the GPUs that the profile uses.
- **Vision projector.** For a model with a projector, choose *Keep the vision projector in system RAM* to plan without it on the GPU. Its VRAM goes to the model, and images and audio are read on the CPU, more slowly. The box starts with the model's current setting.
- **Helper model.** Autoconfigure reads the model card with a small model of its own (install it on Settings). Without it, settings come only from what fits.
- **Draft models.** For speculative decoding, Autotune tries installed draft models that are related to the model: same publisher as its base model, or same family. A finetune made for something else is left out. If no installed model is related, it tries the smallest ones that load.
- **Noise.** Autotune keeps the simpler setting unless the faster one wins by more than the run-to-run variation. Each stage is a normal benchmark job; you can read it on Benchmarks.
