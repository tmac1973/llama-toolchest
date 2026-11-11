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
- **Noise.** Autotune keeps the simpler setting unless the faster one wins by more than the run-to-run variation. [How Autotune picks a winner](#picking) has the full rule. Each stage is a normal benchmark job; you can read it on Benchmarks.

## What the use cases measure { #use-cases }

Autotune asks what you use the model for because the answer changes which settings are fastest. Speculative decoding guesses the next few tokens and has the model check them in one step. When the answer repeats the prompt, as when you ask for an edited copy of a file, most guesses are right and generation can be several times faster. When the answer is new text, few guesses are right and the same setting gains little or nothing.

Each use case runs one or two hidden benchmark presets, called *workloads*:

| Use case | Workloads | Prompt | Answer |
| --- | --- | --- | --- |
| **General chat** | chat (`autotune-chat`) | A passage of prose with a request to respond to it, at 512 and 4096 tokens | 256 tokens of new text |
| **Coding and editing** | coding (`autotune-code`) | A Python file of about 1536 tokens (a different one each run), with a request to rename a method, add type hints and return the whole file | 512 tokens, mostly the file again |
| **Mixed** (the default) | chat and coding | Both of the above | Both of the above |

Every setting runs each workload three times. The coding workload edits a different file each time: n-gram speculative decoding remembers text it has seen between requests, so with the same file every time it would guess the later answers from the first one and make n-gram settings look faster than they are. Every setting edits the same three files, so the comparison stays fair. *Mixed* measures every setting with both workloads, so it takes about twice as long. Choose it when you use the model for both kinds of work, or when you are not sure.

## How each goal is scored { #goals }

Autotune scores every setting on three goals and reads each goal from a fixed prompt size:

| Goal | Chat workload | Coding workload |
| --- | --- | --- |
| **Fastest generation**: tokens written per second | Read on the 512-token prompt | Read on the 1536-token prompt |
| **Fastest prompt**: tokens of input read per second | Read on the 4096-token prompt | Read on the 1536-token prompt |
| **Fastest response**: seconds for a whole answer | A 512-token prompt and a 256-token answer | A 1536-token prompt and a 512-token answer |

Response time is calculated from the measured prompt and generation speeds for that request size.

Each score is the median of the three runs, so one unusually fast or slow run cannot decide the result. The score also records its *spread*: how much the three runs differed.

With *Mixed*, the two workloads are combined into one score per goal:

- **Speeds** (generation and prompt) are the average of the chat and coding speeds.
- **Response time** is the total of one chat answer and one coding answer.

The results show both parts, for example *Average of two workloads: chat 112.1 → 129.5, coding 115.0 → 133.8 tokens per second*. The first number is the starting profile, and the second is the winner. A setting can win *Mixed* because of a large gain on coding alone, so check the parts when one kind of work matters more to you.

## How Autotune picks a winner { #picking }

Autotune measures in stages, in this order: batch sizes and attention, GPU split and CPU threads, expert cache, speculative decoding, the draft settings, and confirming the finalists. A stage that does not apply to the model is skipped. Each stage picks a winner for each goal, and the next stage builds on those winners.

For each goal, a stage picks its winner like this:

1. **Keep only the settings that tie with the fastest.** A setting is out if another setting is faster by more than the *noise*: the two settings' spreads combined. A gap smaller than that might be chance. A setting with a large spread cannot use its own spread to stay close to a steadier setting that measured clearly faster. A setting that is measurably faster than all the others is the only one left, and it wins, however many changes it makes.
2. **If several tie, take the plainest.** When the settings left are all within the noise of each other, none of them is measurably faster, so the one with the fewest changes from the starting profile wins. Each changed batch size, attention, split, thread or draft value counts as one change. Speculative decoding counts as two or three, because it adds a mechanism and sometimes a second model file. It also uses VRAM and makes speed less steady, so it has to earn its place with a measurable gain.
3. **If they are equally plain, take the one with the faster slow runs.** That is the higher score minus its spread. It prefers a steady setting when two measured about the same, and the faster one when the gap is real.

The last stage measures the finalists and the starting profile again, close together, and only those numbers decide the result. Every goal's winner reaches it. That includes a speculative decoding winner the draft-settings stage had nothing to tune, such as speculation turned off, which is often the fastest at reading prompts. The winner must still beat the starting profile by more than the noise. If it does not, Autotune keeps your profile and says that it is already the fastest for that goal. The results also show the closest alternative and its margin of error, so you can see whether the gap was small or just uncertain.

Each goal that has a winner gets a saved profile. When one setting wins two or three goals, it is saved once, as one profile for all of them.
