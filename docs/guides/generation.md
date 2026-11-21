# Image, video and music generation

Image and video generation (`sd-server` of stable-diffusion.cpp) and music (`ace-server` of acestep.cpp) can run next to llama-server. They are off until you turn them on.

1. **Turn on** *Generation backends* in Settings (`/settings`).
2. **Build** stable-diffusion.cpp and/or acestep.cpp on Builds (`/builds`) (ROCm, CUDA, or Vulkan in host installs).
3. **Get a model set.** A set is the files one model needs, and the settings it runs with.
    - **Download one** with *Get generation models* on Download Models (`/models/browse?open=generation`). Files go in the `gen-models` folder of the models folder.
    - **See how a set runs:** press *Details* on a catalog set or on a set on Models (`/models`). It shows each file, each setting with the reason for it, and the command that toolchest starts. Values in angle brackets, such as `<port>`, are set for each instance.
    - **Use your own files:** press *Add a custom set* on Models (`/models`). Choose the backend and the recipe. The recipe list comes from stable-diffusion.cpp's own recipe files (see [Where recipes come from](#recipes)). It has three groups: the recipes the catalog uses, the other recipes, and the recipes llama-toolchest cannot run. These last ones are greyed, and each shows why, for example a file or an input that llama-toolchest cannot pass to sd-server. Under the list, the form shows the chosen recipe's description from stable-diffusion.cpp, when it has one. The form then shows one field for each file the recipe needs. Type part of a file name to see matching files in the models folder. Toolchest checks that each file exists and is a model file. It never deletes the files of a custom set. A new set's extra launch arguments and test size start from the recipe's own defaults, such as its steps, CFG scale and sampler. sd-server uses a recipe's defaults only when it loads the recipe itself, and llama-toolchest gives it the files one by one, so the form writes them out as flags. Change or clear them as you need. When you choose another recipe, the fields you did not change take that recipe's defaults, and the ones you changed stay. A copy keeps the settings of the set it was copied from.
    - **Start from a known set:** in *Details*, press *Copy as custom set*. The form opens with the recipe, settings and sizes of that set. Files that are on this machine are filled in. For a file that is not, the field shows the name of the catalog file. Change what you need, then save. Use this for a different quantization, a fine-tune, or files you already have from ComfyUI.
    - **Change a custom set:** press *Edit*. Its name, backend and recipe stay; to change them, copy the set. Stop its instance first. After you change its files or settings, measure the instance again.
    - **Files in a container:** the models folder is `/data/models` inside the container. Toolchest cannot see other host folders. Put the files under the models folder, or mount their folder under it.
4. **Create an instance**: on Models (`/models`), under *Generation models*, press *New instance* and choose a build and GPU. It shows on Server (`/server`).
5. **Measure it.** One test generation records the peak VRAM. The reservation is that plus 10%, rounded up to 512 MiB. *Start* on an unmeasured instance offers to measure first. [Generation reservations](../gpu/reservations.md#measure) explains video sizes, edits, budgets and when Measure stops.
6. **Start it**, or turn on auto-start. Change it with *Edit* while it is stopped. Changes other than auto-start clear the reservation.

## Where recipes come from { #recipes }

A recipe is stable-diffusion.cpp's description of a model: which files it needs, and its default settings. Each stable-diffusion.cpp build keeps the recipe files of the commit it was built from. The custom set form uses the recipes of the newest build. When there is no build yet, or the newest one is older than this feature, it uses a copy built into llama-toolchest. So a new model that stable-diffusion.cpp adds can be used after the next build, without a llama-toolchest update.

An instance starts only on a build that knows its set's recipe. If you run a set on an older build that does not have the recipe, the start is refused and says so; choose a newer build for the instance. A set whose recipe the current list does not know is kept. *Details* shows a warning, and *Copy* and *Edit* still open it.

## GPU memory { #gpu-memory }

Generation backends share the GPUs with the LLMs. These pages explain how:

- [Generation reservations](../gpu/reservations.md): the room that each instance keeps free, and how Measure sets it.
- [LLMs and generation backends on one GPU](../gpu/taking-turns.md): the job queue, and how LLMs and backends take turns when both do not fit.
- [Measured reservations](../reference/measured-reservations.md): what each catalog set needed on two test cards.
