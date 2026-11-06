# Troubleshooting

## Start does nothing, or "no compiled build available" { #start-does-nothing-or-no-compiled-build-available }

There is no successful build. Build llama.cpp on Builds.

## Model will not load, or llama-server crashes { #model-will-not-load-or-llama-server-crashes }

Check Server Logs. The Server Logs start again with each router start, so for an earlier crash use [Engine notes](guides/server.md#engine-notes): it keeps each crash with its error lines. If you use tensor parallelism, try a layer-split GPU Assignment.

## Model missing from `/v1/models` { #model-missing-from-v1models }

Check its *On* switch, then restart the server.

## Stuck on "loading" { #stuck-on-loading }

It probably failed to start; check Server Logs.

## Out of VRAM { #out-of-vram }

Run Autoconfigure, or lower GPU Layers or Context Size, use KV Cache Quant, or lower Max loaded models. If a load that fits by the estimate still runs out of memory, raise the [GPU memory margin](gpu/memory-margin.md#oom) of that GPU.

## No sampling presets { #no-sampling-presets }

None were found for this model. Try *↻ Refresh presets* later.

## "could not be parsed" or "was written by a newer version" { #store-read-only }

A settings file under `<data>/config` (for example `models.json`, `builds.json` or `benchmarks.json`) could not be read. The page that uses it shows a warning and stops saving to it, so nothing overwrites the file. Fix the file, restore it from a backup, or upgrade llama-toolchest if a newer version wrote it. Then restart llama-toolchest.

llama.cpp: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
