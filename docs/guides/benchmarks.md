# Benchmarks

A **job** runs every combination of the chosen models, builds and presets. Each result records the build flags, settings and GPU state, so it stays useful after you change things. A job never changes your saved model settings. A job (or an Autotune run) started while the server is still starting, as just after the app restarts, waits up to two minutes for it to come up instead of failing at once.

## Speed presets { #speed-presets }

| Preset | Prompts (tokens) | Generated | Runs | Time |
| --- | --- | --- | --- | --- |
| `internal-quick` | 256 | 128 | 1 | ~10 s |
| `internal-standard` | 128 / 512 / 2048 | 128 | 3 | ~1 min |
| `internal-thorough` | 128 / 512 / 2048 / 8192 | 256 | 5 | ~5 min |
| `internal-long-ctx` | 32K | 512 | 1 |  |
| `internal-long-ctx-thorough` | 8K / 16K / 32K / 64K | 512 | 3 | ~20 min |
| `benchy-quick`, `benchy-standard` | Run through [llama-benchy](https://github.com/eugr/llama-benchy) |  |  |  |

Prompt sizes are approximate; the real count (`prompt_tokens`) is usually 10–25% lower. Results show **PP t/s** (prompt reading speed), **TG t/s** (generation speed) and **TTFT** (time to first token).

## Parameters, sweeps and profiles { #parameters-sweeps-and-profiles }

- Each parameter uses the setting of the model by default. One value applies to the whole job. Several values make a sweep, with one result per value. Parameters marked **↻** reload the model for each value; the preview shows the number of reloads.
- Saved profiles show under each model as start points. Each checked profile is measured on its own, exactly as saved, so the parameters turn off. The job keeps a copy of each profile. ⚠ marks a profile saved on a build that the job does not include.
- **Memory used**: each result records the VRAM that its load used (needs *Model loading detail* 4). It shows the cost of a faster setting, such as a larger micro-batch.
- **Actions**: *Detail*, *Use*, *Compare* (pick rows, across jobs), *Export* (CSV or JSON) and *Retry failed*.
- Results marked *⚠ unverified* are older than a fix to parameter overrides. Run them again if the settings matter.

## Using a result's settings { #use-a-result }

**Use** on a finished result saves the settings it was measured with as a profile of its model. Each result records the whole config it ran under when it ran. A profile saved from it therefore has the same sampling, chat template and other settings that the cell had, even if you changed the model since then.

The dialog suggests a name from the sweep point or the profile the cell started from; type over it to change it. It warns when the name replaces a saved profile, and lists the settings that differ from the model's current ones.

- **Save** adds the profile only. Apply it later from the model's *Configure* panel.
- **Save and apply** also makes it the model's settings. A loaded model uses them the next time it loads.

The profile is marked *(benchmark)*, with a note naming the job and what the result measured. Results recorded before this feature existed did not keep their whole config, so *Use* is turned off on them. Run the cell again to save it.

## Capability benchmarks { #capability-benchmarks }

These score how *good* a model is, not how fast. They load the model directly, so **the server stops while one runs** and starts again after it. Sampling settings have no effect. The KV cache stays at full precision unless the job sets `kv_cache_quant`. Test data downloads at first use.

| Test | Measures | Use it to | Reading the score | Quick / full |
| --- | --- | --- | --- | --- |
| **perplexity** | How well it predicts Wikipedia text | Check a file works, or a setting change | Lower is better. Only comparable within one model. | 2–5 min / 15–40 min |
| **kl-divergence** | How far a smaller quant drifts from a larger one | Choose which size of a model to keep | 0 means identical. Needs two quants of one model; the reference is measured once and stored (can be tens of GB). | 2–5 min / 15–40 min |
| **hellaswag** | Common sense: pick the right sentence ending | Compare different models | % correct. Guessing ~25%, people ~95.6%. | 1–5 min / 1–3 h |
| **winogrande** | What "it" or "they" refers to | A second opinion beside HellaSwag | % correct. Guessing ~50%, people ~94%. | 1–3 min / 5–15 min |

The number after **±** is the margin of error. If the ranges of two results overlap, the test found no real difference. Use `-full` for numbers you keep. *How to read this score* on a finished run shows good and bad values.
