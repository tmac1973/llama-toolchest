# Server tab

## Server card { #server-card }

State, the build to run (*Auto* picks the newest), the eviction mode in effect with a link to change it, and Start / Stop / Restart.

## Available Models { #available-models }

!!! note "Since v2.51.0"
    Embedding and reranking models show in their own sections.

Every enabled chat model, with a load icon and a copy icon for its public name. Embedding and reranking models follow in their own sections. Hover a row to compare the VRAM measured at the last load (weights, KV cache and buffers, per GPU) with the estimate. Measured figures need *Model loading detail* 4 or higher in Settings (`/settings`). The use that a card reports is a few hundred MiB higher, for driver overhead.

## API Endpoint and Agent Configs { #api-endpoint-and-agent-configs }

The base URL for clients. *Agent Configs* gives config files for opencode, Kilo Code, pi, Goose and Crush, and shows where each file goes. The files are a snapshot: download them again after you change the served models or their context size. With an API key set, the files read it from `LLAMA_TOOLCHEST_API_KEY`.

## Live Performance { #live-performance }

Average generation and prompt speed per model, taken from real chat requests.

## Server Logs { #server-logs }

Live output from llama-server. To change the height, drag the bottom edge. Double-click the edge for the default height.

## Engine notes { #engine-notes }

The line above the Server Logs counts what llama-server and the generation backends did in the last 24 hours. It turns amber when a model or a generation instance crashed or failed to load, unless a job expected that failure (see below). *Review* shows more: router starts and stops, loads, failed loads, crashes and unloads, a table for each model with its time loaded, average load time, requests and tokens, and a separate table for the generation instances. Choose the period at the top: 24 hours, 7 days, 30 days or all.

- A **crash** is a model that stopped with an error and nobody asked it to stop. An unload, an eviction, or a stop that had to be forced is not a crash. A **failed load** is a model that stopped before it finished loading.
- Each crash shows the time, the program that failed (llama-server, sd-server or ace-server), the model or instance, the cause, the exit status, the build, the GPUs, and the error lines from the log. *What llama-server wrote* (or sd-server, or ace-server) shows the last lines of that program's output.
- The **generation instances** are recorded too: sd-server for images and video, ace-server for music. Each start that passes its health check counts as a load, each stop you or toolchest asked for as an unload, an exit with an error as a crash, and an exit before the health check passes as a failed start. Their table shows starts, failed starts, crashes, stops, time running and average start time. Generation requests are not counted. When the same failure happens several times in a row, it shows once, with the number of times and the time range.
- The **cause** comes from the error lines:

    | Cause | What the log says | What it means |
    | --- | --- | --- |
    | GPU fault | `illegal memory access`, `misaligned address`, `unspecified launch failure`, or a failed GPU operation such as `ggml_cuda_compute_forward: CONCAT failed` with no memory error | A setting does not cause this. A bug in the backend or the driver, or a hardware problem, does. Look at these even when they happen during a job. |
    | Out of VRAM | `out of memory`, `cudaMalloc failed`, `failed to allocate`, `cannot make enough memory available` | A GPU ran out of memory, while the model loaded or later. |
    | Out of system memory | `std::bad_alloc`, `cannot allocate memory`, or exit status 137 | System RAM ran out, or the system stopped the process. |
    | Setting not supported | `GGML_ASSERT(...)`, `not supported`, `unknown argument` | The backend refused the setting or an option in the extra arguments. It fails the same way each time. |
    | Other | none of the above | The error lines do not show the cause. |

    For sd-server, only the lines after its last `retrying with` line count. sd.cpp writes that line when a step ran out of memory and it tries again with less. When it then stops and the later lines show no GPU fault, the cause is out of VRAM. The memory errors before the retry are not the cause, because sd.cpp recovered from them.

    When a generation backend gets an option it does not know, it writes the error and then its whole help text. The help text is left out of the error lines and the output, so the error itself stays visible. For a failed start, all of the output is searched for error lines; for a crash, only the last lines are.

    The cause is read again each time you open Review, so older events get it too.

- Events while a benchmark or Autotune job holds the router show the job's name. The requests of a job are not counted as model usage. A generation instance that Measure started shows the job "Measure".
- A job tries settings to find out which ones work, so some of them fail. Measure tries sizes in the same way. A failure in a job is **expected** when its cause is out of VRAM, out of system memory, or setting not supported. Review counts expected failures and marks them, but they do not turn the line amber. A GPU fault or an unknown failure in a job still does.
- Requests count by the hour, for completions, chat, embeddings and rerank. A request that gets an error answer counts as a failed request.
- The data is kept in `engine-notes.json` in the data folder, so it stays after a restart. It stays until you delete it: use *Delete old data* at the bottom of Review. The same data is at `GET /api/service/notes?period=24h` (or `7d`, `30d`, `all`) as JSON, with the cause of each failure. *Download JSON* at the bottom of Review saves the whole file (`GET /api/service/notes/download`), for example to attach to a bug report. The log lines in it can contain file paths and model names.
- A crash also writes a `WARN` line to the toolchest log.
- Notes from before toolchest recorded the generation instances hold llama-server events only.
