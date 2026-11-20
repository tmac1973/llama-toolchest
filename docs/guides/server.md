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

Average generation and prompt speed per model, taken from real chat requests to llama-server and to [fork instances](forks.md). A fork instance has its own row, under its instance name. The figures are kept in memory, so a restart starts them again.

- **Cache** shows how much of the prompt llama-server took from its prompt cache instead of processing it again, for example `92% · 14/15`. The percentage is the part of all prompt tokens that came from the cache. The count is how many requests reused the cache at all, out of the requests that reported it. The first request of a conversation always misses, which is why the count can be lower than the number of requests.
- A chat client sends the whole conversation on every turn, and llama-server reuses the part it already processed. So on a model that serves conversations, later turns should come almost entirely from the cache. **A low percentage means the prompt changes near the start** from turn to turn: a date or time in the system prompt, a client that rewrites or shortens the history, or tools sent in a different order. Each turn then processes the whole conversation again. To find what changes, open [Live traffic](#live-traffic) and use *Show rendered prompt* on two turns in a row.
- An engine that does not report its cache use shows `—`.
- **Reset** clears the figures of one model. Use it after you change a setting, to see only the requests that came after the change.

## Server Logs { #server-logs }

Live output from llama-server. To change the height, drag the bottom edge. Double-click the edge for the default height.

## Live traffic { #live-traffic }

*Live traffic*, in the Server Logs header, opens a window that shows the requests clients send to the API and the responses as they arrive. It is for troubleshooting: you see the request as the engine gets it, and what the engine sends back, token by token.

- **What it records.** Every POST that goes through toolchest's API to llama-server (the router) or to a [fork instance](forks.md): chat, completions, embeddings, rerank, speech and the rest. Model lists and other GETs are left out. The request shown is the one sent on to the engine, after toolchest added the model's sampling defaults and system prompt.
- **When it records.** Only while the window is open. With the window closed, toolchest records nothing, keeps nothing and writes nothing to disk. Requests from before you opened it do not show.
- **Parallel requests.** Each request has its own row, newest first, with its model, its engine, its path and its state (waiting, streaming, or the status and time). Responses never mix: select a row to see that one request. *Model* and *Engine* filter the list. *Engine* tells the router apart from each fork instance, so you can watch, for example, only the Avalon instance. *Pause* keeps the list as it is while you read; the responses of the requests in it still come in. *Clear* empties the list. The window keeps the last 50 requests.
- **What a request shows.** The request body, the answer put together from the streamed pieces (the reasoning, the text and any tool calls, each in its own block, and why the engine stopped (for example `length` when it reached `max_tokens`)), and the raw response. *Copy* copies either one.
- **Show rendered prompt.** For a chat request, this asks the same engine for the prompt its chat template makes of the request, with llama-server's `/apply-template`. It shows how the engine sees the messages, which helps with chat template problems. The model must be loaded, or the fork instance running. The engine renders the request again when you press the button, so if the template changed since, the result can differ from what it used. An engine without `/apply-template` says so.
- **What it leaves out.** Headers are never recorded, so API keys never show. Base64 data, such as images and audio, is shortened, and bodies and responses are cut at 1 MiB; a note says so. Multipart requests (audio files, image edits) show only the model they name.
- **What it cannot see.** Requests sent straight to an engine's own port, which do not go through toolchest, and `/v1/realtime` voice sessions. Generation requests (images, video, music) are not shown either.

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
