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

The line above the Server Logs counts what llama-server did in the last 24 hours. It turns amber when a model crashed or failed to load. *Review* shows more: router starts and stops, loads, failed loads, crashes and unloads, and a table for each model with its time loaded, average load time, requests and tokens. Choose the period at the top: 24 hours, 7 days, 30 days or all.

- A **crash** is a model that stopped with an error and nobody asked it to stop. An unload, an eviction, or a stop that had to be forced is not a crash. A **failed load** is a model that stopped before it finished loading.
- Each crash shows the time, the model, the exit status, the build, the GPUs, and the error lines from the log. *What llama-server wrote* shows the last lines of that model's output.
- Events while a benchmark or Autotune job holds the router show the job's name. The requests of a job are not counted as model usage.
- Requests count by the hour, for completions, chat, embeddings and rerank. A request that gets an error answer counts as a failed request.
- The data is kept in `engine-notes.json` in the data folder, so it stays after a restart. It stays until you delete it: use *Delete old data* at the bottom of Review. The same data is at `GET /api/service/notes?period=24h` (or `7d`, `30d`, `all`) as JSON.
- A crash also writes a `WARN` line to the toolchest log.
