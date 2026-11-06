# Models

Downloaded GGUFs, grouped by publisher and base model.

## Filter, Collapse / Expand All { #filter-collapse-expand-all }

Search by org, name, quant or architecture.

## On switch { #on-switch }

Enable or disable the model.

## Configure { #configure }

Per-model settings and saved profiles (see [Model config](model-config.md)).

## Autoconfigure, Autotune { #autoconfigure-autotune }

See [below](autoconfigure.md).

## ↻ and ⊘ { #restart-markers }

Changes waiting for a server restart. Click to restart; it asks first, since a restart stops running requests.

## VRAM Est. { #vram-est }

*Base* is the weights; *peak* adds the full KV cache at the configured context.

## Downloads card { #downloads-card }

One row per download: *Pause*, *Resume* or *Discard* (deletes the partial files). Interrupted downloads come back paused after a restart.

## `incomplete`, orphan { #incomplete-orphan }

An unfinished download, or a GGUF missing from disk. Neither can be enabled.

## Update { #update }

Compares the model files with its repository on HuggingFace or ModelScope, for a publisher that uploads a fixed version under the same name. It compares SHA-256 checksums and lists the changed files: the model, its shards, and its projector and MTP head when they are in the same repository. Nothing downloads until you confirm, and then only the changed files. A file from before this check has no checksum, so toolchest reads it once: about a minute per 20 GB. A new file replaces the old one only when it is complete and its checksum matches. A loaded model keeps the old files until the server restarts (↻). To continue a paused update, click Update again.

## Remove { #remove }

*Keep Files* or *Delete Files*.
