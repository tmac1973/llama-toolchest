# Getting started

## Install { #install }

On Linux, clone the repository and run the setup script:

```bash
git clone https://github.com/tmac1973/llama-toolchest.git
cd llama-toolchest
./setup.sh install              # default: container install
# or, for a containerless install on the host:
./setup.sh install --host
```

The script detects your GPU and container runtime (Docker or Podman), installs missing prerequisites, shows a summary, then builds and starts llama-toolchest. When it finishes, the web UI is at `http://localhost:3000`.

| Mode | When to use it |
|---|---|
| Container (default) | Most users. The GPU SDKs and the build tools stay inside the container, away from your system. |
| Host (`--host`) | You already have a working GPU driver and want a leaner setup, or you need Vulkan. It installs the released `.deb`/`.rpm` and runs it as a systemd service. |

Other ways to install:

- **macOS:** on an Apple Silicon Mac, the same `./setup.sh install` runs a macOS setup with no container. See [macOS](macos.md).
- **By hand:** install the released packages without `setup.sh`. See [Manual install](manual-install.md).
- **HTTPS and a login:** for a server that others can reach, see [Secure install](secure.md).

The [README](https://github.com/tmac1973/llama-toolchest#installation) lists the other `setup.sh` options: SDK flags, switching between container and host, ROCm versions, and the full command list.

## First steps { #first-steps }

Open `http://localhost:3000`, then:

1. **Build llama.cpp.** On Builds (`/builds`), pick the profile for your hardware (CUDA, ROCm, Vulkan, CPU) and start a build.
2. **Download a model.** On Download Models (`/models/browse`), *Find recommended LLMs* lists models that fit this machine, or search for a GGUF repo and pick a quant (`Q4_K_M` is a good default).
3. **Enable it.** On Models (`/models`), turn on the model's *On* switch.
4. **Autoconfigure it.** Press *Autoconfigure* on the model and say how long your conversations or documents are. Review the settings and press *Save and apply*. It calculates what fits in VRAM and test-loads it. For the advice of the publisher too, install the helper model on Settings (`/settings`) first.
5. **Start the server.** On Server (`/server`), press *Start*.
6. **Autotune it (optional).** Press *Autotune* on the model to measure faster speed settings. Then press *Use these settings* on the result you want. It takes a while, and the server cannot answer other requests while it runs.
7. **Send a request** to the OpenAI-compatible address in the *API Endpoint* card, with the model's public name in the `model` field.
