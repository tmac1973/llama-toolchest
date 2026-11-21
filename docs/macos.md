# macOS { #macos }

llama-toolchest runs natively on Apple Silicon Macs (M1 and newer). It builds llama.cpp, stable-diffusion.cpp and acestep.cpp with Metal, so chat, image and music models all run on the GPU. There is no container on a Mac: the program runs as a launchd agent of your user and starts when you log in.

Working macOS support starts with release 2.63.0. Older macOS downloads start, but their llama.cpp builds cannot run.

## What works and what doesn't { #support }

| Feature | On Apple Silicon |
|---|---|
| llama.cpp builds | Metal and CPU |
| Image, video and music (sd.cpp, acestep.cpp) | Metal |
| Autoconfigure, load checks, eviction | Yes, against the Mac's GPU memory limit |
| Curated llama.cpp forks | No: they need ROCm or CUDA |
| Container and secure (Caddy) installs | No: Linux only |

Intel Macs are not supported, and there is no release download for them.

## Install with setup.sh { #install }

You need the Xcode Command Line Tools (the compiler) and [Homebrew](https://brew.sh). The setup script checks both, and offers to install `cmake` and `ninja` with Homebrew when they are missing.

```bash
git clone https://github.com/tmac1973/llama-toolchest.git
cd llama-toolchest
./setup.sh install
```

`./setup.sh` sees that it runs on a Mac and uses the macOS setup. It downloads the latest release, checks it against the release checksums, installs it as `~/.local/bin/llama-toolchest`, and starts it. When it finishes, open `http://localhost:3000`, build llama.cpp with the **metal** profile on Builds, and carry on with [Getting started](getting-started.md).

Run `./setup.sh install` again at any time to update to the newest release. Options:

- `--port 3100` uses another port for the web UI and API on a new install.
- `LT_VERSION=2.63.0 ./setup.sh install` installs a given release.
- `-y` answers yes to every question, for scripted installs.

If the Command Line Tools are missing, the script opens Apple's installer. Run it again once that is done.

## Starting, stopping and logs { #service }

```bash
./setup.sh status     # what is installed and running
./setup.sh down       # stop llama-toolchest and the models it runs
./setup.sh up         # start it again
./setup.sh restart
./setup.sh logs       # follow the log
```

| What | Where |
|---|---|
| Program | `~/.local/bin/llama-toolchest` |
| Config file | `~/Library/Application Support/llama-toolchest/llama-toolchest.yaml` |
| Models, builds, settings | `~/Library/Application Support/llama-toolchest/` |
| Log | `~/Library/Logs/llama-toolchest/llama-toolchest.log` |
| launchd agent | `~/Library/LaunchAgents/io.github.tmac1973.llama-toolchest.plist` |

The agent starts toolchest with Homebrew's `bin` folder on its `PATH`, so builds find `cmake` and `ninja` even though launchd does not read your shell profile. If you install Homebrew somewhere unusual, run `./setup.sh install` again to pick up the new path.

## Uninstall { #uninstall }

```bash
./setup.sh uninstall
```

This stops toolchest and removes the program and the launchd agent. It then asks whether to delete your models, builds and settings too; the answer defaults to keeping them, and `-y` never deletes them.

## Manual install { #manual-install }

If you'd rather not use the script, download `llama-toolchest_<VERSION>_macos_arm64.tar.gz` from the [releases page](https://github.com/tmac1973/llama-toolchest/releases) and unpack it. Install the build tools first:

```bash
xcode-select --install
brew install cmake ninja
```

The binary is not notarized by Apple. A file downloaded in a browser carries a quarantine flag, and Gatekeeper refuses to run it ("cannot be opened because the developer cannot be verified"). Remove the flag, then start it:

```bash
xattr -d com.apple.quarantine ./llama-toolchest
./llama-toolchest
```

Start it from a terminal where `cmake` and `ninja` are on the `PATH`. A copy downloaded with `curl`, as `setup.sh` does, has no quarantine flag.

## Memory on a Mac { #memory }

A Mac has no separate graphics memory: the CPU and the GPU share the RAM. macOS lets the GPU use only part of it, about two thirds of the RAM on a 16 GB Mac and about three quarters above that. llama.cpp reports that limit as the GPU's memory, and toolchest shows and plans with the same figure (on a 24 GB Mac, 17.8 GiB).

Because it is one pool, Autoconfigure counts the part of a model on the GPU and the part kept in system memory together against the RAM, less a safety margin for macOS and your other programs. A model that would fit a PC with a separate GPU of the same size can be *Too large* on a Mac with the same RAM. That is correct: the PC has the GPU's memory on top of its RAM.

To let the GPU use more of the RAM, raise the limit (it resets at reboot) and restart toolchest:

```bash
sudo sysctl iogpu.wired_limit_mb=20480    # 20 GiB on a 24 GB Mac
./setup.sh restart
```

Leave macOS a few GiB: if the GPU takes too much, the Mac starts to swap and everything slows down.

### When the GPU's share is full { #cpu-share }

When a model does not fit in the GPU's share, Autoconfigure lets the CPU run part of it. Nothing is copied to other memory; the CPU simply does that part of the work. For a mixture-of-experts model it moves only expert weights, which costs the least speed, and the Recommended list and the Fit column on Download Models say **Experts on the CPU**. A dense model gets some of its layers run on the CPU instead, which is much slower. Raising the limit above helps in both cases.

When the CPU runs part of a model, Autoconfigure gives it one thread per performance core. The efficiency cores are left out, because work split over both kinds waits for the slower cores. Autotune measures half the performance cores, all of them, and every core.
