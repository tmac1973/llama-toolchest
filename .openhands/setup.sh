#!/usr/bin/env bash
# Runs every time OpenHands starts work on this repo.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

# The sandbox image ships git/make/gcc/node/python but NOT Go, so install a
# toolchain matching go.mod into the user's home. Roughly 70 MB per fresh
# sandbox; there is no supported way to mount a shared cache into it.
need=$(awk '/^go /{print $2; exit}' go.mod)
if ! command -v go >/dev/null 2>&1; then
  if [ -x "$HOME/.local/go/bin/go" ]; then
    export PATH="$HOME/.local/go/bin:$PATH"
  else
    v=$(curl -s 'https://go.dev/dl/?mode=json' | jq -r '.[0].version')
    echo "installing $v (go.mod requires >= $need)"
    curl -sSL "https://go.dev/dl/${v}.linux-amd64.tar.gz" -o /tmp/go.tgz || exit 1
    mkdir -p "$HOME/.local" && tar -C "$HOME/.local" -xzf /tmp/go.tgz || exit 1
    rm -f /tmp/go.tgz
    export PATH="$HOME/.local/go/bin:$PATH"
  fi
fi
export PATH="$HOME/.local/go/bin:$PATH"
go version || exit 1
go mod download || { echo "go mod download failed"; exit 1; }
command -v node >/dev/null && node -v
echo "note: CGO/gcc present; llama.cpp sidecars are NOT built here - make build only."
exit 0
