#!/usr/bin/env bash
# The repo's pre-PR gate, in one place so setup.sh and the stop hook agree.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
go vet ./... && go test -race ./...
