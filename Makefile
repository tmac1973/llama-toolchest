.PHONY: build agent run dev start stop restart clean \
       docker docker-rebuild docker-compose-up docker-compose-down docker-compose-logs \
       package package-snapshot scrape-sampling-presets

PID_FILE = bin/llama-toolchest.pid

# Auto-detect GPU vendor (override with: make docker-rebuild GPU=cuda)
GPU ?= $(shell ./setup.sh detect 2>/dev/null || echo "rocm")
COMPOSE_FILE = docker-compose.$(GPU).yml

# Local development
build:
	go build -o bin/llama-toolchest ./cmd/llama-toolchest
	go build -o bin/agent ./cmd/agent

agent:
	go build -o bin/agent ./cmd/agent

run: build
	./bin/llama-toolchest --config config.yaml

dev:
	go run ./cmd/llama-toolchest --config config.yaml

start: build
	@echo "Starting llama-toolchest..."
	@./bin/llama-toolchest --config config.yaml & echo $$! > $(PID_FILE)
	@echo "PID $$(cat $(PID_FILE)) written to $(PID_FILE)"

stop:
	@if [ -f $(PID_FILE) ]; then \
		kill $$(cat $(PID_FILE)) 2>/dev/null && echo "Stopped PID $$(cat $(PID_FILE))" || true; \
		rm -f $(PID_FILE); \
	fi
	@PID=$$(lsof -ti:3000 2>/dev/null) && kill $$PID 2>/dev/null && echo "Killed process $$PID on :3000" || true

restart: stop start

clean: stop
	rm -rf bin/

# Container builds (vendor-aware)
docker:
	docker compose -f $(COMPOSE_FILE) build

docker-rebuild:
	docker compose -f $(COMPOSE_FILE) down
	docker compose -f $(COMPOSE_FILE) build --no-cache
	docker compose -f $(COMPOSE_FILE) up -d

docker-compose-up:
	docker compose -f $(COMPOSE_FILE) up -d

docker-compose-down:
	docker compose -f $(COMPOSE_FILE) down

docker-compose-logs:
	docker compose -f $(COMPOSE_FILE) logs -f

# Release packaging via GoReleaser. `package-snapshot` builds dist/ artifacts
# from the current commit without publishing — used by the dev container
# rebuild flow and for verifying the release config before tagging.
# `package` is reserved for the CI workflow (it expects a clean tag);
# locally you almost always want package-snapshot.
package-snapshot:
	goreleaser release --snapshot --clean --skip=publish

package:
	goreleaser release --clean

# Refresh internal/models/sampling_presets_data.go by scraping HuggingFace.
# The generated file is committed — run this manually when you want to pull
# in new presets (e.g. after a popular model release).
scrape-sampling-presets:
	go run ./scripts/scrape-sampling-presets

# Runs the benchmark job form's parameter-control JavaScript against a
# DOM stub. Part of `go test ./...`, but skipped when node is missing —
# this target fails loudly instead, so CI can't quietly lose the
# coverage.
.PHONY: js-test
js-test:
	@command -v node >/dev/null || { echo "node is required for js-test"; exit 1; }
	go test ./internal/api/ -run TestParameterControlsJS -v
