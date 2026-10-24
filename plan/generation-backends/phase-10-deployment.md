# Phase 10 — Deployment: Images, Ports, Quadlet and Caddy

**Depends on:** Phase 02 (port range settings and `GenPublicPortStart`),
Phase 03 (sd.cpp web UI option needs `pnpm`), Phase 05 (instances to start
in the tests), Phase 08 (ACE-Step compatibility routes at the root), Phase 09
(dashboard web UI links used in the tests) · **Enables:** Phase 11
(container-mode verification)

## Goal

Make the feature work in both install modes.

**Container images** gain:
- `nodejs` and `pnpm`, so sd.cpp can build its web UI
- the published port range (default 8081–8088) for backend web UIs

**`setup.sh`** asks for the start of the range, the same way it asks for the
chat UI port. Quadlet units publish the range.

**Caddy secure mode** gets:
- one block per web UI port, with basic auth, like the chat block
- the ACE-Step compatibility routes, which use Bearer auth only, like `/v1`

Vulkan stays host-mode only, so the images do not change for Vulkan.

## Files touched

- `Dockerfile.rocm`, `Dockerfile.cuda`, `Dockerfile.rocm-next`. `Dockerfile.cpu`
  is unchanged, because generation backends need a GPU profile (Phase 03):
  - Install `nodejs` and `pnpm` (`npm install -g pnpm@9.15.4`; see step 1).
  - `EXPOSE 8081-8088`.
  - Add `gen_port_start: 8081` and `gen_port_count: 8` to the default yaml.
- `docker-compose.gen-ports.yml` (new): an override, used only when
  `LLAMA_TOOLCHEST_GEN_PORTS=1` (step 6). It:
  - publishes
    `${LLAMA_TOOLCHEST_BIND:-}${LLAMA_TOOLCHEST_GEN_HOST_PORT_START:-8081}-${LLAMA_TOOLCHEST_GEN_HOST_PORT_END:-8088}:8081-8088`
    on the app service
  - passes `LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START`, set to the host start (or
    to `CADDY_GEN_PORT_START` in secure mode), and `LLAMA_TOOLCHEST_GEN_PORTS=1`
    into the container environment
  - in secure mode, publishes `${CADDY_GEN_PORT_START:-8081}-…` on the Caddy
    service as well
- `docker-compose.cpu.yml`, `docker-compose.cuda.yml`,
  `docker-compose.rocm.yml`, `docker-compose.rocm-next.yml`,
  `docker-compose.secure.yml`: unchanged.
- `setup.sh`:
  - `LLAMA_TOOLCHEST_GEN_HOST_PORT_START="8081"` next to L22, with
    `LLAMA_TOOLCHEST_GEN_HOST_PORT_END` set to start + 7.
  - These host-side names are different from the in-container settings, so
    `.env` can never change the ports the backends listen on inside the
    container.
  - Quadlet units pass `LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START` as an
    `Environment=` line.
  - Prompt in `prompt_ports` (L809-874) as "Generation web UI ports (8
    ports)".
  - Conflict checks for every port in the range (L737-739, L781).
  - `.env` write (L1145-1154).
  - Quadlet `PublishPort` lines for the range (L1837-1841, L1884-1886).
  - Summary output (L820-827).
- `scripts/lib/host.sh` (L1715) and `scripts/lib/migrate.sh` (L169): write
  `gen_port_start` and `gen_port_count` in host mode and during migration.
- `Caddyfile.template`:
  - A `@@GEN_ADDRESSES@@` block repeated for n = 0…7. Each one listens on
    public port `CADDY_GEN_PORT_START + n` and uses basic auth plus
    `reverse_proxy llama-toolchest:<8081 + n>`. The upstream is always the
    port inside the container, whatever the public port is.
  - In the site block, add `/release_task` and `/query_result` to the matcher
    that already sends `/v1/*` with Bearer auth and without basic auth.
- `docs/secure.md`, `docs/manual-install.md`: document the ports, the Caddy
  blocks, and that host installs need `pnpm` on `PATH` for the sd.cpp web UI.
- `.goreleaser.yaml` (L70-95): add `nodejs` as a recommended (not required)
  package dependency for `.deb` and `.rpm`, so host installs can build the web
  UI.

## Steps

1. **Add nodejs and pnpm to each image** through the image's own package
   manager:
   - Fedora: `dnf install nodejs npm`
   - Ubuntu/Debian: `apt-get install nodejs npm`
   
   Then run `npm install -g pnpm@9.15.4` on each of the three GPU images. Pin
   that exact pnpm version in all three Dockerfiles.
2. **Publish the range in compose** with a single range mapping. Inside the
   container the range is always 8081–8088. The host side follows
   `LLAMA_TOOLCHEST_GEN_HOST_PORT_START`.
3. **Keep the container's port settings fixed.**
   - In container mode, `gen_port_start`/`gen_port_count` stay at 8081/8,
     because that is the inside of the container. Settings shows them
     read-only (Phase 02).
   - Only the host mapping and `GenPublicPortStart`, set through
     `LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START`, follow the setup prompt.
4. **Avoid port clashes in secure mode.** Today the app's loopback debug
   inference port is `SECURE_APP_DEBUG_INF_PORT`, default 8081 (`setup.sh`
   `write_env_file`). That clashes with the range Caddy will publish. In secure
   mode:
   - The default `SECURE_APP_DEBUG_INF_PORT` changes from 8081 to 18080.
   - The app publishes its range on loopback only, at
     `127.0.0.1:18081-18088`, by setting `LLAMA_TOOLCHEST_BIND=127.0.0.1:` and
     `LLAMA_TOOLCHEST_GEN_HOST_PORT_START=18081`.
   - Caddy alone publishes the public range, from `CADDY_GEN_PORT_START`,
     default 8081.
   - `LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START` is set to `CADDY_GEN_PORT_START`,
     so dashboard links point at Caddy.
   - Reconfiguring an existing secure install moves its debug port, and
     `setup.sh` prints that change in its summary.
5. **Render the Caddy blocks.** `setup.sh` renders one block per port in the
   range, using `CADDY_GEN_ADDRESS_n="${DOMAIN}:$((CADDY_GEN_PORT_START + n))"`
   and upstream port `8081 + n`, with the same basic auth credentials as the
   chat block.
6. **Make the port publishing opt-in.** This keeps the overview's promise that
   nothing changes until the feature is turned on.
   - `setup.sh` asks "Publish ports for generation backend web UIs? [y/N]",
     with no as the default. The answer is stored as
     `LLAMA_TOOLCHEST_GEN_PORTS=1` in `.env`.
   - **With no:**
     - no port range is published
     - no Caddy gen blocks are rendered
     - the secure-mode debug port stays at 8081
     - compose and Quadlet output is byte-identical to today
   - **With yes:** steps 2–5 apply.
     - The range publish lives in a separate compose override,
       `docker-compose.gen-ports.yml`, added to the compose file list only
       when the flag is set.
     - Quadlet `PublishPort` lines are written only when the flag is set.
   - The API (`/v1/images/*`, `/v1/music/*`, and the ACE-Step routes on port
     3000) works whatever the answer. Only the backend web UIs need the
     published ports.
   - The Settings page shows "Web UI ports are not published; rerun setup.sh
     and answer yes to open the backend web UIs" when `GenerationEnabled` is
     on and the app runs in a container without the range published.
     Toolchest detects this from `LLAMA_TOOLCHEST_GEN_PORTS` in its own
     environment, which compose passes through.
7. **Ask during setup.** The prompts (the yes/no above, then the range start)
   happen on fresh installs and on `setup.sh` reconfigure runs, so existing
   installs can turn the ports on later or move the range if it clashes.

## Build gate

- `go build ./...`
- `go test ./...`
- `make installer-test` (`scripts/test-lib.sh`)
- `docker build -f Dockerfile.rocm .` and `docker build -f Dockerfile.cuda .`
  succeed.

## Test plan

- **`installer-test`:** add cases for:
  - answering no, which leaves `.env`, the compose file list, the Quadlet
    units and the Caddyfile byte-identical to today
  - answering yes: the range prompt default, a range that collides with 8080
    (rejected), the Quadlet `PublishPort` lines, and the secure-mode debug
    port moving to 18080
- **Manual, container mode on asmodean (ROCm image):**
  - Install with `setup.sh`.
  - Build sd.cpp with the web UI option on, and confirm pnpm is found.
  - Start a Ming-Image instance and open its web UI through the dashboard link
    on the published port.
- **Manual, secure mode:**
  - With `setup.sh --secure`, the web UI port asks for basic auth.
  - `POST /release_task` with the Bearer key works.
  - Without the key it gets 401.
- **Manual, host mode:** the `.rpm` install on asmodean writes the new yaml
  keys.

## Commit

`feat(deploy): publish generation web UI ports and add pnpm to images`

## Rollback

Revert the commit and rerun `setup.sh` to regenerate the compose `.env`,
Quadlet units and Caddyfile without the range. The images built from the
reverted Dockerfiles no longer contain nodejs.
