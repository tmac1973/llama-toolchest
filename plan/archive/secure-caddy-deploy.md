# Plan: Optional Secure Deploy (Caddy reverse proxy) + Scriptable `setup.sh`

> Offer a one-command **secure** container install that puts the llama-toolchest
> UI and API behind **HTTPS + a single admin login**, using Caddy as a reverse
> proxy in a second container. No Go code changes. Along the way, give
> `setup.sh` a proper **non-interactive mode** and an **interactive
> host-vs-container prompt**, so every install path is both guided *and*
> scriptable.

---

## 1. Motivation

llama-toolchest is reachable today with **no authentication** on the management
UI or `/api/*`, and the only auth that exists (`api_key` on `/v1/*`) is sent in
**plaintext over HTTP**. A few external users are now running it, so we want to
*offer* (not force) a hardened deployment for anyone exposing it beyond a
trusted LAN. The registry data and serving already work; this is purely a
deployment/packaging concern.

Constraints from the maintainer:
- **No Go changes.** Solve it entirely in compose + Caddy + `setup.sh` + docs.
- **Low effort to ship and to use** — easy with *or* without Caddy.
- Must work **interactively and non-interactively** (flags / env), and we must
  not regress the existing plain install.

---

## 2. Current state (audit)

### 2.1 Security surfaces
| Surface | Port | Auth today |
|---|---|---|
| Management UI (`/`, `/server`, `/models`, `/settings`…) | 3000 | none |
| Management API (`/api/*` — delete models, start/stop, settings) | 3000 | none |
| OpenAI proxy (`/v1/*`) | 3000 | optional Bearer `api_key`, plaintext |
| llama.cpp router (own chat UI + raw inference) | 8080 | none |

Two non-obvious facts that shape the design:
- **8080 bypasses the `api_key`.** The token only guards 3000's `/v1`; the raw
  router on 8080 is published and unauthenticated, so a client can reach
  inference without ever touching the token.
- **The "Open Chat UI →" link points straight at 8080.** Built in
  `internal/api/server.go:560-565` as `scheme://host:LlamaPort`, where `scheme`
  and `host` come from `ExternalURL`. `apiURL` (`server.go:459`) is likewise
  `ExternalURL + "/v1"`. So **both rendered links derive their scheme from
  `ExternalURL`** — the reason they show `http` today is simply that
  `ExternalURL` is an `http://…` value. `LlamaPort` is dual-purpose: it's both
  the port in that external link *and* the in-container port the app uses to
  reach the router (`proxy.go:35`, `settings.go:119`), so it can't be
  repurposed to point the link elsewhere without Go changes.

### 2.2 `setup.sh` interactivity vs. flags
- **Flags are boolean-only.** The parser (`main()`, ~setup.sh:1507-1538) is a
  `case` loop of bare toggles; nothing ever `shift`s a value. There is **no
  `--flag value` / `--flag=value` support**.
- **Partial env-var inputs** exist (`INSTALL_MODE`, `HOST_INSTALL_MODE`, `GPU`,
  `LT_VERSION`, `LLAMA_TOOLCHEST_MODELS_DIR`).
- **No deliberate non-interactive mode.** `prompt_confirm` / `prompt_ports` /
  `prompt_models_dir` call `read` unconditionally; there is no `--yes`/`-y`, no
  `ASSUME_YES`, and no `[ -t 0 ]` TTY guard. EOF behavior is accidental:
  `prompt_confirm` is usually called as `if ! prompt_confirm …`, and in that
  condition context `set -e` is suppressed so an EOF read falls through to the
  default and auto-accepts — but `prompt_ports`' bare `read` loops run with
  `set -e` active and **hard-abort on EOF** in the port-conflict path. Net:
  scriptable installs work by luck in the happy path and crash otherwise.
- **No interactive mode selection.** `install` defaults to `container`
  (`INSTALL_MODE="${INSTALL_MODE:-container}"`, setup.sh:36) and only switches
  to host via `--host`/`--from-source`/`--cuda`/etc. There is no prompt.
- **No args → help.** `command="${1:-help}"` (setup.sh:1501) → prints usage and
  exits. Bare `./setup.sh` does not start an install.
- **Single compose chokepoint.** `compose_cmd()` (setup.sh:878) already stacks
  `-f docker-compose.models.yml` conditionally — the exact seam for adding a
  secure overlay.

---

## 3. Goals / non-goals

**Goals**
1. A second-container **Caddy** secure deploy: HTTPS + single admin Basic-Auth
   account, selectable as **self-signed (internal CA)** or **Let's Encrypt**.
2. Close the 8080 bypass and keep the in-app chat/API links working over TLS,
   with **zero Go changes**.
3. A **general non-interactive foundation** for `setup.sh` (`--yes`, TTY guard,
   `--flag value` parsing) that benefits every install, not just the secure one.
4. An **interactive host-vs-container prompt** so host mode is discoverable
   without knowing `--host`.
5. Plain (non-secure) install behavior is **byte-for-byte unchanged** by default.
6. Secure mode works under **both compose and Podman Quadlet (systemd)**:
   `up`/`down`/`enable`/`disable` and boot-autostart manage *both* containers.

**Non-goals**
- No RBAC / multi-user / per-user keys — exactly one admin account.
- No Go-side auth, sessions, or login page.
- No change to the default no-args behavior (still prints help).

Note: secure mode is supported under **both** compose and Podman Quadlet, so
`up`/`down`/`enable`/boot-autostart behave identically to plain installs — see §6A.

---

## 4. Architecture — the secure overlay

Caddy becomes the **only LAN-facing container**. The app publishes nothing to
the LAN (loopback-only, for host-local debugging). Caddy reaches the app over
the compose network by service name and terminates TLS at the edge; the
Caddy↔app hop is plain HTTP but never leaves the host.

```
                       ┌──────────── Caddy (TLS + Basic Auth) ───────────┐
LAN ── 443  ──────────►│  / , /api/*   → llama-toolchest:3000  (gated)    │
LAN ── 8080 ──────────►│  chat UI      → llama-toolchest:8080  (gated)    │
LAN ── 443/v1 ────────►│  /v1/*        → llama-toolchest:3000  (Bearer)   │
LAN ── 80   ──────────►│  → 308 redirect to 443                          │
                       └──────────────── internal compose net ───────────┘
   llama-toolchest 3000/8080 bound to 127.0.0.1 only (not LAN-reachable)
```

### 4.1 Port handling
Compose **appends** `ports` lists across overlay files, so an overlay can't
*remove* a published port. Therefore the base files are parameterized with a
bind-interface prefix:

```yaml
# docker-compose.{cuda,rocm,cpu}.yml
ports:
  - "${LLAMA_TOOLCHEST_BIND:-}${LLAMA_TOOLCHEST_PORT:-3000}:3000"
  - "${LLAMA_TOOLCHEST_BIND:-}${LLAMA_TOOLCHEST_INFERENCE_PORT:-8080}:8080"
```

- Default `LLAMA_TOOLCHEST_BIND` is empty → `"3000:3000"` exactly as today
  (**no behavior change** for plain installs).
- Secure `.env` sets `LLAMA_TOOLCHEST_BIND=127.0.0.1:` → `"127.0.0.1:3000:3000"`,
  binding loopback-only. Caddy still reaches the app via the compose network.

### 4.2 Chat link + API endpoint over TLS
The secure `.env` sets `ExternalURL=https://<domain-or-host>`. Because both
links derive their scheme from `ExternalURL`:
- `apiURL`  → `https://host/v1`     → Caddy:443 → app:3000 `/v1`
- `chatURL` → `https://host:8080`   → Caddy:8080 → app:8080 (router chat UI)

Caddy listens on **8080 too** (TLS + auth), so the existing `https://host:8080`
link resolves to Caddy instead of the raw router — the chat link works
unchanged, now authenticated. **Invariant:** `ExternalURL` must match the Caddy
front (`https://…`); the installer owns this, users shouldn't hand-edit it.

### 4.3 Auth model
- **UI + `/api/*`**: Caddy `basic_auth` (single account, bcrypt hash).
- **`/v1/*`**: **exempted** from `basic_auth` (OpenAI clients send
  `Authorization: Bearer`, which collides with Basic). Relies on the app's
  existing optional `api_key`, now over TLS.
- **Consequence:** anyone who pointed a client at `:8080/v1` directly now hits
  Basic-Auth. Intended — `docs/secure.md` directs programmatic clients to the
  443 `/v1` endpoint.

---

## 5. Caddyfile design

Templated via env placeholders so no secrets are baked into the file:

```caddyfile
{
    # ACME email only used in Let's Encrypt mode; harmless otherwise.
    email {$ACME_EMAIL}
}

# Site address is the domain (LE) or :443 (self-signed/internal CA).
{$CADDY_SITE_ADDRESS} {
    @v1 path /v1 /v1/*
    handle @v1 {
        reverse_proxy llama-toolchest:3000
    }
    handle {
        basic_auth {
            {$CADDY_AUTH_USER} {$CADDY_AUTH_HASH}
        }
        reverse_proxy llama-toolchest:3000
    }
}

# Router chat UI, behind the same login.
{$CADDY_CHAT_ADDRESS} {
    basic_auth {
        {$CADDY_AUTH_USER} {$CADDY_AUTH_HASH}
    }
    reverse_proxy llama-toolchest:8080
}
```

- **Self-signed / internal CA:** `CADDY_SITE_ADDRESS=:443`,
  `CADDY_CHAT_ADDRESS=:8080`. Caddy serves its internal-CA cert; browser shows a
  trust warning until the root is imported. Works on any IP/LAN, zero config.
- **Let's Encrypt:** `CADDY_SITE_ADDRESS=<domain>`,
  `CADDY_CHAT_ADDRESS=<domain>:8080`, `ACME_EMAIL` set. Requires the domain's
  public DNS → this host and reachable ACME challenge (HTTP-01 on 80 or TLS-ALPN
  on 443). The same cert covers the `:8080` site.

`docker-compose.secure.yml` (sketch):
```yaml
services:
  caddy:
    image: caddy:2
    container_name: llama-toolchest-caddy
    depends_on: [llama-toolchest]
    ports:
      - "80:80"
      - "443:443"
      - "${LLAMA_TOOLCHEST_INFERENCE_PORT:-8080}:8080"
    environment:
      CADDY_SITE_ADDRESS: "${CADDY_SITE_ADDRESS}"
      CADDY_CHAT_ADDRESS: "${CADDY_CHAT_ADDRESS}"
      CADDY_AUTH_USER: "${CADDY_AUTH_USER}"
      CADDY_AUTH_HASH: "${CADDY_AUTH_HASH}"
      ACME_EMAIL: "${ACME_EMAIL:-}"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro,z
      - caddy-data:/data:z      # persisted certs / ACME account
      - caddy-config:/config:z
    restart: unless-stopped
volumes:
  caddy-data:
  caddy-config:
```

---

## 6. `setup.sh` changes

### 6.1 Non-interactive foundation (general)
- Add `--yes` / `-y` and `ASSUME_YES=1`.
- Add a TTY guard: compute `INTERACTIVE` from `[ -t 0 ]` (and `ASSUME_YES`).
  When **not** interactive and `--yes` not given for a flow that needs input,
  **fail fast** with a message listing the required flags — instead of today's
  luck-of-the-draw EOF behavior.
- `prompt_confirm`: return success immediately when `ASSUME_YES`.
- `prompt_ports` / `prompt_models_dir` / all new prompts: when non-interactive,
  use the value already supplied via flag/env and skip `read`.

### 6.2 Value-bearing flag parsing
Extend the `main()` `case` loop to consume `--flag value` and `--flag=value`
(a small helper that detects `=` or shifts the next arg). Needed for §6.4.

### 6.3 Interactive host-vs-container prompt (NEW)
For the `install` command, when **mode was not set explicitly**
(`INSTALL_MODE_EXPLICIT=false`, i.e. no `--host`/`--container`/`--from-*`/
`--cuda`/`--rocm`/`--vulkan` and no `INSTALL_MODE` env) **and** interactive,
prompt before the host short-circuit (~setup.sh:1630):

```
Install mode:
  1) Container (Docker/Podman) — recommended, isolated     [default]
  2) Host (install directly on this machine)
Choose [1]:
```

- Choosing host sets `INSTALL_MODE=host` and continues into the existing
  host flow (which already prompts for SDK backends).
- Non-interactive without a mode flag keeps the current default (`container`);
  no new required flag.
- Only `install` prompts. `up/down/logs/rebuild/quick` keep auto-detecting the
  existing install.
- No-args behavior is unchanged (still prints help).

### 6.4 Secure flags + flow
| Flag | Env | Meaning |
|---|---|---|
| `--secure` / `--no-secure` | `SECURE=1` | enable/disable the Caddy stack |
| `--tls self-signed\|letsencrypt` | `TLS_MODE` | cert strategy (default self-signed) |
| `--domain <fqdn>` | `DOMAIN` | required for Let's Encrypt |
| `--acme-email <email>` | `ACME_EMAIL` | LE account registration |
| `--auth-user <name>` | `AUTH_USER` | admin username (default `admin`) |
| `--auth-hash <bcrypt>` | — | precomputed hash (preferred non-interactive) |
| `--auth-pass-file <path>` | `AUTH_PASS` | plaintext source, hashed during install |

Interactive flow (slots in after `prompt_models_dir`, only when not already
answered by flags), container mode only:
```
Enable access control + HTTPS (Caddy reverse proxy)? [y/N]
  └─ TLS: 1) self-signed (LAN, browser warning)  2) Let's Encrypt (public domain)
        └─ if 2: Domain? ___    ACME email? ___
  └─ Admin username [admin]: ___
  └─ Admin password: ***   (confirm) ***
```

### 6.5 Secret handling (decided: hash/file/env, no `--auth-pass`)
Never accept the plaintext password on argv (leaks via `ps`/history).
Precedence:
1. `--auth-hash` — precomputed bcrypt; nothing secret in argv. Best for CI.
2. `--auth-pass-file` / `AUTH_PASS` env — plaintext stays out of argv; hashed
   during install.
3. Interactive `read -rs` (confirm twice).

Hashing uses the Caddy image itself so there's no host dependency:
`$CONTAINER_CMD run --rm caddy:2 caddy hash-password --plaintext "$pw"`.
Only the resulting bcrypt hash is written to `.env` (`CADDY_AUTH_HASH`).

### 6.6 Wiring
- A `SECURE` global makes `compose_cmd()` append `-f docker-compose.secure.yml`.
- Secure mode works under both compose and Quadlet (§6A); no path is disabled.
- `write_env_file()` additions when secure: `LLAMA_TOOLCHEST_BIND=127.0.0.1:`,
  `ExternalURL=https://<domain-or-host>`, `CADDY_SITE_ADDRESS`,
  `CADDY_CHAT_ADDRESS`, `CADDY_AUTH_USER`, `CADDY_AUTH_HASH`, `ACME_EMAIL`.
- Final summary prints the `https://…` URLs and (self-signed) the
  trust-the-root hint.

## 6A. Dual-container Quadlet (systemd) support

The single-container Quadlet path today (`generate_quadlet`, setup.sh:1053; one
`.container` named by the hardcoded `PODMAN_SERVICE_NAME`) is the rootless-Podman
autostart mechanism — `up`/`down`/`logs`/`enable`/`disable`/`uninstall` all key
off that one service name. Secure mode needs the same systemd integration for
**two** containers, so these paths become **set-aware**.

### 6A.1 Unit layout (rootless Podman, secure mode)
Generate three units instead of one, mirroring the compose topology so a single
`Caddyfile` serves both:
- `llama-toolchest.network` — shared Podman network providing name-based DNS.
- `llama-toolchest.container` — the app. `Network=llama-toolchest.network`, GPU
  args + data volume as today, but **no host `PublishPort`** (Caddy fronts it).
- `llama-toolchest-caddy.container` — Caddy. `Network=llama-toolchest.network`,
  `PublishPort=443/80/${INFERENCE_PORT}`, the `CADDY_*`/`ACME_EMAIL` env, and
  `Caddyfile` + `caddy-data`/`caddy-config` volume mounts. Ordered after the app:
  `After=llama-toolchest.service` + `Requires=llama-toolchest.service`.

**Why `.network` + two `.container`, not a `.pod`:** a pod shares one network
namespace, and both the router and Caddy's chat site bind **8080** — they would
collide in a shared namespace. Separate containers on a network (as in compose)
avoid the collision and let Caddy reach `llama-toolchest:3000` / `:8080` by name,
so the **same Caddyfile works for compose and Quadlet** (service-name DNS).

### 6A.2 Boot autostart
Each `.container` unit carries `[Install] WantedBy=default.target`, so the
Quadlet generator auto-activates **both** at boot; `Requires=`/`After=` orders
Caddy after the app. Rootless still needs `loginctl enable-linger` (already
handled). Docker / rootful-Podman don't use Quadlet — autostart is the
`restart: unless-stopped` policy, which both the base app service and the
secure overlay's `caddy` service already declare; `enable`/`disable` updates the
policy on **both** containers.

### 6A.3 Set-aware command paths
Introduce a `quadlet_units()` helper returning the unit basenames for the current
install — `["llama-toolchest"]` plain, or
`["llama-toolchest", "llama-toolchest-caddy"]` (+ the `.network`) secure — and
refactor the single-name call sites to iterate it:
- `container_up`: start app → caddy (or start caddy and let `Requires=` pull the
  app); `container_down`: stop caddy → app.
- `has_quadlet`: unchanged trigger (app `.container` present); a `secure_quadlet`
  check additionally detects the caddy unit.
- `autostart_enable`/`disable` (rootless): write/remove the whole unit set +
  `daemon-reload`; boot activation stays automatic via `WantedBy=`.
- `container_uninstall`: remove all units + the `.network` + (optionally) the
  `caddy-data`/`caddy-config` volumes.
- `container_logs`: `journalctl --user -u llama-toolchest -u llama-toolchest-caddy`.

This keeps `up`/`down`/`enable`/`disable`/boot-autostart behaving identically
whether the install is plain (one unit) or secure (two units + network).

---

## 7. File inventory

**New**
- `docker-compose.secure.yml` — Caddy overlay.
- `Caddyfile` — env-templated proxy + auth + TLS.
- `docs/secure.md` — deploy guide.
- `plan/archive/secure-caddy-deploy.md` — this document.

**Edited**
- `docker-compose.cuda.yml`, `docker-compose.rocm.yml`, `docker-compose.cpu.yml`
  — add `${LLAMA_TOOLCHEST_BIND:-}` prefix to published ports (no-op by default).
- `setup.sh` — §6 + §6A (non-interactive foundation, value-flag parsing,
  host/container prompt, secure flow, compose wiring, env writing, usage text,
  and the set-aware Quadlet refactor: `quadlet_units()` helper + multi-unit
  generation/up/down/enable/disable/uninstall/logs).
- `README.md` — pointer to `docs/secure.md`.

*(No Go files touched.)*

---

## 8. Behavior matrix

| Scenario | Result |
|---|---|
| `./setup.sh` (no args) | prints help (unchanged) |
| `install`, interactive, no mode flag | **prompts** host vs container (NEW) |
| `install` (plain, container) | identical to today; `BIND` empty, no Caddy |
| `install --secure` interactive | prompts TLS mode, domain/email, user, password |
| `install --secure …flags… --yes` | fully non-interactive |
| non-interactive, missing required flag | **fails fast** listing the needed flag (NEW) |
| secure + Quadlet (rootless Podman) | network + app + caddy units; `up`/`down`/`enable` manage both |
| secure + Docker / rootful Podman | both containers `unless-stopped`; `enable`/`disable` updates both |
| `up`/`down` in secure mode | starts/stops **both** containers (compose or Quadlet) |
| boot autostart in secure mode | both containers start at boot, Caddy ordered after the app |
| API client at `:8080/v1` in secure mode | now behind Basic-Auth → use `443 /v1` |

---

## 9. Example invocations

```bash
# Guided (now asks host vs container, then optionally secure):
./setup.sh install

# Plain container, fully scripted:
./setup.sh install --container --yes

# Secure, self-signed, scripted (precomputed hash — no secret in argv):
./setup.sh install --secure --tls self-signed \
  --auth-user admin --auth-hash "$(caddy hash-password --plaintext 's3cret')" --yes

# Secure, Let's Encrypt, scripted (password via file, hashed during install):
AUTH_PASS="$(cat ~/secret)" ./setup.sh install --secure --tls letsencrypt \
  --domain llm.example.com --acme-email me@example.com --auth-user admin --yes
```

---

## 10. `docs/secure.md` outline
1. What the secure deploy gives you (TLS + single admin login; what's protected).
2. Quick start — self-signed (LAN), one command; trusting the internal CA root.
3. Quick start — Let's Encrypt (domain prerequisites, ports 80/443 reachable).
4. Interactive vs non-interactive (flag/env reference table).
5. Changing the username/password (re-run, or regenerate hash + edit `.env`).
6. Client guidance — UI via 443; programmatic via `https://host/v1` + Bearer.
7. Limitations — compose-only (no Quadlet); 8080 now authenticated.
8. Troubleshooting — cert warnings, ACME failures, port conflicts.

---

## 11. Acceptance checklist
- [ ] `docker-compose.secure.yml` adds Caddy; app ports not LAN-published in secure mode.
- [ ] Base compose files take `${LLAMA_TOOLCHEST_BIND:-}` with **no default-behavior change**.
- [ ] `Caddyfile` gates UI/`/api`, exempts `/v1`, fronts 8080 chat, supports both TLS modes.
- [ ] Chat link + API endpoint resolve over TLS via Caddy (`ExternalURL=https://…`).
- [ ] `setup.sh`: `--yes`/TTY guard; `--flag value`/`--flag=value` parsing.
- [ ] `setup.sh`: interactive host-vs-container prompt on `install`.
- [ ] `setup.sh`: secure flags + interactive flow; secrets never on argv.
- [ ] Password hashed via the Caddy image; only the bcrypt hash hits `.env`.
- [ ] Dual-container Quadlet: `.network` + app + caddy units, ordered + boot-enabled (§6A).
- [ ] `up`/`down`/`logs`/`enable`/`disable`/`uninstall` are set-aware (manage both containers).
- [ ] Docker / rootful-Podman: `enable`/`disable` toggles restart policy on both containers.
- [ ] Plain install verified unchanged (diff the rendered compose + `.env`, single Quadlet unit).
- [ ] `docs/secure.md` written; README points to it; `usage()` documents new flags.

---

## 12. Out of scope / future
- Go-side native login (single port, single cert, drop the `:8080` site). The
  cleaner long-term shape, but deliberately deferred — this plan needs no Go.
- Per-client API keys / rotating credentials.
- mTLS / SSO / OIDC.
