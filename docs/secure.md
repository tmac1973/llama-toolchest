# Securing llama-toolchest (HTTPS + admin login)

By default llama-toolchest serves its web UI, management API, and inference
endpoints over **plain HTTP with no authentication**. That's fine on a trusted
LAN or behind a VPN/Tailscale, but if the server is reachable from an untrusted
network you'll want HTTPS and a login in front of it.

The **secure install** option does this by running a [Caddy](https://caddyserver.com)
reverse proxy in a second container: Caddy terminates TLS and enforces a single
admin Basic-Auth account, and llama-toolchest itself is bound to loopback so
nothing but Caddy faces the network.

> ## ⚠️ Best-effort security — please read
>
> The bundled Caddy configuration is provided as a **convenience starting
> point, AS-IS and without warranty**. It has **not** been hardened for any
> specific threat model. **You are responsible for reviewing and auditing it**
> before relying on it — including TLS/cipher policy, the strength of the
> authentication, which ports are exposed, rate limiting, request-size limits,
> logging/PII, and your host firewall and network policy.
>
> The rendered config lives at `./Caddyfile` (generated from
> [`Caddyfile.template`](../Caddyfile.template)). Read it. If this server is
> reachable from an untrusted network, treat the security of the deployment as
> **your** responsibility.

---

## What it protects (and what it doesn't)

| Surface | Without secure install | With secure install |
|---|---|---|
| Web UI (`/`, `/server`, `/settings`, …) | open, HTTP | HTTPS + admin login |
| Management API (`/api/*`) | open, HTTP | HTTPS + admin login |
| OpenAI API (`/v1/*`) | optional Bearer key, HTTP | HTTPS; Bearer `api_key` (no Basic Auth, so clients work) |
| Router chat UI / raw inference (`:8080`) | open, HTTP, bypasses the key | HTTPS + admin login via Caddy; raw port no longer LAN-exposed |
| ACE-Step music API (`/release_task`, `/query_result`) | optional Bearer key, HTTP | HTTPS; Bearer `api_key`, like `/v1/*` |
| Generation backend web UIs (`:8081`–`:8088`, opt-in) | open, HTTP | HTTPS + admin login via Caddy, one site per port |

**Not** addressed (your responsibility): OS/container hardening, firewalling,
keeping images patched, rate limiting, DDoS, securing the host itself, and
anything specific to your environment. One admin account is the only identity —
there is no multi-user/RBAC.

---

## Prerequisites

- A **container** install (Docker or Podman). Secure mode is container-only; it
  can't be combined with `--host`.
- For **Let's Encrypt**: a public DNS name pointing at the host, and inbound
  ports **80 and 443** reachable from the internet.
- For **self-signed**: nothing extra — works on any LAN/IP.
- **Rootless Podman:** binding ports 80/443 requires lowering the kernel's
  unprivileged-port floor. The installer detects this and offers to set
  `net.ipv4.ip_unprivileged_port_start=80` (host-wide, persisted, needs sudo).
  Decline if you'd rather run rootful Podman, or set `CADDY_HTTP_PORT` /
  `CADDY_HTTPS_PORT` to values ≥ 1024 in `.env` (and adjust your URL
  accordingly).

---

## Quick start — self-signed (LAN / homelab)

```bash
./setup.sh install --secure
```

You'll be asked for the TLS mode (pick **self-signed**), an admin username, and
a password. That's it. Caddy serves its own internal-CA certificate.

- Open `https://<host>` — your browser will show a **certificate warning** until
  you trust Caddy's root CA (it's self-signed; that's expected). To remove the
  warning, export Caddy's root and import it into your OS/browser trust store:
  ```bash
  docker exec llama-toolchest-caddy \
    cat /data/caddy/pki/authorities/local/root.crt > caddy-root.crt
  # then import caddy-root.crt into your trust store
  ```

## Quick start — Let's Encrypt (public domain)

```bash
./setup.sh install --secure --tls letsencrypt \
  --domain llm.example.com --acme-email you@example.com
```

Caddy automatically obtains and renews a trusted certificate. Make sure
`llm.example.com` resolves to this host and ports 80/443 are reachable.

---

## Non-interactive / scripted installs

Every prompt can be supplied by a flag or environment variable, and `--yes`
skips confirmations (required when stdin isn't a TTY).

| Flag | Env | Meaning |
|---|---|---|
| `--secure` / `--no-secure` | `SECURE=1` | enable/disable the Caddy proxy |
| `--tls self-signed\|letsencrypt` | `TLS_MODE` | certificate strategy (default `self-signed`) |
| `--domain <fqdn>` | `DOMAIN` | public domain (required for Let's Encrypt) |
| `--acme-email <email>` | `ACME_EMAIL` | ACME account email (recommended for LE) |
| `--auth-user <name>` | `AUTH_USER` | admin username (default `admin`) |
| `--auth-hash <bcrypt>` | — | precomputed bcrypt hash (best for CI) |
| `--auth-pass-file <path>` | `AUTH_PASS` | password source, hashed during install |
| `-y`, `--yes` | `ASSUME_YES=1` | skip confirmations |

**The plaintext password is never accepted on argv** (it would leak via `ps`
and shell history). Provide it as a precomputed hash, a file, or the `AUTH_PASS`
env var.

```bash
# Self-signed, precomputed hash — nothing secret on the command line:
./setup.sh install --secure --tls self-signed \
  --auth-user admin \
  --auth-hash "$(docker run --rm docker.io/library/caddy:2 caddy hash-password --plaintext 's3cret')" \
  --yes

# Let's Encrypt, password from the environment (hashed during install):
AUTH_PASS="$(cat ~/secret)" ./setup.sh install --secure --tls letsencrypt \
  --domain llm.example.com --acme-email you@example.com --auth-user admin --yes
```

---

## Day-2 operations

These work the same as a normal install and manage **both** containers:

```bash
./setup.sh up        # start app + Caddy
./setup.sh down      # stop both
./setup.sh logs      # follow both
./setup.sh enable    # start on boot (both)
./setup.sh disable   # cancel boot autostart
./setup.sh uninstall # remove both containers (data + cert volumes are kept)
```

- **Docker / rootful Podman:** autostart is the containers' `unless-stopped`
  restart policy.
- **Rootless Podman:** autostart uses systemd **Quadlet** units — a shared
  network plus an app unit and a Caddy unit (ordered after the app). Both are
  written to `~/.config/containers/systemd/` and activate on boot via linger.

### Changing the username or password

Re-run the install with new credentials, or regenerate the hash and edit
`./Caddyfile` directly:

```bash
docker run --rm docker.io/library/caddy:2 caddy hash-password --plaintext 'newpass'
# replace the hash after the username in ./Caddyfile, then:
./setup.sh down && ./setup.sh up
```

### Connecting API clients

Point OpenAI-compatible clients at the **HTTPS** endpoint and authenticate with
the app's Bearer `api_key` (set under Settings), **not** the Basic-Auth login:

```
https://<host>/v1
```

The router's raw `:8080` is fronted by Caddy with the admin login, so it's for
the browser chat UI — not for programmatic clients.

---

## How it works (and what to audit)

- **Ports.** In secure mode the app binds to `127.0.0.1` only (loopback), so it
  is not reachable from the network; Caddy publishes `80` (HTTP→HTTPS redirect +
  ACME), `443` (UI/API/`/v1`), and `8080` (chat UI). The app's raw router is
  reachable only from the host itself, on a loopback debug port.
- **Generation web UI ports (opt-in).** If you answer yes to "Publish ports for
  generation backend web UIs?" (or pass `--gen-ports`), Caddy also publishes
  one port per generation backend port, `8081`–`8088` by default
  (`--gen-port-start` moves them, and `--gen-port-count` or the install
  question sets how many, up to 64). Each port is its own Caddy site with the
  admin login, forwarding to the backend's port inside the app container. The
  app's own copy of the range is on loopback only, at `127.0.0.1:18081`
  onwards, and its loopback debug port moves from
  `8081` to `18080` so it doesn't clash. The installer prints that move. With
  no (the default) none of this is added. The image and music APIs work
  either way, because they go through `443`.
- **`ExternalURL`.** The installer sets `LLAMA_TOOLCHEST_EXTERNAL_URL=https://…`
  so the UI's chat/API links render with the externally reachable URL. If you
  change domains, update the variable in `.env`: while it is set it wins over
  Settings → Server URL, and it is never written to the config file, so
  switching back to plain mode returns to the URL saved in Settings.
- **TLS termination.** Browser ↔ Caddy is encrypted; Caddy ↔ app is plain HTTP
  over the private container network and never leaves the host.
- **Auth split.** Everything except `/v1/*`, `/release_task` and
  `/query_result` requires Basic Auth. Those are left to the app's Bearer
  `api_key` so OpenAI and ACE-Step clients (which send
  `Authorization: Bearer …`) aren't broken by a Basic-Auth challenge. The
  ACE-Step routes are at the server root because that is where ACE-Step
  clients call them.

Read [`Caddyfile.template`](../Caddyfile.template) and your rendered
`./Caddyfile` to confirm all of the above matches your expectations before
exposing the server.

---

## Limitations

- Container installs only (no `--host`).
- A single admin account (no multi-user, no RBAC, no SSO/OIDC/mTLS).
- The bundled config is a starting point — **audit it for your threat model.**

## Troubleshooting

- **Browser cert warning (self-signed):** expected. Trust Caddy's root CA (see
  above) or switch to Let's Encrypt.
- **Let's Encrypt fails to issue:** confirm the domain resolves to this host and
  ports 80/443 are reachable from the internet; check `./setup.sh logs`.
- **Generation web UI port already in use:** the installer checks every port
  in the range and asks for another first port. In a scripted install, pass
  `--gen-port-start <port>`.
- **Port already in use:** something else holds 80/443/8080. Stop it, or adjust
  `CADDY_HTTP_PORT` / `CADDY_HTTPS_PORT` / `CADDY_CHAT_PORT` in `.env`. These
  change only the host ports: Caddy still listens on 80, 443 and 8080 inside
  its container. Re-run `./setup.sh install` afterwards, so the external URL
  and the in-app chat link (`LLAMA_TOOLCHEST_CHAT_PUBLIC_PORT`) use the new ports. A common case is a **host-mode** llama-toolchest on
  port 3000 — the installer detects it and offers to stop the host service.
- **`rootlessport cannot expose privileged port 80` (rootless Podman):** the
  kernel won't let an unprivileged user bind ports < 1024. Accept the
  installer's offer to lower `net.ipv4.ip_unprivileged_port_start`, or set
  `CADDY_HTTP_PORT`/`CADDY_HTTPS_PORT` ≥ 1024, or run rootful Podman.
- **Chat link points at `http://localhost:8080`:** `ExternalURL` wasn't applied;
  set it under Settings or via `LLAMA_TOOLCHEST_EXTERNAL_URL`.
