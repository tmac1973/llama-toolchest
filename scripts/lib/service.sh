# scripts/lib/service.sh
#
# Helpers for managing the systemd llama-toolchest.service unit on a host
# install. Sourced by setup.sh / scripts/lib/host.sh; not standalone.
#
# Two scopes:
#   - user (default when run as non-root): unit lives in
#     ~/.config/systemd/user/, controlled with `systemctl --user`
#   - system (when run as root): unit lives in
#     /etc/systemd/system/, controlled with plain `systemctl`
#
# Public functions:
#   service_scope                 → echoes "user" or "system"; the single
#                                   place the installer decides scope
#                                   (host paths, unit paths, Quadlet dir)
#   service_unit_path             → echoes the destination unit-file path
#   service_systemctl <args...>   → systemctl with --user added in user scope
#   service_install <src>         → copy unit file from <src> to dest, daemon-reload
#   service_uninstall             → stop, disable, remove unit file, daemon-reload
#   service_enable                → enable + start
#   service_disable               → stop + disable (leaves unit file in place)
#   service_start                 → start (without changing enabled state)
#   service_stop                  → stop (without changing enabled state)
#   service_restart               → restart
#   service_status                → show status (multi-line)
#   service_is_active             → returns 0 if active, 1 otherwise
#
# All functions write progress to stdout via the calling script's log/ok/warn
# helpers if they're in scope; otherwise plain echo.

readonly SERVICE_NAME="llama-toolchest.service"

# Root means a system-wide install; anyone else gets a per-user install.
# host.sh and setup.sh (Quadlet units) call this too, so all three agree.
service_scope() {
    if [[ $EUID -eq 0 ]]; then
        echo "system"
    else
        echo "user"
    fi
}

service_unit_path() {
    local scope; scope="$(service_scope)"
    if [[ "$scope" == "system" ]]; then
        echo "/etc/systemd/system/${SERVICE_NAME}"
    else
        echo "${HOME}/.config/systemd/user/${SERVICE_NAME}"
    fi
}

# Run systemctl with the right --user / --system flag. setup.sh uses this
# for the Podman Quadlet units as well.
service_systemctl() {
    if [[ "$(service_scope)" == "user" ]]; then
        systemctl --user "$@"
    else
        systemctl "$@"
    fi
}

service_install() {
    local src="$1"
    local dst; dst="$(service_unit_path)"

    if [[ ! -f "$src" ]]; then
        echo "ERROR: service unit file not found: $src" >&2
        return 1
    fi

    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    service_systemctl daemon-reload

    # User services need lingering enabled to start on boot without the user
    # being logged in. Skip if already enabled.
    if [[ "$(service_scope)" == "user" ]] && command -v loginctl >/dev/null 2>&1; then
        if ! loginctl show-user "$USER" 2>/dev/null | grep -q "^Linger=yes"; then
            echo "Enabling user lingering so the service can start on boot without you being logged in (sudo required)..."
            sudo loginctl enable-linger "$USER" || true
        fi
    fi
}

service_uninstall() {
    local dst; dst="$(service_unit_path)"

    service_systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    service_systemctl disable "$SERVICE_NAME" 2>/dev/null || true

    if [[ -f "$dst" ]]; then
        rm -f "$dst"
        service_systemctl daemon-reload
    fi
}

service_enable() {
    service_systemctl enable --now "$SERVICE_NAME"
}

service_disable() {
    service_systemctl disable --now "$SERVICE_NAME"
}

service_restart() {
    service_systemctl restart "$SERVICE_NAME"
}

# Plain start/stop — used by `setup.sh up`/`down` to toggle the running
# state without touching whether the unit is enabled at boot.
service_start() {
    service_systemctl start "$SERVICE_NAME"
}

service_stop() {
    service_systemctl stop "$SERVICE_NAME"
}

service_status() {
    service_systemctl status "$SERVICE_NAME" --no-pager 2>&1 || true
}

service_is_active() {
    service_systemctl is-active --quiet "$SERVICE_NAME"
}
