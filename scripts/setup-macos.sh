#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# llama-toolchest setup for macOS. setup.sh hands over to this script when it
# runs on a Mac, so `./setup.sh install` works the same way there.
#
# It checks the build tools (Xcode Command Line Tools, Homebrew, cmake, ninja,
# git) and offers to install the missing ones, installs the released binary in
# ~/.local/bin, and runs llama-toolchest as a launchd agent of the logged-in
# user, so it starts at login and restarts if it stops.
#
# Written for the bash 3.2 that macOS ships: no associative arrays, mapfile or
# ${var,,}.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly REPO="tmac1973/llama-toolchest"
readonly RELEASE_API="https://api.github.com/repos/${REPO}/releases/latest"
readonly RELEASE_DOWNLOAD="https://github.com/${REPO}/releases/download"
readonly LABEL="io.github.tmac1973.llama-toolchest"

readonly BIN_DIR="${HOME}/.local/bin"
readonly BIN_PATH="${BIN_DIR}/llama-toolchest"
readonly APP_DIR="${HOME}/Library/Application Support/llama-toolchest"
readonly CONFIG_PATH="${APP_DIR}/llama-toolchest.yaml"
readonly LOG_DIR="${HOME}/Library/Logs/llama-toolchest"
readonly LOG_PATH="${LOG_DIR}/llama-toolchest.log"
readonly PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

ASSUME_YES=false
PORT=""
LOCAL_BINARY=""
INTERACTIVE=true
[[ -t 0 ]] || INTERACTIVE=false

if [[ -t 1 ]]; then
    BLUE=$'\033[0;34m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'; NC=$'\033[0m'
else
    BLUE=""; GREEN=""; YELLOW=""; RED=""; NC=""
fi
log()   { echo "${BLUE}==>${NC} $*"; }
ok()    { echo "${GREEN}  ✓${NC} $*"; }
warn()  { echo "${YELLOW}  ⚠${NC} $*" >&2; }
err()   { echo "${RED}  ✗${NC} $*" >&2; }
fatal() { err "$@"; exit 1; }

# confirm asks a yes/no question, "yes" being the default. With --yes it
# answers yes; without a terminal and without --yes it answers no.
confirm() {
    local reply
    if [[ "$ASSUME_YES" == true ]]; then return 0; fi
    if [[ "$INTERACTIVE" != true ]]; then return 1; fi
    read -r -p "  $1 [Y/n] " reply
    case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

# brew_bin prints the brew command, also when Homebrew is installed but not
# on PATH (an ssh session that does not read the login profile).
brew_bin() {
    if command -v brew >/dev/null 2>&1; then command -v brew; return; fi
    local b
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        if [[ -x "$b" ]]; then echo "$b"; return; fi
    done
    return 1
}

# service_path is the PATH the launchd agent runs with. launchd does not read
# the shell profile, so without Homebrew's bin directory the builds could not
# find cmake or ninja.
service_path() {
    local p="/usr/bin:/bin:/usr/sbin:/sbin"
    local brew prefix
    if brew="$(brew_bin)"; then
        prefix="$("$brew" --prefix)"
        p="${prefix}/bin:${prefix}/sbin:${p}"
    fi
    [[ -d /usr/local/bin && ":$p:" != *":/usr/local/bin:"* ]] && p="/usr/local/bin:${p}"
    echo "$p"
}

# ─── Checks ──────────────────────────────────────────────────────────────────

check_arch() {
    case "$(uname -m)" in
        arm64) ok "Apple Silicon ($(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo arm64))" ;;
        x86_64)
            warn "This is an Intel Mac. llama-toolchest runs here, but only with the CPU build:"
            warn "Metal and the GPU memory planning need Apple Silicon."
            ;;
        *) fatal "Unsupported CPU: $(uname -m)" ;;
    esac
}

have_clt() { xcode-select -p >/dev/null 2>&1; }

# missing_tools prints the build tools that are not found, one per line.
missing_tools() {
    local t brew prefix
    brew="$(brew_bin 2>/dev/null || true)"
    prefix=""
    [[ -n "$brew" ]] && prefix="$("$brew" --prefix)"
    for t in cmake ninja git; do
        if command -v "$t" >/dev/null 2>&1; then continue; fi
        if [[ -n "$prefix" && -x "${prefix}/bin/${t}" ]]; then continue; fi
        echo "$t"
    done
}

# check_deps reports every prerequisite. With install set it offers to
# install what is missing; it returns 1 when something is still missing.
check_deps() {
    local install="${1:-false}" missing brew
    local result=0

    if have_clt; then
        ok "Xcode Command Line Tools ($(xcode-select -p))"
    else
        err "Xcode Command Line Tools are not installed (the C/C++ compiler for builds)"
        if [[ "$install" == true ]] && confirm "Open the installer for the Command Line Tools now?"; then
            xcode-select --install || true
            echo "  Finish the installer window that opened, then run this command again."
            exit 1
        fi
        echo "  Install them with:  xcode-select --install"
        result=1
    fi

    if brew="$(brew_bin)"; then
        ok "Homebrew ($brew)"
    else
        err "Homebrew is not installed"
        echo "  Install it from https://brew.sh, then run this command again."
        return 1
    fi

    missing="$(missing_tools | tr '\n' ' ')"
    missing="${missing% }"
    if [[ -z "$missing" ]]; then
        ok "cmake, ninja and git"
        return "$result"
    fi
    err "Missing build tools: $missing"
    if [[ "$install" == true ]] && confirm "Install them with Homebrew (brew install $missing)?"; then
        # shellcheck disable=SC2086 # word splitting is the point
        HOMEBREW_NO_AUTO_UPDATE=1 "$brew" install $missing
        ok "Installed $missing"
        return "$result"
    fi
    echo "  Install them with:  brew install $missing"
    return 1
}

port_in_use() {
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

# ─── Install ─────────────────────────────────────────────────────────────────

latest_version() {
    curl -fsSL "$RELEASE_API" | grep -m1 '"tag_name":' | sed 's/.*"v\{0,1\}\([^"]*\)".*/\1/'
}

# install_binary puts the program in BIN_PATH: LOCAL_BINARY when given,
# otherwise the release LT_VERSION (default: the latest), checked against
# the release's checksums.txt.
install_binary() {
    mkdir -p "$BIN_DIR"
    if [[ -n "$LOCAL_BINARY" ]]; then
        [[ -x "$LOCAL_BINARY" ]] || fatal "$LOCAL_BINARY is not an executable file"
        cp "$LOCAL_BINARY" "${BIN_PATH}.new"
        mv "${BIN_PATH}.new" "$BIN_PATH"
        xattr -d com.apple.quarantine "$BIN_PATH" 2>/dev/null || true
        ok "Installed $LOCAL_BINARY as $BIN_PATH"
        return
    fi

    local version arch asset tmp
    version="${LT_VERSION:-}"
    if [[ -z "$version" ]]; then
        version="$(latest_version)" || fatal "Could not read the latest release from GitHub"
    fi
    version="${version#v}"
    arch="arm64"
    [[ "$(uname -m)" == "x86_64" ]] && arch="amd64"
    asset="llama-toolchest_${version}_macos_${arch}.tar.gz"

    tmp="$(mktemp -d)"
    log "Downloading llama-toolchest ${version}"
    curl -fSL --progress-bar -o "${tmp}/${asset}" "${RELEASE_DOWNLOAD}/v${version}/${asset}" \
        || fatal "Download failed: ${RELEASE_DOWNLOAD}/v${version}/${asset}"
    curl -fsSL -o "${tmp}/checksums.txt" "${RELEASE_DOWNLOAD}/v${version}/checksums.txt" \
        || fatal "Could not download the release checksums"
    (cd "$tmp" && grep " ${asset}\$" checksums.txt | shasum -a 256 -c - >/dev/null) \
        || fatal "Checksum mismatch for $asset"
    ok "Checksum verified"
    tar -xzf "${tmp}/${asset}" -C "$tmp" llama-toolchest
    mv "${tmp}/llama-toolchest" "$BIN_PATH"
    chmod 755 "$BIN_PATH"
    # A file downloaded with curl has no quarantine flag, but remove it in
    # case: Gatekeeper refuses binaries that are not notarized.
    xattr -d com.apple.quarantine "$BIN_PATH" 2>/dev/null || true
    rm -rf "$tmp"
    ok "Installed $BIN_PATH ($("$BIN_PATH" -version 2>/dev/null | head -1))"
}

# write_config sets the web UI port in the config file, keeping a file that
# is already there.
write_config() {
    local port="$1"
    mkdir -p "$APP_DIR"
    if [[ -f "$CONFIG_PATH" ]]; then
        ok "Keeping the config file $CONFIG_PATH"
        return
    fi
    cat > "$CONFIG_PATH" <<EOF
# llama-toolchest configuration. Settings changed in the web UI are saved here.
listen_addr: ":${port}"
external_url: "http://localhost:${port}"
EOF
    ok "Wrote $CONFIG_PATH"
}

write_plist() {
    mkdir -p "$(dirname "$PLIST_PATH")" "$LOG_DIR"
    cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${BIN_PATH}</string>
        <string>-config</string>
        <string>${CONFIG_PATH}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>$(service_path)</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>${HOME}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>${LOG_PATH}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_PATH}</string>
</dict>
</plist>
EOF
    plutil -lint "$PLIST_PATH" >/dev/null || fatal "The launchd file $PLIST_PATH is not valid"
    ok "Wrote $PLIST_PATH"
}

# launchd_domain is the launchd domain of this user: gui/<uid> when the user
# is logged in at the screen, otherwise user/<uid> (an ssh-only session).
launchd_domain() {
    local uid
    uid="$(id -u)"
    if launchctl print "gui/${uid}" >/dev/null 2>&1; then
        echo "gui/${uid}"
    else
        echo "user/${uid}"
    fi
}

is_loaded() {
    launchctl print "$(launchd_domain)/${LABEL}" >/dev/null 2>&1
}

service_start() {
    [[ -f "$PLIST_PATH" ]] || fatal "llama-toolchest is not installed (run: ./setup.sh install)"
    if is_loaded; then
        launchctl kickstart -k "$(launchd_domain)/${LABEL}"
    else
        launchctl bootstrap "$(launchd_domain)" "$PLIST_PATH"
    fi
}

# service_stop unloads the agent and waits until it is gone: bootout returns
# while the program is still stopping, and a bootstrap before then fails
# with "Operation already in progress".
service_stop() {
    local i
    is_loaded || return 0
    launchctl bootout "$(launchd_domain)/${LABEL}" 2>/dev/null || true
    for i in $(seq 1 30); do
        is_loaded || return 0
        sleep 1
    done
    warn "llama-toolchest is still stopping after 30 seconds"
}

configured_port() {
    local p=""
    [[ -f "$CONFIG_PATH" ]] && p="$(sed -n 's/^listen_addr: *"\{0,1\}[^:]*:\([0-9]*\)"\{0,1\}.*/\1/p' "$CONFIG_PATH" | head -1)"
    echo "${p:-3000}"
}

wait_for_ui() {
    local port="$1" i
    for i in $(seq 1 30); do
        if curl -fs -o /dev/null "http://localhost:${port}/server"; then return 0; fi
        sleep 1
    done
    return 1
}

cmd_install() {
    log "Checking this Mac"
    check_arch
    check_deps true || fatal "Install the missing prerequisites above, then run this command again."

    local port
    port="${PORT:-$(configured_port)}"
    service_stop
    if port_in_use "$port"; then
        fatal "Port $port is in use. Choose another with: ./setup.sh install --port <port>"
    fi

    log "Installing llama-toolchest"
    install_binary
    write_config "$port"
    write_plist
    service_start
    port="$(configured_port)"
    if wait_for_ui "$port"; then
        ok "llama-toolchest is running"
    else
        warn "llama-toolchest did not answer yet; see the log: $LOG_PATH"
    fi
    echo ""
    echo "  Web UI and API:  http://localhost:${port}   (OpenAI API at /v1)"
    echo "  It starts when you log in. Stop or start it with: ./setup.sh down | up"
    echo "  Log file:        $LOG_PATH"
    echo "  Next: open the web UI, go to Builds and build llama.cpp with the metal profile."
}

cmd_uninstall() {
    log "Removing llama-toolchest"
    service_stop
    rm -f "$PLIST_PATH" "$BIN_PATH"
    ok "Removed the launchd agent and $BIN_PATH"
    if [[ -d "$APP_DIR" ]]; then
        echo "  Your models, builds and settings are kept in:"
        echo "    $APP_DIR"
        # Never with --yes alone: deleting downloaded models is asked for.
        if [[ "$ASSUME_YES" != true ]] && confirm "Delete them too ($(du -sh "$APP_DIR" 2>/dev/null | cut -f1))? This cannot be undone."; then
            rm -rf "$APP_DIR"
            ok "Deleted $APP_DIR"
        else
            echo "  Kept. Delete the folder yourself to remove them."
        fi
    fi
}

cmd_status() {
    check_arch
    check_deps false || true
    if [[ -x "$BIN_PATH" ]]; then
        ok "Installed: $BIN_PATH ($("$BIN_PATH" -version 2>/dev/null | head -1))"
    else
        echo "  Not installed"
        return
    fi
    if is_loaded; then
        ok "Running as launchd agent $LABEL on http://localhost:$(configured_port)"
    else
        echo "  Installed but not running (start it with: ./setup.sh up)"
    fi
}

usage() {
    cat <<'USAGE'
llama-toolchest setup for macOS

Usage: ./setup.sh <command> [options]

Commands:
  install     Check and install prerequisites (Xcode Command Line Tools,
              Homebrew packages cmake, ninja, git), install the latest
              release in ~/.local/bin and start it as a launchd agent that
              runs at login. Run it again to update.
  uninstall   Stop and remove the program and the launchd agent. Asks
              before deleting models, builds and settings.
  up | start  Start llama-toolchest
  down | stop Stop llama-toolchest (and the models it runs)
  restart     Restart llama-toolchest
  logs        Follow the log file
  status      Show what is installed and running
  deps        Check the prerequisites and print install commands

Options:
  --port <n>       Port for the web UI and API on a new install (default 3000)
  --binary <path>  Install this llama-toolchest binary instead of downloading
                   a release (for testing a build)
  -y, --yes        Answer yes to every question

Environment:
  LT_VERSION=1.2.3   Install this release instead of the latest

Containers and the --host/--cuda/--rocm/--vulkan options are Linux only.
USAGE
}

main() {
    local command="${1:-help}"
    shift || true
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -y|--yes)   ASSUME_YES=true ;;
            --port)     PORT="${2:-}"; shift ;;
            --port=*)   PORT="${1#*=}" ;;
            --binary)   LOCAL_BINARY="${2:-}"; shift ;;
            --binary=*) LOCAL_BINARY="${1#*=}" ;;
            --host)     ;; # a Mac install is always a host install
            -h|--help)  usage; exit 0 ;;
            *) err "Unknown option on macOS: $1"; echo ""; usage; exit 1 ;;
        esac
        shift
    done
    if [[ -n "$PORT" && ! "$PORT" =~ ^[0-9]+$ ]]; then
        fatal "--port needs a number"
    fi

    case "$command" in
        install|update) cmd_install ;;
        uninstall)      cmd_uninstall ;;
        up|start)       service_start; ok "Started (http://localhost:$(configured_port))" ;;
        down|stop)      service_stop; ok "Stopped" ;;
        restart)        service_start; ok "Restarted" ;;
        logs)           touch "$LOG_PATH" 2>/dev/null || true; tail -n 100 -f "$LOG_PATH" ;;
        status)         cmd_status ;;
        deps)           check_arch; check_deps false ;;
        help|-h|--help) usage ;;
        *) err "Unknown command on macOS: $command"; echo ""; usage; exit 1 ;;
    esac
}

main "$@"
