#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# llama-toolchest setup — distro-agnostic, runtime-agnostic setup and launcher
# ─────────────────────────────────────────────────────────────────────────────

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CDI_SYSTEM_DIR="/etc/cdi"
readonly CDI_USER_DIR="${HOME}/.config/containers/cdi"

# ─── Global state (populated by detect_* functions) ──────────────────────────

GPU_VENDOR=""           # cuda, rocm, cpu
GPU_INFO=""             # human-readable GPU description
AMD_GFX_VERSION=""      # HSA_OVERRIDE_GFX_VERSION value (empty = not needed)
AMD_GFX_TARGET=""       # detected gfx target, e.g. gfx1201 (empty = unknown)
HOST_VIDEO_GID=""       # host video group GID
HOST_RENDER_GID=""      # host render group GID

LLAMA_TOOLCHEST_PORT="3000"            # host port for management UI
LLAMA_TOOLCHEST_INFERENCE_PORT="8080"  # host port for inference API
LLAMA_TOOLCHEST_MODELS_DIR=""          # host path for model storage (empty = use docker volume)

CONTAINER_CMD=""        # docker or podman
COMPOSE_CMD=""          # "docker compose" or "podman-compose" or "podman compose"
CONTAINER_VERSION=""
COMPOSE_VERSION=""

# Tracks whether the user explicitly picked a mode this run (via --host /
# --container / --cuda / etc. or the INSTALL_MODE env var). Stateful
# commands (up/down/logs) auto-detect from disk only when the user did NOT
# pick — otherwise the explicit choice wins. Set BEFORE the default so an
# inherited INSTALL_MODE env var still counts as explicit.
INSTALL_MODE_EXPLICIT="${INSTALL_MODE:+true}"
INSTALL_MODE_EXPLICIT="${INSTALL_MODE_EXPLICIT:-false}"
INSTALL_MODE="${INSTALL_MODE:-container}"  # host or container; default container preserves existing behavior
HOST_INSTALL_MODE="${HOST_INSTALL_MODE:-package}"  # for --host: "package" (download released .deb/.rpm) or "source" (go build locally)
# --from-source in CONTAINER mode: build the binary from this tree and install
# it into the image over the packaged one. Container mode only reads this;
# host mode uses HOST_INSTALL_MODE above.
FROM_SOURCE=false
LOCAL_BINARY=""         # path, relative to the build context, of that binary
HOST_SDK_BACKENDS=()    # backends to install host SDKs for (--cuda/--rocm/--vulkan); empty means autodetect+prompt
MIGRATE_DIRECTION=""    # "to-host" or "to-container" when command=migrate

# ── Experimental ROCm container variant ──
# ROCm 10 is published ONLY as a container image: repo.radeon.com's el9, el10,
# rhel9 and rhel10 paths all stop at 7.2.4, and so does the amdgpu-install path,
# so Dockerfile.rocm cannot reach anything newer however long we wait. The 7.14.x
# line is in the same position. The "next" variant builds on an AMD-published
# ROCm image instead — see Dockerfile.rocm-next.
#
# "" = undecided (the prompt will ask, and empty behaves as stable everywhere),
# "stable" = Dockerfile.rocm, "next" = Dockerfile.rocm-next.
# Container-only: there are no 10.x packages for a host install to use.
ROCM_VARIANT="${ROCM_VARIANT:-}"
ROCM_BASE_IMAGE="${ROCM_BASE_IMAGE:-}"
# Set BEFORE any default, so an inherited env var counts as explicit and
# suppresses both the prompt and the value stored in .env. Mirrors
# INSTALL_MODE_EXPLICIT.
ROCM_VARIANT_EXPLICIT=false
if [[ -n "$ROCM_VARIANT" || -n "$ROCM_BASE_IMAGE" ]]; then
    ROCM_VARIANT_EXPLICIT=true
fi
# A base image on its own means the experimental variant; naming a ROCm release
# and then getting the stable one would be a surprise.
if [[ -n "$ROCM_BASE_IMAGE" && -z "$ROCM_VARIANT" ]]; then
    ROCM_VARIANT="next"
fi
case "$ROCM_VARIANT" in
    ""|stable|next) ;;
    *) echo "ROCM_VARIANT must be 'stable' or 'next' (got '${ROCM_VARIANT}')" >&2; exit 1 ;;
esac
# The AMD repository a bare tag is qualified against, and the release used when
# no tag is given. Check https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags for
# what exists.
readonly ROCM_NEXT_IMAGE_REPO="docker.io/rocm/dev-ubuntu-24.04"
readonly ROCM_NEXT_DEFAULT_TAG="10.0.0-full"
# GPU targets ROCm 10.0.0 ships code for, read from the package list of
# rocm/dev-ubuntu-24.04:10.0.0-full (amdrocm-core-sdk10.0-gfx*). RDNA 1 and
# newer, plus the CDNA datacenter parts — ROCm 10 does NOT require RDNA 4.
readonly ROCM10_GFX_TARGETS="gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx1250 gfx908 gfx90a gfx942 gfx950"

# ── Non-interactive + secure-install state ──
# ASSUME_YES (--yes/-y): skip confirmations and never block on a prompt.
# INTERACTIVE: computed in main() from whether stdin is a TTY.
case "${ASSUME_YES:-}" in 1|true|yes|y|Y) ASSUME_YES=true ;; *) ASSUME_YES=false ;; esac
INTERACTIVE=true
# SECURE (--secure): deploy behind the bundled Caddy reverse proxy (HTTPS +
# single-admin Basic Auth). Container mode only. See docs/secure.md.
case "${SECURE:-}" in 1|true|yes) SECURE=true; SECURE_EXPLICIT=true ;; *) SECURE=false; SECURE_EXPLICIT="${SECURE_EXPLICIT:-false}" ;; esac
TLS_MODE="${TLS_MODE:-}"        # self-signed | letsencrypt
DOMAIN="${DOMAIN:-}"            # FQDN, required for letsencrypt
ACME_EMAIL="${ACME_EMAIL:-}"    # account email for letsencrypt
AUTH_USER="${AUTH_USER:-}"      # admin username (defaults to "admin" later)
AUTH_HASH=""                    # bcrypt hash: from --auth-hash, or computed during install
AUTH_PASS_FILE=""               # --auth-pass-file: path to a file holding the plaintext password
AUTH_PASS="${AUTH_PASS:-}"      # env-provided plaintext (hashed during install); never accepted on argv

DISTRO_ID=""            # debian, ubuntu, fedora, arch, cachyos, opensuse-leap, etc.
DISTRO_NAME=""          # Pretty name from os-release
DISTRO_FAMILY=""        # debian, fedora, arch, suse
PKG_MANAGER=""          # apt, dnf, pacman, zypper

ACTIONS=()              # list of human-readable actions to perform
PREREQS=()             # list of prerequisite action keys

# ─── Utility ─────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()   { echo -e "${BLUE}==>${NC} $*"; }
ok()    { echo -e "${GREEN}  ✓${NC} $*"; }
warn()  { echo -e "${YELLOW}  ⚠${NC} $*" >&2; }
err()   { echo -e "${RED}  ✗${NC} $*" >&2; }
fatal() { err "$@"; exit 1; }

need_cmd() {
    command -v "$1" &>/dev/null
}

run_sudo() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

# ─── Detection: GPU ──────────────────────────────────────────────────────────

# Detect host video/render group GIDs for container device access.
# Container group_add needs the host's actual GIDs, not names,
# because the container's /etc/group may have different GID mappings.
detect_host_gpu_gids() {
    if need_cmd getent; then
        HOST_VIDEO_GID="$(getent group video 2>/dev/null | cut -d: -f3)" || true
        HOST_RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3)" || true
    fi
    # Fallback: try parsing /etc/group directly
    if [[ -z "$HOST_VIDEO_GID" ]]; then
        HOST_VIDEO_GID="$(grep '^video:' /etc/group 2>/dev/null | cut -d: -f3)" || true
    fi
    if [[ -z "$HOST_RENDER_GID" ]]; then
        HOST_RENDER_GID="$(grep '^render:' /etc/group 2>/dev/null | cut -d: -f3)" || true
    fi
}

# Detect the AMD GPU gfx target from sysfs and determine if
# HSA_OVERRIDE_GFX_VERSION is needed for ROCm compatibility.
detect_amd_gfx_version() {
    local gfx_target=""

    # Try rocminfo first (may not be installed on host)
    if need_cmd rocminfo; then
        gfx_target="$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1)" || true
    fi

    # Fallback: read from sysfs ip_discovery or amdgpu firmware
    if [[ -z "$gfx_target" && -d /sys/class/drm ]]; then
        for card_dir in /sys/class/drm/card[0-9]*/device; do
            if [[ -f "$card_dir/vendor" && "$(cat "$card_dir/vendor")" == "0x1002" ]]; then
                # Try to read gfx target from pp_dpm_sclk or firmware info
                local fw_ver
                fw_ver="$(cat "$card_dir/gpu_id" 2>/dev/null)" || true
                break
            fi
        done
    fi

    [[ -z "$gfx_target" ]] && return
    AMD_GFX_TARGET="$gfx_target"

    # Map gfx target to HSA_OVERRIDE_GFX_VERSION
    # Only set the override for GPUs not natively supported by ROCm 7.2
    case "$gfx_target" in
        # RDNA 4 — natively supported in ROCm 7.2
        gfx1200|gfx1201)
            AMD_GFX_VERSION=""
            ;;
        # RDNA 3 — natively supported
        gfx1100|gfx1101|gfx1102|gfx1103)
            AMD_GFX_VERSION=""
            ;;
        # RDNA 2 — natively supported
        gfx1030|gfx1031|gfx1032|gfx1033|gfx1034|gfx1035|gfx1036)
            AMD_GFX_VERSION=""
            ;;
        # RDNA 1 — needs override
        gfx1010|gfx1011|gfx1012|gfx1013)
            AMD_GFX_VERSION="10.1.0"
            ;;
        # Vega — needs override
        gfx900|gfx902|gfx904|gfx906|gfx908|gfx909)
            AMD_GFX_VERSION="9.0.0"
            ;;
        *)
            # Unknown target — leave empty, let ROCm try natively
            AMD_GFX_VERSION=""
            ;;
    esac
}

detect_gpu() {
    # NVIDIA: check for nvidia-smi AND that it can talk to a GPU
    if need_cmd nvidia-smi; then
        if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
            GPU_VENDOR="cuda"
            GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)"
            GPU_INFO="${GPU_INFO%%$'\n'*}"
            return
        fi
    fi
    # NVIDIA: fallback — device node exists but nvidia-smi missing/broken
    if [[ -e /dev/nvidia0 ]]; then
        GPU_VENDOR="cuda"
        GPU_INFO="NVIDIA GPU detected (nvidia-smi unavailable)"
        return
    fi

    # AMD: check for ROCm kernel driver
    if [[ -e /dev/kfd ]]; then
        GPU_VENDOR="rocm"
        GPU_INFO="AMD GPU"
        if need_cmd rocminfo; then
            local name
            # rocminfo lists CPU agents before GPU agents. Match GPU marketing
            # names (Radeon, Instinct, FirePro) rather than excluding CPU names,
            # so we don't break if AMD introduces new CPU branding.
            name="$(rocminfo 2>/dev/null | grep 'Marketing Name' | sed 's/.*: *//' \
                | grep -iE 'Radeon|Instinct|FirePro' | head -1)" || true
            [[ -n "$name" ]] && GPU_INFO="$name"
        elif [[ -d /sys/class/drm ]]; then
            for card_dir in /sys/class/drm/card[0-9]*/device; do
                if [[ -f "$card_dir/vendor" && "$(cat "$card_dir/vendor")" == "0x1002" ]]; then
                    GPU_INFO="AMD GPU ($(cat "$card_dir/device" 2>/dev/null || echo "unknown"))"
                    break
                fi
            done
        fi
        detect_amd_gfx_version
        detect_host_gpu_gids
        return
    fi

    GPU_VENDOR="cpu"
    GPU_INFO="No supported GPU detected"
}

# ─── Detection: Container runtime ────────────────────────────────────────────

detect_container_runtime() {
    local user_override="${RUNTIME:-}"

    if [[ -n "$user_override" ]]; then
        case "$user_override" in
            docker)
                need_cmd docker || fatal "RUNTIME=docker specified but docker is not installed"
                CONTAINER_CMD="docker"
                ;;
            podman)
                need_cmd podman || fatal "RUNTIME=podman specified but podman is not installed"
                CONTAINER_CMD="podman"
                ;;
            *)
                fatal "Unknown RUNTIME=$user_override (expected: docker or podman)"
                ;;
        esac
    else
        # Auto-detect: prefer docker if available, fall back to podman
        if need_cmd docker && docker info &>/dev/null 2>&1; then
            # Make sure it's real Docker, not podman emulating docker
            if docker --version 2>/dev/null | grep -qi podman; then
                CONTAINER_CMD="podman"
            else
                CONTAINER_CMD="docker"
            fi
        elif need_cmd podman; then
            CONTAINER_CMD="podman"
        else
            fatal "No container runtime found. Install Docker or Podman first."
        fi
    fi

    # Get version (use `read` to grab first line — avoids SIGPIPE with pipefail)
    CONTAINER_VERSION="$($CONTAINER_CMD --version 2>/dev/null || true)"
    CONTAINER_VERSION="${CONTAINER_VERSION%%$'\n'*}"

    # Detect compose command
    if [[ "$CONTAINER_CMD" == "docker" ]]; then
        if docker compose version &>/dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
            COMPOSE_VERSION="$(docker compose version 2>/dev/null || true)"
        elif need_cmd docker-compose; then
            COMPOSE_CMD="docker-compose"
            COMPOSE_VERSION="$(docker-compose --version 2>/dev/null || true)"
        else
            fatal "Docker is installed but neither 'docker compose' plugin nor 'docker-compose' found"
        fi
    else
        # Podman: try podman compose (Podman 5+), then podman-compose
        if podman compose version &>/dev/null 2>&1; then
            COMPOSE_CMD="podman compose"
            COMPOSE_VERSION="$(podman compose version 2>/dev/null || true)"
        elif need_cmd podman-compose; then
            COMPOSE_CMD="podman-compose"
            COMPOSE_VERSION="$(podman-compose --version 2>/dev/null || true)"
        else
            fatal "Podman is installed but neither 'podman compose' nor 'podman-compose' found"
        fi
    fi
    COMPOSE_VERSION="${COMPOSE_VERSION%%$'\n'*}"
}

# ─── Detection: Linux distribution ───────────────────────────────────────────

detect_distro() {
    if [[ ! -f /etc/os-release ]]; then
        fatal "Cannot detect distribution: /etc/os-release not found"
    fi

    # shellcheck disable=SC1091
    source /etc/os-release

    DISTRO_ID="${ID:-unknown}"
    DISTRO_NAME="${PRETTY_NAME:-$DISTRO_ID}"
    local id_like="${ID_LIKE:-}"

    # Map to distro family and package manager
    case "$DISTRO_ID" in
        debian|ubuntu|pop|linuxmint|elementary|zorin|kali)
            DISTRO_FAMILY="debian"
            PKG_MANAGER="apt"
            ;;
        fedora|rhel|centos|rocky|alma|nobara)
            DISTRO_FAMILY="fedora"
            PKG_MANAGER="dnf"
            ;;
        arch|cachyos|endeavouros|manjaro|garuda|artix)
            DISTRO_FAMILY="arch"
            PKG_MANAGER="pacman"
            ;;
        opensuse-leap|opensuse-tumbleweed|sles)
            DISTRO_FAMILY="suse"
            PKG_MANAGER="zypper"
            ;;
        *)
            # Fall back to ID_LIKE
            if [[ "$id_like" == *"debian"* || "$id_like" == *"ubuntu"* ]]; then
                DISTRO_FAMILY="debian"
                PKG_MANAGER="apt"
            elif [[ "$id_like" == *"fedora"* || "$id_like" == *"rhel"* ]]; then
                DISTRO_FAMILY="fedora"
                PKG_MANAGER="dnf"
            elif [[ "$id_like" == *"arch"* ]]; then
                DISTRO_FAMILY="arch"
                PKG_MANAGER="pacman"
            elif [[ "$id_like" == *"suse"* ]]; then
                DISTRO_FAMILY="suse"
                PKG_MANAGER="zypper"
            else
                warn "Unknown distro: $DISTRO_ID (ID_LIKE=$id_like)"
                warn "Will skip automatic prerequisite installation"
                DISTRO_FAMILY="unknown"
                PKG_MANAGER=""
            fi
            ;;
    esac
}

# ─── Prerequisite checks ─────────────────────────────────────────────────────

has_nvidia_toolkit() {
    need_cmd nvidia-ctk
}

has_cdi_spec() {
    # Check both system and user CDI directories
    [[ -f "$CDI_SYSTEM_DIR/nvidia.yaml" ]] || [[ -f "$CDI_USER_DIR/nvidia.yaml" ]]
}

docker_has_nvidia_runtime() {
    docker info 2>/dev/null | grep -qi "nvidia"
}

selinux_enforcing() {
    need_cmd getenforce && [[ "$(getenforce 2>/dev/null)" == "Enforcing" ]]
}

selinux_device_bool_set() {
    need_cmd getsebool && getsebool container_use_devices 2>/dev/null | grep -q "on"
}

# Container-mode dependency report. Verifies the prereqs needed to run
# `setup.sh install` in container mode and prints remediation commands
# for anything missing. Returns 0 if everything's present, 1 otherwise.
container_deps() {
    local rc=0
    local inst_cmd
    case "$DISTRO_FAMILY" in
        debian) inst_cmd="sudo apt-get install -y" ;;
        fedora) inst_cmd="sudo dnf install -y" ;;
        arch)   inst_cmd="sudo pacman -S --needed" ;;
        suse)   inst_cmd="sudo zypper install -y" ;;
        *)      inst_cmd="<distro ${DISTRO_FAMILY:-unknown}>" ;;
    esac

    echo "Container install dependencies (distro: ${DISTRO_FAMILY:-unknown}, GPU: ${GPU_VENDOR:-unknown}):"
    echo ""
    echo "  Container runtime:"
    if [[ -n "$CONTAINER_CMD" ]]; then
        printf "    %-25s %s\n" "$CONTAINER_CMD" "OK ($CONTAINER_VERSION)"
    else
        printf "    %-25s %s\n" "docker or podman" "not installed"
        case "$DISTRO_FAMILY" in
            debian) echo "            $inst_cmd docker.io  (or follow https://docs.docker.com/engine/install/)" ;;
            fedora) echo "            $inst_cmd podman      (or docker-ce from Docker's repo)" ;;
            arch)   echo "            $inst_cmd docker      (or podman)" ;;
            *)      echo "            install Docker or Podman from your distro" ;;
        esac
        rc=1
    fi

    if [[ -n "$COMPOSE_CMD" ]]; then
        printf "    %-25s %s\n" "compose" "OK ($COMPOSE_VERSION)"
    else
        printf "    %-25s %s\n" "compose" "not installed"
        case "$CONTAINER_CMD" in
            docker) echo "            $inst_cmd docker-compose-plugin" ;;
            podman) echo "            $inst_cmd podman-compose" ;;
            *)      echo "            install docker-compose-plugin or podman-compose" ;;
        esac
        rc=1
    fi
    echo ""

    # GPU-specific integration. Only flag what's relevant for this host.
    if [[ "$GPU_VENDOR" == "cuda" ]]; then
        echo "  NVIDIA GPU integration:"
        if has_nvidia_toolkit; then
            printf "    %-25s %s\n" "NVIDIA Container Toolkit" "OK"
        else
            printf "    %-25s %s\n" "NVIDIA Container Toolkit" "not installed"
            echo "            ./setup.sh install   # this script auto-installs the toolkit"
            rc=1
        fi
        if [[ "$CONTAINER_CMD" == "docker" ]]; then
            if has_nvidia_toolkit && docker_has_nvidia_runtime; then
                printf "    %-25s %s\n" "Docker NVIDIA runtime" "OK"
            else
                printf "    %-25s %s\n" "Docker NVIDIA runtime" "not configured"
                echo "            sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
                rc=1
            fi
        fi
        if [[ "$CONTAINER_CMD" == "podman" ]]; then
            if has_cdi_spec; then
                printf "    %-25s %s\n" "Podman CDI spec" "OK"
            else
                printf "    %-25s %s\n" "Podman CDI spec" "missing"
                echo "            sudo nvidia-ctk cdi generate --output=$CDI_SYSTEM_DIR/nvidia.yaml"
                rc=1
            fi
        fi
        echo ""
    fi

    if [[ "$GPU_VENDOR" == "rocm" && "$DISTRO_FAMILY" == "fedora" ]] && selinux_enforcing; then
        echo "  SELinux (rocm on Fedora/RHEL):"
        if selinux_device_bool_set; then
            printf "    %-25s %s\n" "container_use_devices" "OK"
        else
            printf "    %-25s %s\n" "container_use_devices" "off"
            echo "            sudo setsebool -P container_use_devices on"
            rc=1
        fi
        echo ""
    fi

    # Optional: uv inside the container (for llama-benchy presets).
    # Only checkable when the container is running. Newer images (built
    # after the uv layer was added to the Dockerfiles) include it
    # automatically; older images need a rebuild.
    echo "  Benchmark integration (optional — for llama-benchy presets):"
    if container_exists llama-toolchest && $CONTAINER_CMD container inspect -f '{{.State.Running}}' llama-toolchest 2>/dev/null | grep -qx "true"; then
        if $CONTAINER_CMD exec llama-toolchest sh -c 'command -v uvx' >/dev/null 2>&1; then
            printf "    %-25s %s\n" "uv (in container)" "OK"
        else
            printf "    %-25s %s\n" "uv (in container)" "missing — llama-benchy benchmark presets won't run"
            echo "            ./setup.sh rebuild   # the Dockerfile installs uv; rebuild to pick it up"
        fi
    else
        printf "    %-25s %s\n" "uv (in container)" "container not running — cannot verify"
    fi
    echo ""

    if [[ $rc -eq 0 ]]; then
        ok "All container dependencies satisfied."
    else
        warn "One or more dependencies are missing — see commands above."
    fi
    return $rc
}

check_prerequisites() {
    PREREQS=()
    ACTIONS=()

    if [[ "$GPU_VENDOR" == "cuda" ]]; then
        # NVIDIA Container Toolkit is needed for both Docker and Podman
        if ! has_nvidia_toolkit; then
            PREREQS+=("install_nvidia_toolkit")
            ACTIONS+=("Install NVIDIA Container Toolkit")
        fi

        if [[ "$CONTAINER_CMD" == "docker" ]]; then
            if has_nvidia_toolkit && ! docker_has_nvidia_runtime; then
                PREREQS+=("configure_docker_nvidia")
                ACTIONS+=("Configure Docker NVIDIA runtime + restart Docker daemon")
            elif ! has_nvidia_toolkit; then
                # Will need to configure after install
                PREREQS+=("configure_docker_nvidia")
                ACTIONS+=("Configure Docker NVIDIA runtime + restart Docker daemon")
            fi
        fi

        if [[ "$CONTAINER_CMD" == "podman" ]]; then
            if ! has_cdi_spec; then
                PREREQS+=("generate_cdi_spec")
                ACTIONS+=("Generate NVIDIA CDI spec for Podman")
            fi
        fi
    fi

    # SELinux: needed for ROCm device access on Fedora/RHEL
    if [[ "$GPU_VENDOR" == "rocm" ]] && selinux_enforcing && ! selinux_device_bool_set; then
        PREREQS+=("selinux_device_bool")
        ACTIONS+=("Enable SELinux container_use_devices boolean")
    fi

    # Build + run is always an action. Names the file via dockerfile() rather
    # than interpolating GPU_VENDOR, so the ROCm variant is reflected here as
    # well as in print_summary — otherwise the two disagree.
    ACTIONS+=("Build container image ($(dockerfile))")
    ACTIONS+=("Start llama-toolchest service")
}

# ─── Prerequisite installation ────────────────────────────────────────────────

install_nvidia_toolkit_apt() {
    log "Adding NVIDIA Container Toolkit apt repository..."
    # Install prerequisites for adding repos
    run_sudo apt-get update -qq
    run_sudo apt-get install -y -qq curl gpg

    # Add NVIDIA GPG key and repo
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | run_sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | run_sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    run_sudo apt-get update -qq
    run_sudo apt-get install -y nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
}

install_nvidia_toolkit_dnf() {
    log "Adding NVIDIA Container Toolkit dnf repository..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        | run_sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null

    run_sudo dnf install -y nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
}

install_nvidia_toolkit_pacman() {
    log "Installing NVIDIA Container Toolkit via pacman..."
    run_sudo pacman -Sy --noconfirm nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
}

install_nvidia_toolkit_zypper() {
    log "Adding NVIDIA Container Toolkit zypper repository..."
    run_sudo zypper ar -f \
        https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        nvidia-container-toolkit 2>/dev/null || true

    run_sudo zypper --gpg-auto-import-keys install -y nvidia-container-toolkit
    ok "NVIDIA Container Toolkit installed"
}

install_nvidia_toolkit() {
    case "$PKG_MANAGER" in
        apt)    install_nvidia_toolkit_apt ;;
        dnf)    install_nvidia_toolkit_dnf ;;
        pacman) install_nvidia_toolkit_pacman ;;
        zypper) install_nvidia_toolkit_zypper ;;
        *)      fatal "Cannot install NVIDIA Container Toolkit: unsupported package manager" ;;
    esac
}

configure_docker_nvidia() {
    log "Configuring Docker NVIDIA runtime..."
    run_sudo nvidia-ctk runtime configure --runtime=docker
    log "Restarting Docker daemon..."
    run_sudo systemctl restart docker
    ok "Docker NVIDIA runtime configured"
}

generate_cdi_spec() {
    log "Generating NVIDIA CDI spec..."
    run_sudo mkdir -p "$CDI_SYSTEM_DIR"
    run_sudo nvidia-ctk cdi generate --output="$CDI_SYSTEM_DIR/nvidia.yaml"
    ok "CDI spec written to $CDI_SYSTEM_DIR/nvidia.yaml"

    # Verify
    if need_cmd nvidia-ctk; then
        log "Verifying CDI devices..."
        nvidia-ctk cdi list 2>/dev/null | head -5 || true
    fi
}

selinux_device_bool() {
    log "Enabling SELinux container_use_devices..."
    run_sudo setsebool -P container_use_devices 1
    ok "SELinux boolean set"
}

install_prerequisites() {
    for prereq in "${PREREQS[@]}"; do
        case "$prereq" in
            install_nvidia_toolkit)  install_nvidia_toolkit ;;
            configure_docker_nvidia) configure_docker_nvidia ;;
            generate_cdi_spec)       generate_cdi_spec ;;
            selinux_device_bool)     selinux_device_bool ;;
            *)                       warn "Unknown prerequisite: $prereq" ;;
        esac
    done
}

# ─── Port configuration ───────────────────────────────────────────────────────

is_port_available() {
    local port="$1"
    # Check if something is already listening on the port
    if need_cmd ss; then
        ! ss -tlnH "sport = :${port}" 2>/dev/null | grep -q .
    elif need_cmd netstat; then
        ! netstat -tln 2>/dev/null | grep -q ":${port} "
    else
        # Can't check — assume available
        return 0
    fi
}

# Portable container-existence check. Podman has `container exists`; Docker
# does NOT — it only works via `container inspect` exiting non-zero. Use
# this helper everywhere instead of the podman-specific subcommand so the
# logic behaves the same on both runtimes.
container_exists() {
    local name="$1"
    [[ -z "$CONTAINER_CMD" ]] && return 1
    $CONTAINER_CMD container inspect "$name" >/dev/null 2>&1
}

# Returns 0 if the host port is bound by one of our own containers — i.e.,
# a llamactl (pre-rename) or llama-toolchest container that the install
# flow is going to stop and replace anyway. Lets prompt_ports treat such
# bindings as "not really a conflict" instead of asking the user to pick
# different ports.
is_port_held_by_our_container() {
    local port="$1"
    [[ -z "$CONTAINER_CMD" ]] && return 1
    for cname in llamactl llama-toolchest llama-toolchest-caddy; do
        container_exists "$cname" || continue
        # `docker/podman port <c>` outputs lines like "3000/tcp -> 0.0.0.0:3001".
        # Match the trailing :PORT to confirm this container is the binder.
        local mappings
        mappings="$($CONTAINER_CMD port "$cname" 2>/dev/null)" || mappings=""
        if echo "$mappings" | grep -qE ":${port}(\s|$)"; then
            return 0
        fi
        # `port` can come back empty (host networking, pasta, a stopped
        # container that still reserves the port via its userspace proxy). Fall
        # back to the published-port list from inspect, where the host port
        # shows up as "HostPort":"3000", so we still recognize our own
        # container instead of flagging a foreign-process conflict.
        if $CONTAINER_CMD inspect --format '{{json .HostConfig.PortBindings}}' "$cname" 2>/dev/null \
            | grep -qE "\"${port}\""; then
            return 0
        fi
    done
    return 1
}

# describe_port_holder prints "name (pid N)" for whatever is listening on a TCP
# port, or nothing if it can't tell. Used to make port-conflict errors
# actionable instead of a cryptic "address already in use" after a long build.
describe_port_holder() {
    local port="$1" info
    need_cmd ss || return 0
    info="$(ss -tlnpH "sport = :${port}" 2>/dev/null | grep -oE '"[^"]+",pid=[0-9]+' | head -1)" || true
    [[ -z "$info" ]] && return 0
    local pname="${info%%\",pid=*}"; pname="${pname#\"}"
    printf '%s (pid %s)' "$pname" "${info##*pid=}"
}

# preflight_port_check aborts BEFORE the (slow) image build if any host port the
# new stack needs is occupied by something we don't manage — e.g. a host-mode
# llama-toolchest running directly on the machine. Our own containers are
# cleared by stop_existing_containers first, so anything left is a real
# conflict. If it's our own host process, offer to stop it.
preflight_port_check() {
    local ports=()
    if [[ "$SECURE" == true ]]; then
        ports=("$LLAMA_TOOLCHEST_PORT" "${SECURE_APP_DEBUG_INF_PORT:-8081}" \
               "${CADDY_HTTP_PORT:-80}" "${CADDY_HTTPS_PORT:-443}" "${CADDY_CHAT_PORT:-8080}")
    else
        ports=("$LLAMA_TOOLCHEST_PORT" "$LLAMA_TOOLCHEST_INFERENCE_PORT")
    fi

    local p holder conflict=false saw_host=false
    for p in "${ports[@]}"; do
        is_port_available "$p" && continue
        holder="$(describe_port_holder "$p")"
        err "Port ${p} is already in use${holder:+ by ${holder}}."
        conflict=true
        [[ "$holder" == llama-toolchest* || "$holder" == llamactl* ]] && saw_host=true
    done
    [[ "$conflict" == false ]] && return 0

    echo ""
    if [[ "$saw_host" == true ]] && host_is_installed; then
        echo "  That's a llama-toolchest installed directly on the host. Running it"
        echo "  alongside the container conflicts now AND again on every reboot"
        echo "  (the host service stays enabled), so just stopping it isn't enough."
        if prompt_confirm "Uninstall the host install (stop, disable, remove binary) and continue?"; then
            host_uninstall
            local still=false
            for p in "${ports[@]}"; do is_port_available "$p" || still=true; done
            [[ "$still" == false ]] && { ok "Host install removed; continuing."; return 0; }
            err "Ports are still busy after removing the host install."
        fi
        echo "  Or remove it yourself, then re-run:  ./setup.sh uninstall --host"
    else
        echo "  Stop the process(es) above, or pick different ports, then re-run."
    fi
    fatal "Port conflict — not building until the ports are free."
}

# ensure_unprivileged_ports: rootless Podman cannot bind ports below the kernel's
# net.ipv4.ip_unprivileged_port_start (default 1024), but the secure stack
# publishes 80/443. Offer to lower that floor (persisted across reboots, which
# autostart needs). Only relevant for rootless Podman in secure mode.
ensure_unprivileged_ports() {
    [[ "$SECURE" == true ]] || return 0
    [[ "$CONTAINER_CMD" == "podman" && $EUID -ne 0 ]] || return 0
    need_cmd sysctl || return 0

    local min_port=65535 p
    for p in "${CADDY_HTTP_PORT:-80}" "${CADDY_HTTPS_PORT:-443}" "${CADDY_CHAT_PORT:-8080}"; do
        (( p < min_port )) && min_port="$p"
    done

    local current
    current="$(sysctl -n net.ipv4.ip_unprivileged_port_start 2>/dev/null || echo 1024)"
    (( min_port >= current )) && return 0   # already permitted

    echo ""
    warn "Rootless Podman can't bind privileged port ${min_port} (kernel floor is ${current})."
    echo "  The secure stack publishes ports 80/443. Either lower the floor"
    echo "  (host-wide, one-time, persisted), or set CADDY_HTTP_PORT/CADDY_HTTPS_PORT"
    echo "  to >= 1024 in .env and adjust your URL — or run rootful Podman."
    echo ""
    if prompt_confirm "Lower net.ipv4.ip_unprivileged_port_start to ${min_port}? (needs sudo)"; then
        run_sudo sysctl -w "net.ipv4.ip_unprivileged_port_start=${min_port}" \
            || fatal "Failed to set the sysctl. Set CADDY_* ports >= 1024 in .env, or run rootful."
        if echo "net.ipv4.ip_unprivileged_port_start=${min_port}" \
            | run_sudo tee /etc/sysctl.d/99-llama-toolchest-ports.conf >/dev/null 2>&1; then
            ok "Privileged ports enabled for rootless containers (persisted)."
        else
            warn "Applied for this session but couldn't persist it; it may reset on reboot."
        fi
    else
        fatal "Privileged ports 80/443 are unavailable to rootless Podman. Set CADDY_HTTP_PORT/CADDY_HTTPS_PORT >= 1024 in .env (and adjust the URL), or run rootful. See docs/secure.md."
    fi
}

prompt_ports() {
    # Non-interactive: accept the configured/flagged ports as-is, no prompting.
    if [[ "$INTERACTIVE" != true || "$ASSUME_YES" == true ]]; then
        return
    fi

    echo ""
    echo -e "${BOLD}Port configuration${NC}"
    echo ""
    echo "  Current ports:"
    echo "    Management UI:  ${LLAMA_TOOLCHEST_PORT}"
    echo "    Inference API:  ${LLAMA_TOOLCHEST_INFERENCE_PORT}"
    echo ""

    # Check if current ports are available. If a port is bound by an existing
    # llama-toolchest (or pre-rename llamactl) container, that container is
    # going to be stopped+removed during install, so it's not a real conflict.
    local ports_ok=true
    for cfg in "UI:$LLAMA_TOOLCHEST_PORT" "Inference:$LLAMA_TOOLCHEST_INFERENCE_PORT"; do
        local label="${cfg%%:*}"
        local p="${cfg##*:}"
        if is_port_available "$p"; then
            continue
        fi
        if is_port_held_by_our_container "$p"; then
            log "Port ${p} is held by an existing llama-toolchest container; it'll be replaced during install."
        else
            warn "Port ${p} (${label}) is already in use by another process"
            ports_ok=false
        fi
    done

    if [[ "$ports_ok" == true ]]; then
        if prompt_confirm "Use these ports?"; then
            return
        fi
    else
        echo ""
        echo "  One or more ports are in use by something other than llama-toolchest."
        echo "  Pick alternative ports, or stop the offending process and re-run."
    fi

    echo ""
    local port
    while true; do
        read -rp "$(echo -e "  ${BOLD}Management UI port${NC} [${LLAMA_TOOLCHEST_PORT}]: ")" port
        port="${port:-$LLAMA_TOOLCHEST_PORT}"
        if [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )); then
            LLAMA_TOOLCHEST_PORT="$port"
            break
        fi
        err "Invalid port number: $port"
    done

    while true; do
        read -rp "$(echo -e "  ${BOLD}Inference API port${NC} [${LLAMA_TOOLCHEST_INFERENCE_PORT}]: ")" port
        port="${port:-$LLAMA_TOOLCHEST_INFERENCE_PORT}"
        if [[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )); then
            if [[ "$port" == "$LLAMA_TOOLCHEST_PORT" ]]; then
                err "Cannot use the same port as management UI ($LLAMA_TOOLCHEST_PORT)"
                continue
            fi
            LLAMA_TOOLCHEST_INFERENCE_PORT="$port"
            break
        fi
        err "Invalid port number: $port"
    done
}

prompt_models_dir() {
    # Non-interactive: keep the configured/flagged value (env or .env), no prompt.
    if [[ "$INTERACTIVE" != true || "$ASSUME_YES" == true ]]; then
        return
    fi

    echo ""
    echo -e "${BOLD}Model storage${NC}"
    echo ""
    if [[ -n "$LLAMA_TOOLCHEST_MODELS_DIR" ]]; then
        echo "  Current: ${LLAMA_TOOLCHEST_MODELS_DIR} (host directory)"
    else
        echo "  Current: Docker volume (default)"
    fi
    echo ""
    echo "  Mount a host directory so models persist even if the"
    echo "  container volume is removed."
    echo ""

    local path
    read -rp "$(echo -e "  ${BOLD}Host path${NC} [${LLAMA_TOOLCHEST_MODELS_DIR:-none}]: ")" path

    if [[ -z "$path" ]]; then
        # Keep current setting (or none)
        return
    fi

    if [[ "$path" == "none" || "$path" == "-" ]]; then
        LLAMA_TOOLCHEST_MODELS_DIR=""
        echo "  → Models will use Docker volume"
        return
    fi

    # Expand ~ to home directory
    path="${path/#\~/$HOME}"

    # Resolve to absolute path
    if [[ "$path" != /* ]]; then
        path="$(cd "$SCRIPT_DIR" && realpath -m "$path" 2>/dev/null || echo "$SCRIPT_DIR/$path")"
    fi

    # Create if it doesn't exist
    if [[ ! -d "$path" ]]; then
        log "Creating directory: $path"
        mkdir -p "$path" || { err "Cannot create $path"; return; }
    fi

    LLAMA_TOOLCHEST_MODELS_DIR="$path"
    export LLAMA_TOOLCHEST_MODELS_DIR
    echo "  → Models will be stored at: $path"
}

load_env_ports() {
    local env_file="${SCRIPT_DIR}/.env"
    [[ ! -f "$env_file" ]] && return 0

    # Legacy rename: pre-rebrand .env files used LLAMACTL_*. Detect and rewrite
    # in place so the user's port and models-dir customizations carry over.
    if grep -qE '^LLAMACTL_(PORT|INFERENCE_PORT|MODELS_DIR|HOST)=' "$env_file" 2>/dev/null; then
        log "Migrating legacy LLAMACTL_* vars in .env to LLAMA_TOOLCHEST_*..."
        sed -i \
            -e 's/^LLAMACTL_PORT=/LLAMA_TOOLCHEST_PORT=/' \
            -e 's/^LLAMACTL_INFERENCE_PORT=/LLAMA_TOOLCHEST_INFERENCE_PORT=/' \
            -e 's/^LLAMACTL_MODELS_DIR=/LLAMA_TOOLCHEST_MODELS_DIR=/' \
            -e 's/^LLAMACTL_HOST=/LLAMA_TOOLCHEST_HOST=/' \
            "$env_file"
    fi

    local val
    val="$(grep '^LLAMA_TOOLCHEST_PORT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
    [[ -n "$val" ]] && LLAMA_TOOLCHEST_PORT="$val" || true
    val="$(grep '^LLAMA_TOOLCHEST_INFERENCE_PORT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
    [[ -n "$val" ]] && LLAMA_TOOLCHEST_INFERENCE_PORT="$val" || true
    val="$(grep '^LLAMA_TOOLCHEST_MODELS_DIR=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
    [[ -n "$val" ]] && LLAMA_TOOLCHEST_MODELS_DIR="$val" || true

    # Remember the ROCm container variant so up/down/rebuild reuse it without
    # re-passing a flag. An explicit flag or env var this run always wins. The
    # restored value does NOT suppress the install prompt — it becomes the
    # prompt's default, which is what lets an experimental machine be returned
    # to stable interactively.
    if [[ "$ROCM_VARIANT_EXPLICIT" != true ]]; then
        val="$(grep '^ROCM_VARIANT=' "$env_file" 2>/dev/null | cut -d= -f2-)" || true
        [[ -n "$val" ]] && ROCM_VARIANT="$val" || true
        val="$(grep '^ROCM_BASE_IMAGE=' "$env_file" 2>/dev/null | cut -d= -f2-)" || true
        [[ -n "$val" ]] && ROCM_BASE_IMAGE="$val" || true
    fi

    # Remember a prior secure install so up/down/rebuild re-add the Caddy
    # overlay without needing --secure again. An explicit --secure/--no-secure
    # on this run always wins.
    if [[ "$SECURE_EXPLICIT" != true ]]; then
        val="$(grep '^LLAMA_TOOLCHEST_SECURE=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ "$val" == "1" || "$val" == "true" ]] && SECURE=true || true
    fi

    # Reload the secure-mode values so a rebuild/quick (which re-runs
    # write_env_file but NOT configure_secure) reproduces the same .env instead
    # of resetting the external URL/ports to defaults.
    if [[ "$SECURE" == true ]]; then
        val="$(grep '^LLAMA_TOOLCHEST_EXTERNAL_URL=' "$env_file" 2>/dev/null | cut -d= -f2-)" || true
        [[ -n "$val" ]] && SECURE_EXTERNAL_URL="$val" || true
        val="$(grep '^LLAMA_TOOLCHEST_INFERENCE_PORT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && SECURE_APP_DEBUG_INF_PORT="$val" || true
        val="$(grep '^CADDY_HTTP_PORT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && CADDY_HTTP_PORT="$val" || true
        val="$(grep '^CADDY_HTTPS_PORT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && CADDY_HTTPS_PORT="$val" || true
        val="$(grep '^CADDY_CHAT_PORT=' "$env_file" 2>/dev/null | cut -d= -f2)" || true
        [[ -n "$val" ]] && CADDY_CHAT_PORT="$val" || true
    fi
}

# Legacy rename: copy contents of pre-rebrand 'llamactl-data' Docker volume
# into the new 'llama-toolchest-data' volume so the renamed container starts
# with the user's existing models, builds, and config. No-op once the new
# volume exists. Called from install/rebuild/quick before the new container
# is started.
migrate_legacy_volume() {
    local old_vol="llamactl-data"
    local new_vol="llama-toolchest-data"

    [[ -z "$CONTAINER_CMD" ]] && return 0

    # Old volume gone → nothing to do (fresh install or already migrated).
    if ! $CONTAINER_CMD volume inspect "$old_vol" >/dev/null 2>&1; then
        return 0
    fi

    # New volume already there → assume migration happened previously, or the
    # user has both manually. Leave both alone; warn so they know.
    if $CONTAINER_CMD volume inspect "$new_vol" >/dev/null 2>&1; then
        warn "Both '$old_vol' and '$new_vol' volumes exist; using '$new_vol' as-is and leaving '$old_vol' untouched."
        return 0
    fi

    echo ""
    warn "Detected legacy Docker volume '$old_vol' from the pre-rename project."
    log "Will copy its contents into '$new_vol' so the renamed container keeps your existing models, builds, and configs."
    log "Disk usage will roughly double during the copy. The old volume stays in place — remove it manually after confirming the new install works."
    echo ""
    if ! prompt_confirm "Migrate volume now?"; then
        err "Migration is required to continue. Run './setup.sh install' again when ready, or remove '$old_vol' manually if you don't need its contents."
        exit 1
    fi

    # Stop and remove any container that may be holding the volume open OR
    # binding the inference port. We handle three legacy container names:
    #   - llamactl: original pre-rename container
    #   - llama-toolchest: post-rename container that may be running an old
    #     image (binary inside still calls itself llamactl) against the
    #     pre-rename volume. Common state for users who rebuilt their
    #     container partway through the rename.
    # Also strip any pre-rename podman Quadlet unit so systemd doesn't
    # bring the old container back on next boot to fight the new one.
    local old_quadlet_user="${HOME}/.config/containers/systemd/llamactl.container"
    local old_quadlet_sys="/etc/containers/systemd/llamactl.container"
    for unit in "$old_quadlet_user" "$old_quadlet_sys"; do
        if [[ -f "$unit" ]]; then
            log "Removing legacy Quadlet unit: $unit"
            if [[ "$unit" == "$old_quadlet_sys" ]]; then
                sudo systemctl stop llamactl.service 2>/dev/null || true
                sudo rm -f "$unit"
                sudo systemctl daemon-reload 2>/dev/null || true
            else
                systemctl --user stop llamactl.service 2>/dev/null || true
                rm -f "$unit"
                systemctl --user daemon-reload 2>/dev/null || true
            fi
        fi
    done

    for cname in llamactl llama-toolchest; do
        if container_exists "$cname"; then
            log "Stopping and removing existing container '$cname'..."
            $CONTAINER_CMD stop "$cname" >/dev/null 2>&1 || true
            $CONTAINER_CMD rm "$cname" >/dev/null 2>&1 || true
        fi
    done

    # Compose recognizes "its own" volumes by the labels it stamps on them.
    # If we create the volume bare, the next `docker compose up` warns:
    #   volume "X" already exists but was not created by Docker Compose
    # Pre-stamp the labels so the next compose run treats it as its own.
    # Project name defaults to the lowercase basename of the compose-file
    # directory (where setup.sh is being run from).
    local compose_project; compose_project="$(basename "$SCRIPT_DIR" | tr '[:upper:]' '[:lower:]')"

    log "Creating new volume $new_vol..."
    $CONTAINER_CMD volume create \
        --label "com.docker.compose.project=${compose_project}" \
        --label "com.docker.compose.volume=${new_vol}" \
        "$new_vol" >/dev/null

    log "Copying contents (can take several minutes for large model collections)..."
    if ! $CONTAINER_CMD run --rm \
        -v "${old_vol}:/from:ro" \
        -v "${new_vol}:/to" \
        docker.io/library/alpine:3 \
        sh -c "set -e; \
               cp -a /from/. /to/; \
               if [ -f /to/config/llamactl.yaml ] && [ ! -f /to/config/llama-toolchest.yaml ]; then \
                   mv /to/config/llamactl.yaml /to/config/llama-toolchest.yaml; \
                   echo 'Renamed config llamactl.yaml -> llama-toolchest.yaml'; \
               fi"; then
        err "Volume copy failed. The partially-populated '$new_vol' can be removed with:"
        echo "    $CONTAINER_CMD volume rm $new_vol"
        exit 1
    fi

    ok "Volume migrated."
    log "The old volume '$old_vol' is preserved. Once you've confirmed the new install works, reclaim its disk with:"
    echo "    $CONTAINER_CMD volume rm $old_vol"
    echo ""
}

# ─── Container operations ────────────────────────────────────────────────────

compose_file() {
    if [[ "$GPU_VENDOR" == "rocm" && "${ROCM_VARIANT:-}" == "next" ]]; then
        echo "docker-compose.rocm-next.yml"
        return
    fi
    echo "docker-compose.${GPU_VENDOR}.yml"
}

# compose_cmd builds the full compose command with all required -f flags.
compose_cmd() {
    local cmd="$COMPOSE_CMD -f $(compose_file)"
    # Add models volume override if a host directory is configured
    if [[ -n "${LLAMA_TOOLCHEST_MODELS_DIR:-}" ]]; then
        cmd+=" -f docker-compose.models.yml"
    fi
    # Add the Caddy reverse-proxy overlay for secure installs.
    if [[ "$SECURE" == true ]]; then
        cmd+=" -f docker-compose.secure.yml"
    fi
    echo "$cmd"
}

dockerfile() {
    if [[ "$GPU_VENDOR" == "rocm" && "${ROCM_VARIANT:-}" == "next" ]]; then
        echo "Dockerfile.rocm-next"
        return
    fi
    echo "Dockerfile.${GPU_VENDOR}"
}

has_quadlet() {
    [[ "$CONTAINER_CMD" == "podman" && $EUID -ne 0 ]] \
        && [[ -f "${QUADLET_USER_DIR}/${PODMAN_SERVICE_NAME}.container" ]]
}

# Write .env file for docker-compose variable substitution
write_env_file() {
    local env_file="${SCRIPT_DIR}/.env"
    : > "$env_file"

    if [[ "$SECURE" == true ]]; then
        # Secure mode: only Caddy faces the network. Bind the app to loopback
        # (host-local debugging) and shift its inference publish off 8080 so it
        # doesn't collide with Caddy's published chat port. The public surface
        # is Caddy's 80/443/8080; see docs/secure.md.
        {
            echo "LLAMA_TOOLCHEST_SECURE=1"
            echo "LLAMA_TOOLCHEST_BIND=127.0.0.1:"
            echo "LLAMA_TOOLCHEST_PORT=${LLAMA_TOOLCHEST_PORT}"
            echo "LLAMA_TOOLCHEST_INFERENCE_PORT=${SECURE_APP_DEBUG_INF_PORT:-8081}"
            echo "LLAMA_TOOLCHEST_EXTERNAL_URL=${SECURE_EXTERNAL_URL:-https://localhost}"
            echo "CADDY_HTTP_PORT=${CADDY_HTTP_PORT:-80}"
            echo "CADDY_HTTPS_PORT=${CADDY_HTTPS_PORT:-443}"
            echo "CADDY_CHAT_PORT=${CADDY_CHAT_PORT:-8080}"
        } >> "$env_file"
    else
        # Port configuration
        echo "LLAMA_TOOLCHEST_PORT=${LLAMA_TOOLCHEST_PORT}" >> "$env_file"
        echo "LLAMA_TOOLCHEST_INFERENCE_PORT=${LLAMA_TOOLCHEST_INFERENCE_PORT}" >> "$env_file"
    fi

    # Model storage — bind-mount a host directory so models survive volume removal
    if [[ -n "$LLAMA_TOOLCHEST_MODELS_DIR" ]]; then
        echo "LLAMA_TOOLCHEST_MODELS_DIR=${LLAMA_TOOLCHEST_MODELS_DIR}" >> "$env_file"
        export LLAMA_TOOLCHEST_MODELS_DIR
    fi

    # GPU-specific settings
    if [[ -n "$AMD_GFX_VERSION" ]]; then
        echo "HSA_OVERRIDE_GFX_VERSION=${AMD_GFX_VERSION}" >> "$env_file"
    fi
    if [[ -n "$HOST_VIDEO_GID" ]]; then
        echo "HOST_VIDEO_GID=${HOST_VIDEO_GID}" >> "$env_file"
    fi
    if [[ -n "$HOST_RENDER_GID" ]]; then
        echo "HOST_RENDER_GID=${HOST_RENDER_GID}" >> "$env_file"
    fi

    # Which ROCm container was chosen. ROCM_BASE_IMAGE has to be in .env rather
    # than merely exported: docker-compose.rocm-next.yml substitutes it into a
    # build arg, and compose reads .env from the project directory for that.
    if [[ "$GPU_VENDOR" == "rocm" ]]; then
        echo "ROCM_VARIANT=${ROCM_VARIANT:-stable}" >> "$env_file"
        if [[ "${ROCM_VARIANT:-}" == "next" ]]; then
            ROCM_BASE_IMAGE="$(rocm_base_image)"
            echo "ROCM_BASE_IMAGE=${ROCM_BASE_IMAGE}" >> "$env_file"
            export ROCM_BASE_IMAGE
        fi
    fi
}

# ─── Secure (Caddy reverse proxy) configuration ──────────────────────────────

print_secure_disclaimer() {
    echo ""
    echo -e "${YELLOW}  ⚠  BEST-EFFORT SECURITY${NC}"
    echo "     The bundled Caddy config is a convenience starting point, provided"
    echo "     AS-IS with no warranty and NOT hardened for any specific threat"
    echo "     model. You are responsible for reviewing and auditing it (TLS, auth,"
    echo "     exposed ports, rate limiting, firewall/network policy) before relying"
    echo "     on it. See docs/secure.md and the rendered ./Caddyfile."
    echo ""
}

# Prompt for host vs container when the user didn't choose explicitly.
prompt_install_mode() {
    echo ""
    echo -e "${BOLD}Install mode${NC}"
    echo ""
    echo "  1) Container  (Docker/Podman) — isolated, recommended"
    echo "  2) Host       (install directly on this machine)"
    echo ""
    local choice
    read -rp "$(echo -e "  ${BOLD}Choose${NC} [1]: ")" choice
    case "${choice:-1}" in
        1) INSTALL_MODE="container" ;;
        2) INSTALL_MODE="host" ;;
        *) err "Invalid choice: $choice"; prompt_install_mode; return ;;
    esac
    INSTALL_MODE_EXPLICIT=true
}

# ─── Experimental ROCm variant ───────────────────────────────────────────────

# rocm_base_image echoes the full base image reference for the "next" variant.
# A value containing a "/" is a complete reference and passes through unchanged
# (so another AMD repository, e.g. dev-ubuntu-26.04, can be named); a bare tag
# is qualified with the AMD repository; empty takes the default release.
rocm_base_image() {
    if [[ -z "$ROCM_BASE_IMAGE" ]]; then
        echo "${ROCM_NEXT_IMAGE_REPO}:${ROCM_NEXT_DEFAULT_TAG}"
    elif [[ "$ROCM_BASE_IMAGE" == */* ]]; then
        echo "$ROCM_BASE_IMAGE"
    else
        echo "${ROCM_NEXT_IMAGE_REPO}:${ROCM_BASE_IMAGE}"
    fi
}

# rocm_variant_label describes a variant for a message. Takes the image
# reference as a second argument rather than reading the global, so it can
# describe the PREVIOUS variant as well as the current one.
rocm_variant_label() {
    case "${1:-stable}" in
        next) echo "experimental${2:+ — ${2}}" ;;
        *)    echo "stable — ROCm 7.2.4 on Fedora" ;;
    esac
}

# prompt_rocm_variant asks which ROCm container to build. The default is
# whichever variant is already installed, so pressing Enter keeps it rather
# than silently moving an experimental machine back to stable.
prompt_rocm_variant() {
    local cur="${ROCM_VARIANT:-stable}" default_n=1 cur_ref cur_shown mark1="" mark2=""
    if [[ "$cur" == "next" ]]; then
        default_n=2
        mark2="   (current)"
    else
        mark1="   (current)"
    fi
    # Two values, deliberately. cur_ref is the full reference and is what an
    # empty answer keeps — reducing it to a bare tag would silently move a
    # machine pinned to another AMD repository (dev-ubuntu-26.04, say) onto the
    # default one. cur_shown is what the prompt displays: just the tag for the
    # usual repository, the whole reference when it is not the usual one, so
    # what is on offer is never ambiguous.
    cur_ref="$(rocm_base_image)"
    if [[ "$cur_ref" == "${ROCM_NEXT_IMAGE_REPO}:"* ]]; then
        cur_shown="${cur_ref##*:}"
    else
        cur_shown="$cur_ref"
    fi

    echo ""
    echo -e "${BOLD}ROCm version${NC}"
    echo ""
    echo "  1) Stable        ROCm 7.2.4 on Fedora — the tested default${mark1}"
    echo "  2) Experimental  ROCm 10 on AMD's Ubuntu image${mark2}"
    echo ""
    echo "     ROCm 10 is published only as a container image, so this is the only"
    echo "     way to run it. It supports RDNA 1 and newer, and the CDNA cards."
    echo "     Your kernel has to be new enough for your own card's driver, which"
    echo "     is checked below — the version differs by card, not by ROCm."
    echo "     Any llama.cpp builds you already have were compiled against the other"
    echo "     ROCm version and will need rebuilding. None of them are deleted, so"
    echo "     switching back makes them work again."
    echo ""
    local choice
    read -rp "$(echo -e "  ${BOLD}Choose${NC} [${default_n}]: ")" choice
    case "${choice:-$default_n}" in
        1) ROCM_VARIANT="stable"; ROCM_BASE_IMAGE="" ;;
        2)
            ROCM_VARIANT="next"
            local tag
            read -rp "$(echo -e "  ${BOLD}Base image tag${NC} [${cur_shown}]: ")" tag
            ROCM_BASE_IMAGE="${tag:-$cur_ref}"
            ;;
        *) err "Invalid choice: $choice"; prompt_rocm_variant; return ;;
    esac
}

# validate_rocm_base_image fails before anything is built when the tag does not
# exist, so a typo costs seconds instead of a partial 20 GB pull. Uses the
# container runtime that container mode already requires, rather than adding a
# curl or skopeo dependency.
validate_rocm_base_image() {
    local ref
    ref="$(rocm_base_image)"
    if [[ -z "$CONTAINER_CMD" ]]; then
        warn "Cannot check ${ref} yet — no container runtime detected. The build will fail later if that tag does not exist."
        return 0
    fi
    log "Checking the ROCm base image exists: ${ref}"
    if ! $CONTAINER_CMD manifest inspect "$ref" >/dev/null 2>&1; then
        err "ROCm base image not found: ${ref}"
        log "The tag may not exist, or the registry may be unreachable."
        log "Available tags: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags"
        exit 1
    fi
    ok "Base image found."
}

# rocm_gfx_kernel_floor echoes "<major.minor> <family>" for the detected card:
# the kernel version in which the AMD driver gained support for it. Nothing when
# the card is unknown or has no figure worth quoting, in which case no version
# comparison is made rather than an invented one.
#
# The kernel a card needs depends on the CARD, not on ROCm — ROCm 10 itself
# supports RDNA 1 and newer (see ROCM10_GFX_TARGETS). Figures from the Gentoo
# AMDGPU wiki, which tracks them per generation:
# https://wiki.gentoo.org/wiki/AMDGPU
# The CDNA parts (gfx908/90a/942/950) are deliberately absent: no figure is
# quoted for them here, so none is asserted.
rocm_gfx_kernel_floor() {
    case "${1:-}" in
        gfx1200|gfx1201)                     echo "6.12 RDNA 4" ;;
        gfx1150|gfx1151|gfx1152|gfx1153)     echo "6.10 RDNA 3.5" ;;
        gfx1100|gfx1101|gfx1102|gfx1103)     echo "6.0 RDNA 3" ;;
        gfx1030|gfx1031|gfx1032|gfx1033|gfx1034|gfx1035|gfx1036)
                                             echo "5.9 RDNA 2" ;;
        gfx1010|gfx1011|gfx1012|gfx1013)     echo "5.3 RDNA 1" ;;
        gfx900|gfx902|gfx904|gfx906|gfx909|gfx90c)
                                             echo "4.15 Vega" ;;
    esac
}

# check_rocm_host_kernel reports whether this machine's GPU and driver can be
# expected to work with the experimental variant, before a 20 GB pull.
#
# Two separate questions, and neither is a refusal — the experimental path is
# for people who want to try things:
#   1. Does ROCm 10 ship code for this card at all?
#   2. Is the host kernel new enough for the AMD driver to drive it? The answer
#      depends on the card, not on ROCm, so the floor is looked up per target
#      and no comparison is made when the target is unknown.
check_rocm_host_kernel() {
    local running major minor drv="(not reported)" floor family fl_major fl_minor
    running="$(uname -r)"
    major="${running%%.*}"
    minor="${running#*.}"; minor="${minor%%.*}"
    major="${major//[^0-9]/}"; minor="${minor//[^0-9]/}"
    major="${major:-0}"; minor="${minor:-0}"

    # Present only for DKMS installs; an in-tree amdgpu has no version file,
    # which is not a problem, so it reads as "not reported".
    if [[ -r /sys/module/amdgpu/version ]]; then
        drv="$(cat /sys/module/amdgpu/version)"
    fi

    # The compute interface. Without it nothing else matters.
    if [[ ! -e /dev/kfd ]]; then
        warn "/dev/kfd is missing — the AMD compute driver is not loaded, so the container will not reach the GPU."
    fi

    # 1. Is the card in ROCm 10's target list?
    if [[ -n "$AMD_GFX_TARGET" ]]; then
        if [[ " ${ROCM10_GFX_TARGETS} " != *" ${AMD_GFX_TARGET} "* ]]; then
            warn "ROCm 10 does not ship code for ${AMD_GFX_TARGET}, which is what this machine reports."
            log "It covers RDNA 1 and newer and the CDNA cards. Yours may still work through"
            log "HSA_OVERRIDE_GFX_VERSION, but it is not a supported target."
        fi
    else
        log "Could not read this machine's GPU target, so its driver requirements were not checked."
    fi

    # 2. Is the kernel new enough for THIS card's driver?
    read -r floor family <<<"$(rocm_gfx_kernel_floor "$AMD_GFX_TARGET")"
    if [[ -z "$floor" ]]; then
        log "Host kernel ${running}, amdgpu ${drv}. No kernel requirement is on record for"
        log "${AMD_GFX_TARGET:-this GPU}, so it was not checked — if the container cannot see the card,"
        log "the host driver is the first thing to suspect."
        return 0
    fi
    fl_major="${floor%%.*}"
    fl_minor="${floor##*.}"
    if (( major < fl_major || (major == fl_major && minor < fl_minor) )); then
        warn "Host kernel ${running} is older than ${floor}, where the AMD driver gained ${family} support (amdgpu ${drv})."
        log "That is a requirement of your card and the host kernel, not of ROCm — the"
        log "container carries no kernel components. Continuing anyway; the GPU may not work."
    else
        ok "Host kernel ${running} is new enough for ${family} (needs ${floor}; amdgpu ${drv})."
    fi
}

# warn_rocm_variant_switch says what changing ROCm version means for existing
# llama.cpp builds. It never deletes anything: a binary linked against one ROCm
# still works if you switch back.
warn_rocm_variant_switch() {
    local env_file="${SCRIPT_DIR}/.env" prev="stable" prev_img="" cur_img="" v
    if [[ -f "$env_file" ]]; then
        v="$(grep '^ROCM_VARIANT=' "$env_file" 2>/dev/null | cut -d= -f2-)" || true
        if [[ -n "$v" ]]; then prev="$v"; fi
        prev_img="$(grep '^ROCM_BASE_IMAGE=' "$env_file" 2>/dev/null | cut -d= -f2-)" || true
    fi
    if [[ "${ROCM_VARIANT:-stable}" == "next" ]]; then
        cur_img="$(rocm_base_image)"
    fi
    if [[ "$prev" == "${ROCM_VARIANT:-stable}" && "$prev_img" == "$cur_img" ]]; then
        return 0
    fi

    echo ""
    warn "The ROCm version is changing."
    log "  from: $(rocm_variant_label "$prev" "$prev_img")"
    log "    to: $(rocm_variant_label "${ROCM_VARIANT:-stable}" "$cur_img")"
    log "Your llama.cpp builds were compiled against the old ROCm and will not load"
    log "under the new one. Rebuild them from the Builds page once this finishes."
    log "Nothing is deleted — switching back makes the old builds work again."
    echo ""
}

# build_local_binary compiles this working tree for the container and echoes
# nothing; it sets LOCAL_BINARY to the path the compose build arg expects,
# relative to the build context (the repository root).
#
# Why a binary and not a package: the image still installs the released
# .deb/.rpm, which is what brings in cmake, ninja, git and the compiler that
# llama.cpp builds need, plus the systemd units. Only the program is replaced.
# That keeps the dev image faithful to a release in everything except the code,
# and needs nothing but the Go toolchain — no goreleaser, no nfpm.
#
# The web templates and static files are compiled in via go:embed, so the one
# file carries the UI as well as the code.
build_local_binary() {
    need_cmd go || fatal "--from-source needs the Go toolchain to build this tree. Install Go, or drop --from-source to install the released package."

    local out="dist/llama-toolchest-local" arch version commit
    arch="$(go env GOARCH)"
    version="$(git -C "$SCRIPT_DIR" describe --tags --always --dirty 2>/dev/null || echo dev)"
    commit="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"

    log "Building llama-toolchest from this tree (${version}, linux/${arch})..."
    mkdir -p "${SCRIPT_DIR}/dist"
    # Same settings as the release build (.goreleaser.yaml): CGO off so the
    # binary runs on whatever base image the variant uses, and the version
    # stamped in so the container reports the tree it came from rather than
    # claiming to be a release.
    ( cd "$SCRIPT_DIR" && CGO_ENABLED=0 GOOS=linux GOARCH="$arch" \
        go build -trimpath \
          -ldflags "-s -w -X main.version=${version} -X main.commit=${commit} -X main.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          -o "$out" ./cmd/llama-toolchest ) \
        || fatal "Building from source failed. Fix the build, or drop --from-source."

    LOCAL_BINARY="$out"
    export LOCAL_BINARY
    ok "Built ${out} (${version})."
}

# caddy_hash_password hashes plaintext via the caddy image, feeding the secret
# on STDIN (newline-terminated — hash-password reads a line) so it never appears
# in argv / ps. Echoes the resulting hash. Caddy's basic_auth accepts both
# bcrypt ($2…) and argon2id ($argon2id$…); the image default is bcrypt.
caddy_hash_password() {
    local plaintext="$1" hash
    hash="$(printf '%s\n' "$plaintext" | $CONTAINER_CMD run --rm -i "$CADDY_IMAGE" caddy hash-password 2>/dev/null || true)"
    hash="${hash%%$'\n'*}"
    [[ "$hash" == \$* ]] || return 1
    printf '%s' "$hash"
}

# render_caddyfile fills Caddyfile.template into ./Caddyfile. Values are
# substituted with bash parameter expansion (not sed) so the bcrypt hash's
# special chars ($ / .) are handled literally.
render_caddyfile() {
    local tpl="${SCRIPT_DIR}/Caddyfile.template" out="${SCRIPT_DIR}/Caddyfile"
    [[ -f "$tpl" ]] || fatal "Missing $tpl"
    local content; content="$(<"$tpl")"
    content="${content//@@GLOBAL@@/$CADDY_GLOBAL}"
    content="${content//@@TLS@@/$CADDY_TLS}"
    content="${content//@@SITE_ADDRESS@@/$CADDY_SITE_ADDRESS}"
    content="${content//@@CHAT_ADDRESS@@/$CADDY_CHAT_ADDRESS}"
    content="${content//@@AUTH_USER@@/$AUTH_USER}"
    content="${content//@@AUTH_HASH@@/$AUTH_HASH}"
    printf '%s\n' "$content" > "$out"
    chmod 600 "$out" 2>/dev/null || true
    ok "Rendered ./Caddyfile — review it before exposing the server."
}

# configure_secure resolves all secure-install settings (flags > prompts >
# defaults), computes the bcrypt hash, and renders ./Caddyfile. Container-only.
configure_secure() {
    # Offer it interactively when the user didn't pass --secure/--no-secure.
    if [[ "$SECURE_EXPLICIT" != true && "$INTERACTIVE" == true && "$ASSUME_YES" != true ]]; then
        if prompt_confirm "Enable access control + HTTPS (Caddy reverse proxy)?"; then
            SECURE=true
        fi
    fi
    [[ "$SECURE" == true ]] || return 0

    print_secure_disclaimer

    # ── TLS mode ──
    if [[ -z "$TLS_MODE" ]]; then
        if [[ "$INTERACTIVE" == true && "$ASSUME_YES" != true ]]; then
            echo -e "${BOLD}TLS certificate${NC}"
            echo "  1) Self-signed / internal CA — works on any LAN/IP; browser shows a warning"
            echo "  2) Let's Encrypt            — trusted cert; needs a public domain + ports 80/443"
            local c; read -rp "$(echo -e "  ${BOLD}Choose${NC} [1]: ")" c
            case "${c:-1}" in 1) TLS_MODE="self-signed" ;; 2) TLS_MODE="letsencrypt" ;; *) fatal "Invalid choice: $c" ;; esac
        else
            TLS_MODE="self-signed"
        fi
    fi
    case "$TLS_MODE" in
        self-signed|letsencrypt) ;;
        *) fatal "--tls must be 'self-signed' or 'letsencrypt' (got '$TLS_MODE')" ;;
    esac

    # ── Domain / ACME email (letsencrypt) ──
    if [[ "$TLS_MODE" == letsencrypt ]]; then
        if [[ -z "$DOMAIN" && "$INTERACTIVE" == true && "$ASSUME_YES" != true ]]; then
            read -rp "$(echo -e "  ${BOLD}Public domain (FQDN)${NC}: ")" DOMAIN
        fi
        [[ -n "$DOMAIN" ]] || fatal "Let's Encrypt requires a domain (--domain <fqdn>)"
        if [[ -z "$ACME_EMAIL" && "$INTERACTIVE" == true && "$ASSUME_YES" != true ]]; then
            read -rp "$(echo -e "  ${BOLD}ACME account email${NC} (optional, recommended): ")" ACME_EMAIL
        fi
        [[ -n "$ACME_EMAIL" ]] || warn "No ACME email set; Let's Encrypt registration will be anonymous."
    fi

    # ── Admin credentials ──
    [[ -n "$AUTH_USER" ]] || AUTH_USER="admin"
    if [[ "$INTERACTIVE" == true && "$ASSUME_YES" != true && -z "$AUTH_HASH" && -z "$AUTH_PASS_FILE" && -z "$AUTH_PASS" ]]; then
        local u; read -rp "$(echo -e "  ${BOLD}Admin username${NC} [${AUTH_USER}]: ")" u
        [[ -n "$u" ]] && AUTH_USER="$u"
    fi

    # ── Resolve the bcrypt hash (plaintext never touches argv) ──
    if [[ -z "$AUTH_HASH" ]]; then
        local pw=""
        if [[ -n "$AUTH_PASS_FILE" ]]; then
            [[ -r "$AUTH_PASS_FILE" ]] || fatal "Cannot read --auth-pass-file: $AUTH_PASS_FILE"
            pw="$(< "$AUTH_PASS_FILE")"; pw="${pw%%$'\n'*}"
        elif [[ -n "$AUTH_PASS" ]]; then
            pw="$AUTH_PASS"
        elif [[ "$INTERACTIVE" == true && "$ASSUME_YES" != true ]]; then
            local pw2
            read -rsp "$(echo -e "  ${BOLD}Admin password${NC}: ")" pw; echo ""
            read -rsp "$(echo -e "  ${BOLD}Confirm password${NC}: ")" pw2; echo ""
            [[ "$pw" == "$pw2" ]] || fatal "Passwords do not match"
        fi
        [[ -n "$pw" ]] || fatal "No admin password. Provide --auth-hash, --auth-pass-file, or AUTH_PASS (or run interactively)."
        log "Hashing password via the caddy image..."
        AUTH_HASH="$(caddy_hash_password "$pw")" || fatal "Failed to hash password (is the caddy:2 image pullable?). Alternatively pass --auth-hash."
        unset pw
    fi

    # ── Derive Caddy site addresses + the external URL for the app's links ──
    local ext_host
    if [[ "$TLS_MODE" == letsencrypt ]]; then
        CADDY_SITE_ADDRESS="$DOMAIN"
        CADDY_CHAT_ADDRESS="${DOMAIN}:${CADDY_CHAT_PORT:-8080}"
        CADDY_GLOBAL=$'{\n\temail '"${ACME_EMAIL}"$'\n}'
        CADDY_TLS="# Let's Encrypt cert is provisioned automatically for the domain"
        ext_host="$DOMAIN"
    else
        # Bare port + internal CA. on_demand mints a self-signed cert for
        # whatever hostname/IP the client uses (localhost, LAN IP, hostname),
        # so a bare :443 site still presents a certificate instead of failing
        # the TLS handshake.
        CADDY_SITE_ADDRESS=":${CADDY_HTTPS_PORT:-443}"
        CADDY_CHAT_ADDRESS=":${CADDY_CHAT_PORT:-8080}"
        CADDY_GLOBAL="# self-signed / internal CA — no ACME"
        CADDY_TLS=$'tls internal {\n\t\ton_demand\n\t}'
        ext_host="${DOMAIN:-$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo localhost)}"
    fi
    local scheme_port=""
    [[ "${CADDY_HTTPS_PORT:-443}" != "443" ]] && scheme_port=":${CADDY_HTTPS_PORT}"
    SECURE_EXTERNAL_URL="https://${ext_host}${scheme_port}"

    render_caddyfile
}

container_up() {
    if has_quadlet; then
        log "Starting llama-toolchest via systemd (Quadlet)..."
        # Start in order (app first, then Caddy). Each service's Requires=/After=
        # also pulls its deps, but starting explicitly keeps output clear.
        local svc
        while read -r svc; do
            systemctl_cmd start "$svc"
        done < <(quadlet_services)
    else
        $(compose_cmd) up -d
    fi
}

container_down() {
    if has_quadlet; then
        log "Stopping llama-toolchest via systemd (Quadlet)..."
        # Stop in reverse order (Caddy first, then app).
        local svc
        while read -r svc; do
            systemctl_cmd stop "$svc" 2>/dev/null || true
        done < <(quadlet_services | tac)
    else
        $(compose_cmd) down
    fi
}

# stop_existing_containers clears any running/stopped llama-toolchest containers
# before a fresh `up`, so their published ports are free. This covers copies a
# plain `compose up` won't manage: a different compose project name, a Quadlet
# service, the pre-rename "llamactl" container, or a manually-started one.
stop_existing_containers() {
    # If Quadlet owns the containers, stop the services first so systemd doesn't
    # immediately restart what we remove.
    if has_quadlet; then
        local svc
        while read -r svc; do
            systemctl_cmd stop "$svc" 2>/dev/null || true
        done < <(quadlet_services | tac)
    fi
    local cname
    for cname in llama-toolchest-caddy llama-toolchest llamactl; do
        if container_exists "$cname"; then
            log "Removing existing container so the new one can bind its ports: $cname"
            $CONTAINER_CMD stop "$cname" 2>/dev/null || true
            $CONTAINER_CMD rm "$cname" 2>/dev/null || true
        fi
    done
}

container_install() {
    migrate_legacy_volume
    stop_existing_containers
    preflight_port_check
    ensure_unprivileged_ports
    write_env_file
    $(compose_cmd) up -d --build
}

container_rebuild() {
    local quadlet_active=false
    has_quadlet && quadlet_active=true

    migrate_legacy_volume
    container_down
    stop_existing_containers
    preflight_port_check
    ensure_unprivileged_ports
    write_env_file
    $(compose_cmd) build --no-cache

    if [[ "$quadlet_active" == true ]]; then
        container_up
    else
        $(compose_cmd) up -d
    fi
}

# Quick: pull the latest released llama-toolchest package and reinstall
# it inside the existing image, reusing every other cached layer (ROCm
# SDK, base OS install, etc.). The Dockerfile's package-install RUN
# step takes LLAMA_TOOLCHEST_VERSION as a build arg via the compose
# file; passing a different version invalidates only that layer's
# cache. If we can't reach the GH API we fall back to "latest" (the
# Dockerfile resolves it at build time, but layer caching still
# works — same arg → cache hit, different actual remote version → no
# refresh).
container_quick_rebuild() {
    migrate_legacy_volume
    container_down
    write_env_file

    local latest=""
    if declare -F host_latest_release_version >/dev/null 2>&1; then
        latest="$(host_latest_release_version 2>/dev/null || true)"
    fi
    if [[ -n "$latest" ]]; then
        log "Resolved latest release: v${latest}"
        export LLAMA_TOOLCHEST_VERSION="$latest"
    else
        warn "Couldn't resolve latest release tag; falling back to 'latest' (build-time resolution)."
    fi

    $(compose_cmd) up -d --build
}

container_logs() {
    if has_quadlet; then
        local journal_args=()
        local svc
        while read -r svc; do
            journal_args+=(-u "$svc")
        done < <(quadlet_services)
        journalctl --user "${journal_args[@]}" -n 100 -f
    else
        $(compose_cmd) logs -f
    fi
}

# ─── Auto-start (enable/disable) ─────────────────────────────────────────────

readonly QUADLET_USER_DIR="${HOME}/.config/containers/systemd"
readonly QUADLET_SYSTEM_DIR="/etc/containers/systemd"
readonly PODMAN_SERVICE_NAME="llama-toolchest"
# Fully-qualified so it resolves under Podman hosts with no unqualified-search
# registries configured (Docker ignores the docker.io/library/ prefix).
readonly CADDY_IMAGE="docker.io/library/caddy:2"

quadlet_dir() {
    if [[ $EUID -eq 0 ]]; then
        echo "$QUADLET_SYSTEM_DIR"
    else
        echo "$QUADLET_USER_DIR"
    fi
}

systemctl_cmd() {
    if [[ $EUID -eq 0 ]]; then
        systemctl "$@"
    else
        systemctl --user "$@"
    fi
}

# Get the restart policy for the container
get_restart_policy() {
    $CONTAINER_CMD inspect --format '{{.HostConfig.RestartPolicy.Name}}' llama-toolchest 2>/dev/null || echo ""
}

is_autostart_enabled() {
    if [[ "$CONTAINER_CMD" == "docker" ]]; then
        # Docker: check if the container has a restart policy that auto-starts
        local policy
        policy="$(get_restart_policy)"
        [[ "$policy" == "always" || "$policy" == "unless-stopped" ]]
    else
        # Podman rootless needs a Quadlet unit to survive reboot
        # Podman rootful can use restart policy like Docker
        if [[ $EUID -eq 0 ]]; then
            local policy
            policy="$(get_restart_policy)"
            [[ "$policy" == "always" || "$policy" == "unless-stopped" ]]
        else
            [[ -f "$(quadlet_dir)/${PODMAN_SERVICE_NAME}.container" ]]
        fi
    fi
}

get_volume_name() {
    # Get the actual volume name from the running container, or fall back to compose default
    $CONTAINER_CMD inspect --format '{{range .Mounts}}{{.Name}}{{end}}' llama-toolchest 2>/dev/null \
        || echo "llama-toolchest-data"
}

# quadlet_services echoes the systemd service names for the active mode, in
# START order (app first, then Caddy). Reverse for stop. The .network unit is
# pulled in automatically by the containers' Network= dependency, so it isn't
# listed here.
quadlet_services() {
    echo "${PODMAN_SERVICE_NAME}.service"
    if [[ "$SECURE" == true ]]; then
        echo "${PODMAN_SERVICE_NAME}-caddy.service"
    fi
}

# generate_quadlet_app emits the app .container unit. In secure mode it joins
# the shared network and publishes NOTHING (Caddy fronts it); otherwise it
# publishes the UI/inference ports to the host as before.
generate_quadlet_app() {
    local image_name="localhost/llama-toolchest:latest"
    local volume_name
    volume_name="$(get_volume_name)"
    local gpu_args=""

    if [[ "$GPU_VENDOR" == "cuda" ]]; then
        # --ipc=host: a tensor-parallel split has the cards exchange partial
        # results through NCCL, which needs shared memory. The container's
        # own /dev/shm is too small for it, and the load aborts with
        # "CUDA error: unhandled system error" after the weights have loaded.
        gpu_args="AddDevice=nvidia.com/gpu=all
PodmanArgs=--ipc=host"
    elif [[ "$GPU_VENDOR" == "rocm" ]]; then
        local hsa_env=""
        if [[ -n "$AMD_GFX_VERSION" ]]; then
            hsa_env="Environment=HSA_OVERRIDE_GFX_VERSION=${AMD_GFX_VERSION}"
        fi
        gpu_args="AddDevice=/dev/kfd
AddDevice=/dev/dri
SecurityLabelDisable=true
PodmanArgs=--ipc=host
GroupAdd=${HOST_VIDEO_GID:-video}
GroupAdd=${HOST_RENDER_GID:-render}
${hsa_env}"
    fi

    # Secure mode: no published ports (Caddy is the only network-facing
    # container), join the shared network, and set ExternalURL so links render
    # with the externally reachable https URL.
    local net_args="" ext_url_arg=""
    if [[ "$SECURE" == true ]]; then
        net_args="Network=${PODMAN_SERVICE_NAME}.network"
        ext_url_arg="Environment=LLAMA_TOOLCHEST_EXTERNAL_URL=${SECURE_EXTERNAL_URL:-https://localhost}"
    fi

    cat <<EOF
# Auto-generated by llama-toolchest setup.sh
# GPU backend: ${GPU_VENDOR}
# Runtime: ${CONTAINER_CMD}
# Mode: $([[ "$SECURE" == true ]] && echo "secure (behind Caddy)" || echo "direct")

[Unit]
Description=llama-toolchest - local LLM management
After=network-online.target

[Container]
Image=${image_name}
ContainerName=llama-toolchest
$(if [[ "$SECURE" == true ]]; then
    echo "PublishPort=127.0.0.1:${LLAMA_TOOLCHEST_PORT}:3000"
    echo "PublishPort=127.0.0.1:${SECURE_APP_DEBUG_INF_PORT:-8081}:8080"
else
    echo "PublishPort=${LLAMA_TOOLCHEST_PORT}:3000"
    echo "PublishPort=${LLAMA_TOOLCHEST_INFERENCE_PORT}:8080"
fi)
${net_args}
${ext_url_arg}
Volume=${volume_name}:/data:z
${gpu_args}

[Service]
Restart=on-failure
TimeoutStartSec=900

[Install]
WantedBy=default.target
EOF
}

# generate_quadlet_network emits the shared network for the secure two-container
# stack, giving Caddy name-based DNS to reach the app (matches compose).
generate_quadlet_network() {
    cat <<EOF
# Auto-generated by llama-toolchest setup.sh
[Unit]
Description=llama-toolchest reverse-proxy network

[Network]
NetworkName=llama-toolchest
EOF
}

# generate_quadlet_caddy emits the Caddy .container unit (secure mode only).
# Ordered after the app so the upstream is up first.
generate_quadlet_caddy() {
    cat <<EOF
# Auto-generated by llama-toolchest setup.sh
[Unit]
Description=llama-toolchest Caddy reverse proxy (HTTPS + admin login)
After=${PODMAN_SERVICE_NAME}.service
Requires=${PODMAN_SERVICE_NAME}.service

[Container]
Image=${CADDY_IMAGE}
ContainerName=${PODMAN_SERVICE_NAME}-caddy
Network=${PODMAN_SERVICE_NAME}.network
PublishPort=${CADDY_HTTP_PORT:-80}:80
PublishPort=${CADDY_HTTPS_PORT:-443}:443
PublishPort=${CADDY_CHAT_PORT:-8080}:8080
Volume=${SCRIPT_DIR}/Caddyfile:/etc/caddy/Caddyfile:ro,z
Volume=${PODMAN_SERVICE_NAME}-caddy-data:/data:z
Volume=${PODMAN_SERVICE_NAME}-caddy-config:/config:z

[Service]
Restart=on-failure
TimeoutStartSec=300

[Install]
WantedBy=default.target
EOF
}

# quadlet_write_units (re)writes all unit files for the active mode into the
# Quadlet dir, removing any stale ones from the other mode first.
quadlet_write_units() {
    local qdir="$1"
    mkdir -p "$qdir"
    quadlet_remove_unit_files "$qdir"   # clear stale caddy/network when toggling off
    generate_quadlet_app > "${qdir}/${PODMAN_SERVICE_NAME}.container"
    if [[ "$SECURE" == true ]]; then
        generate_quadlet_network > "${qdir}/${PODMAN_SERVICE_NAME}.network"
        generate_quadlet_caddy   > "${qdir}/${PODMAN_SERVICE_NAME}-caddy.container"
    fi
}

# quadlet_remove_unit_files deletes every unit file this script may have written
# (app + caddy + network), regardless of the current mode, so toggling secure
# off doesn't leave orphaned units behind.
quadlet_remove_unit_files() {
    local qdir="$1"
    rm -f "${qdir}/${PODMAN_SERVICE_NAME}.container" \
          "${qdir}/${PODMAN_SERVICE_NAME}-caddy.container" \
          "${qdir}/${PODMAN_SERVICE_NAME}.network"
}

autostart_enable() {
    if is_autostart_enabled; then
        ok "Auto-start is already enabled"
        return
    fi

    if [[ "$CONTAINER_CMD" == "docker" || $EUID -eq 0 ]]; then
        # Docker (any user) or rootful Podman: restart policy on the container(s).
        if ! $CONTAINER_CMD inspect llama-toolchest &>/dev/null; then
            fatal "Container 'llama-toolchest' not found. Run './setup.sh install' first."
        fi
        log "Setting restart policy to 'unless-stopped'..."
        $CONTAINER_CMD update --restart unless-stopped llama-toolchest
        if [[ "$SECURE" == true ]] && $CONTAINER_CMD inspect llama-toolchest-caddy &>/dev/null; then
            $CONTAINER_CMD update --restart unless-stopped llama-toolchest-caddy
        fi
        ok "Auto-start enabled"
    else
        # Podman rootless: need Quadlet + linger to survive reboot. In secure
        # mode this writes the app, Caddy, and network units together.
        local qdir
        qdir="$(quadlet_dir)"

        if [[ "$SECURE" == true && ! -f "${SCRIPT_DIR}/Caddyfile" ]]; then
            fatal "Secure mode but ./Caddyfile is missing — run './setup.sh install --secure' first."
        fi

        # Stop any containers running outside systemd so the units can take over
        # without a name conflict.
        local was_running=false cname
        for cname in llama-toolchest llama-toolchest-caddy; do
            if podman container exists "$cname" 2>/dev/null; then
                was_running=true
                log "Stopping existing container $cname so systemd can take over..."
                podman stop "$cname" 2>/dev/null || true
                podman rm "$cname" 2>/dev/null || true
            fi
        done

        if [[ "$SECURE" == true ]]; then
            log "Pulling caddy image..."
            podman pull "$CADDY_IMAGE" >/dev/null 2>&1 \
                || warn "Could not pre-pull $CADDY_IMAGE; the unit will pull on first start."
        fi

        log "Installing Quadlet units in ${qdir}..."
        quadlet_write_units "$qdir"
        systemctl_cmd daemon-reload

        # Enable lingering so user services run without an active login session
        local linger_status
        linger_status="$(loginctl show-user "$USER" --property=Linger 2>/dev/null || true)"
        if [[ "$linger_status" != *"yes"* ]]; then
            log "Enabling loginctl linger for user $USER..."
            if ! loginctl enable-linger "$USER" 2>/dev/null; then
                run_sudo loginctl enable-linger "$USER"
            fi
        fi

        # Start now if something was already running; otherwise the units
        # auto-activate on boot via WantedBy= (no explicit enable needed).
        if [[ "$was_running" == true ]]; then
            container_up
        fi

        ok "Auto-start enabled via Podman Quadlet"
        echo ""
        echo "  llama-toolchest will auto-start on boot."
        if [[ "$SECURE" == true ]]; then
            echo "  Manage with: systemctl --user {start,stop,status} ${PODMAN_SERVICE_NAME} ${PODMAN_SERVICE_NAME}-caddy"
        else
            echo "  Manage with: systemctl --user {start,stop,status} ${PODMAN_SERVICE_NAME}"
        fi
    fi
}

autostart_disable() {
    if ! is_autostart_enabled; then
        ok "Auto-start is already disabled"
        return
    fi

    if [[ "$CONTAINER_CMD" == "docker" || $EUID -eq 0 ]]; then
        log "Setting restart policy to 'no'..."
        $CONTAINER_CMD update --restart no llama-toolchest 2>/dev/null || true
        $CONTAINER_CMD update --restart no llama-toolchest-caddy 2>/dev/null || true
        ok "Auto-start disabled"
    else
        local qdir
        qdir="$(quadlet_dir)"

        log "Removing Quadlet units from ${qdir}..."
        # Stop services (Caddy first, then app), drop the unit files, reload.
        local svc
        while read -r svc; do
            systemctl_cmd stop "$svc" 2>/dev/null || true
        done < <(quadlet_services | tac)
        quadlet_remove_unit_files "$qdir"
        systemctl_cmd daemon-reload
        ok "Auto-start disabled"
    fi
}

# ─── Uninstall ────────────────────────────────────────────────────────────────

container_uninstall() {
    local actions=()
    local has_autostart=false
    local has_container=false
    local has_image=false
    local image_name="localhost/llama-toolchest:latest"

    # Check what exists
    if is_autostart_enabled; then
        has_autostart=true
        actions+=("Disable auto-start on boot")
    fi

    if container_exists llama-toolchest; then
        has_container=true
        actions+=("Stop and remove container 'llama-toolchest'")
    fi

    # Secure installs also have a Caddy sidecar.
    local has_caddy=false
    if container_exists llama-toolchest-caddy; then
        has_caddy=true
        actions+=("Stop and remove container 'llama-toolchest-caddy'")
    fi

    # `image exists` is podman-specific; on Docker we use inspect.
    local image_check
    if [[ "$CONTAINER_CMD" == "docker" ]]; then
        image_check="docker image inspect"
    else
        image_check="$CONTAINER_CMD image exists"
    fi
    if $image_check "$image_name" >/dev/null 2>&1; then
        has_image=true
        actions+=("Remove image '${image_name}'")
    fi

    if [[ ${#actions[@]} -eq 0 ]]; then
        ok "Nothing to uninstall — llama-toolchest is not installed"
        return
    fi

    echo ""
    echo -e "${BOLD}The following will be removed:${NC}"
    echo ""
    local i=1
    for action in "${actions[@]}"; do
        echo -e "  ${i}. ${action}"
        ((i++))
    done
    echo ""
    # Compose prefixes volume names with the project directory name
    local volume_name
    volume_name="$($CONTAINER_CMD inspect --format '{{range .Mounts}}{{.Name}}{{end}}' llama-toolchest 2>/dev/null || echo "llama-toolchest-data")"
    echo -e "  ${YELLOW}Note:${NC} The data volume (models, builds, config) will be kept."
    echo -e "        To remove it: ${CONTAINER_CMD} volume rm ${volume_name}"
    if [[ "$has_caddy" == true ]]; then
        echo -e "        Caddy certs/state are kept too: ${CONTAINER_CMD} volume rm llama-toolchest-caddy-data llama-toolchest-caddy-config"
    fi
    echo ""

    if ! prompt_confirm "Proceed with uninstall?"; then
        echo "Aborted."
        exit 0
    fi

    echo ""

    if [[ "$has_autostart" == true ]]; then
        autostart_disable
    fi

    if [[ "$has_caddy" == true ]]; then
        log "Stopping and removing Caddy container..."
        $CONTAINER_CMD stop llama-toolchest-caddy 2>/dev/null || true
        $CONTAINER_CMD rm llama-toolchest-caddy 2>/dev/null || true
        ok "Caddy container removed"
    fi

    if [[ "$has_container" == true ]]; then
        log "Stopping and removing container..."
        $CONTAINER_CMD stop llama-toolchest 2>/dev/null || true
        $CONTAINER_CMD rm llama-toolchest 2>/dev/null || true
        ok "Container removed"
    fi

    if [[ "$has_image" == true ]]; then
        log "Removing image..."
        $CONTAINER_CMD rmi "$image_name" 2>/dev/null || true
        ok "Image removed"
    fi

    echo ""
    ok "llama-toolchest uninstalled"
}

# ─── Summary and confirmation ─────────────────────────────────────────────────

print_summary() {
    local cf df
    cf="$(compose_file)"
    df="$(dockerfile)"

    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  llama-toolchest setup${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}GPU${NC}           ${GPU_INFO}"
    echo -e "  ${CYAN}Backend${NC}       ${GPU_VENDOR}"
    echo -e "  ${CYAN}Runtime${NC}       ${CONTAINER_VERSION}"
    echo -e "  ${CYAN}Compose${NC}       ${COMPOSE_VERSION}"
    echo -e "  ${CYAN}Distro${NC}        ${DISTRO_NAME}"
    echo -e "  ${CYAN}Dockerfile${NC}    ${df}"
    echo -e "  ${CYAN}Compose file${NC}  ${cf}"
    if [[ "$GPU_VENDOR" == "rocm" ]]; then
        if [[ "${ROCM_VARIANT:-stable}" == "next" ]]; then
            echo -e "  ${CYAN}ROCm${NC}          experimental — $(rocm_base_image)"
        else
            echo -e "  ${CYAN}ROCm${NC}          stable (7.2.4, Fedora)"
        fi
    fi
    if [[ -n "${LOCAL_BINARY:-}" ]]; then
        echo -e "  ${CYAN}Program${NC}       built from this tree (${LOCAL_BINARY})"
    else
        echo -e "  ${CYAN}Program${NC}       released package from GitHub"
    fi
    echo -e "  ${CYAN}UI port${NC}       ${LLAMA_TOOLCHEST_PORT}"
    echo -e "  ${CYAN}Inference port${NC} ${LLAMA_TOOLCHEST_INFERENCE_PORT}"
    if [[ -n "$LLAMA_TOOLCHEST_MODELS_DIR" ]]; then
        echo -e "  ${CYAN}Models dir${NC}    ${LLAMA_TOOLCHEST_MODELS_DIR}"
    fi
    if [[ -n "$AMD_GFX_VERSION" ]]; then
        echo -e "  ${CYAN}HSA Override${NC}  ${AMD_GFX_VERSION}"
    fi

    local autostart_status="disabled"
    if is_autostart_enabled; then
        autostart_status="enabled"
    fi
    echo -e "  ${CYAN}Auto-start${NC}    ${autostart_status}"
    echo ""

    if [[ ! -f "${SCRIPT_DIR}/${cf}" ]]; then
        err "Compose file ${cf} not found!"
        echo "  Available compose files:"
        ls -1 "${SCRIPT_DIR}"/docker-compose.*.yml 2>/dev/null | sed 's|.*/|    |' || echo "    (none)"
        echo ""
        fatal "Cannot proceed without compose file"
    fi

    if [[ ${#ACTIONS[@]} -gt 0 ]]; then
        echo -e "  ${BOLD}Actions:${NC}"
        local i=1
        for action in "${ACTIONS[@]}"; do
            echo -e "    ${i}. ${action}"
            ((i++))
        done
        echo ""
    fi

    # Show if any actions need sudo
    if [[ ${#PREREQS[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}Note:${NC} Prerequisite steps require sudo"
        echo ""
    fi
}

# require_value validates that a value-bearing flag actually got a value (and
# not the next flag). Echoes the value so callers can capture it.
require_value() {
    local flag="$1" val="${2:-}"
    if [[ -z "$val" || "$val" == -* ]]; then
        fatal "Flag $flag requires a value"
    fi
    printf '%s' "$val"
}

prompt_confirm() {
    local prompt="$1"
    # --yes auto-confirms; a non-TTY without --yes can't answer, so fail loudly
    # rather than block or silently assume.
    if [[ "$ASSUME_YES" == true ]]; then
        return 0
    fi
    if [[ "$INTERACTIVE" != true ]]; then
        fatal "Non-interactive shell and --yes not given; cannot answer: '${prompt}'. Re-run with --yes (and any required flags)."
    fi
    local answer
    read -rp "$(echo -e "${BOLD}${prompt}${NC} [Y/n] ")" answer
    case "${answer:-Y}" in
        [Yy]*|"") return 0 ;;
        *)        return 1 ;;
    esac
}

# ─── Library: host install + service helpers ─────────────────────────────────

# shellcheck source=scripts/lib/service.sh
source "${SCRIPT_DIR}/scripts/lib/service.sh"
# shellcheck source=scripts/lib/host.sh
source "${SCRIPT_DIR}/scripts/lib/host.sh"
# shellcheck source=scripts/lib/migrate.sh
source "${SCRIPT_DIR}/scripts/lib/migrate.sh"

# Detect which install modes are present on this machine. Echoes one of:
#   host       — only a host install (binary on disk + systemd unit)
#   container  — only a container install (named llama-toolchest exists)
#   both       — both are present (e.g. mid-migration); caller must disambiguate
#   none       — nothing installed
# Used by up/down/logs/status to route to the right backend without forcing
# the user to remember which install they have.
detect_install_mode() {
    local has_host=false has_container=false
    host_is_installed && has_host=true

    # Container detection needs $CONTAINER_CMD; populate it if not set yet.
    if [[ -z "${CONTAINER_CMD:-}" ]]; then
        detect_container_runtime 2>/dev/null || true
    fi
    if [[ -n "${CONTAINER_CMD:-}" ]] && container_exists llama-toolchest; then
        has_container=true
    fi

    if [[ "$has_host" == true && "$has_container" == true ]]; then
        echo "both"
    elif [[ "$has_host" == true ]]; then
        echo "host"
    elif [[ "$has_container" == true ]]; then
        echo "container"
    else
        echo "none"
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────

usage() {
    cat <<'USAGE'
llama-toolchest setup — auto-detect GPU + container runtime, build & run

Usage: ./setup.sh <command> [--host|--container]

Install modes:
  --container     (default) Run llama-toolchest inside a Docker/Podman
                  container. Isolates the GPU SDK install; works with the
                  existing flow.
  --host          Run llama-toolchest directly on the host system. By
                  default, downloads and installs the latest released
                  .deb/.rpm package (use --from-source to opt out).
  --from-package  Implies --host. Download the latest GitHub release
                  for this distro+arch and install via dnf/apt. This is
                  the default for --host.
  --from-source   Build the binary from the local source via `go build`
                  instead of installing a release package. Useful for
                  testing uncommitted changes. On its own it implies
                  --host; with --container it builds this tree into the
                  container image instead, where the released package is
                  still installed for its dependencies and only the
                  program itself is replaced.

Backend selection (host mode, additive — stack flags to install multiple
SDKs in a single run; each implies --host):
  --cuda          Install the CUDA toolkit
  --rocm          Install the ROCm SDK
  --vulkan        Install the Vulkan SDK (loader headers, glslc, vulkaninfo)
                  Vulkan-only is fine; combined with --cuda or --rocm gives
                  you a portable fallback alongside the vendor backend.

Experimental ROCm container (AMD, container mode; these do NOT imply --host,
unlike --rocm above — ROCm 10 is published only as a container image):
  --rocm-next     Build on AMD's own ROCm image instead of Fedora + RPMs,
                  which is the only way to get ROCm 10. Supports RDNA 1 and
                  newer and the CDNA cards; the host kernel your card needs
                  depends on the card, and setup.sh checks it.
  --rocm-image T  Same, with the base image pinned to tag T (for example
                  10.0.0-full). A full image reference works too. See the
                  ROCm section of the README.

If no backend flag and no GPU= env is set, setup.sh auto-detects the
primary GPU and asks whether to add Vulkan as a secondary SDK. With no
--host/--container flag, `install` asks interactively which mode to use.

Pin a specific released version with `LT_VERSION=1.0.0 ./setup.sh
install --host` to avoid hitting the GH API for the latest tag.

Secure install (container only — Caddy reverse proxy for HTTPS + a single
admin login in front of the UI/API):
  --secure / --no-secure   Enable/disable the Caddy reverse proxy. Without
                           either flag, `install` asks interactively.
  --tls self-signed|letsencrypt   Cert strategy (default self-signed).
  --domain <fqdn>          Public domain (required for Let's Encrypt).
  --acme-email <email>     ACME account email (Let's Encrypt; recommended).
  --auth-user <name>       Admin username (default "admin").
  --auth-hash <bcrypt>     Precomputed bcrypt hash (from `caddy hash-password`).
                           Preferred for non-interactive installs.
  --auth-pass-file <path>  Read the admin password from a file (hashed during
                           install). The plaintext is NEVER accepted on argv;
                           use this, AUTH_PASS=… in the env, or --auth-hash.

  ⚠  BEST-EFFORT SECURITY: the bundled Caddy config is a convenience starting
     point, AS-IS and not hardened for any specific threat model. You are
     responsible for auditing it before relying on it. See docs/secure.md.

Non-interactive:
  -y, --yes                Skip confirmation prompts; required when stdin is
                           not a TTY. Combine with the flags above (and
                           --host/--container) for a fully scripted install.

Lifecycle:
  install     Detect, install prerequisites, build, and start
  uninstall   Stop and remove (container or host install)
  migrate     Move state between container and host installs. Takes
              one of --to-host / --to-container. Snapshots the
              registry, brings up the destination, restores the
              registry. Refuses if the destination side already
              exists. builds.json is wiped — llama.cpp must be
              rebuilt on the new side (binaries don't cross the
              container/host boundary).
  quick       Container only: pull the latest released package and
              reinstall it inside the existing image. Reuses the GPU
              SDK / base-OS layers — only the package-install layer
              re-runs. Use this for routine upgrades to a new release.
  rebuild     Container only: full rebuild with no cache, then start.
              Use this when you've changed the Dockerfile or want to
              refresh the GPU SDK layers as well.

Runtime (works for both host and container installs — auto-detected):
  up          Start a stopped install (container or host service)
  down        Stop a running install
  logs        Follow logs (Ctrl-C to stop)

Auto-start (container only — for host installs use systemctl directly):
  enable      Enable auto-start on boot
  disable     Disable auto-start on boot

Info:
  status      Show detected environment and planned actions, then exit
              (works for both modes; --host adds backend SDK report)
  deps        Verify all packages needed to build and run, with copy-
              paste install commands for anything missing. --host
              checks the host package, llama.cpp build toolchain
              (cmake/ninja/git/gcc), and per-backend GPU SDKs (cuda/
              rocm/vulkan). Without --host, checks container runtime,
              compose, and GPU integration (NVIDIA toolkit / SELinux).
              Exits non-zero if anything is missing.
  detect      Print detected GPU backend (cuda/rocm/cpu/vulkan/metal) and exit.
              For AMD it also prints which ROCm container variant would be used
  help        Show this help message

Host-mode lifecycle is managed via systemd directly:
  systemctl --user start|stop|status llama-toolchest    (user install)
  sudo systemctl start|stop|status llama-toolchest      (system install, when run as root)

Environment variables:
  GPU=cuda|rocm|vulkan|cpu      Override GPU auto-detection (single
                                backend; for multi-SDK host installs
                                use --cuda/--rocm/--vulkan flags instead)
  ROCM_VARIANT=stable|next      Which ROCm container to build (container mode,
                                AMD only). Same as --rocm-next; stored in .env
                                so rebuild/up/down reuse it.
  ROCM_BASE_IMAGE=<tag|ref>     ROCm base image for the experimental variant,
                                e.g. 10.0.0-full. Implies ROCM_VARIANT=next.
  RUNTIME=docker|podman         Override container runtime auto-detection
  INSTALL_MODE=host|container   Same as --host / --container
  ASSUME_YES=1                  Same as --yes
  SECURE=1                      Same as --secure
  AUTH_PASS=…                   Admin password for a scripted secure install
                                (hashed during install; keeps it off argv)

Port configuration is stored in .env (see .env.example for details).
You can edit .env directly instead of using the interactive setup.

Examples:
  ./setup.sh install                    # detect, install prereqs, build & run (asks mode)
  ./setup.sh install --host             # install latest released package on the host
  ./setup.sh install --rocm --vulkan    # host install, install both ROCm and Vulkan SDKs
  ./setup.sh install --rocm-image 10.0.0-full  # container install on ROCm 10 (experimental)
  ./setup.sh install --vulkan           # host install, Vulkan SDK only (cross-vendor)
  ./setup.sh install --from-source      # host install, build from local source
  ./setup.sh install --container --from-source  # container running this working tree
  ./setup.sh install --secure           # container install behind Caddy (asks TLS/login)
  ./setup.sh install --secure --tls self-signed \
      --auth-user admin --auth-hash "$(caddy hash-password --plaintext s3cret)" --yes
                                        # fully scripted self-signed secure install
  AUTH_PASS="$(cat ~/pw)" ./setup.sh install --secure --tls letsencrypt \
      --domain llm.example.com --acme-email me@example.com --yes
                                        # fully scripted Let's Encrypt secure install
  ./setup.sh migrate --to-host          # snapshot container state, install on host
  ./setup.sh migrate --to-container     # opposite direction
  ./setup.sh status --host              # show host install status
  ./setup.sh uninstall --host           # remove host install
  ./setup.sh status                     # container dry run
  ./setup.sh quick                      # fast container rebuild
  GPU=cpu ./setup.sh install            # force CPU-only backend (container)
  LT_VERSION=1.0.0 ./setup.sh install --host  # pin a specific package version
USAGE
}

main() {
    local command="${1:-help}"
    shift || true

    cd "$SCRIPT_DIR"

    # ── Parse flags after the command ──
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host)         INSTALL_MODE="host"; INSTALL_MODE_EXPLICIT=true ;;
            --container)    INSTALL_MODE="container"; INSTALL_MODE_EXPLICIT=true ;;
            # --from-source means "build from this tree" in BOTH modes. It
            # still implies --host on its own, as it always has; combined with
            # --container it builds the tree into the container image instead.
            # The mode is only defaulted when the user has not named one, so
            # --container --from-source and --from-source --container agree.
            --from-source)
                HOST_INSTALL_MODE="source"
                FROM_SOURCE=true
                if [[ "$INSTALL_MODE_EXPLICIT" != true ]]; then
                    INSTALL_MODE="host"; INSTALL_MODE_EXPLICIT=true
                fi
                ;;
            --from-package) INSTALL_MODE="host"; HOST_INSTALL_MODE="package"; INSTALL_MODE_EXPLICIT=true ;;
            # Backend flags are additive and host-mode-only (vulkan can't run
            # in containers without driver passthrough we don't manage; for
            # consistency cuda/rocm flags also imply --host). Stack them:
            # `./setup.sh install --rocm --vulkan` installs both SDKs.
            --cuda)         INSTALL_MODE="host"; INSTALL_MODE_EXPLICIT=true; HOST_SDK_BACKENDS+=("cuda") ;;
            --rocm)         INSTALL_MODE="host"; INSTALL_MODE_EXPLICIT=true; HOST_SDK_BACKENDS+=("rocm") ;;
            --vulkan)       INSTALL_MODE="host"; INSTALL_MODE_EXPLICIT=true; HOST_SDK_BACKENDS+=("vulkan") ;;
            # The ROCm container variant. Unlike --rocm (which installs the host
            # SDK), these are container-only and do NOT imply --host: ROCm 10
            # exists only as a container image. --rocm-image both selects the
            # experimental variant and pins the release, so one flag is enough.
            --rocm-next)      ROCM_VARIANT="next"; ROCM_VARIANT_EXPLICIT=true ;;
            --rocm-image)     ROCM_BASE_IMAGE="$(require_value "$1" "${2:-}")"; ROCM_VARIANT="next"; ROCM_VARIANT_EXPLICIT=true; shift ;;
            --rocm-image=*)   ROCM_BASE_IMAGE="${1#*=}"; ROCM_VARIANT="next"; ROCM_VARIANT_EXPLICIT=true ;;
            # Direction flags for the `migrate` command. Mutually exclusive
            # — passing both is an error. Each implies its target mode for
            # the purpose of post-flag dispatch.
            --to-host)
                [[ -n "$MIGRATE_DIRECTION" ]] && { err "--to-host and --to-container are mutually exclusive"; exit 1; }
                MIGRATE_DIRECTION="to-host" ;;
            --to-container)
                [[ -n "$MIGRATE_DIRECTION" ]] && { err "--to-host and --to-container are mutually exclusive"; exit 1; }
                MIGRATE_DIRECTION="to-container" ;;
            # ── Non-interactive ──
            -y|--yes)       ASSUME_YES=true ;;
            # ── Secure (Caddy reverse proxy) install ──
            --secure)       SECURE=true;  SECURE_EXPLICIT=true ;;
            --no-secure)    SECURE=false; SECURE_EXPLICIT=true ;;
            --tls)          TLS_MODE="$(require_value "$1" "${2:-}")"; shift ;;
            --tls=*)        TLS_MODE="${1#*=}" ;;
            --domain)       DOMAIN="$(require_value "$1" "${2:-}")"; shift ;;
            --domain=*)     DOMAIN="${1#*=}" ;;
            --acme-email)   ACME_EMAIL="$(require_value "$1" "${2:-}")"; shift ;;
            --acme-email=*) ACME_EMAIL="${1#*=}" ;;
            --auth-user)    AUTH_USER="$(require_value "$1" "${2:-}")"; shift ;;
            --auth-user=*)  AUTH_USER="${1#*=}" ;;
            --auth-hash)    AUTH_HASH="$(require_value "$1" "${2:-}")"; shift ;;
            --auth-hash=*)  AUTH_HASH="${1#*=}" ;;
            --auth-pass-file)   AUTH_PASS_FILE="$(require_value "$1" "${2:-}")"; shift ;;
            --auth-pass-file=*) AUTH_PASS_FILE="${1#*=}" ;;
            -h|--help)      usage; exit 0 ;;
            *)
                err "Unknown flag: $1"
                echo ""
                usage
                exit 1
                ;;
        esac
        shift || true
    done

    # --secure implies container mode; reject an explicit host combo.
    if [[ "$SECURE" == true ]]; then
        if [[ "$INSTALL_MODE" == "host" ]]; then
            fatal "--secure (Caddy reverse proxy) is container-only; it can't be combined with --host."
        fi
        INSTALL_MODE="container"
    fi

    # Interactivity: a non-TTY stdin means we can't prompt. Prompts then either
    # use flag/env values or fail fast asking for --yes (see prompt_confirm).
    [[ -t 0 ]] || INTERACTIVE=false

    # ── Validate flag/command combinations ──
    if [[ -n "$MIGRATE_DIRECTION" && "$command" != "migrate" ]]; then
        err "--to-host / --to-container are only valid with the 'migrate' command"
        exit 1
    fi

    # ── Validate command ──
    case "$command" in
        install|uninstall|up|down|rebuild|quick|logs|detect|status|enable|disable|migrate|deps) ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            err "Unknown command: $command"
            echo ""
            usage
            exit 1
            ;;
    esac

    # ── Interactive install: offer host vs container when not chosen ──
    # Only `install` prompts; stateful commands auto-detect below. A --secure
    # run already forced container mode (and INSTALL_MODE_EXPLICIT), so it skips.
    if [[ "$command" == "install" && "$INSTALL_MODE_EXPLICIT" != true && "$INTERACTIVE" == true && "$ASSUME_YES" != true ]]; then
        prompt_install_mode
    fi

    # ── Auto-route stateful commands by detecting which install is present ──
    # up/down/logs are mode-agnostic from the user's POV: they want to start,
    # stop, or tail logs for "the install that's there." Detect it instead of
    # forcing the user to remember --host vs --container. install/migrate/etc.
    # set up new state and don't auto-route.
    if [[ "$INSTALL_MODE_EXPLICIT" != "true" ]]; then
        case "$command" in
            up|down|logs)
                local detected
                detected="$(detect_install_mode)"
                case "$detected" in
                    host)      INSTALL_MODE="host" ;;
                    container) INSTALL_MODE="container" ;;
                    both)
                        err "Both host and container installs detected — pass --host or --container to disambiguate."
                        exit 1
                        ;;
                    none)
                        err "No llama-toolchest install detected. Run './setup.sh install' first."
                        exit 1
                        ;;
                esac
                ;;
        esac
    fi

    # ── uninstall: detect and remove whatever is installed (both, if present) ──
    # Without an explicit --host/--container, uninstall used to assume container
    # and silently leave a host install behind. Detect and route instead.
    if [[ "$command" == "uninstall" && "$INSTALL_MODE_EXPLICIT" != "true" ]]; then
        local detected
        detected="$(detect_install_mode)"
        case "$detected" in
            none)
                ok "Nothing to uninstall — llama-toolchest is not installed."
                exit 0
                ;;
            host)      INSTALL_MODE="host" ;;       # handled by the host short-circuit
            container) INSTALL_MODE="container" ;;  # handled by the container path
            both)
                log "Both host and container installs detected — removing both."
                detect_container_runtime
                detect_distro
                load_env_ports
                container_uninstall
                echo ""
                host_uninstall
                exit 0
                ;;
        esac
    fi

    # ── migrate: hybrid command, needs both container and host context ──
    if [[ "$command" == "migrate" ]]; then
        if [[ -z "$MIGRATE_DIRECTION" ]]; then
            err "migrate requires --to-host or --to-container"
            echo ""
            usage
            exit 1
        fi
        detect_distro
        detect_container_runtime
        if [[ -n "${GPU:-}" ]]; then
            GPU_VENDOR="$GPU"
            GPU_INFO="(manually set: $GPU)"
        else
            detect_gpu
        fi
        case "$MIGRATE_DIRECTION" in
            to-host)
                # Resolve SDK backends the same way `install --host` does:
                # explicit flags win; otherwise auto-detect + prompt for vulkan.
                if [[ ${#HOST_SDK_BACKENDS[@]} -eq 0 ]]; then
                    HOST_SDK_BACKENDS=("$GPU_VENDOR")
                    if [[ -z "${GPU:-}" ]] && [[ "$GPU_VENDOR" == "cuda" || "$GPU_VENDOR" == "rocm" ]]; then
                        if prompt_confirm "Also install the Vulkan SDK on the host?"; then
                            HOST_SDK_BACKENDS+=("vulkan")
                        fi
                    fi
                fi
                for _b in "${HOST_SDK_BACKENDS[@]:-}"; do
                    if [[ "$_b" != "vulkan" ]]; then GPU_VENDOR="$_b"; break; fi
                done
                unset _b
                migrate_to_host
                ;;
            to-container)
                load_env_ports
                migrate_to_container
                ;;
        esac
        exit 0
    fi

    # ── Host mode: short circuit before any container detection ──
    if [[ "$INSTALL_MODE" == "host" ]]; then
        # Detect GPU so HSA_OVERRIDE_GFX_VERSION etc. are populated for the
        # unit override.
        if [[ -n "${GPU:-}" ]]; then
            GPU_VENDOR="$GPU"
            GPU_INFO="(manually set: $GPU)"
        else
            detect_gpu
        fi
        # Detect distro family so host_install_from_package knows which
        # package manager + extension to use, and host_install_gpu_sdk
        # knows which package names to install.
        detect_distro

        # Resolve which backend SDKs to install on the host. Precedence:
        #   1. Explicit --cuda/--rocm/--vulkan flags (already in HOST_SDK_BACKENDS)
        #   2. GPU= env override (single backend, back-compat)
        #   3. Auto-detected primary, plus an interactive prompt to also
        #      install the Vulkan SDK (since AMD/NVIDIA users frequently
        #      want it as a portable fallback).
        # The first non-vulkan entry wins as GPU_VENDOR for systemd unit
        # overrides; vulkan-only is fine, just no overrides needed.
        if [[ "$command" == "install" && ${#HOST_SDK_BACKENDS[@]} -eq 0 ]]; then
            if [[ -n "${GPU:-}" ]]; then
                HOST_SDK_BACKENDS=("$GPU_VENDOR")
            else
                HOST_SDK_BACKENDS=("$GPU_VENDOR")
                if [[ "$GPU_VENDOR" == "cuda" || "$GPU_VENDOR" == "rocm" ]]; then
                    if prompt_confirm "Also install the Vulkan SDK on the host? (provides a portable backend in addition to $GPU_VENDOR)"; then
                        HOST_SDK_BACKENDS+=("vulkan")
                    fi
                fi
            fi
        fi
        # Pick a primary for unit overrides: first non-vulkan backend.
        for _b in "${HOST_SDK_BACKENDS[@]:-}"; do
            if [[ "$_b" != "vulkan" ]]; then GPU_VENDOR="$_b"; break; fi
        done
        unset _b

        # The experimental ROCm variant is container-only. Say so rather than
        # silently installing 7.2.4 when the user asked for 10: repo.radeon.com
        # has no 10.x packages in any path, so a host install cannot provide it.
        if [[ "${ROCM_VARIANT:-}" == "next" && "$command" == "install" ]]; then
            warn "ROCm 10 is published only as a container image, so a host install cannot use it."
            log "Host mode will install the ROCm SDK from repo.radeon.com as usual."
            log "For ROCm 10, use container mode:  ./setup.sh install --rocm-image ${ROCM_NEXT_DEFAULT_TAG}"
            echo ""
        fi

        case "$command" in
            install)   host_install ;;
            uninstall) host_uninstall ;;
            status)    host_status ;;
            deps)      host_deps ;;
            detect)    echo "$GPU_VENDOR" ;;
            up)        host_up ;;
            down)      host_down ;;
            logs)      host_logs ;;
            enable|disable|rebuild|quick)
                err "'$command' is not supported in --host mode."
                log "For host installs, manage autostart via systemd directly:"
                if [[ "$(host_scope 2>/dev/null)" == "system" ]]; then
                    echo "    sudo systemctl enable|disable llama-toolchest"
                else
                    echo "    systemctl --user enable|disable llama-toolchest"
                fi
                exit 1
                ;;
        esac
        exit 0
    fi

    # ── Detect everything ──
    if [[ -n "${GPU:-}" ]]; then
        GPU_VENDOR="$GPU"
        GPU_INFO="(manually set: $GPU)"
    else
        detect_gpu
    fi

    # Short-circuit for detect command. The first line stays exactly as it was
    # — it is the machine-readable backend name and something may be parsing it.
    if [[ "$command" == "detect" ]]; then
        echo "$GPU_VENDOR"
        if [[ "$GPU_VENDOR" == "rocm" ]]; then
            load_env_ports
            echo "rocm variant: ${ROCM_VARIANT:-stable}"
            if [[ "${ROCM_VARIANT:-}" == "next" ]]; then
                echo "rocm base image: $(rocm_base_image)"
            fi
        fi
        exit 0
    fi

    detect_container_runtime
    detect_distro
    load_env_ports

    # ── Commands that don't need prerequisite checks ──
    case "$command" in
        up)
            container_up
            ok "llama-toolchest started"
            exit 0
            ;;
        down)
            container_down
            ok "llama-toolchest stopped"
            exit 0
            ;;
        logs)
            container_logs
            exit 0
            ;;
        enable)
            autostart_enable
            exit 0
            ;;
        disable)
            autostart_disable
            exit 0
            ;;
        uninstall)
            container_uninstall
            exit 0
            ;;
        quick)
            log "Quick rebuild (cached)..."
            container_quick_rebuild
            ok "llama-toolchest is running"
            echo ""
            echo "  Web UI:     http://localhost:${LLAMA_TOOLCHEST_PORT}"
            echo ""
            exit 0
            ;;
        deps)
            container_deps
            exit $?
            ;;
    esac

    # ── Which ROCm container (AMD only, container mode) ──
    # Asked only on an interactive install that did not name a variant. A value
    # restored from .env becomes the prompt's DEFAULT (see prompt_rocm_variant),
    # not a reason to skip asking — otherwise a machine already on the
    # experimental variant could never be returned to stable interactively.
    # rebuild/up/down never ask and reuse the stored value.
    if [[ "$GPU_VENDOR" == "rocm" && "$command" == "install" \
          && "$ROCM_VARIANT_EXPLICIT" != true && "$INTERACTIVE" == true && "$ASSUME_YES" != true ]]; then
        prompt_rocm_variant
    fi
    : "${ROCM_VARIANT:=stable}"

    # Pre-flight for the experimental variant, before the summary and the
    # build confirmation so all three messages land together and can be read
    # before committing to a 20 GB pull.
    if [[ "$GPU_VENDOR" == "rocm" && "$ROCM_VARIANT" == "next" ]]; then
        validate_rocm_base_image
        check_rocm_host_kernel
    fi
    if [[ "$GPU_VENDOR" == "rocm" && "$command" != "status" ]]; then
        warn_rocm_variant_switch
    fi

    # ── Build from this tree, when asked (container mode) ──
    # Before the summary, so the summary can say the image will carry a local
    # build, and before the confirmation, so a compile error costs seconds
    # rather than surfacing after the image layers are done.
    if [[ "$FROM_SOURCE" == true && "$command" != "status" ]]; then
        build_local_binary
    fi

    # ── Check prerequisites and show summary (install, rebuild, status) ──
    # AFTER the variant is settled, not before: check_prerequisites builds the
    # actions list by calling dockerfile(), so running it first made the list
    # say Dockerfile.rocm while the summary said Dockerfile.rocm-next.
    check_prerequisites

    print_summary

    if [[ "$command" == "status" ]]; then
        exit 0
    fi

    # ── Configure ports and storage ──
    prompt_ports
    prompt_models_dir

    # ── Secure (Caddy reverse proxy) — install only; rebuild reuses .env ──
    if [[ "$command" == "install" ]]; then
        configure_secure
    fi

    # ── Install prerequisites if needed ──
    if [[ ${#PREREQS[@]} -gt 0 ]]; then
        if ! prompt_confirm "Install prerequisites?"; then
            echo "Aborted."
            exit 0
        fi
        echo ""
        install_prerequisites
        echo ""
    fi

    # ── Build and run ──
    if ! prompt_confirm "Build and start llama-toolchest?"; then
        echo "Aborted."
        exit 0
    fi

    echo ""
    case "$command" in
        install)  container_install ;;
        rebuild)  container_rebuild ;;
    esac

    echo ""
    ok "llama-toolchest is running"
    echo ""
    if [[ "$SECURE" == true ]]; then
        echo "  Web UI:     ${SECURE_EXTERNAL_URL:-https://<host>}      (login required)"
        echo "  Chat UI:    ${SECURE_EXTERNAL_URL:-https://<host>}:${CADDY_CHAT_PORT:-8080}"
        echo "  API:        ${SECURE_EXTERNAL_URL:-https://<host>}/v1   (Bearer api_key)"
        if [[ "${TLS_MODE:-}" == "self-signed" ]]; then
            echo ""
            echo "  Note: self-signed TLS — your browser will warn until you trust"
            echo "        Caddy's root CA. See docs/secure.md."
        fi
        print_secure_disclaimer
    else
        echo "  Web UI:     http://localhost:${LLAMA_TOOLCHEST_PORT}"
        echo "  Inference:  http://localhost:${LLAMA_TOOLCHEST_INFERENCE_PORT}"
    fi
    echo "  Logs:       ./setup.sh logs"
    echo "  Stop:       ./setup.sh down"
    echo "  Auto-start: ./setup.sh enable"
    echo ""
}

main "$@"
