#!/usr/bin/env bash
# scripts/test-lib.sh
#
# Plain-bash tests for the installer libraries (scripts/lib/*.sh). They
# cover the functions that compute something from their inputs and
# environment: scope and path helpers, version and port parsing, package
# name choice, ROCm prefix probing and the migration config rewrite.
# Commands that would touch the system (systemctl, nvidia-smi, rpm,
# apt-cache, uname, ...) are replaced with shell functions, and every file
# the tests write goes into a temporary directory.
#
# Run:   make installer-test     (or: bash scripts/test-lib.sh)
# Exits non-zero if any check fails; prints "ALL PASS" otherwise.
# No dependencies beyond bash and coreutils; the migration path tests also
# need jq and are skipped without it.

set -u

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$TEST_DIR")"
LIB_DIR="${TEST_DIR}/lib"

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

# Results go to a file so checks made inside subshells still count.
RESULTS="${TMP_ROOT}/results"
: > "$RESULTS"

# The libraries print through these helpers from setup.sh. Keep them
# quiet so the test output only shows check results.
log()  { :; }
ok()   { :; }
warn() { :; }
err()  { :; }

pass() { echo "PASS" >> "$RESULTS"; }
fail() {
    echo "FAIL" >> "$RESULTS"
    echo "FAIL: $*" >&2
}

# check_eq <name> <expected> <actual>
check_eq() {
    if [[ "$2" == "$3" ]]; then
        pass
    else
        fail "$1"$'\n'"    expected: [$2]"$'\n'"    actual:   [$3]"
    fi
}

# check_true <name> <command...> — passes when the command returns 0.
check_true() {
    local name="$1"; shift
    if "$@"; then pass; else fail "$name (returned non-zero)"; fi
}

# check_false <name> <command...> — passes when the command returns non-zero.
check_false() {
    local name="$1"; shift
    if "$@"; then fail "$name (returned 0)"; else pass; fi
}

skip() { echo "SKIP: $*"; }

# A fresh HOME per run, so path helpers never point at the real one.
export HOME="${TMP_ROOT}/home"
mkdir -p "$HOME"
unset XDG_CONFIG_HOME XDG_DATA_HOME ROCM_PATH LLAMA_TOOLCHEST_PORT \
      LLAMA_TOOLCHEST_INFERENCE_PORT AMD_GFX_VERSION CONTAINER_CMD \
      HOST_INSTALL_MODE DISTRO_FAMILY DISTRO_ID

# shellcheck source=scripts/lib/service.sh
source "${LIB_DIR}/service.sh"
# shellcheck source=scripts/lib/host.sh
source "${LIB_DIR}/host.sh"
# shellcheck source=scripts/lib/migrate.sh
source "${LIB_DIR}/migrate.sh"

# quadlet_dir lives in setup.sh, which runs main() when sourced. Load only
# that function, plus the two directory constants it reads.
QUADLET_USER_DIR="${HOME}/.config/containers/systemd"
QUADLET_SYSTEM_DIR="/etc/containers/systemd"
eval "$(sed -n '/^quadlet_dir() {/,/^}/p' "${REPO_DIR}/setup.sh")"
# gfx_target_from_kfd, likewise.
eval "$(sed -n '/^gfx_target_from_kfd() {/,/^}/p' "${REPO_DIR}/setup.sh")"

if [[ $EUID -eq 0 ]]; then
    MY_SCOPE="system"
else
    MY_SCOPE="user"
fi

# ─── Scope ───────────────────────────────────────────────────────────────────

test_scope_current_user() {
    check_eq "service_scope matches EUID" "$MY_SCOPE" "$(service_scope)"
    if [[ "$MY_SCOPE" == "user" ]]; then
        check_eq "service_unit_path (user)" \
            "${HOME}/.config/systemd/user/llama-toolchest.service" "$(service_unit_path)"
        check_eq "quadlet_dir (user)" "$QUADLET_USER_DIR" "$(quadlet_dir)"
        check_eq "host_bin_dir (user)" "${HOME}/.local/bin" "$(host_bin_dir)"
        check_eq "host_config_dir (user, no XDG)" \
            "${HOME}/.config/llama-toolchest" "$(host_config_dir)"
        check_eq "host_data_dir (user, no XDG)" \
            "${HOME}/.local/share/llama-toolchest" "$(host_data_dir)"
        check_eq "host_binary_path (user)" \
            "${HOME}/.local/bin/llama-toolchest" "$(host_binary_path)"
        check_eq "host_config_path (user)" \
            "${HOME}/.config/llama-toolchest/llama-toolchest.yaml" "$(host_config_path)"
        check_eq "host_config_dir honours XDG_CONFIG_HOME" "/x/cfg/llama-toolchest" \
            "$(XDG_CONFIG_HOME=/x/cfg host_config_dir)"
        check_eq "host_data_dir honours XDG_DATA_HOME" "/x/data/llama-toolchest" \
            "$(XDG_DATA_HOME=/x/data host_data_dir)"

        (
            systemctl() { echo "systemctl $*"; }
            check_eq "service_systemctl adds --user" \
                "systemctl --user daemon-reload" "$(service_systemctl daemon-reload)"
            check_eq "service_systemctl passes every argument" \
                "systemctl --user enable --now llama-toolchest.service" \
                "$(service_systemctl enable --now "$SERVICE_NAME")"
        )
    fi
}

# EUID is read-only in bash, so the system scope is checked by re-running
# the helpers as root inside an unprivileged user namespace.
test_scope_system() {
    if [[ "$MY_SCOPE" == "system" ]]; then
        skip "already root; system scope covered by the current-user checks"
        return
    fi
    if ! command -v unshare >/dev/null 2>&1 || ! unshare -r true 2>/dev/null; then
        skip "unshare -r not available; system scope not tested"
        return
    fi
    local out
    out="$(HOME="$HOME" unshare -r bash -c '
        set -u
        log() { :; }; ok() { :; }; warn() { :; }; err() { :; }
        source "$1/service.sh"; source "$1/host.sh"
        QUADLET_USER_DIR=u QUADLET_SYSTEM_DIR=/etc/containers/systemd
        eval "$(sed -n "/^quadlet_dir() {/,/^}/p" "$2/setup.sh")"
        systemctl() { echo "systemctl $*"; }
        echo "$(service_scope)|$(service_unit_path)|$(quadlet_dir)|$(host_bin_dir)|$(host_config_dir)|$(host_data_dir)|$(service_systemctl daemon-reload)"
    ' _ "$LIB_DIR" "$REPO_DIR")"
    check_eq "system scope as root" \
        "system|/etc/systemd/system/llama-toolchest.service|/etc/containers/systemd|/usr/local/bin|/etc/llama-toolchest|/var/lib/llama-toolchest|systemctl daemon-reload" \
        "$out"
}

# ─── Versions ────────────────────────────────────────────────────────────────

test_version_ge() {
    check_true  "12.6 >= 12.6"   host_version_ge 12.6 12.6
    check_true  "12.8 >= 12.6"   host_version_ge 12.8 12.6
    check_true  "1.10 >= 1.9"    host_version_ge 1.10 1.9
    check_true  "13.0 >= 12.9"   host_version_ge 13.0 12.9
    check_true  "6.4.1 >= 6.4"   host_version_ge 6.4.1 6.4
    check_false "12.4 >= 12.6"   host_version_ge 12.4 12.6
    check_false "1.9 >= 1.10"    host_version_ge 1.9 1.10
    check_false "6.0 >= 6.1"     host_version_ge 6.0 6.1
}

test_installed_cuda_version() (
    rpm() {
        printf '%s\n' \
            cuda-toolkit-12-4-config-common-12.4.127-1.noarch \
            cuda-toolkit-12-10-config-common-12.10.1-1.noarch \
            cuda-toolkit-12-8-config-common-12.8.90-1.noarch \
            cuda-toolkit-config-common-12.8.90-1.noarch
    }
    DISTRO_FAMILY=debian
    check_eq "host_installed_cuda_version skips non-Fedora" "" "$(host_installed_cuda_version)"
    DISTRO_FAMILY=fedora
    check_eq "host_installed_cuda_version picks the newest" "12.10" "$(host_installed_cuda_version)"
    rpm() { :; }
    check_eq "host_installed_cuda_version with nothing installed" "" "$(host_installed_cuda_version)"
)

test_gpu_compute_cap() (
    nvidia-smi() { printf '8.6\n12.0\n7.5\n'; }
    check_eq "host_gpu_compute_cap picks the highest" "12.0" "$(host_gpu_compute_cap)"
    nvidia-smi() { printf '8.6\n8.6\n8.6\n'; }
    check_eq "host_gpu_compute_cap with identical GPUs" "8.6" "$(host_gpu_compute_cap)"
    nvidia-smi() { printf '8.9 \n'; }
    check_eq "host_gpu_compute_cap strips spaces" "8.9" "$(host_gpu_compute_cap)"
)

test_rocm_version() (
    local fake="${TMP_ROOT}/hipconfig"
    host_find_rocm_tool() { echo "$fake"; }
    printf '#!/bin/sh\necho "6.3.42131-fa1d09cbd"\n' > "$fake"; chmod +x "$fake"
    check_eq "host_rocm_version reads major.minor" "6.3" "$(host_rocm_version)"
    printf '#!/bin/sh\necho "no version here"\n' > "$fake"
    check_false "host_rocm_version rejects output without a version" host_rocm_version
    host_find_rocm_tool() { return 1; }
    check_false "host_rocm_version without hipconfig" host_rocm_version
)

# ─── Packages ────────────────────────────────────────────────────────────────

test_pkg_arch_ext() (
    uname() { echo "$FAKE_ARCH"; }
    FAKE_ARCH=x86_64;  check_eq "host_pkg_arch x86_64"  "amd64" "$(host_pkg_arch)"
    FAKE_ARCH=aarch64; check_eq "host_pkg_arch aarch64" "arm64" "$(host_pkg_arch)"
    FAKE_ARCH=arm64;   check_eq "host_pkg_arch arm64"   "arm64" "$(host_pkg_arch)"
    FAKE_ARCH=riscv64; check_false "host_pkg_arch unknown" host_pkg_arch

    DISTRO_FAMILY=fedora; check_eq "host_pkg_ext fedora" "rpm" "$(host_pkg_ext)"
    DISTRO_FAMILY=debian; check_eq "host_pkg_ext debian" "deb" "$(host_pkg_ext)"
    DISTRO_FAMILY=arch;   check_false "host_pkg_ext arch" host_pkg_ext
    DISTRO_FAMILY="";     check_false "host_pkg_ext empty" host_pkg_ext
)

test_rhel_crb() (
    rpm()   { [[ "$*" == "-E %rhel" ]] && echo "$FAKE_RHEL"; }
    uname() { echo x86_64; }
    FAKE_RHEL=10
    DISTRO_ID=rocky;     check_eq "host_rhel_crb_repo rocky" "crb" "$(host_rhel_crb_repo)"
    DISTRO_ID=almalinux; check_eq "host_rhel_crb_repo almalinux" "crb" "$(host_rhel_crb_repo)"
    DISTRO_ID=centos;    check_eq "host_rhel_crb_repo centos" "crb" "$(host_rhel_crb_repo)"
    DISTRO_ID=ol;        check_eq "host_rhel_crb_repo ol" "ol10_codeready_builder" "$(host_rhel_crb_repo)"
    FAKE_RHEL=9 DISTRO_ID=rhel
    check_eq "host_rhel_crb_repo rhel" \
        "codeready-builder-for-rhel-9-x86_64-rpms" "$(host_rhel_crb_repo)"
    # On Fedora, rpm leaves %rhel unexpanded.
    FAKE_RHEL=%rhel
    DISTRO_ID=fedora; check_false "host_rhel_crb_repo fedora" host_rhel_crb_repo
    DISTRO_ID=ubuntu; check_false "host_rhel_crb_repo ubuntu" host_rhel_crb_repo
    DISTRO_ID=rocky;  check_false "host_rhel_crb_repo with no %rhel" host_rhel_crb_repo

    # dnf resolves ninja-build only once crb is enabled; every call is logged.
    CALLS="${TMP_ROOT}/crb-calls"
    # shellcheck disable=SC2032 # host.sh runs the real dnf under sudo
    dnf() {
        echo "dnf $*" >> "$CALLS"
        case "$*" in
            "-q repoquery ninja-build") [[ -f "${TMP_ROOT}/crb-on" ]] && echo "ninja-build-0:1.11.1-9.el10.x86_64" ;;
            "config-manager --help")    [[ -f "${TMP_ROOT}/plugins" ]] ;;
            "install -y dnf-plugins-core") touch "${TMP_ROOT}/plugins" ;;
            "config-manager --set-enabled crb") touch "${TMP_ROOT}/crb-on" ;;
            *) return 1 ;;
        esac
    }
    run_sudo() { "$@"; }
    subscription-manager() { echo "subscription-manager $*" >> "$CALLS"; touch "${TMP_ROOT}/crb-on"; }
    reset() { rm -f "$CALLS" "${TMP_ROOT}/crb-on" "${TMP_ROOT}/plugins"; touch "$CALLS"; }

    FAKE_RHEL=10 DISTRO_ID=rocky
    prompt_confirm() { return 0; }

    reset; touch "${TMP_ROOT}/crb-on"
    check_true "host_ensure_rhel_crb with crb already enabled" host_ensure_rhel_crb
    check_eq "host_ensure_rhel_crb changes nothing when ninja-build resolves" \
        "dnf -q repoquery ninja-build" "$(cat "$CALLS")"

    reset
    check_true "host_ensure_rhel_crb enables crb" host_ensure_rhel_crb
    check_eq "host_ensure_rhel_crb installs dnf-plugins-core, then enables crb" \
        "dnf -q repoquery ninja-build
dnf config-manager --help
dnf install -y dnf-plugins-core
dnf config-manager --set-enabled crb
dnf -q repoquery ninja-build" "$(cat "$CALLS")"

    reset; touch "${TMP_ROOT}/plugins"
    host_ensure_rhel_crb
    check_false "host_ensure_rhel_crb skips dnf-plugins-core when present" \
        grep -q "dnf-plugins-core" "$CALLS"

    reset
    prompt_confirm() { return 1; }
    check_false "host_ensure_rhel_crb fails when declined" eval "host_ensure_rhel_crb >/dev/null"
    check_false "host_ensure_rhel_crb enables nothing when declined" \
        grep -q "set-enabled" "$CALLS"
    prompt_confirm() { return 0; }

    reset
    FAKE_RHEL=9 DISTRO_ID=rhel
    check_true "host_ensure_rhel_crb on rhel" host_ensure_rhel_crb
    check_true "host_ensure_rhel_crb on rhel uses subscription-manager" \
        grep -qx "subscription-manager repos --enable codeready-builder-for-rhel-9-x86_64-rpms" "$CALLS"

    # Enabling the repo doesn't help: report failure.
    reset
    FAKE_RHEL=10 DISTRO_ID=rocky
    run_sudo() { [[ "$*" == *--set-enabled* ]] && return 0; "$@"; }
    check_false "host_ensure_rhel_crb fails when ninja-build stays missing" host_ensure_rhel_crb

    reset
    FAKE_RHEL=%rhel DISTRO_ID=fedora
    check_true "host_ensure_rhel_crb no-op on fedora" host_ensure_rhel_crb
    check_eq "host_ensure_rhel_crb doesn't touch dnf on fedora" "" "$(cat "$CALLS")"
)

test_apt_helpers() (
    apt-cache() {
        case "$2" in
            good)  printf 'good:\n  Installed: (none)\n  Candidate: 6.4.0-1\n' ;;
            none)  printf 'none:\n  Installed: (none)\n  Candidate: (none)\n' ;;
            *)     return 100 ;;
        esac
    }
    check_true  "host_apt_pkg_available with a candidate" host_apt_pkg_available good
    check_false "host_apt_pkg_available with Candidate: (none)" host_apt_pkg_available none
    check_false "host_apt_pkg_available for an unknown package" host_apt_pkg_available missing

    dpkg-query() {
        case "$3" in
            inst)    printf 'installed' ;;
            removed) printf 'config-files' ;;
            *)       return 1 ;;
        esac
    }
    check_true  "host_dpkg_installed installed" host_dpkg_installed inst
    check_false "host_dpkg_installed removed but not purged" host_dpkg_installed removed
    check_false "host_dpkg_installed unknown" host_dpkg_installed nope

    host_dpkg_installed()     { [[ " $INSTALLED " == *" $1 "* ]]; }
    host_apt_pkg_available()  { [[ " $AVAILABLE " == *" $1 "* ]]; }
    INSTALLED="" AVAILABLE="librocblas-dev"
    check_eq "host_apt_rocm_pkg picks the name apt can resolve" \
        "librocblas-dev" "$(host_apt_rocm_pkg rocblas-dev librocblas-dev)"
    INSTALLED="" AVAILABLE="rocblas-dev librocblas-dev"
    check_eq "host_apt_rocm_pkg prefers the first name" \
        "rocblas-dev" "$(host_apt_rocm_pkg rocblas-dev librocblas-dev)"
    INSTALLED="librocblas-dev" AVAILABLE="rocblas-dev"
    check_eq "host_apt_rocm_pkg says nothing when one is installed" \
        "" "$(host_apt_rocm_pkg rocblas-dev librocblas-dev)"
    check_true "host_apt_rocm_pkg returns 0 when one is installed" \
        host_apt_rocm_pkg rocblas-dev librocblas-dev
    INSTALLED="" AVAILABLE=""
    check_false "host_apt_rocm_pkg with no resolvable name" \
        host_apt_rocm_pkg rocblas-dev librocblas-dev

    host_rocm_prefer_amd_packages() { return 0; }
    check_eq "host_rocm_optional_names rocwmma, AMD first" \
        "rocwmma-dev librocwmma-dev" "$(host_rocm_optional_names rocwmma)"
    host_rocm_prefer_amd_packages() { return 1; }
    check_eq "host_rocm_optional_names rccl, distro first" \
        "librccl-dev rccl-dev" "$(host_rocm_optional_names rccl)"
    check_false "host_rocm_optional_names unknown" host_rocm_optional_names foo
)

test_backend_applicable() {
    check_true  "host_backend_applicable vulkan" host_backend_applicable vulkan
    check_false "host_backend_applicable unknown" host_backend_applicable metal
}

# ─── ROCm prefix probing ─────────────────────────────────────────────────────

test_rocm_prefix_candidates() (
    host_rocm_prefix()    { return 1; }
    host_find_rocm_tool() { return 1; }
    check_eq "candidates with nothing found" \
        $'/opt/rocm\n/usr' "$(host_rocm_prefix_candidates)"

    ROCM_PATH=/custom/rocm
    host_rocm_prefix()    { echo /custom/rocm; }
    host_find_rocm_tool() {
        case "$1" in
            hipconfig) echo /usr/bin/hipconfig ;;
            hipcc)     echo /opt/rocm-6.4.0/bin/hipcc ;;
        esac
    }
    check_eq "candidates are ordered and de-duplicated" \
        $'/custom/rocm\n/usr\n/opt/rocm-6.4.0\n/opt/rocm' "$(host_rocm_prefix_candidates)"
)

test_find_rocm_tool_and_prefix() (
    local root="${TMP_ROOT}/rocm"
    local tool="llt-test-tool-$$"
    mkdir -p "$root/bin"
    printf '#!/bin/sh\n' > "$root/bin/$tool"; chmod +x "$root/bin/$tool"

    ROCM_PATH="$root"
    check_eq "host_find_rocm_tool looks in ROCM_PATH/bin" "$root/bin/$tool" "$(host_find_rocm_tool "$tool")"
    check_false "host_find_rocm_tool misses an absent tool" host_find_rocm_tool "llt-absent-$$"
    check_eq "host_rocm_prefix prefers ROCM_PATH" "$root" "$(host_rocm_prefix)"
)

test_rocm_cmake_pkgs() (
    local a="${TMP_ROOT}/pfx-a" b="${TMP_ROOT}/pfx-b"
    mkdir -p "$a/lib/cmake/hip-lang" "$a/lib64/cmake/hip" "$b/lib/x86_64-linux-gnu/cmake/rocblas"
    touch "$a/lib/cmake/hip-lang/hip-lang-config.cmake" \
          "$a/lib64/cmake/hip/hip-config.cmake" \
          "$b/lib/x86_64-linux-gnu/cmake/rocblas/rocblas-config.cmake"
    host_rocm_prefix_candidates() { printf '%s\n' "$a" "" "$b"; }
    check_true  "cmake pkg in lib/cmake"      host_rocm_have_cmake_pkg hip-lang
    check_true  "cmake pkg in lib64/cmake"    host_rocm_have_cmake_pkg hip
    check_true  "cmake pkg in multiarch dir"  host_rocm_have_cmake_pkg rocblas
    check_false "cmake pkg absent"            host_rocm_have_cmake_pkg hipblas
    check_eq "host_rocm_missing_cmake_pkgs" "hipblas" "$(host_rocm_missing_cmake_pkgs)"
)

test_rocm_optional_absent() (
    local p="${TMP_ROOT}/pfx-opt"
    mkdir -p "$p/include"
    host_rocm_prefix() { echo "$p"; }
    check_eq "both optional extras absent" "rocWMMA RCCL" "$(host_rocm_optional_absent)"
    mkdir -p "$p/include/rocwmma"; touch "$p/include/rocwmma/rocwmma.hpp" "$p/include/rccl.h"
    check_eq "both optional extras present" "" "$(host_rocm_optional_absent)"
    host_rocm_prefix() { return 1; }
    check_eq "no ROCm install" "" "$(host_rocm_optional_absent)"
)

# ─── Config files ────────────────────────────────────────────────────────────

test_effective_port() (
    local cfg; cfg="$(host_config_path)"
    rm -f "$cfg"
    check_eq "port default" "3000" "$(host_effective_port)"
    check_eq "port from env" "4100" "$(LLAMA_TOOLCHEST_PORT=4100 host_effective_port)"

    mkdir -p "$(dirname "$cfg")"
    local line expected
    while IFS='|' read -r line expected; do
        printf 'data_dir: "/x"\n%s\nlog_level: info\n' "$line" > "$cfg"
        check_eq "port from [$line]" "$expected" "$(host_effective_port)"
    done <<'EOF'
listen_addr: ":3001"|3001
listen_addr: :3002|3002
listen_addr: "0.0.0.0:3003"|3003
  listen_addr:   "[::]:3004"  # comment|3004
listen_addr: ""|3000
listen_addr: "localhost"|3000
listen_addr: ':3005'|3005
EOF
    rm -f "$cfg"
)

test_write_config() (
    local cfg; cfg="$(host_config_path)"
    rm -f "$cfg"
    LLAMA_TOOLCHEST_PORT=3555
    host_write_config
    check_true "host_write_config creates the file" test -f "$cfg"
    check_eq "host_write_config port" "3555" "$(host_effective_port)"
    check_true "host_write_config creates the models dir" test -d "$(host_data_dir)/models"
    echo "# edited" > "$cfg"
    host_write_config
    check_eq "host_write_config leaves an existing file alone" "# edited" "$(cat "$cfg")"
    rm -f "$cfg"
)

test_write_unit_override() (
    service_systemctl() { echo "$*" >> "${TMP_ROOT}/systemctl.log"; }
    local dir file
    dir="$(service_unit_path).d"
    file="$dir/override.conf"

    HOST_INSTALL_MODE=source AMD_GFX_VERSION=11.0.0
    host_write_unit_override
    check_eq "override (source + gfx)" \
        "[Service]
ExecStart=
ExecStart=$(host_binary_path) --config $(host_config_path)
Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0" "$(cat "$file" 2>/dev/null)"

    HOST_INSTALL_MODE=package AMD_GFX_VERSION=10.3.0
    host_write_unit_override
    check_eq "override (package + gfx)" \
        "[Service]
Environment=HSA_OVERRIDE_GFX_VERSION=10.3.0" "$(cat "$file" 2>/dev/null)"

    HOST_INSTALL_MODE=package; unset AMD_GFX_VERSION
    host_write_unit_override
    check_false "override removed when nothing to record" test -e "$file"
    check_eq "daemon-reload after each change" \
        $'daemon-reload\ndaemon-reload\ndaemon-reload' "$(cat "${TMP_ROOT}/systemctl.log")"
)

# ─── Migration ───────────────────────────────────────────────────────────────

test_migrate_config() (
    local src="${TMP_ROOT}/src.yaml" dst="${TMP_ROOT}/dst.yaml"
    cat > "$src" <<'EOF'
# test config
listen_addr: ":3456"
data_dir: "/home/u/.local/share/llama-toolchest"
models_dir: "/srv/models"
llama_port: 8080
external_url: "https://ai.example.test"
hf_token: "hf_abc"
api_key: "sk-123"
log_level: "debug"
models_max: 3
auto_start: false
EOF
    LLAMA_TOOLCHEST_PORT=""
    LLAMA_TOOLCHEST_INFERENCE_PORT=8181
    LLAMA_TOOLCHEST_MODELS_DIR=""
    migrate_write_translated_config "$src" "$dst" to-container
    check_eq "to-container sets LLAMA_TOOLCHEST_PORT from listen_addr" "3456" "$LLAMA_TOOLCHEST_PORT"
    # A custom host models folder is bind-mounted at /data/models, so it
    # goes to .env, not into the container config.
    check_eq "to-container moves models_dir to the bind mount" "/srv/models" "$LLAMA_TOOLCHEST_MODELS_DIR"
    check_eq "to-container body" \
'listen_addr: ":3000"
data_dir: "/data"
llama_port: 8181
external_url: "https://ai.example.test"
hf_token: "hf_abc"
api_key: "sk-123"
log_level: "debug"
models_max: 3
auto_start: false' "$(sed '1,2d' "$dst")"
    check_false "_migrate_get is removed afterwards" declare -F _migrate_get

    # Container to host: the host port comes from the container's mapping.
    CONTAINER_CMD=fake_runtime
    fake_runtime() {
        cat <<'JSON'
            "PortBindings": {
                "3000/tcp": [
                    {
                        "HostIp": "",
                        "HostPort": "3999"
                    }
JSON
    }
    unset LLAMA_TOOLCHEST_INFERENCE_PORT
    LLAMA_TOOLCHEST_MODELS_DIR=""
    printf 'listen_addr: ":3000"\ndata_dir: "/data"\nmodels_dir: "/data/models"\n' > "$src"
    migrate_write_translated_config "$src" "$dst" to-host
    check_eq "to-host body with defaults" \
"listen_addr: \":3999\"
data_dir: \"$(host_data_dir)\"
llama_port: 8080
external_url: \"\"
hf_token: \"\"
api_key: \"\"
log_level: \"info\"
models_max: 1
auto_start: true" "$(sed '1,2d' "$dst")"

    # The container's own /data/models path never reaches the host config;
    # a bind-mounted host folder does.
    LLAMA_TOOLCHEST_MODELS_DIR="/srv/models"
    migrate_write_translated_config "$src" "$dst" to-host
    check_eq "to-host takes models_dir from the bind mount" 'models_dir: "/srv/models"' "$(grep '^models_dir:' "$dst")"
    LLAMA_TOOLCHEST_MODELS_DIR=""

    unset CONTAINER_CMD
    migrate_write_translated_config "$src" "$dst" to-host
    check_eq "to-host without a runtime falls back to :3000" \
        'listen_addr: ":3000"' "$(sed -n 3p "$dst")"

    check_false "unknown direction fails" migrate_write_translated_config "$src" "$dst" sideways
)

test_migrate_model_paths() (
    if ! command -v jq >/dev/null 2>&1; then
        skip "jq not installed; migrate_translate_model_paths not tested"
        return
    fi
    local f="${TMP_ROOT}/models.json"
    cat > "$f" <<'EOF'
{"models": [{"id": "a", "file_path": "/data/models/a.gguf"}],
 "configs": {
   "c1": {"mmproj_path": "/data/models/a-mmproj.gguf", "draft_model_path": "/data/models/sub/d.gguf"},
   "c2": {"mmproj_path": "/data/models2/b.gguf"},
   "c3": {"name": "plain"}
 }}
EOF
    migrate_translate_model_paths "$f" "/data/models/" "/home/u/models"
    check_eq "mmproj path rewritten" "/home/u/models/a-mmproj.gguf" "$(jq -r '.configs.c1.mmproj_path' "$f")"
    check_eq "draft path rewritten" "/home/u/models/sub/d.gguf" "$(jq -r '.configs.c1.draft_model_path' "$f")"
    check_eq "similar prefix left alone" "/data/models2/b.gguf" "$(jq -r '.configs.c2.mmproj_path' "$f")"
    check_eq "config without paths unchanged" '{"name":"plain"}' "$(jq -c '.configs.c3' "$f")"
    check_eq "model file_path untouched" "/data/models/a.gguf" "$(jq -r '.models[0].file_path' "$f")"

    check_true "missing file is not an error" \
        migrate_translate_model_paths "${TMP_ROOT}/absent.json" /a /b

    # jq prints its own parse error; keep it out of the test output.
    quiet_translate() { migrate_translate_model_paths "$@" 2>/dev/null; }
    printf 'not json' > "$f"
    check_false "invalid JSON reports failure" quiet_translate "$f" /a /b
    check_eq "invalid JSON left unchanged" "not json" "$(cat "$f")"
)


# ─── ROCm 10 (stable.repo.amd.com) ───────────────────────────────────────────

# Which stable.repo.amd.com directory each distro maps to, from os-release.
test_rocm10_repo_dist() (
    HOST_FS_ROOT="${TMP_ROOT}/osr"
    mkdir -p "${HOST_FS_ROOT}/etc"
    osr() { printf '%s\n' "$@" > "${HOST_FS_ROOT}/etc/os-release"; }

    osr 'ID=ubuntu' 'VERSION_ID="24.04"' 'UBUNTU_CODENAME=noble'
    check_eq "rocm10 repo: Ubuntu 24.04" "ubuntu2404" "$(host_rocm10_repo_dist)"
    osr 'ID=ubuntu' 'VERSION_ID="26.04"' 'UBUNTU_CODENAME=resolute'
    check_eq "rocm10 repo: Ubuntu 26.04" "ubuntu2604" "$(host_rocm10_repo_dist)"
    osr 'ID=linuxmint' 'ID_LIKE="ubuntu debian"' 'VERSION_ID="22"' 'UBUNTU_CODENAME=noble'
    check_eq "rocm10 repo: Mint goes by its Ubuntu base" "ubuntu2404" "$(host_rocm10_repo_dist)"
    osr 'ID=debian' 'VERSION_ID="13"'
    check_eq "rocm10 repo: Debian 13" "debian13" "$(host_rocm10_repo_dist)"
    osr 'ID=rocky' 'ID_LIKE="rhel centos fedora"' 'VERSION_ID="10.0"'
    check_eq "rocm10 repo: Rocky 10" "rhel10" "$(host_rocm10_repo_dist)"
    osr 'ID=fedora' 'VERSION_ID=44'
    check_eq "rocm10 repo: Fedora 44 uses rhel10" "rhel10" "$(host_rocm10_repo_dist)"
    osr 'ID=fedora' 'VERSION_ID=42'
    check_false "rocm10 repo: none for untested Fedora 42" host_rocm10_repo_dist
    osr 'ID=debian' 'VERSION_ID="11"'
    check_false "rocm10 repo: none for Debian 11" host_rocm10_repo_dist
)

# Install-state checks against the layouts measured in distrobox
# (plan/rocm-multi-install.md), built with relative links.
test_rocm10_install_state() (
    fake_install() {  # <dir> — the parts the checks look at
        mkdir -p "$1/lib/cmake/hip-lang" "$1/lib/llvm/bin"
        : > "$1/lib/cmake/hip-lang/hip-lang-config.cmake"
    }

    # Distro ROCm plus 10.1: /opt/rocm is a directory of links into core-10.1.
    HOST_FS_ROOT="${TMP_ROOT}/state-distro"
    fake_install "${HOST_FS_ROOT}/opt/rocm/core-10.1"
    ln -s core-10.1 "${HOST_FS_ROOT}/opt/rocm/core-10"
    ln -s core-10.1/lib "${HOST_FS_ROOT}/opt/rocm/lib"
    check_true  "distro+10.1: 10.x installed" host_rocm10_installed
    check_false "distro+10.1: no old AMD packages" host_rocm_old_amd_installed
    check_eq    "distro+10.1: nothing taken over" "" "$(host_rocm_taken_over_links)"

    # AMD 7.2.4 then 10.1, the user's machine: 10.1 nested inside 7.2.4,
    # whose llvm and amdgcn links it took over.
    HOST_FS_ROOT="${TMP_ROOT}/state-mixed"
    fake_install "${HOST_FS_ROOT}/opt/rocm-7.2.4"
    ln -s rocm-7.2.4 "${HOST_FS_ROOT}/opt/rocm"
    fake_install "${HOST_FS_ROOT}/opt/rocm-7.2.4/core-10.1"
    mkdir -p "${HOST_FS_ROOT}/opt/rocm-7.2.4/core-10.1/amdgcn"
    ln -s core-10.1/lib/llvm "${HOST_FS_ROOT}/opt/rocm-7.2.4/llvm"
    ln -s core-10.1/amdgcn "${HOST_FS_ROOT}/opt/rocm-7.2.4/amdgcn"
    check_true "mixed: 10.x installed" host_rocm10_installed
    check_true "mixed: old AMD packages installed" host_rocm_old_amd_installed
    check_eq   "mixed: taken-over links reported" \
        $'/opt/rocm-7.2.4/llvm\n/opt/rocm-7.2.4/amdgcn' "$(host_rocm_taken_over_links)"

    # AMD 7.2.4 alone: its own llvm link points inside it.
    HOST_FS_ROOT="${TMP_ROOT}/state-old"
    fake_install "${HOST_FS_ROOT}/opt/rocm-7.2.4"
    ln -s lib/llvm "${HOST_FS_ROOT}/opt/rocm-7.2.4/llvm"
    check_false "old only: no 10.x" host_rocm10_installed
    check_true  "old only: old AMD packages installed" host_rocm_old_amd_installed
    check_eq    "old only: nothing taken over" "" "$(host_rocm_taken_over_links)"
)

# Installing 10.1 is refused beside AMD's old packages, before anything is
# asked or touched.
test_rocm10_refused_beside_old_amd() (
    HOST_FS_ROOT="${TMP_ROOT}/refuse"
    mkdir -p "${HOST_FS_ROOT}/opt/rocm-7.2.4/lib"
    host_rocm10_installed() { return 1; }
    prompt_confirm() { echo asked >> "${TMP_ROOT}/refuse.log"; return 0; }
    run_sudo() { echo "$*" >> "${TMP_ROOT}/refuse.log"; }
    check_false "10.1 install refused beside AMD 7.2.4" host_install_rocm10
    check_false "nothing asked or run" test -s "${TMP_ROOT}/refuse.log"
)

# HOST_ROCM is validated, and with no choice possible it stays "distro" —
# the behaviour from before ROCm 10 was an option.
test_choose_rocm() (
    HOST_ROCM=both; host_choose_rocm
    check_eq "HOST_ROCM given: kept" "both" "$HOST_ROCM"
    HOST_ROCM=""; INTERACTIVE=false; ASSUME_YES=false
    host_rocm10_installed() { return 1; }
    host_rocm10_repo_dist() { echo ubuntu2404; }
    host_rocm10_supports_gpu() { return 0; }
    host_choose_rocm
    check_eq "non-interactive: distro" "distro" "$HOST_ROCM"
    HOST_ROCM=""; INTERACTIVE=true
    host_rocm10_repo_dist() { return 1; }
    host_choose_rocm
    check_eq "no 10.x for this distro: distro, unasked" "distro" "$HOST_ROCM"
)

# ROCm 10's alternatives links in /opt/rocm are not AMD's old line, so the
# old repo is never offered beside them; AMD's old line still is.
test_rocm10_not_old_line() (
    host_rocm_apt_repo_configured() { return 1; }
    HOST_FS_ROOT="${TMP_ROOT}/oldline-10"
    mkdir -p "${HOST_FS_ROOT}/opt/rocm/core-10.1/lib"
    ln -s core-10.1/lib "${HOST_FS_ROOT}/opt/rocm/lib"
    check_false "10.x /opt/rocm is not the old AMD line" host_rocm_prefer_amd_packages

    HOST_FS_ROOT="${TMP_ROOT}/oldline-7"
    mkdir -p "${HOST_FS_ROOT}/opt/rocm-7.2.4/lib"
    ln -s rocm-7.2.4 "${HOST_FS_ROOT}/opt/rocm"
    check_true "AMD 7.2.4 is the old AMD line" host_rocm_prefer_amd_packages
)

# The gfx target from KFD topology, with no ROCm installed: the CPU node
# (0) is skipped and the stepping is hex.
test_gfx_target_from_kfd() (
    local t="${TMP_ROOT}/kfd" v want
    for v in 110000:gfx1100 90010:gfx90a 120001:gfx1201 90402:gfx942; do
        want="${v##*:}"; v="${v%%:*}"
        rm -rf "$t"; mkdir -p "$t/0" "$t/1"
        echo "gfx_target_version 0" > "$t/0/properties"
        printf 'cpu_cores_count 0\ngfx_target_version %s\n' "$v" > "$t/1/properties"
        check_eq "gfx_target_from_kfd ${v}" "$want" "$(gfx_target_from_kfd "$t")"
    done
    rm -rf "$t"; mkdir -p "$t/0"
    echo "gfx_target_version 0" > "$t/0/properties"
    check_false "gfx_target_from_kfd with no GPU node" gfx_target_from_kfd "$t"
)

# ─── Run ─────────────────────────────────────────────────────────────────────

test_scope_current_user
test_scope_system
test_version_ge
test_installed_cuda_version
test_gpu_compute_cap
test_rocm_version
test_pkg_arch_ext
test_rhel_crb
test_apt_helpers
test_backend_applicable
test_rocm_prefix_candidates
test_find_rocm_tool_and_prefix
test_rocm_cmake_pkgs
test_rocm_optional_absent
test_effective_port
test_write_config
test_write_unit_override
test_migrate_config
test_migrate_model_paths
test_rocm10_repo_dist
test_rocm10_install_state
test_rocm10_refused_beside_old_amd
test_choose_rocm
test_rocm10_not_old_line
test_gfx_target_from_kfd

passes="$(grep -c '^PASS$' "$RESULTS")"
fails="$(grep -c '^FAIL$' "$RESULTS")"
echo "${passes} passed, ${fails} failed"
if [[ "$fails" -ne 0 ]]; then
    exit 1
fi
echo "ALL PASS"
