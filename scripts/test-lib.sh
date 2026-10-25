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
    LLAMA_TOOLCHEST_PORT=3000 LLAMA_TOOLCHEST_INFERENCE_PORT=8080
    host_write_config
    check_eq "host_write_config gen ports" $'gen_port_start: 8081\ngen_port_count: 8' "$(grep '^gen_port' "$cfg")"
    rm -f "$cfg"
)

# The host generation range starts at 8081 and moves past the UI or
# llama-server port when either is inside it.
test_host_gen_port_start() (
    check_eq "gen start default" "8081" "$(LLAMA_TOOLCHEST_PORT=3000 LLAMA_TOOLCHEST_INFERENCE_PORT=8080 host_gen_port_start)"
    check_eq "gen start past llama port" "8082" "$(LLAMA_TOOLCHEST_PORT=3000 LLAMA_TOOLCHEST_INFERENCE_PORT=8081 host_gen_port_start)"
    check_eq "gen start past both" "8091" "$(LLAMA_TOOLCHEST_PORT=8085 LLAMA_TOOLCHEST_INFERENCE_PORT=8090 host_gen_port_start)"
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
auto_start: false
gen_port_start: 8081
gen_port_count: 8' "$(sed '1,2d' "$dst")"
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
auto_start: true
gen_port_start: 8081
gen_port_count: 8" "$(sed '1,2d' "$dst")"

    # Turned-on generation backends stay on, and the host range moves off a
    # llama-server port that falls inside it.
    printf 'listen_addr: ":3000"\ndata_dir: "/data"\ngeneration_enabled: true\ngen_port_start: 8081\n' > "$src"
    LLAMA_TOOLCHEST_INFERENCE_PORT=8084
    migrate_write_translated_config "$src" "$dst" to-host
    check_eq "to-host keeps generation_enabled" "generation_enabled: true" "$(grep '^generation_enabled:' "$dst")"
    check_eq "to-host gen range clears the llama-server port" "gen_port_start: 8085" "$(grep '^gen_port_start:' "$dst")"
    unset LLAMA_TOOLCHEST_INFERENCE_PORT
    printf 'listen_addr: ":3000"\ndata_dir: "/data"\nmodels_dir: "/data/models"\n' > "$src"

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

# RHEL-family hosts: ROCm comes from EPEL (Fedora's names) or AMD's elN
# repo (rocm-hip-runtime-devel), never the base repos. Each run of dnf
# resolves only what the fake repos below carry: EPEL's set when "epel" is
# on, AMD's once setup has written rocm.repo.
test_rhel_rocm() (
    HOST_FS_ROOT="${TMP_ROOT}/rhel"
    mkdir -p "${HOST_FS_ROOT}/etc/yum.repos.d"
    DISTRO_FAMILY=fedora DISTRO_NAME="Rocky Linux 10.2" FAKE_RHEL=10
    INSTALLED="" EPEL=false
    rpm() {
        case "$1" in
            -E) echo "$FAKE_RHEL" ;;
            -q) [[ " $INSTALLED " == *" $2 "* ]] ;;
            *)  return 1 ;;
        esac
    }
    host_dnf_pkg_available() {
        local avail=""
        [[ "$EPEL" == true ]] && avail+=" rocm-hip-devel rocblas-devel hipblas-devel rocm-cmake"
        host_rocm_dnf_repo_configured \
            && avail+=" rocm-hip-runtime-devel rocblas-devel hipblas-devel rocm-cmake"
        [[ "$avail " == *" $1 "* ]]
    }
    host_rocm_sdk_usable() { return 1; }
    host_rocm_old_amd_installed() { return 1; }
    host_rocm10_installed() { return 1; }
    host_rocm_apt_repo_configured() { return 1; }
    CALLS="${TMP_ROOT}/rhel-calls"
    curl() { echo "curl $*" >> "$CALLS"; [[ "$FAKE_AMD_EL" == true ]]; }
    prompt_confirm() { echo "asked: $*" >> "$CALLS"; [[ "$ANSWER" == yes ]]; }
    run_sudo() {
        if [[ "$1" == tee ]]; then cat > "${HOST_FS_ROOT}$2"; else "$@"; fi
    }
    sudo() { echo "sudo $*" >> "$CALLS"; }
    host_verify_rocm_buildable() { :; }
    host_offer_rocm_optional() { :; }
    reset() {
        rm -f "${HOST_FS_ROOT}"/etc/yum.repos.d/*; : > "$CALLS"
        INSTALLED="" EPEL=false FAKE_AMD_EL=true ANSWER=yes HOST_ROCM=distro
    }

    check_eq "host_rhel_major rocky 10" "10" "$(host_rhel_major)"
    FAKE_RHEL=%rhel
    check_false "host_rhel_major on fedora (%rhel unexpanded)" host_rhel_major
    FAKE_RHEL=10 DISTRO_FAMILY=debian
    check_false "host_rhel_major off the fedora family" host_rhel_major
    DISTRO_FAMILY=fedora

    reset
    echo "baseurl=https://repo.radeon.com/amdgpu/31.10/el/10.1/main/x86_64/" \
        > "${HOST_FS_ROOT}/etc/yum.repos.d/amdgpu.repo"
    check_false "AMD's amdgpu repo is not the ROCm repo" host_rocm_dnf_repo_configured
    echo "baseurl=https://repo.radeon.com/rocm/el10/7.2.4/main" \
        > "${HOST_FS_ROOT}/etc/yum.repos.d/rocm.repo"
    check_true "AMD's ROCm dnf repo is detected" host_rocm_dnf_repo_configured
    check_true "AMD's ROCm dnf repo means AMD's names" host_rocm_prefer_amd_packages

    reset
    INSTALLED="rocm-hip-runtime-devel"
    check_eq "host_rpm_rocm_pkg says nothing when one is installed" \
        "" "$(host_rpm_rocm_pkg rocm-hip-devel rocm-hip-runtime-devel)"
    INSTALLED="" EPEL=true
    check_eq "host_rpm_rocm_pkg picks the name dnf can resolve" \
        "rocm-hip-devel" "$(host_rpm_rocm_pkg rocm-hip-runtime-devel rocm-hip-devel)"
    EPEL=false
    check_false "host_rpm_rocm_pkg with no resolvable name" \
        host_rpm_rocm_pkg rocm-hip-devel rocm-hip-runtime-devel

    # What's missing, per source.
    reset
    check_eq "rhel, no ROCm repo: Fedora's names, all unresolvable" \
        "rocm-hip-devel rocblas-devel hipblas-devel rocm-cmake" \
        "$(host_missing_gpu_sdk_packages rocm)"
    check_eq "rhel, no ROCm repo: all reported unresolvable" \
        "rocm-hip-devel rocblas-devel hipblas-devel rocm-cmake" "$(host_rocm_rhel_unresolvable)"
    EPEL=true
    check_eq "rhel with EPEL: nothing unresolvable" "" "$(host_rocm_rhel_unresolvable)"
    EPEL=false
    echo "baseurl=https://repo.radeon.com/rocm/el10/latest/main" \
        > "${HOST_FS_ROOT}/etc/yum.repos.d/rocm.repo"
    check_eq "rhel with AMD's repo: AMD's HIP runtime name" \
        "rocm-hip-runtime-devel rocblas-devel hipblas-devel rocm-cmake" \
        "$(host_missing_gpu_sdk_packages rocm)"
    EPEL=true
    check_eq "rhel with AMD's repo and EPEL: still AMD's name" \
        "rocm-hip-runtime-devel rocblas-devel hipblas-devel rocm-cmake" \
        "$(host_missing_gpu_sdk_packages rocm)"
    INSTALLED="rocm-hip-runtime-devel rocblas-devel"
    check_eq "rhel: installed components are not missing" \
        "hipblas-devel rocm-cmake" "$(host_missing_gpu_sdk_packages rocm)"
    reset; FAKE_RHEL=%rhel
    check_eq "fedora keeps its own names, unresolved or not" \
        "rocm-hip-devel rocblas-devel hipblas-devel rocm-cmake" \
        "$(host_missing_gpu_sdk_packages rocm)"
    FAKE_RHEL=10

    # Adding AMD's repo.
    reset
    check_true "host_install_amd_rocm_repo_rhel adds the repo" host_install_amd_rocm_repo_rhel
    check_eq "rocm.repo as AMD documents it" \
        "[rocm]
name=ROCm (repo.radeon.com, EL10)
baseurl=https://repo.radeon.com/rocm/el10/latest/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" \
        "$(cat "${HOST_FS_ROOT}/etc/yum.repos.d/rocm.repo")"
    check_true "probes AMD's EL10 repo before asking" \
        grep -qx "curl -fsI https://repo.radeon.com/rocm/el10/latest/main/repodata/repomd.xml" "$CALLS"
    : > "$CALLS"
    check_true "host_install_amd_rocm_repo_rhel with the repo present" host_install_amd_rocm_repo_rhel
    check_eq "nothing asked when the repo is present" "" "$(cat "$CALLS")"

    reset; FAKE_AMD_EL=false
    check_false "no AMD build for this EL: fails" host_install_amd_rocm_repo_rhel
    check_false "no AMD build for this EL: nothing asked" grep -q "^asked" "$CALLS"
    reset; ANSWER=no
    check_false "declined: fails" host_install_amd_rocm_repo_rhel
    check_false "declined: no repo written" test -e "${HOST_FS_ROOT}/etc/yum.repos.d/rocm.repo"

    # The install step, as setup.sh install --host reaches it on Rocky 10.
    reset
    check_true "rhel install: accepts AMD's repo and installs" host_install_gpu_sdk_packages rocm
    check_eq "rhel install: dnf gets AMD's names" \
        "sudo dnf install -y rocm-hip-runtime-devel rocblas-devel hipblas-devel rocm-cmake" \
        "$(grep '^sudo' "$CALLS")"

    reset; ANSWER=no
    check_false "rhel install: declined with nothing resolvable fails" \
        eval "host_install_gpu_sdk_packages rocm >/dev/null"
    check_false "rhel install: declined runs no doomed dnf" grep -q "^sudo" "$CALLS"

    reset; EPEL=true
    check_true "rhel install with EPEL" host_install_gpu_sdk_packages rocm
    check_false "rhel install with EPEL: AMD's repo not offered" grep -q "^curl" "$CALLS"
    check_eq "rhel install with EPEL: dnf gets Fedora's names" \
        "sudo dnf install -y rocm-hip-devel rocblas-devel hipblas-devel rocm-cmake" \
        "$(grep '^sudo' "$CALLS")"

    # Beside ROCm 10, AMD's old line is never offered.
    reset; HOST_ROCM=both
    check_false "rhel install for both, nothing resolvable: fails" \
        eval "host_install_gpu_sdk_packages rocm >/dev/null"
    check_false "rhel install for both: AMD's old repo not offered" grep -q "^asked" "$CALLS"
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

# ─── Generation web UI ports (setup.sh) ──────────────────────────────────────

# load_setup_fns <name>... loads functions from setup.sh without running it.
load_setup_fns() {
    local fn
    for fn in "$@"; do
        eval "$(sed -n "/^${fn}() {/,/^}/p" "${REPO_DIR}/setup.sh")"
    done
}

# gen_setup_env sets the setup.sh globals the port functions read, for a
# plain (not secure) ROCm install with the default ports.
gen_setup_env() {
    SCRIPT_DIR="${TMP_ROOT}/setup"
    rm -rf "$SCRIPT_DIR"; mkdir -p "$SCRIPT_DIR"
    cp "${REPO_DIR}/Caddyfile.template" "$SCRIPT_DIR/"
    GEN_PORT_COUNT=8 SECURE_GEN_APP_PORT_START=18081 SECURE_GEN_DEBUG_INF_PORT=18080
    GEN_PORTS=false GEN_PORTS_EXPLICIT=false GEN_PUBLIC_PORT_START=8081 GEN_PORT_START_EXPLICIT=false
    SECURE=false SECURE_EXPLICIT=false INTERACTIVE=false ASSUME_YES=true ROCM_VARIANT_EXPLICIT=true
    LLAMA_TOOLCHEST_PORT=3000 LLAMA_TOOLCHEST_INFERENCE_PORT=8080 LLAMA_TOOLCHEST_MODELS_DIR=""
    GPU_VENDOR=rocm ROCM_VARIANT=stable AMD_GFX_VERSION="" HOST_VIDEO_GID=39 HOST_RENDER_GID=105
    COMPOSE_CMD="docker compose" CONTAINER_CMD=docker PODMAN_SERVICE_NAME=llama-toolchest
    CADDY_IMAGE=docker.io/library/caddy:2 SECURE_EXTERNAL_URL=https://box:443
    unset SECURE_APP_DEBUG_INF_PORT CADDY_HTTP_PORT CADDY_HTTPS_PORT CADDY_CHAT_PORT
    get_volume_name() { echo llama-toolchest-data; }
    fatal() { exit 1; }
}

# gen_secure_caddy sets the Caddy values configure_secure would derive for a
# self-signed install.
gen_secure_caddy() {
    SECURE=true
    CADDY_GLOBAL="# self-signed / internal CA — no ACME"
    CADDY_TLS=$'tls internal {\n\t\ton_demand\n\t}'
    CADDY_SITE_ADDRESS=":443" CADDY_CHAT_ADDRESS=":8080"
    AUTH_USER=admin AUTH_HASH='$2a$14$abc/def.ghi'
}

GEN_SETUP_FNS=(gen_app_port_start gen_port_end gen_range_has gen_host_ports
    gen_port_start_problem settle_gen_ports prompt_gen_ports load_env_ports
    write_env_file compose_file compose_cmd generate_quadlet_app
    generate_quadlet_caddy caddy_gen_sites render_caddyfile chat_public_port)

# Answering no (the default) leaves .env, the compose files, the Quadlet
# units and the Caddyfile without any generation port.
test_gen_ports_off() (
    load_setup_fns "${GEN_SETUP_FNS[@]}"
    gen_setup_env
    prompt_gen_ports
    check_eq "gen off: stays off with --yes" "false" "$GEN_PORTS"
    write_env_file
    check_eq "gen off: .env" "LLAMA_TOOLCHEST_PORT=3000
LLAMA_TOOLCHEST_INFERENCE_PORT=8080
LLAMA_TOOLCHEST_CHAT_PUBLIC_PORT=8080
HOST_VIDEO_GID=39
HOST_RENDER_GID=105
ROCM_VARIANT=stable" "$(cat "$SCRIPT_DIR/.env")"
    check_eq "gen off: compose files" "docker compose -f docker-compose.rocm.yml" "$(compose_cmd)"
    check_false "gen off: no Quadlet range" grep -q '808[1-8]' <(generate_quadlet_app)
    check_eq "gen off: host ports" "" "$(gen_host_ports)"
    GPU_VENDOR=cpu GEN_PORTS=true GEN_PORTS_EXPLICIT=true
    prompt_gen_ports
    check_eq "gen off: never on a CPU install" "false" "$GEN_PORTS"
    GPU_VENDOR=rocm GEN_PORTS=false GEN_PORTS_EXPLICIT=false

    gen_secure_caddy
    SECURE_APP_DEBUG_INF_PORT=8081
    settle_gen_ports
    check_eq "gen off: secure debug port stays" "8081" "$SECURE_APP_DEBUG_INF_PORT"
    render_caddyfile
    check_false "gen off: no Caddy gen sites" grep -q 'llama-toolchest:808[1-8]' "$SCRIPT_DIR/Caddyfile"
    check_eq "gen off: Caddyfile ends with the chat site" "}" "$(tail -n 1 "$SCRIPT_DIR/Caddyfile")"
    check_false "gen off: no leftover token" grep -q '@@' "$SCRIPT_DIR/Caddyfile"
    check_eq "gen off: Caddy unit unchanged" "PublishPort=8080:8080" "$(generate_quadlet_caddy | grep '^PublishPort' | tail -n 1)"
    check_eq "gen off: secure compose files" \
        "docker compose -f docker-compose.rocm.yml -f docker-compose.secure.yml" "$(compose_cmd)"
)

# Answering yes publishes 8081-8088 (or the chosen start) everywhere.
test_gen_ports_on() (
    load_setup_fns "${GEN_SETUP_FNS[@]}"
    gen_setup_env
    GEN_PORTS=true GEN_PORTS_EXPLICIT=true
    prompt_gen_ports
    check_eq "gen on: default start" "8081" "$GEN_PUBLIC_PORT_START"
    write_env_file
    check_eq "gen on: .env keys" "LLAMA_TOOLCHEST_GEN_PORTS=1
LLAMA_TOOLCHEST_GEN_HOST_PORT_START=8081
LLAMA_TOOLCHEST_GEN_HOST_PORT_END=8088
LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START=8081" "$(grep GEN_ "$SCRIPT_DIR/.env")"
    check_eq "gen on: compose files" \
        "docker compose -f docker-compose.rocm.yml -f docker-compose.gen-ports.yml" "$(compose_cmd)"
    check_eq "gen on: Quadlet lines" "PublishPort=8081-8088:8081-8088
Environment=LLAMA_TOOLCHEST_GEN_PORTS=1
Environment=LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START=8081" \
        "$(generate_quadlet_app | grep -E '8081-8088|GEN_')"
    check_eq "gen on: host ports" "8081 8088" "$(gen_host_ports | sed -n '1p;$p' | paste -sd' ')"

    # A range that would hold the chat UI port is refused.
    check_eq "gen on: 8080 in the range" \
        "8075-8082 overlaps port 8080, used by the chat UI" "$(gen_port_start_problem 8075)"
    check_eq "gen on: range past 65535" \
        "the first port must be a number from 1024 to 65528" "$(gen_port_start_problem 65530)"
    check_eq "gen on: 9000 is fine" "" "$(gen_port_start_problem 9000)"
    GEN_PUBLIC_PORT_START=8075 GEN_PORT_START_EXPLICIT=true
    check_false "gen on: clashing --gen-port-start fails" eval '(prompt_gen_ports) 2>/dev/null'

    # The choice survives a rebuild through .env.
    GEN_PORTS=false GEN_PORTS_EXPLICIT=false GEN_PUBLIC_PORT_START=8081 GEN_PORT_START_EXPLICIT=false
    printf 'LLAMA_TOOLCHEST_GEN_PORTS=1\nLLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START=9001\n' > "$SCRIPT_DIR/.env"
    load_env_ports
    check_eq "gen on: reloaded from .env" "true 9001" "$GEN_PORTS $GEN_PUBLIC_PORT_START"

    # Switching a plain install to --secure must not take its public chat
    # port (8080) as the loopback debug port; a secure .env's value is kept.
    SECURE=true SECURE_EXPLICIT=true
    unset SECURE_APP_DEBUG_INF_PORT
    printf 'LLAMA_TOOLCHEST_PORT=3000\nLLAMA_TOOLCHEST_INFERENCE_PORT=8080\n' > "$SCRIPT_DIR/.env"
    load_env_ports
    check_eq "plain .env to secure: debug port not reloaded" "unset" "${SECURE_APP_DEBUG_INF_PORT:-unset}"
    printf 'LLAMA_TOOLCHEST_SECURE=1\nLLAMA_TOOLCHEST_INFERENCE_PORT=18080\n' > "$SCRIPT_DIR/.env"
    load_env_ports
    check_eq "secure .env: debug port reloaded" "18080" "${SECURE_APP_DEBUG_INF_PORT:-unset}"

    # Switching a secure install back to plain keeps the default chat port:
    # the secure .env's LLAMA_TOOLCHEST_INFERENCE_PORT is the debug port.
    SECURE=false SECURE_EXPLICIT=true LLAMA_TOOLCHEST_INFERENCE_PORT=8080
    load_env_ports
    check_eq "secure .env to plain: chat port stays 8080" "8080" "$LLAMA_TOOLCHEST_INFERENCE_PORT"
    printf 'LLAMA_TOOLCHEST_INFERENCE_PORT=8181\n' > "$SCRIPT_DIR/.env"
    load_env_ports
    check_eq "plain .env: chat port reloaded" "8181" "$LLAMA_TOOLCHEST_INFERENCE_PORT"
    SECURE_EXPLICIT=false
    printf 'LLAMA_TOOLCHEST_SECURE=true\n' > "$SCRIPT_DIR/.env"
    load_env_ports
    check_eq "secure .env remembered" "true" "$SECURE"
)

# Secure mode: Caddy publishes the public range behind the login, the app's
# copy goes to loopback 18081-18088, and its debug port leaves 8081.
test_gen_ports_secure() (
    load_setup_fns "${GEN_SETUP_FNS[@]}"
    gen_setup_env
    gen_secure_caddy
    SECURE_APP_DEBUG_INF_PORT=8081
    GEN_PORTS=true GEN_PORTS_EXPLICIT=true
    prompt_gen_ports
    check_eq "secure gen: debug port moves to 18080" "18080" "$SECURE_APP_DEBUG_INF_PORT"
    write_env_file
    check_eq "secure gen: .env" "LLAMA_TOOLCHEST_INFERENCE_PORT=18080
LLAMA_TOOLCHEST_GEN_PORTS=1
LLAMA_TOOLCHEST_GEN_HOST_PORT_START=18081
LLAMA_TOOLCHEST_GEN_HOST_PORT_END=18088
LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START=8081
CADDY_GEN_PORT_START=8081
CADDY_GEN_PORT_END=8088" "$(grep -E 'INFERENCE_PORT|GEN_' "$SCRIPT_DIR/.env")"
    check_eq "secure gen: compose files" \
        "docker compose -f docker-compose.rocm.yml -f docker-compose.secure.yml -f docker-compose.gen-ports.yml -f docker-compose.secure-gen-ports.yml" \
        "$(compose_cmd)"
    check_eq "secure gen: app Quadlet publishes on loopback" \
        "PublishPort=127.0.0.1:18080:8080
PublishPort=127.0.0.1:18081-18088:8081-8088" "$(generate_quadlet_app | grep -E '^PublishPort=127.0.0.1:180')"
    check_eq "secure gen: Caddy Quadlet publishes the public range" \
        "PublishPort=8081-8088:8081-8088" "$(generate_quadlet_caddy | grep '^PublishPort' | tail -n 1)"
    check_eq "secure gen: preflight ports" "16" "$(gen_host_ports | wc -l | tr -d ' ')"

    render_caddyfile
    local cf="$SCRIPT_DIR/Caddyfile"
    check_eq "secure gen: 8 Caddy sites" "8" "$(grep -c '^:808[1-8] {' "$cf")"
    check_eq "secure gen: last site forwards to 8088" \
        $':8088 {\n\ttls internal {\n\t\ton_demand\n\t}\n\tbasic_auth {\n\t\tadmin $2a$14$abc/def.ghi\n\t}\n\treverse_proxy llama-toolchest:8088\n}' \
        "$(sed -n '/^:8088 {/,/^}/p' "$cf")"
    check_true "secure gen: ACE-Step routes use the Bearer key" \
        grep -q '@v1 path /v1 /v1/\* /release_task /query_result' "$cf"
    check_false "secure gen: no leftover token" grep -q '@@' "$cf"

    # The public range may not reach into the app's loopback ports.
    check_eq "secure gen: 18075 refused" \
        "18075-18082 overlaps port 18080, used by the app's loopback ports (18080-18088)" "$(gen_port_start_problem 18075)"
    # Let's Encrypt sites carry the domain.
    CADDY_CHAT_ADDRESS="llm.example.test:8080" GEN_PUBLIC_PORT_START=9001
    check_eq "secure gen: domain sites" "llm.example.test:9001 {" "$(caddy_gen_sites | grep -m1 ' {$')"
    check_true "secure gen: moved range still forwards to 8081" \
        grep -q 'reverse_proxy llama-toolchest:8081' <(caddy_gen_sites)
)

# ─── Container install (setup.sh) ────────────────────────────────────────────

# A file the image build reads through a RUN bind mount is relabelled
# container_file_t when SELinux is on, and left alone otherwise.
test_label_for_build_mount() (
    load_setup_fns label_for_build_mount
    SCRIPT_DIR="${TMP_ROOT}/label"
    rm -rf "$SCRIPT_DIR"; mkdir -p "$SCRIPT_DIR/dist"
    touch "$SCRIPT_DIR/dist/llama-toolchest-local"
    local calls="${TMP_ROOT}/chcon.log"
    chcon() { echo "chcon $*" >> "$calls"; }

    : > "$calls"
    selinuxenabled() { return 0; }
    label_for_build_mount dist/llama-toolchest-local
    check_eq "label: relabelled with SELinux on" \
        "chcon -t container_file_t ${SCRIPT_DIR}/dist/llama-toolchest-local" "$(cat "$calls")"

    : > "$calls"
    label_for_build_mount ""
    label_for_build_mount dist/missing.rpm
    check_eq "label: nothing to relabel" "" "$(cat "$calls")"

    selinuxenabled() { return 1; }
    label_for_build_mount dist/llama-toolchest-local
    check_eq "label: SELinux off" "" "$(cat "$calls")"

    # A failed chcon only warns: the build may still work.
    selinuxenabled() { return 0; }
    chcon() { return 1; }
    check_true "label: chcon failure is not fatal" label_for_build_mount dist/llama-toolchest-local
)

# setup.sh quick offers a full rebuild when the latest release raises the
# image revision above the running image's, and stops when the local
# Dockerfile is too old to build the new image.
test_quick_needs_rebuild() (
    load_setup_fns image_revision_from_dockerfile release_image_revision \
        running_image_revision check_quick_needs_rebuild
    IMAGE_REVISION_LABEL="io.github.tmac1973.llama-toolchest.image-revision"
    SCRIPT_DIR="${TMP_ROOT}/quick"
    rm -rf "$SCRIPT_DIR"; mkdir -p "$SCRIPT_DIR"
    LLAMA_TOOLCHEST_PORT=3000 ASSUME_YES=false INTERACTIVE=true
    dockerfile() { echo Dockerfile.rocm; }
    container_exists() { [[ "$HAVE_CONTAINER" == true ]]; }
    CONTAINER_CMD=fake_podman
    fake_podman() {
        case "$1 $2" in
            "container inspect") echo sha256:abc ;;
            "image inspect") echo "$IMAGE_REV" ;;
        esac
    }
    curl() { [[ -n "$RELEASE_REV" ]] || return 22; dockerfile_text "$RELEASE_REV"; }
    dockerfile_text() {
        printf 'FROM fedora:43\n\nLABEL %s="%s"\nARG X=1\n' "$IMAGE_REVISION_LABEL" "$1"
    }
    local_rev() { dockerfile_text "$1" > "$SCRIPT_DIR/Dockerfile.rocm"; }
    container_rebuild() { echo rebuilt; }
    prompt_confirm() { [[ "$ANSWER" == yes ]]; }

    # outcome: "go" when quick continues, else the exit status plus any
    # rebuild marker.
    outcome() {
        local out rc
        out="$(check_quick_needs_rebuild 2.0.0 2>/dev/null; echo "go")"; rc=$?
        out="$(grep -E '^(go|rebuilt)$' <<<"$out" | tr '\n' ' ')"
        out="${out% }"; echo "${out:+$out }rc=$rc"
    }

    check_eq "revision: parsed from Dockerfile" "7" "$(dockerfile_text 7 | image_revision_from_dockerfile)"
    check_eq "revision: none in old Dockerfile" "" "$(printf 'FROM x\nLABEL other="3"\n' | image_revision_from_dockerfile)"
    for f in "${REPO_DIR}"/Dockerfile.*; do
        check_true "revision: $(basename "$f") has a revision" \
            test -n "$(image_revision_from_dockerfile < "$f")"
    done

    HAVE_CONTAINER=true RELEASE_REV=2 IMAGE_REV=2 ANSWER=yes; local_rev 2
    check_eq "quick: image current" "go rc=0" "$(outcome)"
    IMAGE_REV=3
    check_eq "quick: image newer than release" "go rc=0" "$(outcome)"
    RELEASE_REV=""
    check_eq "quick: release revision unreadable" "go rc=0" "$(outcome)"
    RELEASE_REV=3 HAVE_CONTAINER=false
    check_eq "quick: no container yet" "go rc=0" "$(outcome)"

    HAVE_CONTAINER=true IMAGE_REV=2 RELEASE_REV=3; local_rev 3
    check_eq "quick: older image, rebuild accepted" "rebuilt rc=0" "$(outcome)"
    ANSWER=no
    check_eq "quick: older image, rebuild declined" "rc=1" "$(outcome)"
    IMAGE_REV="<no value>" ANSWER=yes
    check_eq "quick: unlabelled image offers rebuild" "rebuilt rc=0" "$(outcome)"
    IMAGE_REV=2; local_rev 2
    check_eq "quick: local Dockerfile too old" "rc=1" "$(outcome)"
)

# Host ports other than 443/8080 for Caddy only change what is published:
# Caddy keeps listening on 443 and 8080 inside its container, and the
# external URL carries the host port.
test_secure_custom_ports() (
    load_setup_fns "${GEN_SETUP_FNS[@]}" configure_secure secure_chat_url
    gen_setup_env
    print_secure_disclaimer() { :; }
    SECURE=true SECURE_EXPLICIT=true TLS_MODE=self-signed DOMAIN=box ACME_EMAIL=""
    AUTH_USER=admin AUTH_HASH='$2a$14$abc/def.ghi' AUTH_PASS="" AUTH_PASS_FILE=""
    CADDY_HTTP_PORT=8000 CADDY_HTTPS_PORT=8443 CADDY_CHAT_PORT=9443
    configure_secure
    local cf="$SCRIPT_DIR/Caddyfile"
    check_eq "custom ports: Caddy sites" ":443 {
:8080 {" "$(grep -E '^[^#[:space:]].* \{$' "$cf")"
    check_eq "custom ports: external URL" "https://box:8443" "$SECURE_EXTERNAL_URL"
    check_eq "custom ports: chat URL" "https://box:9443" "$(secure_chat_url)"
    # The app's own chat link gets Caddy's chat port, not the app's 8080
    # or its loopback debug port.
    SECURE_APP_DEBUG_INF_PORT=18080
    write_env_file
    check_eq "custom ports: chat link port in .env" "LLAMA_TOOLCHEST_CHAT_PUBLIC_PORT=9443" \
        "$(grep CHAT_PUBLIC "$SCRIPT_DIR/.env")"
    check_eq "custom ports: chat link port in Quadlet" "Environment=LLAMA_TOOLCHEST_CHAT_PUBLIC_PORT=9443" \
        "$(generate_quadlet_app | grep CHAT_PUBLIC)"
    SECURE=false LLAMA_TOOLCHEST_INFERENCE_PORT=9080
    check_eq "plain: chat link port is the published port" "Environment=LLAMA_TOOLCHEST_CHAT_PUBLIC_PORT=9080" \
        "$(generate_quadlet_app | grep CHAT_PUBLIC)"
    SECURE=true
    check_eq "custom ports: Caddy Quadlet" "PublishPort=8000:80
PublishPort=8443:443
PublishPort=9443:8080" "$(generate_quadlet_caddy | grep '^PublishPort')"
    check_true "custom ports: compose maps the HTTPS port to 443" \
        grep -qF '"${CADDY_HTTPS_PORT:-443}:443"' "${REPO_DIR}/docker-compose.secure.yml"
    check_true "custom ports: compose maps the chat port to 8080" \
        grep -qF '"${CADDY_CHAT_PORT:-8080}:8080"' "${REPO_DIR}/docker-compose.secure.yml"
    # Caddy's chat site holds 8080 in its container even with the host
    # chat port moved, so a generation range there is refused.
    check_eq "custom ports: gen range over Caddy's 8080" \
        "8075-8082 overlaps port 8080, used by the chat UI inside the Caddy container" \
        "$(gen_port_start_problem 8075)"

    TLS_MODE=letsencrypt DOMAIN=llm.example.test ACME_EMAIL=admin@example.test
    configure_secure
    check_eq "custom ports: Let's Encrypt sites" "llm.example.test {
llm.example.test:8080 {" "$(grep -E '^[^#[:space:]].* \{$' "$cf" | grep -v '^{$')"
    check_eq "custom ports: Let's Encrypt URL" "https://llm.example.test:8443" "$SECURE_EXTERNAL_URL"

    CADDY_HTTPS_PORT=443 CADDY_CHAT_PORT=8080
    configure_secure
    check_eq "default ports: URL has no port" "https://llm.example.test" "$SECURE_EXTERNAL_URL"
    check_eq "default ports: chat URL" "https://llm.example.test:8080" "$(secure_chat_url)"
    SECURE_EXTERNAL_URL='https://[fd00::1]'
    check_eq "chat URL: IPv6 host without a port" "https://[fd00::1]:8080" "$(secure_chat_url)"
)

# An old-named config in the current volume is renamed to the name the
# image's ENTRYPOINT reads. The fake runtime runs the script on a folder
# standing in for the volume (fake_vol: the function has its own vol).
test_migrate_legacy_config_name() (
    load_setup_fns migrate_legacy_config_name
    local fake_vol="${TMP_ROOT}/vol" calls="${TMP_ROOT}/runtime.log"
    local msgs="${TMP_ROOT}/config-name.log"
    ok()   { echo "ok: $*" >> "$msgs"; }
    warn() { echo "warn: $*" >> "$msgs"; }
    HAVE_VOLUME=true
    fake_runtime() {
        echo "$*" >> "$calls"
        case "$1 $2" in
            "volume inspect") [[ "$HAVE_VOLUME" == true ]] ;;
            "run --rm") local script="${!#}"; sh -c "${script//\/data/$fake_vol}" ;;
            *) return 1 ;;
        esac
    }
    CONTAINER_CMD=fake_runtime
    reset() { rm -rf "$fake_vol" "$calls" "$msgs"; mkdir -p "$fake_vol/config"; : > "$calls"; : > "$msgs"; }

    reset
    echo 'data_dir: "/data"' > "$fake_vol/config/llamactl.yaml"
    migrate_legacy_config_name
    check_eq "config name: renamed" 'data_dir: "/data"' "$(cat "$fake_vol/config/llama-toolchest.yaml" 2>/dev/null)"
    check_false "config name: old name gone" test -e "$fake_vol/config/llamactl.yaml"
    check_true "config name: says so" grep -q '^ok: Renamed config/llamactl.yaml' "$msgs"
    check_true "config name: mounts the current volume" grep -q -- '-v llama-toolchest-data:/data' "$calls"

    # Run again: nothing left to do.
    : > "$msgs"
    migrate_legacy_config_name
    check_eq "config name: second run is quiet" "" "$(cat "$msgs")"

    reset
    echo old > "$fake_vol/config/llamactl.yaml"
    echo new > "$fake_vol/config/llama-toolchest.yaml"
    migrate_legacy_config_name
    check_eq "config name: both kept" "old new" \
        "$(cat "$fake_vol/config/llamactl.yaml") $(cat "$fake_vol/config/llama-toolchest.yaml")"
    check_true "config name: both warns" grep -q '^warn: .*has both' "$msgs"

    reset
    rm -rf "$fake_vol/config"
    migrate_legacy_config_name
    check_eq "config name: no config folder" "" "$(cat "$msgs")"

    reset
    HAVE_VOLUME=false
    migrate_legacy_config_name
    check_eq "config name: no volume, no container run" "volume inspect llama-toolchest-data" "$(cat "$calls")"
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
test_host_gen_port_start
test_write_unit_override
test_migrate_config
test_migrate_model_paths
test_rocm10_repo_dist
test_rocm10_install_state
test_rocm10_refused_beside_old_amd
test_rhel_rocm
test_choose_rocm
test_rocm10_not_old_line
test_gfx_target_from_kfd
test_gen_ports_off
test_gen_ports_on
test_gen_ports_secure
test_label_for_build_mount
test_quick_needs_rebuild
test_secure_custom_ports
test_migrate_legacy_config_name

passes="$(grep -c '^PASS$' "$RESULTS")"
fails="$(grep -c '^FAIL$' "$RESULTS")"
echo "${passes} passed, ${fails} failed"
if [[ "$fails" -ne 0 ]]; then
    exit 1
fi
echo "ALL PASS"
